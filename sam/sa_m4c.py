import functools
import logging
import math
from collections import Counter

import torch
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertEncoder, BertLayerNorm,
                                                BertPreTrainedModel)
from torch import nn

from sam.beam_search import BeamSearch
from sam.textvqa_encoders import ImageEncoder
from tools.registry import registry

logger = logging.getLogger(__name__)


class SAM4C(nn.Module):
    """
    SAM4C has two transfomers MMT and TextBert.
    """

    def __init__(self, mmt_config, text_bert_config):
        super().__init__()
        self.mmt_config = mmt_config
        self.text_bert_config = text_bert_config
        self.frcn_encoder_type = "default"
        if not self.mmt_config.use_phoc_fasttext:
            logger.info("Not using Fasttext and PHOC features for OCR")

        self.normalize = self.mmt_config.normalize
        if not self.mmt_config.normalize:
            logger.info("Not normalizing OCR and Object features")

        # auxiliary heads and fusion
        self.aux_spatial_fusion = getattr(self.mmt_config, "aux_spatial_fusion", "mul")
        self.use_aux_heads = getattr(self.mmt_config, "use_aux_heads", False)
        if self.use_aux_heads:
            logger.info("Using spatial-aux heads")
            logger.info(f"Spatial aux fusion type: {self.aux_spatial_fusion}")
        else:
            logger.info("Not using spatial-aux heads")

        # type of spatial-layers
        self.spatial_type = getattr(self.mmt_config, "spatial_type", "top")
        logger.info(f"Using {self.spatial_type} type spatial layers")

        # build the models
        self.build()

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size
        self.bsdecoder = BeamSearch(self.beam_size)
        logger.info(f"Using beam size: {self.beam_size}")

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()
        if self.use_aux_heads:
            self._build_aux_heads()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768
        # self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.text_bert_config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                "bert-base-uncased", config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {
                    "module": self.text_bert,
                    "lr_scale": self.text_bert_config.lr_scale_text_bert,
                }
            )
        else:
            logger.info("NOT initializing text_bert from BERT_BASE")
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                "Projecting text_bert output to {} dim".format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        # (YK) Todo: support for last-layer finetuning
        assert self.frcn_encoder_type == "default"
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type=self.frcn_encoder_type,
            in_dim=2048,
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        # self.finetune_modules.append({
        #     'module': self.obj_faster_rcnn_fc7,
        #     'lr_scale': self.config.lr_scale_frcn,
        # })
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.mmt_config.obj_feature_size, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.mmt_config.obj_drop)

    def _build_ocr_encoding(self):

        # (YK): Todo
        assert self.frcn_encoder_type == "default"
        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type=self.frcn_encoder_type,
            in_dim=2048,
            weights_file="detectron/fc6/fc7_w.pkl",
            bias_file="detectron/fc6/fc7_b.pkl",
            model_data_dir=None,
        )
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.mmt_config.ocr_feature_size, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.mmt_config.ocr_drop)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append(
            {
                "module": self.mmt,
                "lr_scale": self.mmt_config.lr_scale_mmt,
            }
        )

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(
            hidden_size=self.mmt_config.hidden_size,
            query_key_size=self.mmt_config.ptr_query_size,
        )
        num_outputs = len(registry.answer_vocab)
        logger.info(f"Answer vocab-size is: {num_outputs}")
        self.classifier = nn.Linear(self.mmt_config.hidden_size, num_outputs)

    def _build_aux_heads(self):
        # spatial-category classification head
        self.origin_transform = SimpleClassifier(self.mmt_config.hidden_size, 128, 32)
        self.dest_transform = SimpleClassifier(self.mmt_config.hidden_size, 128, 32)
        self.spatial_classifier = nn.Linear(32, 12)

    def forward(self, batch_dict, use_beam_search=False):
        """Main forward method"""
        self._forward_obj_encoding(batch_dict)
        self._forward_ocr_encoding(batch_dict)

        if not use_beam_search:
            self._forward_mmt_and_output(batch_dict)
        else:
            self._forward_beam_search(batch_dict)

        if self.use_aux_heads:
            self._forward_aux(batch_dict)

        results_dict = {
            "textvqa_scores": batch_dict["scores"],
            # "spatial_scores": None if not self.use_aux_heads else batch_dict["spatial_head_out"]
        }

        if "complete_seqs" in batch_dict:
            results_dict["complete_seqs"] = batch_dict["complete_seqs"].squeeze()
            results_dict["topkscores"] = batch_dict["topkscores"].squeeze()
            results_dict["question_id"] = batch_dict["question_id"].squeeze()

        return results_dict

    def _forward_obj_encoding(self, batch_dict):
        # object appearance feature: Faster R-CNN fc7
        obj_fc7 = self.obj_faster_rcnn_fc7(batch_dict["pad_obj_features"])

        if self.normalize:
            obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7

        # remove bbox-area
        obj_bbox = batch_dict["pad_obj_bboxes"][:, :, :-1]
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        batch_dict["obj_mmt_in"] = obj_mmt_in

    def _forward_ocr_encoding(self, batch_dict):
        # OCR FastText feature (300-dim)
        ocr_fasttext = batch_dict["ocr_fasttext"]
        if self.normalize:
            ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = batch_dict["ocr_phoc"]
        if self.normalize:
            ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = batch_dict["pad_ocr_features"]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        if self.normalize:
            ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = ocr_fc6.new_zeros((ocr_phoc.size(0), 50, 50))

        if self.mmt_config.use_phoc_fasttext:
            ocr_feat = torch.cat(
                [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors], dim=-1
            )
        else:
            ocr_feat = torch.cat([ocr_fc7, ocr_order_vectors], dim=-1)

        # remove area
        ocr_bbox = batch_dict["pad_ocr_bboxes"][:, :, :-1]
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        batch_dict["ocr_mmt_in"] = ocr_mmt_in

    def _forward_mmt(self, batch_dict):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(batch_dict)
        batch_dict["text_bert_emb"] = self.text_bert_out_linear(text_bert_out)

        mmt_results = self.mmt(
            batch_dict,
            fixed_ans_emb=self.classifier.weight,
        )
        batch_dict.update(mmt_results)

    def _forward_output(self, batch_dict):
        mmt_dec_output = batch_dict["mmt_dec_output"]
        mmt_ocr_output = batch_dict["mmt_ocr_output"]
        ocr_mask = batch_dict["pad_ocr_mask"]

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        batch_dict["scores"] = scores

    def _forward_mmt_and_output(self, batch_dict):
        if self.training:
            # fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(batch_dict)
            self._forward_output(batch_dict)
        else:
            dec_step_num = batch_dict["train_prev_inds"].size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            batch_dict["train_prev_inds"] = torch.zeros_like(
                batch_dict["train_prev_inds"]
            )
            batch_dict["train_prev_inds"][:, 0] = registry.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(batch_dict)
                self._forward_output(batch_dict)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = batch_dict["scores"].argmax(dim=-1)
                batch_dict["train_prev_inds"][:, 1:] = argmax_inds[:, :-1]

    def _forward_beam_search(self, batch_dict):
        dec_step_num = batch_dict["train_prev_inds"].size(1)
        # Decoding with beam search
        batch_dict = self.bsdecoder.init_batch(batch_dict)

        for t in range(dec_step_num):
            self._forward_mmt(batch_dict)
            self._forward_output(batch_dict)
            finish, batch_dict, batch_size_t = self.bsdecoder.decode(batch_dict, t)
            if finish:
                break

    def _forward_aux(self, batch_dict):
        txt_max_num = batch_dict["question_mask"].size(-1)
        obj_max_num = batch_dict["pad_obj_mask"].size(-1)
        ocr_max_num = batch_dict["pad_ocr_mask"].size(-1)
        mmt_output = batch_dict["mmt_seq_output"]
        obj_ocr_output = mmt_output[
            :, txt_max_num : txt_max_num + obj_max_num + ocr_max_num, :
        ]

        # shape: (bs, obj_ocr_num, hid_dim)
        obj_ocr_origin_transform = self.origin_transform(obj_ocr_output)
        # shape: (bs, obj_ocr_num, obj_ocr_num, hid_dim)
        obj_ocr_origin_transform = obj_ocr_origin_transform.unsqueeze(-2).repeat(
            1, 1, 150, 1
        )

        # shape: (bs, obj_ocr_num, hid_dim)
        obj_ocr_dest_transform = self.dest_transform(obj_ocr_output)
        # shape: (bs, obj_ocr_num, obj_ocr_num, hid_dim)
        obj_ocr_dest_transform = obj_ocr_dest_transform.unsqueeze(-3).repeat(
            1, 150, 1, 1
        )

        # Add and average the features or Multiply
        if self.aux_spatial_fusion == "mul":
            spatial_head_out = obj_ocr_origin_transform * obj_ocr_dest_transform
        elif self.aux_spatial_fusion == "add":
            spatial_head_out = obj_ocr_origin_transform + obj_ocr_dest_transform
        else:
            raise ValueError

        batch_dict["spatial_head_out"] = self.spatial_classifier(spatial_head_out)

    def get_optimizer_parameters(self, base_lr):
        optimizer_param_groups = []

        # base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, batch_dict):
        encoder_inputs = self.embeddings(batch_dict["question_indices"])
        attention_mask = batch_dict["question_mask"]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class SpatialBertSelfAttention(nn.Module):
    """
    Todo: Keep 768 and build zero-mask for 12th head (not needed with identity relation)
    """

    def __init__(self, config, use_implicit=False):
        super(SpatialBertSelfAttention, self).__init__()
        assert hasattr(config, "num_spatial_relations")

        self.num_attention_heads = config.num_spatial_relations
        self.num_spatial_relations = config.num_spatial_relations

        if hasattr(config, "num_implicit_relations") and use_implicit:
            self.num_attention_heads += config.num_implicit_relations
            self.num_implicit_relations = config.num_implicit_relations

        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, self.num_attention_heads)
            )

        self.output_attentions = config.output_attentions
        self.max_seq_len = config.max_seq_length
        self.mask_quadrants = config.attention_mask_quadrants
        self.max_decoding_steps = config.num_decoding_steps

        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if hasattr(config, "no_drop") and config.no_drop:
            logger.info("not using dropout")
            self.dropout = nn.Dropout(0.0)

        self.use_bias = False
        if hasattr(config, "use_bias") and config.use_bias:
            self.use_bias = True
            logger.info("using head biases")
            self.biases = torch.nn.Embedding(1, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, attention_mask, spatial_adj_matrix, head_mask=None
    ):
        """
        spatial_adj_matrix: (bs, num_ocr + num_obj, num_ocr + num_obj, type_relations == num_heads)
        TODO: Is there way you can draw some connections across these heads? Like one-head using some
            other heads K/Q/V, can this modelled as any cross-relationship, think think.

        One Problem: We concatenate the outputs from all the 12 heads, now:
            - Leave others as zeros? (this will happen if there's no rel_i th of this embedding with others) (as it is) [rel_enc1, rel_enc2, ... rel_enc11]
            - Balancing these features might be an issue if I have these zeros popping in at places.
            - Also balancing them even when they aren't zero? Like you had 12 heads contributing before,
                but now the heads might be anywhere from 1-12.

        spatial_adj_matrix has 0s across the diagonal terms, attaching an identity matrix for this

        """
        # build attention-mask from spatial_adj_matrix
        batch_size, ocr_obj_num, _, num_spatial_heads = spatial_adj_matrix.shape
        num_features = hidden_states.size(1)

        # Removing masking all quadrants
        spatial_attention_mask = attention_mask.new_ones(
            (batch_size, num_features, num_features, num_spatial_heads)
        )

        # Add explicit mask
        spatial_attention_mask[
            :,
            self.max_seq_len : self.max_seq_len + ocr_obj_num,
            self.max_seq_len : self.max_seq_len + ocr_obj_num,
            :,
        ] = spatial_adj_matrix

        # Add implicit mask
        if self.num_attention_heads != self.num_spatial_relations:
            assert hasattr(self, "num_implicit_relations")
            implicit_attention_mask = attention_mask.new_ones(
                (batch_size, num_features, num_features, self.num_implicit_relations)
            )
            spatial_attention_mask = torch.cat(
                [spatial_attention_mask, implicit_attention_mask], dim=-1
            )

        assert spatial_attention_mask.shape == (
            batch_size,
            num_features,
            num_features,
            self.num_attention_heads,
        )

        # Mask attention-quadrants (spatial relations only)
        for quadrant in self.mask_quadrants:
            if quadrant == 1:
                spatial_attention_mask[
                    :,
                    : self.max_seq_len,
                    : self.max_seq_len,
                    : self.num_spatial_relations,
                ] = 0
            elif quadrant == 2:
                spatial_attention_mask[
                    :,
                    : self.max_seq_len,
                    self.max_seq_len : self.max_seq_len + ocr_obj_num,
                    : self.num_spatial_relations,
                ] = 0
            elif quadrant == 4:
                spatial_attention_mask[
                    :,
                    self.max_seq_len : self.max_seq_len + ocr_obj_num,
                    : self.max_seq_len,
                    : self.num_spatial_relations,
                ] = 0
            elif quadrant == 7:
                spatial_attention_mask[
                    :,
                    self.max_seq_len + ocr_obj_num :,
                    : self.max_seq_len,
                    : self.num_spatial_relations,
                ] = 0
            elif quadrant == 8:
                spatial_attention_mask[
                    :,
                    self.max_seq_len + ocr_obj_num :,
                    self.max_seq_len : self.max_seq_len + ocr_obj_num,
                    : self.num_spatial_relations,
                ] = 0
            elif quadrant == 9:
                spatial_attention_mask[
                    :,
                    self.max_seq_len + ocr_obj_num :,
                    self.max_seq_len + ocr_obj_num :,
                    : self.num_spatial_relations,
                ] = 0
            else:
                raise ValueError

        spatial_attention_mask = (1.0 - spatial_attention_mask) * -10000.0
        spatial_attention_mask = spatial_attention_mask.permute((0, 3, 1, 2))

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        try:
            # Prevent imbalanced masking
            combined_mask = torch.min(attention_mask, spatial_attention_mask)
            assert len(torch.unique(combined_mask)) == 2
        except:
            print(f"am: {attention_mask.shape} | sm: {spatial_attention_mask.shape}")

        # for entities that are totally masked
        entity_probs_mask = (combined_mask.max(dim=-1)[0] + 10000.0) / 10000.0
        entity_probs_mask = entity_probs_mask.unsqueeze(-1)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + combined_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # zero-out completely masked entities
        attention_probs = attention_probs * entity_probs_mask

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.use_bias:
            context_layer = context_layer + self.biases(
                context_layer.new_zeros(1).long()
            )

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class SpatialBertAttention(nn.Module):
    def __init__(self, config, use_implicit=False):
        super(SpatialBertAttention, self).__init__()
        self.self = SpatialBertSelfAttention(config, use_implicit)
        from pytorch_transformers.modeling_bert import BertSelfOutput

        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = (
            set(heads) - self.pruned_heads
        )  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, spatial_adj_matrix, head_mask=None):
        self_outputs = self.self(
            input_tensor, attention_mask, spatial_adj_matrix, head_mask
        )
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class SpatialBertLayer(nn.Module):
    def __init__(self, config, use_implicit=False):
        super(SpatialBertLayer, self).__init__()
        from pytorch_transformers.modeling_bert import (BertIntermediate,
                                                        BertOutput)

        self.attention = SpatialBertAttention(config, use_implicit)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self, hidden_states, attention_mask, spatial_adj_matrix, head_mask=None
    ):
        attention_outputs = self.attention(
            hidden_states, attention_mask, spatial_adj_matrix, head_mask
        )
        attention_output = attention_outputs[0]
        # Intermediate is dense + activation
        intermediate_output = self.intermediate(attention_output)
        # Output is dense + dropout + residual + layernorm
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertSpatialEncoder(nn.Module):
    def __init__(self, config):
        super(BertSpatialEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer_type_list = config.layer_type_list
        logger.info(f"Layers type list is: {self.layer_type_list}")
        counter = Counter(self.layer_type_list)

        self.num_spatial_layers = counter["s"]
        self.num_normal_layers = counter["n"]
        self.num_implicit_layers = counter["i"]

        logger.info(f"Num Spatial Layers: {self.num_spatial_layers}")
        logger.info(f"Num Normal Layers: {self.num_normal_layers}")
        logger.info(f"Num Implicit Layers: {self.num_implicit_layers}")

        if not hasattr(config, "mix_list") or config.mix_list is None:
            self.mix_list = ["none"] * len(self.layer_type_list)
        else:
            self.mix_list = config.mix_list
        assert len(self.mix_list) == len(self.layer_type_list)
        logger.info(f"Mix list: {self.mix_list}")
        self.matrix_type_map = {
            "none": "1",
            "share3": "3",
            "share5": "5",
            "share7": "7",
            "share9": "9",
        }

        from pytorch_transformers.modeling_bert import BertLayer

        self.normal_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_normal_layers)]
        )
        self.spatial_layers = nn.ModuleList(
            [SpatialBertLayer(config) for _ in range(self.num_spatial_layers)]
        )
        self.implicit_layers = nn.ModuleList(
            [SpatialBertLayer(config, True) for _ in range(self.num_implicit_layers)]
        )

    def forward(self, hidden_states, attention_mask, batch_dict, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        normal_iter = iter(self.normal_layers)
        spatial_iter = iter(self.spatial_layers)
        implicit_iter = iter(self.implicit_layers)

        for layer_type, mix_type in zip(self.layer_type_list, self.mix_list):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if layer_type == "n":
                layer_module = next(normal_iter)
                layer_outputs = layer_module(hidden_states, attention_mask)
            elif layer_type == "s":
                layer_module = next(spatial_iter)
                matrix_type = self.matrix_type_map[mix_type]
                spatial_adj_matrix = batch_dict["spatial_adj_matrices"][matrix_type]
                layer_outputs = layer_module(
                    hidden_states, attention_mask, spatial_adj_matrix
                )
            else:
                raise ValueError
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        assert next(normal_iter, None) is None
        assert next(spatial_iter, None) is None
        assert next(implicit_iter, None) is None

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertSpatialEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, batch_dict, fixed_ans_emb):
        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(
            fixed_ans_emb, batch_dict["ocr_mmt_in"], batch_dict["train_prev_inds"]
        )

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.long, device=dec_emb.device
        )
        encoder_inputs = torch.cat(
            [
                batch_dict["text_bert_emb"],
                batch_dict["obj_mmt_in"],
                batch_dict["ocr_mmt_in"],
                dec_emb,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                batch_dict["question_mask"],
                batch_dict["pad_obj_mask"],
                batch_dict["pad_ocr_mask"],
                dec_mask,
            ],
            dim=1,
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = batch_dict["question_mask"].size(-1)
        obj_max_num = batch_dict["pad_obj_mask"].size(-1)
        ocr_max_num = batch_dict["pad_ocr_mask"].size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(
            dec_max_num, encoder_inputs.device
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, batch_dict, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size

        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        # default value of 0.1 is used
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ocr_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


try:
    from apex.normalization.fused_layer_norm import \
        FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root)."""
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)
