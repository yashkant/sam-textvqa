import torch

from tools.registry import registry


class BeamSearch:
    def __init__(self, beam_size):
        # Lists to store completed sequences and scores
        self._decode_size = beam_size
        self._complete_seqs = []
        self._complete_seqs_scores = []
        self._EOS_IDX = registry.EOS_IDX
        self.completed_ids = None
        self.batch_dict_keys = [
            "pad_obj_features",
            "pad_obj_bboxes",
            "ocr_fasttext",
            "ocr_phoc",
            "pad_ocr_features",
            "pad_ocr_bboxes",
            "question_indices",
            "question_mask",
            "pad_obj_mask",
            "pad_ocr_mask",
            "spatial_adj_matrices",
            "ocr_mmt_in",
            "obj_mmt_in",
            "question_id",
        ]

    def init_batch(self, batch_dict):
        self._complete_seqs = []
        self._complete_seqs_scores = []
        self._EOS_IDX = registry.EOS_IDX
        self.completed_ids = None

        self._batch_size = batch_dict["train_prev_inds"].shape[0]

        self._offset_mat = (
            torch.arange(0, self._batch_size).repeat_interleave(
                self._decode_size, dim=0
            )
            * self._decode_size
        )

        setattr(
            self,
            "top_k_scores",
            batch_dict["train_prev_inds"].new_zeros(
                (self._batch_size * self._decode_size, 1), dtype=torch.float
            ),
        )

        self.seqs = batch_dict["train_prev_inds"].new_full(
            (self._batch_size * self._decode_size, 1),
            registry.BOS_IDX,
            dtype=torch.long,
        )

        batch_dict["topkscores"] = (
            batch_dict["train_prev_inds"]
            .new_full((self._batch_size * self._decode_size, 1), 0.0)
            .detach()
        )
        # Expand all the features
        # Make it [batch_size * beam_size] * ........
        # each element of batch is repeated beam_size number of time and interleaved.
        # a tensor of bs=2 & dim=1 [1, 2] will become [1, 1, 2, 2] of size 4 and dim=1

        for key in self.batch_dict_keys + ["train_prev_inds"]:
            if key in batch_dict:
                if isinstance(batch_dict[key], dict):
                    for k, v in batch_dict[key].items():
                        batch_dict[key][k] = batch_dict[key][k].repeat_interleave(
                            self._decode_size, dim=0
                        )
                else:
                    batch_dict[key] = batch_dict[key].repeat_interleave(
                        self._decode_size, dim=0
                    )
        return batch_dict

    def decode(self, batch_dict, t):
        vocab_size = batch_dict["scores"].shape[-1]
        current_scores = torch.log(torch.sigmoid(batch_dict["scores"][:, t, :]))

        if self.completed_ids is not None:
            current_scores[self.completed_ids, :] = -float("Inf")
            current_scores[
                self.completed_ids, self._EOS_IDX
            ] = 0  # make EOS probability highest.

        current_scores += batch_dict["topkscores"].expand_as(current_scores)

        # If time is zero, then need to look into only one beam for one image
        if t == 0:
            ignore_ids = (
                (torch.arange(0, self._batch_size) * self._decode_size).view(-1, 1)
                + torch.arange(1, self._decode_size).view(1, -1)
            ).view(-1)

            current_scores[ignore_ids, :] = -float("Inf")

        topkscores = current_scores.reshape(self._batch_size, -1)
        value, indices = topkscores.topk(
            self._decode_size, dim=-1, largest=True, sorted=True
        )

        prev_position = indices / vocab_size
        new_position = indices % vocab_size

        prev_position = prev_position.view(-1) + self._offset_mat.to(
            prev_position.device
        )
        new_position = new_position.view(-1)

        # New positions and prev_positions found!
        # Build input index:
        batch_dict["train_prev_inds"] = self.add_next_word(
            batch_dict["train_prev_inds"], prev_position, new_position, t
        )
        # Need to store all the scores
        batch_dict["topkscores"] = batch_dict["topkscores"][prev_position] + value.view(
            -1
        ).unsqueeze(1)

        # Build data for next round
        for key in self.batch_dict_keys:
            if isinstance(batch_dict[key], dict):
                for k, v in batch_dict[key].items():
                    batch_dict[key][k] = batch_dict[key][k][prev_position]
            else:
                batch_dict[key] = batch_dict[key][prev_position]

        # Find complete sequences for each image in the batch!
        if t + 1 < batch_dict["train_prev_inds"].shape[1]:
            self.completed_ids = self.find_complete_inds(
                batch_dict["train_prev_inds"], t + 1
            )
        else:
            # no beam got completed!
            self.completed_ids = torch.arange(batch_dict["train_prev_inds"].shape[0])

        finish = False
        if (len(self.completed_ids) == self._batch_size * self._decode_size) or (
            batch_dict["train_prev_inds"].shape[1] == t + 1
        ):
            # Save completed sequences!
            batch_dict["complete_seqs"] = batch_dict["train_prev_inds"][
                self.completed_ids, :
            ]
            finish = True

        return finish, batch_dict, 0

    def find_complete_inds(self, seqs, t):
        completed_ids = (seqs[:, t] == self._EOS_IDX).nonzero()
        return completed_ids

    def get_result(self):
        if len(self._complete_seqs_scores) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            i = self._complete_seqs_scores.index(max(self._complete_seqs_scores))
            captions = torch.FloatTensor(self._complete_seqs[i]).unsqueeze(0)
        return captions

    def add_next_word(self, seqs, prev_word_inds, next_word_inds, t):
        new_seqs = seqs[prev_word_inds]
        if t + 1 < new_seqs.shape[1]:
            new_seqs[:, t + 1] = next_word_inds
        return new_seqs

    def update_data(self, data, prev_word_inds, next_word_inds, incomplete_inds):
        data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)
        h1 = data["state"]["td_hidden"][0][prev_word_inds[incomplete_inds]]
        c1 = data["state"]["td_hidden"][1][prev_word_inds[incomplete_inds]]
        h2 = data["state"]["lm_hidden"][0][prev_word_inds[incomplete_inds]]
        c2 = data["state"]["lm_hidden"][1][prev_word_inds[incomplete_inds]]
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data
