import logging
import multiprocessing as mp
import os

import _pickle as cPickle
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from tqdm import tqdm

from sam.spatial_utils import torch_broadcast_adj_matrix
from tools.objects_to_byte_tensor import enc_obj2bytes

from ._image_features_reader import ImageFeaturesH5Reader
from .processors import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_imdb(path_holder, name):
    # train and val features are in the same file
    if name in ["train", "val", "test"]:
        imdb_path = path_holder.format(name)
    else:
        assert False, "data split is not recognized."

    if registry.debug:
        imdb_path = path_holder.format("debug")

    logger.info(f"IMDB path: {imdb_path}")
    imdb_data = ImageDatabase(imdb_path)

    # build entries with only the essential keys
    entries = []
    store_keys = [
        "question",
        "question_id",
        "image_id",
        "answers",
        "image_height",
        "image_width",
        "google_ocr_tokens_filtered",
    ]

    logger.info(f"Building Entries for {name}")
    for idx, instance in enumerate(imdb_data):
        entry = dict([(key, instance[key]) for key in store_keys if key in instance])
        entries.append(entry)
    del imdb_data

    return entries


class TextVQADataset(Dataset):
    def _set_attrs(self, task_cfg):

        keys = [
            ("max_seq_length", None),
            ("textvqa_ocr", None),
            ("textvqa_obj", None),
            ("textvqa_imdb", None),
            ("max_obj_num", None),
            ("max_ocr_num", None),
            ("mix_list", None),
            ("debug", False),
            ("vocab_type", "4k"),
            ("dynamic_sampling", True),
            ("distance_threshold", 0.5),
            ("heads_type", "none"),
            ("clean_answers", True),
        ]

        for key, default in keys:
            if key in task_cfg:
                setattr(self, key, task_cfg[key])
            elif default is not None:
                setattr(self, key, default)
            else:
                print(f"Missing key: {key}")

        registry_keys = [("vocab_type", "4k"), ("distance_threshold", 0.5)]

        for key, default in registry_keys:
            if key in task_cfg:
                registry[key] = task_cfg[key]
            elif default is not None:
                registry[key] = default
            else:
                print(f"Missing key: {key}")

    def set_head_types(self, task_cfg):
        head_types = []
        if "mix_list" in task_cfg:
            for head_type in set(task_cfg["mix_list"]):
                if head_type in self.matrix_type_map:
                    head_types.extend(self.matrix_type_map[head_type])
        self.head_types = list(set(head_types))
        self.head_types.sort()

    def __init__(
        self, split, tokenizer, padding_index=0, processing_threads=32, task_cfg=None
    ):
        super().__init__()
        self.split = split
        self._set_attrs(task_cfg)

        # train + val features are in a single file!
        if split in ["train", "val"]:
            format_str = "trainval"
        else:
            format_str = "test"

        self.obj_features_reader = ImageFeaturesH5Reader(
            features_path=self.textvqa_obj.format(format_str), in_memory=True
        )
        self.ocr_features_reader = ImageFeaturesH5Reader(
            features_path=self.textvqa_ocr.format(format_str), in_memory=True
        )

        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.processing_threads = processing_threads
        self.matrix_type_map = {
            "share3": ["3"],
            "share5": ["3", "5"],
            "share7": ["3", "5", "7"],
            "share9": ["3", "5", "7", "9"],
        }
        # check head types to process
        self.set_head_types(task_cfg)
        self.needs_spatial = len(self.head_types) > 0

        registry.vocab_type = self.vocab_type
        registry.distance_threshold = self.distance_threshold
        # registry.mix_list = task_cfg["M4C"].get("mix_list", ["none"])
        logger.info(f"Dynamic Sampling is {self.dynamic_sampling}")
        logger.info(f"distance_threshold is {self.distance_threshold}")
        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Clean Answers is {self.clean_answers}")
        logger.info(f"needs_spatial is {self.needs_spatial}")

        cache_path = task_cfg["textvqa_spatial_cache"].format(self.split)
        logger.info(f"Cache Name:  {cache_path}")

        if not os.path.exists(cache_path) or self.debug:
            # initialize processors (only once)
            if "processors" not in registry:
                self.processors = Processors(
                    self._tokenizer, vocab_type=self.vocab_type
                )
                registry.processors = self.processors
            else:
                self.processors = registry.processors

            # load imdbs
            self.entries = load_imdb(self.textvqa_imdb, split)

            # convert questions to tokens, create masks, segment_ids
            self.process()

            # process spatial graphs
            if self.needs_spatial:
                self.process_spatials()

            # cache entries
            if not self.debug:
                cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            if "processors_only_registry" not in registry:
                # only initialize the M4C processor (for registry)
                self.processors = Processors(
                    self._tokenizer, only_registry=True, vocab_type=self.vocab_type
                )
                registry.processors_only_registry = self.processors
            else:
                self.processors = registry.processors_only_registry

            # load cache!
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def process(self):
        # Fill the boxes from readers
        for entry in tqdm(self.entries, desc="Processing Entries"):
            # tensorize
            entry["question_id"] = torch.tensor(entry["question_id"])
            entry["image_height"] = torch.tensor(entry["image_height"])
            entry["image_width"] = torch.tensor(entry["image_width"])

            # process question
            processed_question = self.processors.bert_processor(
                {"question": entry["question"]}
            )
            entry["question_indices"] = processed_question["token_inds"]
            entry["num_question_tokens"] = processed_question["token_num"]
            entry["question_mask"] = processed_question["tokens_mask"]

            # process ocr-tokens
            cleaned_ocr_tokens = [
                Processors.word_cleaner(word)
                for word in entry["google_ocr_tokens_filtered"]
            ]

            # fasttext features
            ft_processed_tokens = self.processors.fasttext_processor(
                {"tokens": cleaned_ocr_tokens}
            )
            entry["ocr_fasttext"] = ft_processed_tokens["padded_token_indices"]
            entry["ocr_tokens"] = ft_processed_tokens["padded_tokens"]
            entry["ocr_length"] = ft_processed_tokens["length"]
            entry["cleaned_ocr_tokens"] = cleaned_ocr_tokens

            # phoc features
            phoc_processed_tokens = self.processors.phoc_processor(
                {"tokens": cleaned_ocr_tokens}
            )
            entry["ocr_phoc"] = phoc_processed_tokens["padded_phoc_features"]

            # biggest keys are: ocr_phoc, ocr_fasttext and targets (that goes into caching)
            remove_keys = [
                "sampled_idx_seq",
                "google_ocr_info_filtered",
                "google_ocr_tokens_filtered",
            ]
            for key in remove_keys:
                entry.pop(key, None)

    def process_spatials(self):
        pad_obj_ocr_bboxes_list = []
        for entry in tqdm(self.entries, desc="Reading object/ocr features"):
            # Adding spatial graph matrix
            obj_features, obj_num_boxes, obj_bboxes, _ = self.obj_features_reader[
                entry["image_id"]
            ]
            obj_features, obj_num_boxes, obj_bboxes = (
                obj_features[1:],
                obj_num_boxes - 1,
                obj_bboxes[1:],
            )
            _, _, pad_obj_bboxes = self._pad_features(
                obj_features,
                obj_bboxes,
                obj_num_boxes,
                self.max_obj_num,
                tensorize=False,
            )
            ocr_features, ocr_num_boxes, ocr_bboxes, _ = self.ocr_features_reader[
                entry["image_id"]
            ]
            ocr_features, ocr_num_boxes, ocr_bboxes = (
                ocr_features[1:],
                ocr_num_boxes - 1,
                ocr_bboxes[1:],
            )
            _, _, pad_ocr_bboxes = self._pad_features(
                ocr_features,
                ocr_bboxes,
                ocr_num_boxes,
                self.max_ocr_num,
                tensorize=False,
            )

            # Append bboxes to the list
            pad_obj_ocr_bboxes_list.append(
                np.concatenate([pad_obj_bboxes[:, :-1], pad_ocr_bboxes[:, :-1]], axis=0)
            )

        logger.info(f"Building Spatial Graphs with {self.processing_threads} threads")
        with mp.Pool(self.processing_threads) as pool:
            results = list(
                tqdm(
                    pool.imap(SpatialProcessor, pad_obj_ocr_bboxes_list),
                    total=len(pad_obj_ocr_bboxes_list),
                )
            )

        assert len(results) == len(self.entries)
        for result, entry in zip(results, self.entries):
            entry["spatial_adj_matrix_shared"] = result
            # entry["spatial_gauss_bias_shared"] = result[1]

    def __len__(self):
        return len(self.entries)

    def _pad_features(self, features, bboxes, num_boxes, max_feat_num, tensorize=True):
        mix_num_boxes = min(int(num_boxes), max_feat_num)
        mask = [1] * (int(mix_num_boxes))
        while len(mask) < max_feat_num:
            mask.append(0)

        mix_boxes_pad = np.zeros((max_feat_num, 5))
        mix_boxes_pad[:mix_num_boxes] = bboxes[:mix_num_boxes]

        mix_features_pad = np.zeros((max_feat_num, 2048))
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        if not tensorize:
            return mix_features_pad, mask, mix_boxes_pad

        # tensorize
        pad_features = torch.tensor(mix_features_pad).float()
        mask_features = torch.tensor(mask).long()
        pad_bboxes = torch.tensor(mix_boxes_pad).float()

        return pad_features, mask_features, pad_bboxes

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]

        # add object-features and bounding boxes
        obj_features, obj_num_boxes, obj_bboxes, _ = self.obj_features_reader[image_id]
        # remove avg-features
        obj_features, obj_num_boxes, obj_bboxes = (
            obj_features[1:],
            obj_num_boxes - 1,
            obj_bboxes[1:],
        )
        pad_obj_features, pad_obj_mask, pad_obj_bboxes = self._pad_features(
            obj_features, obj_bboxes, obj_num_boxes, self.max_obj_num
        )

        # add ocr-features and bounding boxes
        ocr_features, ocr_num_boxes, ocr_bboxes, _ = self.ocr_features_reader[image_id]

        # remove avg-features
        ocr_features, ocr_num_boxes, ocr_bboxes = (
            ocr_features[1:],
            ocr_num_boxes - 1,
            ocr_bboxes[1:],
        )
        pad_ocr_features, pad_ocr_mask, pad_ocr_bboxes = self._pad_features(
            ocr_features, ocr_bboxes, ocr_num_boxes, self.max_ocr_num
        )

        segment_ids = torch.zeros_like(entry["question_mask"])

        item = edict(
            {
                "pad_obj_features": pad_obj_features,
                "pad_obj_mask": pad_obj_mask,
                "pad_obj_bboxes": pad_obj_bboxes,
                "pad_ocr_features": pad_ocr_features,
                "pad_ocr_mask": pad_ocr_mask,
                "pad_ocr_bboxes": pad_ocr_bboxes,
                "segment_ids": segment_ids,
            }
        )

        if "answers" in entry:
            # process answers (dynamic sampling)
            if self.clean_answers:
                cleaned_answers = [
                    Processors.word_cleaner(word) for word in entry["answers"]
                ]
            else:
                cleaned_answers = entry["answers"]
            cleaned_ocr_tokens = entry["cleaned_ocr_tokens"]
            processed_answers = self.processors.answer_processor(
                {
                    "answers": cleaned_answers,
                    "context_tokens": cleaned_ocr_tokens,
                }
            )
            entry.update(processed_answers)
        else:
            # Empty placeholder
            entry["train_prev_inds"] = torch.zeros(12, dtype=torch.long)
            entry["train_loss_mask"] = torch.zeros(12, dtype=torch.float)
            entry["answers"] = ["nothing-here"] * 10
            entry["targets"] = torch.zeros(12, self.processors.answer_processor.get_vocab_size(), dtype=torch.float)

        if self.needs_spatial:
            # In the first iteration expand all the spatial relation matrices
            if "spatial_adj_matrices" not in entry:
                entry["spatial_adj_matrices"] = {}

                build_map = {
                    "3": ["1", "31", "32"],
                    "5": ["3", "51", "52"],
                    "7": ["5", "71", "72"],
                    "9": ["7", "91", "92"],
                }

                entry["spatial_adj_matrices"]["1"] = torch_broadcast_adj_matrix(
                    torch.from_numpy(entry["spatial_adj_matrix_shared"]["1"])
                )

                entry["spatial_adj_matrices"]["full_spatial"] = (
                    torch.from_numpy(entry["spatial_adj_matrix_shared"]["1"]) != 0
                ).int()

                for head_type in self.head_types:
                    use_matrix_types = build_map[head_type]
                    assert use_matrix_types[0] in entry["spatial_adj_matrices"]
                    init_matrix = entry["spatial_adj_matrices"][use_matrix_types[0]]
                    first_matrix = torch_broadcast_adj_matrix(
                        torch.from_numpy(
                            entry["spatial_adj_matrix_shared"][use_matrix_types[1]]
                        )
                    )
                    second_matrix = torch_broadcast_adj_matrix(
                        torch.from_numpy(
                            entry["spatial_adj_matrix_shared"][use_matrix_types[2]]
                        )
                    )
                    init_matrix = torch.max(init_matrix, first_matrix)
                    init_matrix = torch.max(init_matrix, second_matrix)
                    entry["spatial_adj_matrices"][head_type] = init_matrix

        item.update(entry)

        # remove unwanted keys
        unwanted_keys_item = [
            "spatial_adj_matrix_shared",
            "spatial_adj_matrix",
            "cleaned_ocr_tokens",
            "image_id",
            "image_path",
        ]

        for key in unwanted_keys_item:
            if key in item:
                item.pop(key, None)

        # unwanted_keys_entry = [
        #     'spatial_adj_matrices',
        # ]
        #
        # for key in unwanted_keys_entry:
        #     if key in entry:
        #         entry.pop(key, None)

        # Collate Function doesn't work correctly with lists

        for key, value in item.items():
            if not isinstance(value, torch.Tensor) and not isinstance(value, dict):
                try:
                    item[key] = enc_obj2bytes(value)
                except:
                    print(key)
                    import pdb

                    pdb.set_trace()

        return item


class ImageDatabase(torch.utils.data.Dataset):
    """
    Dataset for IMDB used in Pythia
    General format that we have standardize follows:
    {
        metadata: {
            'version': x
        },
        data: [
            {
                'id': DATASET_SET_ID,
                'set_folder': <directory>,
                'feature_path': <file_path>,
                'info': {
                    // Extra information
                    'questions_tokens': [],
                    'answer_tokens': []
                }
            }
        ]
    }
    """

    def __init__(self, imdb_path):
        super().__init__()
        self.metadata = {}
        self._load_imdb(imdb_path)

    def _load_imdb(self, imdb_path):
        if imdb_path.endswith(".npy"):
            self._load_npy(imdb_path)
        else:
            raise ValueError("Unknown file format for imdb")

    def _load_npy(self, imdb_path):
        self.db = np.load(imdb_path, allow_pickle=True)
        assert isinstance(self.db, np.ndarray)
        assert "image_id" not in self.db
        self.metadata = {"version": 1}
        self.data = self.db
        self.start_idx = 1
        self.metadata.update(self.db[0])
        self._sort()

    def __len__(self):
        return len(self.data) - self.start_idx

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]
        return data

    def get_version(self):
        return self.metadata.get("version", None)

    def _sort(self):
        sorted_data = sorted(
            self.data[self.start_idx :], key=lambda x: x["question_id"]
        )
        self.data[self.start_idx :] = sorted_data
