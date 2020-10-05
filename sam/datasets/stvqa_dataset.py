import logging
import os

import _pickle as cPickle
from torch.utils.data import Dataset

from sam.datasets.textvqa_dataset import ImageDatabase, TextVQADataset

from ._image_features_reader import ImageFeaturesH5Reader
from .processors import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_dataset(path_holder, name):
    """Load entries from Imdb

    (YK): We load questions and answers corresponding to
        the splits, and return entries.
    """

    if name == "train" or name == "val":
        imdb_path = path_holder.format(name)
    elif name == "test":
        # We only evaluate on test-task3
        imdb_path = path_holder.format("test")
    else:
        assert False, "data split is not recognized."

    if registry.debug:
        imdb_path = path_holder.format("debug")

    logger.info(f"Loading IMDB split: {name} | path: {imdb_path}")
    imdb_data = ImageDatabase(imdb_path)

    # build entries with only the essential keys
    entries = []
    store_keys = [
        "question",
        "question_id",
        "image_path",
        "answers",
        "image_height",
        "image_width",
        "google_ocr_tokens_filtered",
        # "google_ocr_info_filtered",
    ]

    logger.info(f"Building Entries for {name}")
    for instance in imdb_data:
        entry = dict([(key, instance[key]) for key in store_keys if key in instance])
        # Also need to add features-dir
        entry["image_id"] = entry["image_path"].split(".")[0] + ".npy"
        entries.append(entry)
    del imdb_data

    return entries


class STVQADataset(TextVQADataset):
    def _set_attrs(self, task_cfg):
        keys = [
            ("max_seq_length", None),
            ("stvqa_ocr", None),
            ("stvqa_obj", None),
            ("stvqa_imdb", None),
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
                # raise ValueError(f"Missing key: {key}")

        registry_keys = [("vocab_type", None), ("distance_threshold", None)]

        for key, default in registry_keys:
            if key in task_cfg:
                registry[key] = task_cfg[key]
            elif default is not None:
                registry[key] = default
            else:
                print(f"Missing key: {key}")
                # raise ValueError(f"Missing key: {key}")

    def __init__(
        self, split, tokenizer, padding_index=0, processing_threads=32, task_cfg=None
    ):
        # Just initialize the grand-parent classs
        Dataset.__init__(self)

        self.clean_answers = None
        self.split = split
        self._max_seq_length = task_cfg["max_seq_length"]
        self._set_attrs(task_cfg)

        # train + val features are in a single file!
        if split in ["train", "val"]:
            format_str = "trainval"
        else:
            format_str = "test"

        self.obj_features_reader = ImageFeaturesH5Reader(
            features_path=self.stvqa_obj.format(format_str), in_memory=True
        )
        self.ocr_features_reader = ImageFeaturesH5Reader(
            features_path=self.stvqa_ocr.format(format_str), in_memory=True
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
        self.path_holder = task_cfg["stvqa_imdb"]

        registry.vocab_type = self.vocab_type
        registry.distance_threshold = self.distance_threshold
        # registry.mix_list = task_cfg["M4C"].get("mix_list", ["none"])

        logger.info(f"Dynamic Sampling is {self.dynamic_sampling}")
        logger.info(f"distance_threshold is {self.distance_threshold}")
        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Clean Answers is {self.clean_answers}")
        logger.info(f"needs_spatial is {self.needs_spatial}")

        cache_path = task_cfg["stvqa_spatial_cache"].format(self.split)
        logger.info(f"Cache Name:  {cache_path}")

        if not os.path.exists(cache_path) or self.debug:
            # Initialize Processors
            if "processors" not in registry:
                self.processors = Processors(
                    self._tokenizer, vocab_type=self.vocab_type
                )
                registry.processors = self.processors
            else:
                self.processors = registry.processors

            self.entries = _load_dataset(self.path_holder, split)
            # convert questions to tokens, create masks, segment_ids
            self.process()

            if self.needs_spatial:
                self.process_spatials()

            if not self.debug:
                cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            if "processors_only_registry" not in registry:
                self.processors = Processors(
                    self._tokenizer, only_registry=True, vocab_type=self.vocab_type
                )  # only initialize the M4C processor (for registry)
                registry.processors_only_registry = self.processors
            else:
                self.processors = registry.processors_only_registry

            # otherwise load cache!
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))
