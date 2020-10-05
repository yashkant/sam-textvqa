import os
import re
from collections import Counter
from itertools import chain

import torch


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, vocab_file, data_root_dir=None):
        if not os.path.exists(vocab_file):
            raise RuntimeError(
                "Vocab file {} for vocab dict doesn't exist".format(vocab_file)
            )

        self.word_list = load_str_list(vocab_file)
        self._build()

    def _build(self):
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (
            self.word2idx_dict[self.UNK_TOKEN]
            if self.UNK_TOKEN in self.word2idx_dict
            else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[self.PAD_TOKEN]
            if self.PAD_TOKEN in self.word2idx_dict
            else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError(
                "word %s not in dictionary (while dictionary does not contain <unk>)"
                % w
            )

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds
