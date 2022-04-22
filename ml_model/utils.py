from functools import reduce
import os
from collections import defaultdict

import math
import json
import torch

# A dull encoder for ascii strings.
# Char level encoder


class CharOneHotEncoder():
    def __init__(self, max_len=2048):
        self.embedding_size = 128
        self.max_len = max_len
        self.max_pow = math.ceil(math.log2(max_len)) + 1
        self.lookup = torch.eye(self.embedding_size)
        self.pow_list = [2**i for i in range(self.max_pow)]

    def str2tensor(self, s: str):
        # s != ''
        return torch.index_select(self.lookup, 0,
                                  torch.tensor([ord(c) for c in s[:self.max_len]]))

    def tslist2tensor(self, tensors: list):
        length = max(
            [tensor.shape[0] if tensor is not None else 1 for tensor in tensors])
        for i in range(self.max_pow):
            if length <= self.pow_list[i]:
                length = self.pow_list[i]
                return torch.stack([self.pad(tensor, length) for tensor in tensors])

    def pad(self, tensor, length):
        if tensor is None:
            return torch.zeros(length, self.embedding_size)
        padding = torch.zeros(length - tensor.shape[0], self.embedding_size)
        return torch.cat([tensor, padding])

    # Accept a list of strs, convert to onehot tensor
    def __call__(self, strs: list):
        if not strs:
            return torch.zeros(1, 1, self.embedding_size)
        return self.tslist2tensor([self.str2tensor(s) if s else None for s in strs])


class DataLoader():

    def __init__(self, pair_dir):
        self.pair_dir = pair_dir
        self.v = 0
        self.fn2label = defaultdict(self.get_label)
        self.samples = reduce(lambda x, y: x + y,
                              map(self.file2samples, os.listdir(pair_dir)))

    # Each file contains
    def file2samples(self, pair_file):
        file_name = os.path.basename(pair_file)
        if not file_name.endswith('.json'):
            return None
        file_path = os.path.join(self.pair_dir, file_name)
        with open(file_path) as f:
            pair_info = json.load(f)
        ret = []
        for k, v in pair_info.items():
            v['label'] = self.fn2label[k[:k.find('(')]]
            ret.append(v)
        return ret

    @staticmethod
    def batch_collector(batch):
        bin_cfg = [item['bin_cfg'] for item in batch]
        bin_bb = [item['bin_bbs'] for item in batch]
        bin_strs = [item['bin_strs'] for item in batch]
        mir_cfg = [item['mir_cfg'] for item in batch]
        mir_bb = [item['mir_bbs'] for item in batch]
        mir_strs = [item['mir_strs'] for item in batch]
        labels = [item['label'] for item in batch]
        return bin_bb, bin_cfg, mir_bb, mir_cfg, labels
        # return bin_cfg, bin_bb, bin_strs, mir_cfg, mir_bb, mir_strs, labels

    def get_label(self):
        v = self.v
        self.v += 1
        return v

    def __getitem__(self, key):
        return self.samples[key]

    def __len__(self):
        return len(self.samples)


def edges2tensor(cfg_list, max_idx):
    dims = (max_idx, max_idx)
    ret = torch.zeros(dims)
    for s, t in cfg_list:
        ret[t, s] = 1
    return ret
