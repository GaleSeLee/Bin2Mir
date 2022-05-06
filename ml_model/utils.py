import json
from random import choice


class DataLoader():

    def __init__(self, file_path, bin_bb_limit=50, mir_bb_limit=100):
        with open(file_path) as f:
            pair_info = json.load(f)
        self.mir2bin = pair_info['mir2bin']
        self.mir_info = pair_info['mir_funcs']
        self.bin_info = pair_info['bin_funcs']
        self.label2identifier = {i: k for i,
                                 k in enumerate(self.mir2bin.keys())}
        self.limit_length(self.mir_info, mir_bb_limit)
        self.limit_length(self.bin_info, bin_bb_limit)

    @staticmethod
    def batch_collector(batch):
        bin_cfg = [item['bin_info']['edge_list'] for item in batch]
        bin_bb = [item['bin_info']['bb_list'] for item in batch]
        mir_cfg = [item['mir_info']['edge_list'] for item in batch]
        mir_bb = [item['mir_info']['bb_list'] for item in batch]
        # mir_strs = [item['mir_strs'] for item in batch]
        # bin_strs = [item['bin_strs'] for item in batch]
        # Seems not important at all
        labels = [item['label'] for item in batch]
        return bin_bb, bin_cfg, mir_bb, mir_cfg, labels
        # return bin_cfg, bin_bb, bin_strs, mir_cfg, mir_bb, mir_strs, labels

    def __getitem__(self, key):
        mir_identifier = self.label2identifier[key]
        bin_identifier = choice(self.mir2bin[mir_identifier])
        return {
            'mir_info': self.mir_info[mir_identifier],
            'bin_info': self.bin_info[bin_identifier],
            'label': key,
        }

    def __len__(self):
        return len(self.label2identifier)

    def get_bin_funcs(self):

        ret = {}
        for k, v in self.mir2bin.items():
            bin_funcs = [self.bin_info[identifier] for identifier in v]
            bin_bb = [func['bb_list'] for func in bin_funcs]
            bin_cfg = [func['edge_list'] for func in bin_funcs]
            ret[k] = bin_bb, bin_cfg
        return ret

    @staticmethod
    def limit_length(d, limit):
        for k in d.keys():
            d[k]['bb_list'] = d[k]['bb_list'][:limit]
            d[k]['edge_list'] = list(filter(
                lambda x: x[-1] < limit and x[-2] < limit,
                d[k]['edge_list']))
        