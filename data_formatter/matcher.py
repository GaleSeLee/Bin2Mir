import os

import json

from data_formatter.analyser import MirAnalyser, BinaryAnalyser
from data_formatter.func import BinFunc, MirFunc


class Matcher():

    def __init__(self, base_dir, crate_list, session):
        self.session = session
        self.base_dir = base_dir
        self.mir_analyser = MirAnalyser(
            base_dir=base_dir,
            crate_list=crate_list,
            session=session
        )
        self.bin_analyser = BinaryAnalyser(
            base_dir=base_dir,
            crate_list=crate_list,
            session=session,
        )
        self.missed_list = []
        self.matched_list = []
        self.dumplicate_list = []

    def match(self):
        for func in self.bin_analyser.activated():
            coarse_matched_list = list(
                filter(lambda x: x.coarse_eq(func), self.mir_analyser.activated()))
            path_matched_list = list(
                filter(lambda x: x.coarse_eq(func, short=False), coarse_matched_list))
            generic_matched_list = list(
                filter(lambda x: x.coarse_eq(func, generic=True), coarse_matched_list))
            rigid_match_list = list(
                filter(lambda x: x == func, coarse_matched_list))
            if len(coarse_matched_list) == 0:
                self.missed_list.append(func.into_str())
            elif len(coarse_matched_list) == 1:
                self.add_pair(func, coarse_matched_list[0])
            elif len(path_matched_list) == 1:
                self.add_pair(func, path_matched_list[0])
            elif len(generic_matched_list) == 1:
                self.add_pair(func, generic_matched_list[0])
            elif len(rigid_match_list) == 1:
                self.add_pair(func, rigid_match_list[0])
            else:
                self.add_dumplicate_match(func, coarse_matched_list)
        self.session.commit()

    def dump_match(self, file_name, **kwargs):
        dump_list = []
        self.bin_analyser.load_matched(**kwargs)
        for func in self.bin_analyser.activated():
            mir_func = self.session.query(MirFunc).filter_by(
                identifier=func.match_mir).first()
            self.bin_analyser.load_func_data(func)
            self.mir_analyser.load_func_data(mir_func)
            dump_list.append(self.full_formatter(func, mir_func))
        with open(os.path.join(self.base_dir, 'pairs', f'{file_name}.json'), 'w') as f:
            json.dump(dump_list, f)

    def clear(self):
        self.bin_analyser.funcs.clear()
        self.mir_analyser.funcs.clear()

    def add_pair(self, bin_func, mir_func):
        # self.matched_list.append((bin_func, mir_func))
        self.session.query(BinFunc).filter_by(identifier=bin_func.identifier).update(
            {'match_mir': mir_func.identifier})
        # self.session.commit()

    def add_dumplicate_match(self, bin_func, mir_list):
        # self.dumplicate_list.append((bin_func.into_str(), [mir_func.into_str() for mir_func in mir_list]))
        self.session.query(BinFunc).filter_by(identifier=bin_func.identifier).update(
            {'coarse_list': ':'.join([mir_func.identifier for mir_func in mir_list])})
        # self.session.commit()

    def dump_statistics(self, file_path: str):

        def crate_cnt(crate_list: list):
            ret_dict = {}
            for crate in filter(lambda x: x is not None, crate_list):
                if crate not in ret_dict:
                    ret_dict[crate] = 1
                else:
                    ret_dict[crate] += 1
            return ret_dict
        bin_crates = crate_cnt(
            ([func.crate for func in self.bin_analyser.funcs]))
        bin_activated_crates = crate_cnt(
            [func.crate for func in self.bin_analyser.activated()])
        matched_bin_crates = crate_cnt(
            ([pair[0].crate for pair in self.matched_list]))
        # matched_mir_crates = crate_cnt(set([pair[1].crate for pair in self.matched_list]))
        dump_dict = {
            'bin_func_info': {
                'total_num': len(self.bin_analyser.funcs),
                'valid_num': len(list(self.bin_analyser.activated())),
                'crates': [key for key in bin_crates.keys()],
                'crate_cnt': bin_crates,
                'activated_crate_cnt': bin_activated_crates
            },
            'matched_func_info': {
                'matched_num': len(self.matched_list),
                'matched_crate_cnt': matched_bin_crates,
                'missed': self.missed_list,
                'dumplicates': self.dumplicate_list
            }
        }
        with open(file_path, 'w') as f:
            json.dump(dump_dict, f, indent=2)

    def dump_random_walks(self, file_name: str):
        bin_random_walks = [func.random_walk(path_num=1) for func in self.bin_analyser.activated()]
        with open(os.path.join(self.base_dir, 'bin_random_walks', f'{file_name}.json'), 'w') as f:
            json.dump(bin_random_walks, f)

    def full_formatter(self, bin_func, mir_func):
        return {
            'bin_cfg': bin_func.edge_list,
            'mir_cfg': [(e[1], e[2]) for e in mir_func.edge_list],
            'bin_bbs': bin_func.bb_list,
            'mir_bbs': [bb.into_str() for bb in mir_func.bb_list],
            # 'bin_strs': bin_func.string_refs,
            # 'mir_strs': mir_func.get_all_strs()
        }

    @staticmethod
    def cfg_formatter(bin_func, mir_func):
        return {
            'bin_cfg': ' ,'.join([f'({e[0]},{e[1]})' for e in bin_func.edge_list]),
            'mir_cfg': ' ,'.join([f'({e[0]},{e[1]})' for e in mir_func.edge_list]),
            'bin_bbs': {i: bb for i, bb in enumerate(bin_func.bb_list)},
            'mir_bbs': {i: bb.contents for i, bb in enumerate(mir_func.mir_bbs)},
        }

    @staticmethod
    def both_hash_str(pair):
        bin_func, mir_func = pair
        return bin_func.string_refs and mir_func.get_all_strs()

    @staticmethod
    def str_only_formatter(bin_func, mir_func):
        return {
            'bin_strs': bin_func.string_refs,
            'mir_strs': mir_func.get_all_strs()
        }
