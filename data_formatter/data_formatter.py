from functools import reduce
import os

import json

from data_formatter.analyser import MirAnalyser, BinaryAnalyser
from data_formatter.func import BinFunc, MirFunc


class Formatter():

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

    def match(self, bin_funcs, mir_funcs, statistic_file=''):
        missed_list, matched_list, dup_list = [], [], []
        for func in self.activated(bin_funcs):
            coarse_matched_list = list(
                filter(lambda x: x.coarse_eq(func), self.activated(mir_funcs)))
            path_matched_list = list(
                filter(lambda x: x.coarse_eq(func, short=False), coarse_matched_list))
            generic_matched_list = list(
                filter(lambda x: x.coarse_eq(func, generic=True), coarse_matched_list))
            rigid_match_list = list(
                filter(lambda x: x == func, coarse_matched_list))
            if len(coarse_matched_list) == 0:
                missed_list.append(func.into_str())
            elif len(coarse_matched_list) == 1:
                self.add_pair(func, coarse_matched_list[0])
                matched_list.append((func.into_str(), coarse_matched_list[0].into_str()))
            elif len(path_matched_list) == 1:
                self.add_pair(func, path_matched_list[0])
                matched_list.append((func.into_str(), path_matched_list[0].into_str()))
            elif len(generic_matched_list) == 1:
                self.add_pair(func, generic_matched_list[0])
                matched_list.append((func.into_str(), generic_matched_list[0].into_str()))
            elif len(rigid_match_list) == 1:
                self.add_pair(func, rigid_match_list[0])
                matched_list.append((func.into_str(), rigid_match_list[0].into_str()))
            else:
                self.add_dumplicate_match(func, coarse_matched_list)
                dup_list.append((func.into_str(), [mir_func.into_str() for mir_func in coarse_matched_list]))
        self.session.commit()
        if statistic_file:
            with open(statistic_file, 'w') as f:
                json.dump({
                    'missed_list': missed_list,
                    'matched_list': matched_list,
                    'dup_list': dup_list
                }, f)

    def dump_match(self, file_name: str, **kwargs):
        dump_list = []
        bin_funcs = self.bin_analyser.load_matched(**kwargs)
        for func in bin_funcs:
            mir_func = self.session.query(MirFunc).filter_by(
                identifier=func.match_mir).first()
            self.bin_analyser.load_func_data(func)
            self.mir_analyser.load_func_data(mir_func)
            dump_list.append(self.full_formatter(func, mir_func))
        with open(os.path.join(self.base_dir, 'pairs', f'{file_name}.json'), 'w') as f:
            json.dump(dump_list, f)

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

    def dump_random_walks(self, funcs: list, file_name: str):
        for func in self.activated(funcs):
            if not func.full_info:
                self.bin_analyser.load_func_data(func)
        bin_random_walks = reduce(lambda x, y: x + y, [func.random_walk(path_num=1) for func in self.activated(funcs)])
        with open(os.path.join(self.base_dir, 'bin_random_walks', f'{file_name}.json'), 'w') as f:
            json.dump(bin_random_walks, f)

    def full_formatter(self, bin_func, mir_func):
        return {
            'bin_label': bin_func.into_str(generic=True),
            'mir_label': mir_func.into_str(generic=True),
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

    @staticmethod
    def activated(funcs):
        return filter(lambda x: x.valid(), funcs)

    @staticmethod
    def matched(funcs):
        return filter(lambda x: x.match_mir != '', funcs)
