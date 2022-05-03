from functools import reduce
import os
import random

import json

from data_formatter.analyser import MirAnalyser, BinaryAnalyser
from data_formatter.func import BinFunc, MirFunc
from data_formatter.utils import statistic_dict


class Formatter():

    # A standard data file layout is expected.
    # Formatter reads all input from "raw_dir", and expects
    #   a file structure like below:
    # raw_dir
    #   | ----- bin_crate_i_dir
    #               | ----- cfg.json          ( bin cfg info )
    #               | ----- lib_crate_i.json' ( mir info )
    #               | ----- bin               ( r, bin func info )
    # Formatter will write all its output to "formatted_dir",
    #   which is shown below:
    # formatted_dir
    #   | ----- statistic
    #   | ----- func_data
    #   | ----- output_data
    # Formatter also requires a db, which can be placed anywhere,
    #   as long as the session is provided
    def __init__(self, raw_dir, formatted_dir, crate_list, session):
        self.session = session
        self.formatted_dir = formatted_dir
        os.makedirs(os.path.join(formatted_dir, 'statistic'), exist_ok=True)
        os.makedirs(os.path.join(formatted_dir, 'func_data'), exist_ok=True)
        os.makedirs(os.path.join(formatted_dir, 'output_data'), exist_ok=True)
        self.mir_analyser = MirAnalyser(
            raw_dir=raw_dir,
            formatted_dir=formatted_dir,
            crate_list=crate_list,
            session=session
        )
        self.bin_analyser = BinaryAnalyser(
            raw_dir=raw_dir,
            formatted_dir=formatted_dir,
            crate_list=crate_list,
            session=session,
        )

    def match(self, bin_funcs, mir_funcs, file_name='matched'):
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
                matched_list.append(
                    (func.into_str(), coarse_matched_list[0].into_str()))
            elif len(path_matched_list) == 1:
                self.add_pair(func, path_matched_list[0])
                matched_list.append(
                    (func.into_str(), path_matched_list[0].into_str()))
            elif len(generic_matched_list) == 1:
                self.add_pair(func, generic_matched_list[0])
                matched_list.append(
                    (func.into_str(), generic_matched_list[0].into_str()))
            elif len(rigid_match_list) == 1:
                self.add_pair(func, rigid_match_list[0])
                matched_list.append(
                    (func.into_str(), rigid_match_list[0].into_str()))
            else:
                self.add_dumplicate_match(func, coarse_matched_list)
                dup_list.append(
                    (func.into_str(), [mir_func.into_str() for mir_func in coarse_matched_list]))
        self.session.commit()
        with open(os.path.join(self.formatted_dir, 'statistic', f'{file_name}.json'), 'w') as f:
            json.dump({
                'missed_list': missed_list,
                'matched_list': matched_list,
                'dup_list': dup_list,
            }, f)

    # kwargs work as filter arguements for bin funcs
    def dump_match(self, file_name='matched_info', max_per_mir=10, **kwargs):
        dump_dict = {
            'mir2bin': {},
            'mir_funcs': {},
            'bin_funcs': {},
        }
        bin_funcs = self.bin_analyser.load_matched(**kwargs)
        random.shuffle(bin_funcs)
        for func in bin_funcs:
            mir_identifier = func.match_mir
            if func.match_mir not in dump_dict['mir2bin']:
                mir_func = self.session.query(MirFunc).filter_by(
                    identifier=mir_identifier).first()
                self.mir_analyser.load_func_data(mir_func)
                dump_dict['mir2bin'][mir_identifier] = []
                dump_dict['mir_funcs'][mir_identifier] = mir_func.into_dict()
            if len(dump_dict['mir2bin'][func.match_mir]) < max_per_mir:
                self.bin_analyser.load_func_data(func)
                dump_dict['mir2bin'][mir_identifier].append(func.identifier)
                dump_dict['bin_funcs'][func.identifier] = func.into_dict()
        with open(os.path.join(self.formatted_dir, 'output_data', f'{file_name}.json'), 'w') as f:
            json.dump(dump_dict, f, indent=2)

    def dump_matched_mir(self, file_name='matched_mir', **kwargs):
        mir_list, bin_list = [], []
        bin_funcs = self.bin_analyser.load_matched(**kwargs)
        for func in bin_funcs:
            mir_func = self.session.query(MirFunc).filter_by(
                identifier=func.match_mir).first()
            mir_list.append(mir_func.into_str(generic=True))
            bin_list.append(func.into_str(generic=True))
        s_dict = statistic_dict(mir_list, bin_list)
        file_path = os.path.join(self.formatted_dir, 'statistic', f'{file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(s_dict, f, indent=2)

    def add_pair(self, bin_func, mir_func):
        self.session.query(BinFunc).filter_by(identifier=bin_func.identifier).update(
            {'match_mir': mir_func.identifier})

    def add_dumplicate_match(self, bin_func, mir_list):
        self.session.query(BinFunc).filter_by(identifier=bin_func.identifier).update(
            {'coarse_list': ':'.join([mir_func.identifier for mir_func in mir_list])})

    def dump_random_walks(self, funcs: list, file_name='random_walks'):
        for func in self.activated(funcs):
            if not func.full_info:
                self.bin_analyser.load_func_data(func)
        bin_random_walks = reduce(
            lambda x, y: x + y, [func.random_walk(path_num=1) for func in self.activated(funcs)])
        with open(os.path.join(self.formatted_dir, 'output_data', f'{file_name}.json'), 'w') as f:
            json.dump(bin_random_walks, f, indent=2)

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
