from functools import reduce
import os
import json

from data_formatter.macro import FunctionAnalErrorCode
from data_formatter.func import BinFunc, MirFunc
from data_formatter.utils import DefaultDict


class BaseAnalyser():

    def __init__(self, base_dir, crate_list, session) -> None:
        self.base_dir = base_dir
        self.crate_list = crate_list
        self.session = session
        self.funcs = []
        self.crate_data = DefaultDict(self.load_data)

    def activated(self):
        return filter(lambda x: x.valid(), self.funcs)

    def new_funcs(self):
        return filter(lambda x: x.is_new, self.activated())

    def db_funcs(self):
        return filter(lambda x: not x.is_new, self.activated())

    def partial_funcs(self):
        return filter(lambda x: not x.full_info, self.activated())

    # Load funcs from db, without concrete info
    def load(self, **kwargs):
        self.funcs.extend(self.session.query(
            self.func_type).filter_by(**kwargs).all())

    # Load concrete data from json for db functions
    def load_data(self, crate):
        file_path = os.path.join(self.base_dir, 'func_data', f'{crate}.json')
        if not os.path.exists(file_path):
            return {}
        with open(file_path) as f:
            return json.load(f)

    # Add concrete data to func, make it have full info
    def load_func_data(self, func):
        func.load_data(
            **self.crate_data[func.get_data_crate()][func.identifier])
        func.full_info = True

    # Analyse from raw inputs
    def load_raw(self):
        raise NotImplementedError

    # Save funcs to db, and concrete info to json
    def save(self):
        for func in self.new_funcs():
            if self.session.query(self.func_type).filter_by(identifier=func.identifier).first() is not None:
                continue
            self.crate_data[func.get_data_crate(
            )][func.identifier] = func.into_dict()
            self.session.add(func)
            func.is_new = False
        data_path = os.path.join(self.base_dir, 'func_data')
        # for crate in self.crate_list:
        for crate in self.crate_data:
            with open(os.path.join(data_path, f'{crate}.json'), 'w') as f:
                json.dump(self.crate_data[crate], f)
        self.session.commit()

    def dump_statistic(self, funcs, statistic_file):
        def crate_cnt(crate_list: list):
            ret_dict = {}
            for crate in filter(lambda x: x is not None, crate_list):
                if crate not in ret_dict:
                    ret_dict[crate] = 1
                else:
                    ret_dict[crate] += 1
            return ret_dict
        valid_funcs = list(filter(lambda x: x.valid(), funcs))
        invalid_funcs = list(filter(lambda x: not x.valid(), funcs))
        valid_crate_cnt = crate_cnt([func.crate for func in valid_funcs])
        total_crate_cnt = crate_cnt([func.crate for func in funcs])
        dump_dict = {
            'Total function num': len(funcs),
            'Valid function num': len(valid_funcs),
            'Function num for each crate': {crate: f'{valid_crate_cnt.get(crate, 0)}/{total_crate_cnt[crate]}' for crate in total_crate_cnt},
            # 'Function num for each invalid': {},
            'Invalid list': [func.origin_decl for func in invalid_funcs]
        }
        with open(statistic_file, 'w') as f:
            json.dump(dump_dict, f)


class BinaryAnalyser(BaseAnalyser):

    def __init__(self, base_dir, crate_list, session) -> None:
        super().__init__(base_dir, crate_list, session)
        self.func_type = BinFunc

    # target crate: the crate which bin file belongs to
    def load_raw(self, target_crate, statistic_file=''):
        funcs = self.load_cfg_info(os.path.join(
            self.base_dir, 'cfgs.json'), target_crate)
        self.load_bin(os.path.join(self.base_dir, 'bin'), funcs)
        if statistic_file:
            self.dump_statistic(funcs, statistic_file)
        self.funcs.extend(filter(lambda x: x.valid(), funcs))
        # self.load_string_info(os.path.join(self.base_dir, 'strings.json'))

    def load_cfg_info(self, path, target_crate):
        with open(path) as f:
            cfg_info = json.load(f)
        funcs = [BinFunc(target_crate, info_dict)
                 for info_dict in cfg_info]
        for func in filter(lambda x: x.valid(), funcs):
            if sum(func.block_length_list) < 10:
                func.errno |= FunctionAnalErrorCode.BinFileError
            if func.crate not in self.crate_list:
                func.errno |= FunctionAnalErrorCode.NotWantedCrate
        return funcs

    def load_bin(self, base_dir, funcs):
        for func in filter(lambda x: x.valid(), funcs):
            func.load_bin(base_dir)

    def load_string_info(self, path):
        raise NotImplementedError
        with open(path) as f:
            string_json = json.load(f)
        # Use dict to speed up match
        funcs_dict = {func.identifier: i for i,
                      func in enumerate(self.new_funcs())}
        string_refs = {
            str_literal: list(filter(lambda x: x in funcs_dict,
                                     str_info['refs'] + str_info['indirect']))
            for str_literal, str_info in string_json.items()}

        pop_list = [k if not v else None for k, v in string_refs.items()]
        for k in filter(lambda x: x is not None, pop_list):
            string_refs.pop(k)
        for k, v in funcs_dict.items():
            self.funcs[v].string_refs = list(
                filter(lambda x: x is not None,
                       [key if k in value else None for key, value in string_refs.items()]))

    def load_matched(self, **kwargs):
        self.funcs.extend(self.session.query(BinFunc).filter(
            BinFunc.match_mir != '').filter_by(**kwargs).all())


class MirAnalyser(BaseAnalyser):

    def __init__(self, base_dir, crate_list, session) -> None:
        super().__init__(base_dir, crate_list, session)
        self.func_type = MirFunc

    def load_raw(self, statistic_file=''):
        funcs = reduce(lambda x, y: x + y,
                       [self.load_mir_info(crate) for crate in self.crate_list])
        if statistic_file:
            self.dump_statistic(funcs, statistic_file)
        self.funcs.extend(filter(lambda x: x.valid() and x.is_new, funcs))

    # Ignore functions already find in db
    def load_mir_info(self, crate):
        crate_json = os.path.join(self.base_dir, f'{crate}.json')
        if not os.path.exists(crate_json):
            print(f'File {crate_json} not exists')
            return
        with open(crate_json) as f:
            crate_info = json.load(f)
        funcs = [MirFunc(
            crate=crate,
            fndef_info=item[0],
            bb_list=item[1]
        )
            for item in crate_info]
        for func in filter(lambda x: x.valid() and
                           not self.session.query(MirFunc)
                           .filter_by(identifier=x.identifier).first() is None, funcs):
            func.is_new = False
        for func in filter(lambda x: x.valid() and x.is_new, funcs):
            func.analyse_bb_list()
        return funcs
