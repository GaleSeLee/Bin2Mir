from functools import reduce
import os
import json

from data_formatter.macro import FunctionAnalErrorCode, ExtendErrorCode
from data_formatter.func import BinFunc, MirFunc
from data_formatter.utils import DefaultDict, statistic_dict


class BaseAnalyser():

    def __init__(self, raw_dir, formatted_dir, crate_list, session) -> None:
        # Wanted lib crates, such as std, core
        self.crate_list = crate_list
        self.raw_dir = raw_dir
        self.formatted_dir = formatted_dir
        self.session = session
        self.crate_data = DefaultDict(self.load_data)

    # Load funcs from db, without concrete info
    def load(self, **kwargs):
        return self.session.query(self.func_type).filter_by(**kwargs).all()

    # Load concrete data from json for db functions
    def load_data(self, crate):
        file_path = os.path.join(
            self.formatted_dir, 'func_data', f'{crate}.json')
        if not os.path.exists(file_path):
            return {}
        with open(file_path) as f:
            return json.load(f)

    # Add concrete data to func, make it have full info
    def load_func_data(self, func):
        # print(func.into_str(), func.crate, func.identifier)
        func.load_data(
            **self.crate_data[func.get_data_crate()][func.identifier])
        func.full_info = True

    # Analyse from raw inputs
    # Return a list of funcs, maybe invalid
    # For mir, target_crate is only used for
    #   statistic output
    def load_raw(self, target_crate) -> list:
        raise NotImplementedError

    # Save valid new funcs to db, and concrete info to json
    def save(self, funcs):
        for func in filter(lambda x: x.valid() and x.is_new, funcs):
            self.save_func(func)
            func.is_new = False
        self.flush_func_data()
        self.session.commit()

    def save_func_data(self, func):
        self.crate_data[func.get_data_crate(
        )][func.identifier] = func.into_dict()

    def flush_func_data(self):
        data_path = os.path.join(self.formatted_dir, 'func_data')
        for crate in self.crate_data:
            with open(os.path.join(data_path, f'{crate}.json'), 'w') as f:
                json.dump(self.crate_data[crate], f, indent=2)

    def dump_statistic(self, funcs, file_name):
        dump_path = os.path.join(
            self.formatted_dir, 'statistic', f'{file_name}.json')
        with open(dump_path, 'w') as f:
            json.dump(self.get_statistic(funcs), f)

    def get_statistic(self, funcs):
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
            'Invalid': statistic_dict([FunctionAnalErrorCode.errno2str(func.errno) for func in invalid_funcs], [func.origin_decl for func in invalid_funcs])
        }
        return dump_dict

    def save_func(self, func):
        # Duplicate MIR Function, like
        #   <T as std::array::SpecArrayClone>::clone::<N>
        #   impl std::iter::adapters::fuse::FuseImpl<I>
        #       for std::iter::Fuse<I> next
        # No idea why there are different function with same
        #   Name, just ignore them by now
        if self.session.query(self.func_type).filter_by(identifier=func.identifier).first() is not None:
            self.session.query(self.func_type).filter_by(
                identifier=func.identifier).update({'duplicate_def': True})
        else:
            self.session.add(func)
            self.save_func_data(func)


class BinaryAnalyser(BaseAnalyser):

    def __init__(self, raw_dir, formatted_dir, crate_list, session) -> None:
        super().__init__(raw_dir, formatted_dir, crate_list, session)
        self.func_type = BinFunc

    # target crate: the crate which bin file belongs to
    def load_raw(self, target_crate):
        funcs = self.load_cfg_info(target_crate)
        self.load_bin(funcs, target_crate)
        self.dump_statistic(funcs, f'{target_crate}_bin_info')
        return funcs
        # self.load_string_info(os.path.join(self.base_dir, 'strings.json'))

    def load_cfg_info(self, target_crate):
        file_path = os.path.join(self.raw_dir, target_crate, 'cfg.json')
        with open(file_path) as f:
            cfg_info = json.load(f)
        funcs = [BinFunc(target_crate, info_dict)
                 for info_dict in cfg_info]
        for func in filter(lambda x: x.valid() and
                           not self.session.query(BinFunc)
                           .filter_by(identifier=x.identifier).first() is None, funcs):
            func.is_new = False
        for func in filter(lambda x: x.valid() and x.is_new, funcs):
            if sum(func.block_length_list) < 10:
                func.errno |= FunctionAnalErrorCode.BinFileError
            if func.crate not in self.crate_list:
                func.errno |= FunctionAnalErrorCode.NotWantedCrate
        return funcs

    def load_bin(self, funcs, target_crate):
        bin_dir = os.path.join(self.raw_dir, target_crate, 'bin')
        for func in filter(lambda x: x.valid(), funcs):
            func.load_bin(bin_dir)

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
        return self.session.query(BinFunc).filter(BinFunc.match_mir != '').filter_by(**kwargs).all()


class MirAnalyser(BaseAnalyser):

    def __init__(self, raw_dir, formatted_dir, crate_list, session) -> None:
        super().__init__(raw_dir, formatted_dir, crate_list, session)
        self.func_type = MirFunc
        self.query_buf = set()

    def load_raw(self, target_crate):
        funcs = reduce(lambda x, y: x + y,
                       [self.load_mir_info(crate, target_crate) for crate in self.crate_list])
        self.dump_statistic(funcs, f'{target_crate}_mir_info')
        return funcs

    # TODO: 
    # It would be better to check whether a function is recur-possible or not
    #   before trying to extend.
    def find_recur(self):
        raise NotImplementedError

    # Ignore functions already find in db
    def load_mir_info(self, crate, target_crate):
        crate_json = os.path.join(self.raw_dir, target_crate, f'{crate}.json')
        if not os.path.exists(crate_json):
            # print(f'File {crate_json} not exists')
            return []
        crate_info = self.format_raw_json(crate_json)
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

    def update_extend(self, func):
        self.save_func_data(func)
        self.session.query(MirFunc).filter_by(identifier=func.identifier).update({
                'extended': True, 'perfect_extended': func.perfect_extended})

    def save_extended(self, funcs):
        for func in filter(lambda x: x.valid() and x.extended, funcs):
            self.update_extend(func)
        self.flush_func_data()
        self.session.commit()

    def get_statistic(self, funcs):
        dump_dict = super().get_statistic(funcs)
        dump_dict['new_funcs_num'] = sum(
            [1 if func.is_new and func.valid() else 0 for func in funcs])
        dup_def_list = [func.into_str() for func in self.session.query(
            MirFunc).filter_by(duplicate_def=True).all()]
        dump_dict['dup_def_num'] = len(dup_def_list)
        dump_dict['dup_def_list'] = dup_def_list
        return dump_dict

    def extend_query(self, decl):
        # TODO: 
        # Handle closure
        if 'move _' in decl:
            return ExtendErrorCode.Closure, None

        # This MirFunc is not normally init, only used to
        #   get the identifier here
        identifier = MirFunc('', '{' + decl + '}', []).identifier

        # Put it here temporaly In detection of recur,
        #   it would be better to prevent inline outside.
        if identifier in self.query_buf:
            return ExtendErrorCode.Recur, None
        self.query_buf.add(identifier)
        targets = self.load(identifier=identifier)
        if not targets:
            return ExtendErrorCode.NotFound, None
        if len(targets) > 1:
            raise ValueError
        target_func = targets[-1]
        if target_func.duplicate_def:
            return ExtendErrorCode.DupDef, None
        if target_func.matched:
            return ExtendErrorCode.Matched, None
        self.load_func_data(target_func)
        target_func.extend_cfg(self.extend_query, self.update_extend)
        return ExtendErrorCode.NoError, target_func

    # There is something wrong with the raw_extractor,
    #   the output is always "padded" with some kind of
    #   truncated info, and is not indented.
    # Before the problem is handled, use this function
    #   to re_format raw json output
    @staticmethod
    def format_raw_json(file_path):
        with open(file_path) as f:
            raw_content = f.read()
            try:
                info_dict = json.loads(raw_content)
            except Exception as e:
                error_str = str(e)
                info_dict = json.loads(
                    raw_content[: int(error_str[error_str.find('(char ') + 6: -1])])
        with open(file_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        return info_dict
