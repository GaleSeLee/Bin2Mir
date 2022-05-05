from functools import reduce
import os
import re
import random
from hashlib import sha1

from pwn import disasm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, Boolean,
    String, ForeignKey
)
from sqlalchemy import orm

from data_formatter.macro import (
    FunctionAnalErrorCode, FunctionType,
    RustDeclaration, MirEdge
)


Base = declarative_base()


class BaseFunc(Base):
    __tablename__ = 'function'
    # Model type, used to distinguish Bin and Mir function
    # Details: https://www.osgeo.cn/sqlalchemy/orm/inheritance.html
    mtype = Column(Integer)
    identifier = Column(String(63), primary_key=True)
    crate = Column(String(10), default='')
    # 0/1/2 -- normal/trait/closure
    ftype = Column(Integer)
    decl_str = Column(String(255))
    # class/enumerate + fn name or simply fn name

    # Only useful when ftype == 1
    impl_trait_str = Column(String(255), default='')
    impl_class_str = Column(String(255), default='')

    __mapper_args__ = {
        'polymorphic_identity': 0,
        'polymorphic_on': mtype
    }

    # When obj = Class() is executed,
    #   the __init__ constructor is invoked normally
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.errno = FunctionAnalErrorCode.NoError
        self.is_new = True
        self.full_info = True
        self.edge_list = []

    # When instances are loaded during a Query operation
    #   init_on_load is called.
    @orm.reconstructor
    def init_on_load(self):
        self.errno = FunctionAnalErrorCode.NoError
        self.is_new = False
        self.full_info = False
        self.decl = RustDeclaration(self._tokenization(self.decl_str))
        if self.ftype == FunctionType.Trait:
            self.impl_class_decl = RustDeclaration(
                self._tokenization(self.impl_class_str))
            self.impl_trait_decl = RustDeclaration(
                self._tokenization(self.impl_trait_str))

    def _gen_identifier(self):
        self.decl_str = self.decl.into_str(generic=True)
        if self.ftype == FunctionType.Trait:
            self.impl_class_str = self.impl_class_decl.into_str(generic=True)
            self.impl_trait_str = self.impl_trait_decl.into_str(generic=True)
        identity_str = self.crate + self.decl_str
        if hasattr(self, 'target_crate'):
            identity_str += self.target_crate + self.bin_file_name
        if self.ftype == FunctionType.Trait:
            identity_str += self.impl_class_str + self.impl_trait_str
        self.identifier = sha1(identity_str.encode()).digest().hex()

    def into_dict(self, all_info=False):
        if all_info:
            raise NotImplementedError
        return {
            'edge_list': self.edge_list,
            'bb_list': self.bb_list,
        }

    # return crate for mir func,
    #   target_crate for bin func
    def get_data_crate(self):
        raise NotImplementedError

    def load_data(self, edge_list, bb_list):
        self.edge_list = edge_list
        self.bb_list = bb_list
        self.full_info = True

    def valid(self):
        return self.errno == FunctionAnalErrorCode.NoError

    def analyse_decl(self, decl):
        tokens = self._tokenization(decl)
        if tokens[0][0] == '':
            # impl Trait for Class
            # Like <std::net::Ipv6Addr as std::fmt::Display>::fmt::fmt_subslice
            self.ftype = FunctionType.Trait
            tmp = tokens[0][1][1:-1].split(' as ')
            if len(tmp) != 2:
                self.errno = FunctionAnalErrorCode.NotSupportedFormat
                return
            impl_class_tokens = self._tokenization(tmp[0].strip())
            impl_trait_tokens = self._tokenization(tmp[1].strip())
            # In case of a default implementation of trait,
            #   regard it as normal function
            if impl_class_tokens[0][0] == 'Self':
                self.ftype = FunctionType.Normal
                self.decl = RustDeclaration(impl_trait_tokens + tokens[1:])
            else:
                self.impl_class_decl = RustDeclaration(impl_class_tokens)
                self.impl_trait_decl = RustDeclaration(impl_trait_tokens)
                self.decl = RustDeclaration(tokens[1:])
        else:
            self.ftype = FunctionType.Normal
            self.decl = RustDeclaration(tokens)

    def into_str(self, generic=True):
        if self.ftype == FunctionType.Closure:
            # Ignore Closure.
            return ''
        if self.ftype == FunctionType.Trait:
            return ''.join([
                f'({self.crate}, {self.ftype}) ',
                f'impl {self.impl_trait_decl.into_str(generic=generic)} ',
                f'for {self.impl_class_decl.into_str(generic=generic)} ',
                f'{self.decl.into_str(generic=generic)}'
            ])
        return f'({self.crate}, {self.ftype}) ' + self.decl.into_str(generic=generic)

    def coarse_eq(self, other, short=True, generic=False):
        if not (self.valid() and other.valid()):
            return False
        if self.ftype != other.ftype:
            return False
        if self.crate != other.crate:
            return False
        if self.ftype == FunctionType.Normal:
            return self.decl.coarse_eq(other.decl, short, generic)
        return (self.impl_trait_decl.coarse_eq(other.impl_trait_decl, short, generic)
                and self.impl_class_decl.coarse_eq(other.impl_class_decl, short, generic)
                and self.decl.coarse_eq(other.decl, short, generic))

    def __eq__(self, other):
        return self.into_str() == other.into_str()

    @staticmethod
    def _tokenization(decl):
        def _match_braket(s: str, start='<', end='>', escapes=['-']):
            assert s[0] == start
            cnt = 0
            for i in range(len(s)):
                c = s[i]
                if c == start:
                    cnt = cnt + 1
                if c == end and s[i - 1] not in escapes:
                    cnt = cnt - 1
                    if cnt == 0:
                        return i + 1
            raise ValueError(f'Failed to match braket of str "{s}"')
        tokens = []
        while decl:
            colon = decl.find('::')
            generic = decl.find('<')
            if colon == -1 and generic == -1:
                tokens.append((decl, ''))
                break
            if colon == -1:
                if generic == 0 and tokens and tokens[-1][1] == '':
                    tokens[-1] = (tokens[-1][0], decl)
                else:
                    tokens.append((decl[:generic], decl[generic:]))
                break
            if generic == -1:
                tokens.extend([(x, '') for x in decl.split('::')])
                break
            elif colon < generic:
                tokens.append((decl[:colon], ''))
                decl = decl[colon + 2:]
            else:
                tmp = decl[:generic]
                decl = decl[generic:]
                generic_ending = _match_braket(decl)
                if tmp == '' and tokens and tokens[-1][1] == '':
                    tokens[-1] = (tokens[-1][0], decl[:generic_ending])
                else:
                    tokens.append((tmp, decl[:generic_ending]))
                decl = decl[generic_ending + 2:]
        return tokens


class MirFunc(BaseFunc):
    __tablename__ = 'mir_func'
    __mapper_args__ = {
        'polymorphic_identity': 1,
    }
    identifier = Column(String(63), ForeignKey(
        'function.identifier'), primary_key=True)
    duplicate_def = Column(Boolean, default=False)

    def __init__(self, crate, fndef_info, bb_list):
        super().__init__()
        self.crate = crate
        self.origin_decl = fndef_info
        self.analyse_def(fndef_info)
        if self.valid():
            self.raw_bb_list = bb_list
            self._gen_identifier()

    def analyse_def(self, fndef_info):
        if fndef_info.startswith('[closure@'):
            self.ftype = FunctionType.Closure
            self.errno = FunctionAnalErrorCode.Closure
        else:
            cur_bra, cur_ket = fndef_info.find('{'), fndef_info.find('}')
            if fndef_info[cur_bra + 1:].find('{') != -1 or fndef_info[cur_ket + 1:].find('}') != -1:
                self.errno = FunctionAnalErrorCode.NotSupportedFormat
                # print(fndef_info)
        if self.valid():
            # self.analyse_signature(fndef_info[:cur_bra].strip())
            self.analyse_decl(fndef_info[cur_bra + 1: cur_ket])

    def analyse_signature(self, sig):
        raise NotImplementedError

    def analyse_bb_list(self):
        self.bb_list = [MirBb(id=bb[0], **bb[1]) for bb in self.raw_bb_list]
        self.raw_bb_list.clear()
        self.edge_list = reduce(
            lambda x, y: x + y,
            [[(t[0], bb.id, t[1])
                for t in filter(lambda x: x[1] is not None, bb.term_dst())]
                for bb in self.bb_list])

    def get_bb_list(self):
        return [bb.into_str() for bb in self.bb_list]

    def get_all_strs(self):
        return reduce(lambda x, y: x + y, [bb.ref_strs for bb in self.bb_list])

    def get_data_crate(self):
        return self.crate

    def into_dict(self, all_info=False):
        ret = super().into_dict(all_info)
        ret['bb_list'] = [bb.into_str() for bb in self.bb_list]
        return ret

    def load_data(self, edge_list, bb_list):
        self.edge_list = edge_list
        self.bb_list = [MirBb(**bb) for bb in bb_list]
        self.full_info = True


class MirBb():

    def __init__(self, id, statements, term, is_cleanup=False, ref_strs=[]):
        self.id = id
        self.term = term
        self.statements = statements
        self.ref_strs = ref_strs
        self.is_cleanup = is_cleanup

    def into_dict(self):
        return {
            'id': self.id,
            'term': self.term,
            'statements': self.statements,
            'is_cleanup': self.is_cleanup,
            'ref_strs': self.ref_strs,
        }

    def into_str(self):
        return '\n'.join(self.statements) + f'\n{self.term_str()}'

    def term_str(self):
        # Return, Abort, Resume, Unreachable, GeneratorDrop
        if isinstance(self.term, str):
            return self.term.lower()
        for term_type, d in self.term.items():
            if term_type == 'Call':
                return f'call {d["args"]}'
            if term_type == 'Assert':
                return f'assert {d["cond"]}'
        return term_type.lower()

    def term_dst(self):
        term = self.term
        if isinstance(term, str):
            return []
        # Only one pair
        for term_type, d in term.items():
            if term_type == 'Call':
                return [(MirEdge.call, d['dest']), (MirEdge.clean, d['cleanup'])]
            if term_type == 'SwitchInt':
                return [(MirEdge.switch, t) for t in set(d['targets'])]
            if term_type in ['Goto', 'InlineAsm']:
                return [(MirEdge.goto, d.get('target', d.get('dest')))]
            if term_type in ['Drop', 'DropAndReplace']:
                return [(MirEdge.drop, d['target']), (MirEdge.unwind, d['unwind'])]
            if term_type == 'Assert':
                return [(MirEdge.ast, d['target']), (MirEdge.clean, d['cleanup'])]
            if term_type == 'Yield':
                return [(MirEdge.yld, d['resume']), (MirEdge.drop, d['drop'])]
            if term_type in ['FalseEdge', 'FalseUnwind']:
                return [
                    (MirEdge.goto, d['real_target']),
                    (MirEdge.false_edge, d.get('imaginary_target', d.get('unwind')))]
        return []


class BinFunc(BaseFunc):
    __tablename__ = 'bin_func'
    __mapper_args__ = {
        'polymorphic_identity': 2,
    }
    identifier = Column(String(63), ForeignKey(
        'function.identifier'), primary_key=True)
    target_crate = Column(String(15), default='')
    match_mir = Column(String(63), ForeignKey(
        'mir_func.identifier'), default='')
    coarse_list = Column(String(255), default='')

    def __init__(self, target_crate, info_dict):
        super().__init__()
        self.target_crate = target_crate
        self.bin_file_name = info_dict['identifier']
        self.addr_ranges = info_dict['addr_ranges']
        self.origin_decl = info_dict['function_name']
        self.edge_list = []
        if '{closure}' in self.origin_decl:
            self.errno = FunctionAnalErrorCode.Closure
            return
        self.analyse_decl(self.origin_decl)
        if self.valid():
            self._gen_identifier()
            self.analyse_bb_list(info_dict['basic_block'])

    def analyse_decl(self, decl):
        super().analyse_decl(self.adjust_decl(decl))
        if not self.valid():
            return
        if self.ftype == FunctionType.Normal:
            self.crate = self.decl.path[0][0]
            self.decl.replace_export()
        elif self.ftype == FunctionType.Trait:
            self.crate = self.impl_class_decl.path[0][0]
            self.impl_class_decl.replace_export()
            self.impl_trait_decl.replace_export()
        if self.crate and not self.crate.islower():
            self.errno = FunctionAnalErrorCode.NotSupportedFormat

    def analyse_bb_list(self, bb_list):
        for bb in bb_list:
            # temporally ignore indirect jump
            for tg in bb['goto']:
                self.edge_list.append((bb['id'], tg))
            for ctg in bb['cond_goto']:
                self.edge_list.append((bb['id'], ctg))
        self.block_length_list = list(
            map(lambda x: x[1] + 1 - x[0], [bb['addr_range'] for bb in bb_list]))

    def load_bin(self, base_dir):
        bin_file_path = os.path.join(base_dir, f'{self.bin_file_name}.bin')
        with open(bin_file_path, 'rb') as f:
            bin_file = f.read()
        file_len = len(bin_file)
        sum_ranges = sum(map(lambda x: 1 + x[1] - x[0], self.addr_ranges))
        sum_bb_ranges = sum(self.block_length_list)
        if not (file_len == sum_ranges == sum_bb_ranges):
            # print(f"Inconsistency in {self}: {file_len} | {sum_ranges} | {sum_bb_ranges}")
            self.errno = FunctionAnalErrorCode.BinFileError
            return
        # Assume blocks are ordered by address,
        #     only need length and origin file to get contents.
        self.bb_list = []
        for l in self.block_length_list:
            self.bb_list.append(
                self.normlize_asm(disasm(bin_file[:l], arch='amd64', byte=False, offset=False)))
            bin_file = bin_file[l:]

    def random_walk(self, path_num=3, block_num=20):
        cfg_d = {}
        for edge in self.edge_list:
            f, t = edge
            if f not in cfg_d:
                cfg_d[f] = [t]
            else:
                cfg_d[f].append(t)
        cur_paths = [[0] for _ in range(path_num)]
        for _ in range(block_num):
            for path in cur_paths:
                nexts = cfg_d.get(path[-1], [])
                if not nexts:
                    continue
                path.append(random.sample(cfg_d[path[-1]], k=1)[-1])
        paths = [','.join([str(i) for i in path]) for path in cur_paths]
        return [reduce(lambda x, y: x + y, '\n'.join([self.bb_list[int(i)] for i in path.split(',')]))
                for path in set(paths)]

    def get_data_crate(self):
        return self.target_crate

    @staticmethod
    # Experimental, under change
    def normlize_asm(instructions):
        def normlize_oprand(oprand):
            oprand = re.sub(r'#.*', '', oprand.replace(' ', ''))
            oprand = re.sub(r'.*PTR', 'ptr_', oprand)
            oprand = re.sub(r'\[(.*)\].*', r'\1', oprand)
            if oprand.startswith('0x'):
                return 'imm'
            oprand = re.sub(r'0x[\dabcdef]*', 'imm',
                            re.sub(r'\*\d', '', oprand))
            oprand = re.sub(r'[+-]', '_', oprand)
            return oprand
        ret = []
        for instruction in instructions.split('\n'):
            if instruction.find(' ') == -1:
                ret.append(instruction)
                continue
            opcode = instruction[:instruction.find(' ')]
            # Not fully understand the instruction,
            #   temporally ignore its oprands
            if opcode == 'rep':
                ret.append('rep')
                continue
            oprands = list(map(
                normlize_oprand,
                instruction[instruction.find(' '):].split(',')
            ))
            ret.append(opcode + ' ' + ','.join(oprands))
        return '\n'.join(ret)

    # Unstable
    @staticmethod
    def adjust_decl(decl):
        return decl.replace('_as_', ' as ').replace('>as_', '> as ').replace('<impl_', '<impl ')


def create_scheme(engine):
    Base.metadata.create_all(engine)
