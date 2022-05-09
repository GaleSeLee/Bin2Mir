class FunctionAnalErrorCode:
    NoError = 0
    NotSupportedFormat = 1
    Dumplicate = 2
    # Inconsistent Length or Too small
    BinFileError = 4
    NotWantedCrate = 8
    Closure = 0x10

    @staticmethod
    def errno2str(errno):
        ret = []
        if errno & 1:
            ret.append('NotSupportedFormat')
        if errno & 2:
            ret.append('Dumplicate')
        if errno & 4:
            ret.append('BinFileError')
        if errno & 8:
            ret.append('NotWantedCrate')
        if errno & 0x10:
            ret.append('Closure')
        return '&'.join(ret)


class ExtendErrorCode:
    NoError  = 0
    Matched  = 1
    NotFound = 2
    Recur    = 3
    Closure  = 4
    DupDef   = 5

    @staticmethod
    def errno2str(errno):
        if errno == 0:
            return 'No Error'
        elif errno == 1:
            return 'Matched'
        elif errno == 2:
            return 'Not Found'
        elif errno == 3:
            return 'Recur'
        elif errno == 4:
            return 'Closure'
        else:
            return 'Duplicate Def'


class FunctionType:
    # method & function
    Normal = 0
    Trait = 1
    Closure = 2

    @staticmethod
    def tno2str(tno):
        if tno == 0:
            return 'Normal'
        if tno == 1:
            return 'Trait'
        if tno == 2:
            return 'Closure'
        raise ValueError(tno)


class RustDeclaration():

    def __init__(self, path: list) -> None:
        self.path = path

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __str__(self):
        return self.into_str()

    def coarse_eq(self, other, short=True, generic=False) -> bool:
        return self.into_str(short, generic) == other.into_str(short, generic)

    def into_str(self, short=False, generic=False):
        return '::'.join([token[0] + (token[1] if generic else '') for token in (self.path[-2:] if short else self.path)])


class BinEdge():
    Conditional = 0
    Unconditional = 1


class MirEdge():
    goto = 0
    switch = 1
    ret = 2
    call = 3
    clean = 4
    drop = 5
    unwind = 6
    # assert
    ast = 7
    abort = 8
    resume = 9
    unreachable = 10
    clean = 11
    false_edge = 12
    # yield
    yld = 13
