# Use a crate::module coarse granularity
# Thus it is possible a moduler in one crate corresponds to
#   multiple crate.
# Do not consider rename into consideration yet
ReExportDict = {
    'std': {
        'core': [
            'any', 'array', 'async_iter', 'cell', 'clone', 'cmp', 'future',
            'hash', 'hint', 'convert', 'default', 'iter', 'pin', 'option',
            'ops', 'mem', 'marker', 'usize', 'result', 'isize', 'ptr', 'intrinsics',
            # Below are from alloc but in the essence are also from core
            'fmt', 'borrow', 'slice', 'str', 'alloc'
        ],
        'alloc': [
            'boxed', 'rc', 'string', 'vec', 'borrow', 'alloc', 'slice', 'str', 'sync'
        ]
    }
}


class FunctionAnalErrorCode:
    NoError = 0
    NotSupportedFormat = 1
    Dumplicate = 2
    # Inconsistent Length or Too small
    BinFileError = 4
    NotWantedCrate = 0x10
    Closure = 0x20


class FunctionType:
    # method & function
    Normal = 0
    Trait = 1
    Closure = 2


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

    # Unstable
    # In mir, we can find std::xxx in core.json or alloc.json
    # Temperally, we change path of bin function but keep the
    #   crate unchanged to match up the mir
    def replace_export(self):
        # Try convert bin 2 mir path, which should be
        #  Single Projection
        # Only convert core & alloc to std for now
        crate = self.path[0][0]
        if not crate in ['core', 'alloc']:
            return
        if len(self.path) > 1 and self.path[1][0] in ReExportDict['std'][crate]:
            self.path[0] = ('std', '')


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
