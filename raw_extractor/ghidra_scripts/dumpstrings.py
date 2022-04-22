from __future__ import print_function

import ghidra
from array import *
import json
import binascii
from hashlib import sha1

from ghidra.app.plugin.core.navigation.locationreferences import ReferenceUtils

from ghidra.util.data import DataTypeParser
from ghidra.util.datastruct import SetAccumulator
from ghidra.program.util import ProgramLocation
from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.address import AddressSet, Address

try:
    from ghidra.ghidra_builtins import *
except:
    pass

memory = currentProgram.getMemory()
ref_manager = currentProgram.getReferenceManager()
dt_manager = currentProgram.getDataTypeManager()
parser = DataTypeParser(dt_manager, dt_manager, None, DataTypeParser.AllowedDataTypes.ALL)
exec_mem_blocks = [m for m in memory.getBlocks() if m.isExecute()]
for m in exec_mem_blocks:
    print('{} ~ {}'.format(m.getStart(), m.getEnd()))
max_execute_addr = max([m.getEnd() for m in exec_mem_blocks])
listing = currentProgram.getListing()
decompile = DecompInterface()

def is_in_exec_mem(addr):
    """returns if addr is in executable memory range
    """
    for m in exec_mem_blocks:
        if m.contains(addr):
            return True
    return False


def prepare_clearup_when_inst(cur_addr, string):
    """only called when preparing
    clear up wrongly recognized instruction (which should be string instead)
    """
    length = len(string)

    for i in range(length):
        addr = cur_addr.add(i)
        inst = getInstructionAt(addr)
        if inst:
            removeInstruction(inst)


def prepare_clearup_conflict_data(cur_addr, string):
    """only called when preparing
    clear up conflicting data
    """

    # conflicting data, clearing up
    cur_data = getDataAt(cur_addr)
    if cur_data:
        return

    while True:
        data = getDataAfter(cur_addr)
        if data is None:
            break
        data_addr = data.getMinAddress()
        if data_addr <= cur_addr.add(len(string)):
            removeData(data)
        else:
            break
    createAsciiString(cur_addr, len(string))


def prepare_seperation(cur_addr, string):
    sym = getSymbolAt(cur_addr)
    data = getDataAt(cur_addr)
    if sym is None or data is None:
        # not reconigzed string
        try:
            createAsciiString(cur_addr, len(string))
        except:

            # Two possible causes of the confliction:
            # 1. data conflicts, because previous analysis already recognized part of current data
            # ==> in this case, just undefine them, and do what WE need to do
            #
            # 2. data conflicts with instruction, due to mysteriously reconigzed string as instruction
            # bytes
            # ==> undefine all included "instructions"
            if getInstructionContaining(cur_addr):
                # XXX: what if -------- | xxxxxx | ----- ? (x stands for inst, --- for data)
                # we might need a loop to check this all
                # case 2
                prepare_clearup_when_inst(cur_addr, string)
            else:
                # case 1:
                prepare_clearup_conflict_data(cur_addr, string)


def seperate_by_xref(s):
    """by following xref, we are able to seperate some strings.
    e.g:
   ref1 ref2
    |  |
    v  v
    aaabbbccc
    by backward inspecting the string, we are able to seperate to:
    - aaa
    - bbbccc
    This might not be complete, but gives some basics.
    """

    cur_addr = s.getAddress()
    string = s.getString(memory)

    # Just in case: we should not be in code segment --
    if is_in_exec_mem(cur_addr):
        return

    prepare_seperation(cur_addr, string)
    sym = getSymbolAt(cur_addr)
    loc = ProgramLocation(currentProgram, cur_addr)

    acc = SetAccumulator()
    ReferenceUtils.getReferences(acc, loc, getMonitor())
    refs = [ref for ref in acc]
    if sym and len(refs) == sym.references:
        # reference is correct, so..
        return

    # print('seperating string: {} @ {}'.format(string, cur_addr))
    removeDataAt(cur_addr)

    last_addr = cur_addr.add(len(string))
    # initial value: itself should be a string
    possible_addrs = [cur_addr]
    presented_offset = set([cur_addr.getOffset()])

    for ref in refs:
        addr = ref.getLocationOfUse()
        upcoming_refs = [ref for ref in ref_manager.getReferencesFrom(addr) if not ref.getToAddress().equals(cur_addr)]
        if len(upcoming_refs) == 0:
            continue

        for r in upcoming_refs:
            to_addr = r.getToAddress()
            if to_addr > cur_addr and to_addr <= cur_addr.add(
                    len(string)) and to_addr.getOffset() not in presented_offset:
                presented_offset.add(to_addr.getOffset())
                possible_addrs.append(to_addr)

    possible_addrs = sorted(possible_addrs, reverse=True)

    for addr in possible_addrs:

        length = last_addr.subtract(addr)

        # print('create string length {} at {} (last addr {})'.format(length, addr, last_addr))
        try:
            createAsciiString(addr, length)
            last_addr = addr
        except:
            print('create string at {} length {} failed (possible conflicts)'.format(addr, length))
            pass


def add_ref_dump(string, addr, text_dict, ref_addr, recur=False):
    if string not in text_dict:
        text_dict[string] = {
            'refs': [],
            'indirect': [],
            'addr': addr.getOffset(),
        }

    func = getFunctionContaining(ref_addr)
    decompile_res = decompile.decompileFunction(func, 30, getMonitor())
    cur_func = decompile_res.getFunction()
    if cur_func:
        func_name = str(cur_func).replace("--", "::")
        # func_id = binascii.hexlify(sha1(func_name + str(next(func.body.getAddressRanges()).getMinAddress().getOffset())).digest())
        func_id = cur_func.getEntryPoint().toString()
        if func_id not in text_dict[string]['refs' if not recur else 'indirect']:
            text_dict[string]['refs' if not recur else 'indirect'].append(func_id)
    # It is often the case that the function references a pointer, 
    #   which references the string literal.
    elif not recur:
        ref_iter = check_refer(ref_addr)
        if ref_iter is not None:
            for indirect_addr in ref_iter:
                add_ref_dump(string, addr, text_dict, indirect_addr.getFromAddress(), recur=True)



def get_text():

    text_dict = {}
    minimum_string_length = 14  # default = 1
    alignment = 1
    require_null_termination = False
    include_all_char_widths = True

    strings_list = findStrings(None, minimum_string_length, 
        alignment, require_null_termination, include_all_char_widths)
    for s in strings_list:
        seperate_by_xref(s)
    data = listing.getDefinedData(True)
    for d in data:
        typ = d.getDataType().getName().lower()
        if 'string' in typ:
            addr = d.getMinAddress()
            string = d.getDefaultValueRepresentation()[1:-1]
            ref_iter = check_refer(addr)
            if ref_iter is not None:
                for ref in ref_iter:
                    add_ref_dump(string, addr, text_dict, ref.getFromAddress())
            
    return text_dict


def check_refer(address):
    ref_iter = ref_manager.getReferencesTo(address)
    if ref_iter:
        return ref_iter
    else:
        None


path = "/Users/liuzhanpeng/working/rust_analy/dumped/strings.json"
with open(path, "w") as f:
    f.write(json.dumps(get_text(), indent=2))
print("OK")
