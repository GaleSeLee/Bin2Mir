from __future__ import print_function

import ghidra
from array import *
import json

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
nonx_mem_blocks = [m for m in memory.getBlocks() if not m.isExecute()]
listing = currentProgram.getListing()
decompile = DecompInterface()
ref_addr_obj = exec_mem_blocks[0].getStart()

def match_addr(ghidra_addr, plain_addr):
    if int(ghidra_addr.toString(), 16) == plain_addr:
        return True
    return False
def addr_in_range(start, end, plain_addr):
    block_start = int(start.toString(), 16)
    block_end = int(end.toString(), 16)
    if block_start <= plain_addr <= block_end:
        return True
    return False
def addr_to_obj(plain_addr):
    ref_addr_long = int(ref_addr_obj.toString(), 16)
    return ref_addr_obj.add(plain_addr - ref_addr_long)


# There are ( probably ) 2 form of using string in rust:
#   1. reference directly, like let s = "Test".
#   2. reference to a pointer, which points to a string.
# 
# Usually the block containing strings is placed
#  after the last executable block.
# While pointers are placed in later block.
# The code below depends on this huristic assumption.
def find_possible_refs(string_list):
    string_ranges = []
    string_set = []
    possible_ref_blocks = []
    possible_pointers = []

    # First, find all strings after last executable block
    # Then, filter out the strings in other blocks
    string_list_1 = []
    max_execute_addr = max([m.getEnd() for m in exec_mem_blocks])
    for s in string_list:
        s_addr = s.getAddress()
        if s_addr < max_execute_addr:
            continue
        string_list_1.append((s_addr, s.getString(memory)))
    min_string_addr = min([string[0] for string in string_list_1])
    for m in nonx_mem_blocks:
        if m.contains(min_string_addr):
            string_block = m
            break
    string_list_2 = []
    for string in string_list_1:
        if string_block.contains(string[0]):
            string_list_2.append(string)

    # Find refs of these string, missed string refs
    #   should be in the same mem block
    possible_blocks = []
    for m in nonx_mem_blocks:
        if m.isWrite() or not m.isInitialized():
            continue
        possible_blocks.append(m)
    cnt_list = [0 for m in possible_blocks]
    for string in string_list_2:
        ref_iter = check_refer(string[0])
        if ref_iter is None:
            continue
        for ref_elem in ref_iter:
            ref_from_addr = ref_elem.getFromAddress()
            for i, m in enumerate(possible_blocks):
                if m.contains(ref_from_addr):
                    cnt_list[i] += 1
    ref_block = possible_blocks[cnt_list.index(max(cnt_list))]


    # traverse undefined data in target_block, if there is a pointer in
    #   there, create data
    iter_addr = ref_block.getStart()
    end_addr =  ref_block.getEnd()
    pointer_type = ghidra.program.model.data.PointerDataType()
    while iter_addr < end_addr:
        d = listing.getDataAt(iter_addr)
        iter_addr = iter_addr.add(8)
        if not d or d.isDefined():
            continue
        data_addr = d.getAddress()
        ref_iter = check_refer(data_addr)
        if ref_iter is None:
            continue
        possible_pointer = memory.getLong(data_addr)
        if addr_in_range(string_block.getStart(), string_block.getEnd(), possible_pointer):
            try:
                createData(data_addr, pointer_type)
            except:
                print("Conflict at", data_addr)


def get_text():
    minimum_string_length = 14  # default = 1
    alignment = 1
    require_null_termination = False
    include_all_char_widths = True
    strings_list = findStrings(None, minimum_string_length, alignment, require_null_termination, include_all_char_widths)
    find_possible_refs(strings_list)


def check_refer(address):
    ref_iter = ref_manager.getReferencesTo(address)
    if ref_iter:
        return ref_iter
    else:
        None

get_text()

