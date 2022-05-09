import json
import os

import ghidra
from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.block import BasicBlockModel
from ghidra.app.util.exporter import BinaryExporter

from java.io import File

try:
    from ghidra.ghidra_builtins import *
except:
    pass


def get_functions():
    """ Initialize all functions as a list of func """
    func_list = []

    func_db = currentProgram.getFunctionManager()
    func_iter = func_db.getFunctions(True)
    while func_iter.hasNext():
        current_func = func_iter.next()
        func_list.append(current_func)
    return func_list


def get_basic_blocks(func_body):
    """ Collect basic blocks from the function body and convert them into a list """
    block_model_iterator = BasicBlockModel(currentProgram)
    b_iter = block_model_iterator.getCodeBlocksContaining(func_body, monitor)

    blocks = []
    while b_iter.hasNext():
        block = b_iter.next()
        blocks.append(block)

    return blocks

def gen_cfg(func):
    """ Generate CFG, LCG and quoted string for target function """
    #decompile = DecompInterface()
    #decompile_res = decompile.decompileFunction(func, 30, getMonitor())
    #cur_high_func = decompile_res.getFunction()

    body = func.getBody()
    blocks = get_basic_blocks(body)
    meta_dict = {
        "function_name": func.comment,
        "basic_block": []
    }

    for block in blocks:
        base_id = blocks.index(block)
        block_dict = {"id": base_id}
        call_list = []
        cfg_list = []
        ind_jumps = []
        cond_jumps = []
        des_iter = block.getDestinations(monitor)

        while des_iter.hasNext():
            des = des_iter.next()
            des_type = des.flowType
            addr = des.getDestinationAddress()

            if des.getDestinationBlock() in blocks:
                target_id = blocks.index(des.getDestinationBlock())
                if des_type is not des_type.CONDITIONAL_JUMP:
                    cfg_list.append(target_id)
                else:
                    cond_jumps.append(target_id)
            elif des_type == des_type.INDIRECTION:
                ind_jumps.append(getSymbolAt(addr).getName())
            else:
                # Ghidra analyser may fail to analyse some function
                target_func = getFunctionAt(addr)
                if target_func and func.comment:
                    call_list.append(func.comment)
                # call_list.append(addr.getOffset())

        block_dict["goto"] = cfg_list
        block_dict["cond_goto"] = cond_jumps 
        block_dict["call"] = call_list
        block_dict['indirect_jump'] = ind_jumps
        block_dict['addr_range'] = map(
                                    lambda x: (x.getMinAddress().getOffset(), x.getMaxAddress().getOffset()),
                                    block.getAddressRanges())[0]
        meta_dict["basic_block"].append(block_dict)
    meta_dict['addr_ranges'] = list(map(
        lambda x: (x.getMinAddress().getOffset(), x.getMaxAddress().getOffset()),
        body.getAddressRanges()))
    # meta_dict["identifier"] = binascii.hexlify(sha1(meta_dict["function_name"] + str(meta_dict['addr_ranges'][0])).digest())
    meta_dict["identifier"] = func.getEntryPoint().toString()

    asv = func.getBody()
    exporter = BinaryExporter()
    base_dir = os.path.join("PATH_DIR", getProgramFile().getName())
    f = File(os.path.join(base_dir, 'bin', meta_dict["identifier"] + ".bin"))
    ret = exporter.export(f, currentProgram, asv, None)

    return meta_dict

if __name__ == '__main__':
    base_dir = os.path.join("PATH_DIR", getProgramFile().getName())
    os.makedirs(os.path.join(base_dir, 'bin'))
    func_list = get_functions()
    func_blocks = []
    for func in func_list:
        if not func.comment:
            continue
        cfg_info = gen_cfg(func)
        if cfg_info:
            func_blocks.append(cfg_info)
    with open(os.path.join(base_dir, 'cfg.json'), 'w') as f:
        json.dump(func_blocks, f, indent=2)
