from functools import reduce
import os
from collections import defaultdict

import json
import torch


class Tokenizer():
    def __init__(self, max_len=2048) -> None:
        self.max_len = max_len

    def sample2labels(self, sample: str) -> list:
        raise NotImplemented

    def sample2tensor(self, sample: str) -> torch.tensor:
        labels = self.sample2labels(sample)
        if not labels:
            return None
        return torch.tensor(self.sample2labels(sample), dtype=torch.long)

    # Return a list of 1d tensor with dype=long,
    #   each tensor represents the label list of each sample.
    # No padding.
    # If sample is of length 0, corresponding element will
    #   be None
    def __call__(self, samples: list) -> list:
        return [self.sample2tensor(sample) for sample in samples]


class MirCharTokenizer(Tokenizer):
    def __init__(self, max_len=2048) -> None:
        super().__init__(max_len)
        self.embedding_size = 128
        self.max_len = max_len

    def sample2labels(self, sample: str):
        return [ord(c) for c in sample[:self.max_len]]


class AsmTokenizer(Tokenizer):
    def __init__(self, max_len=4096) -> None:
        super().__init__(max_len)
        # 0 for train, 1 for test
        self.mode = 0
        self.token_dict = TokenDict()

    # If new_token is set True, create new label, otherwise,
    #   threat all unknown token as 'ukw'
    def train_mode(self):
        self.mode = 0
        self.token_dict.new_token = True

    def test_mode(self):
        self.mode = 1
        self.token_dict.new_token = False

    # Each sample is a list of instrustion, like
    #   ["push rbp", "mov rbp,rsp", ...
    # In seperate mode, return will be list of labels,
    #   each element corresponds to a sample, like:
    #   [1, 2, 3, 2, 4, ...
    def sample2labels(self, sample: list) -> list:
        return reduce(lambda x, y: x + y,
                      [self.instruction2labels(instruction) for instruction in sample[:self.max_len]])

    def instruction2labels(self, instruction: str) -> list:
        tokens = instruction.split()
        opcode = [self.token_dict[tokens[0]]]
        oprands = [] if len(tokens) == 1 else [self.token_dict[oprand]
                                               for oprand in tokens.pop().split(',')]
        return opcode + oprands

    def save_dict(self, model_dir):
        with open(os.path.join(model_dir, 'asm_token_dict.json'), 'w') as f:
            json.dump(self.token_dict.dict, f)

    def load_dict(self, model_dir):
        self.token_dict.load(os.path.join(model_dir, 'asm_token_dict.json'))


class TokenDict():

    def __init__(self):
        self.dict = defaultdict(self.incrementer)
        self.new_token = True
        # It seems that "self.dict['ukw'] = 0"
        #   will not cause all self.incrementer.
        #   Make sense.
        self.dict['ukw'] = 0
        self.dim = 1

    def load(self, file_path):
        with open(file_path) as f:
            self.dict.update(json.load(f))
        self.dim = len(self.dict)

    def incrementer(self):
        if not self.new_token:
            return 0
        v = self.dim
        self.dim += 1
        return v

    def __getitem__(self, key):
        return self.dict[key]

    def __call__(self, key):
        return self.dict[key]

    def __str__(self):
        return str(self.dict)
