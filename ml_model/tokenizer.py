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
    def __init__(self, vocab_size=128, max_len=2048) -> None:
        super().__init__(max_len)
        self.embedding_size = vocab_size
        self.max_len = max_len
        self.lookup = torch.eye(vocab_size)

    # Should gurantee that str is not ''
    def sample2tensor(self, sample: str) -> torch.tensor:
        if not sample:
            return torch.zeros(self.embedding_size)
        labels = [ord(c) for c in sample[:self.max_len]]
        return torch.index_select(self.lookup, 0, torch.tensor(labels, dtype=torch.long))

    def samples2tensor(self, samples: list) -> torch.tensor:
        tensor_list = [self.sample2tensor(sample) for sample in samples]
        max_len = max([tensor.shape[0] for tensor in tensor_list])
        return torch.stack([self._pad2d(tensor, max_len) for tensor in tensor_list])

    @staticmethod
    def _pad2d(tensor, length):
        target_length = 1
        while target_length < length:
            target_length *= 2
        if tensor is None:
            raise ValueError
        padding = torch.zeros(
            target_length - tensor.shape[0], tensor.shape[1])
        return torch.cat([tensor, padding])


class AsmTokenizer(Tokenizer):
    def __init__(self, max_len=4096) -> None:
        super().__init__(max_len)
        self.token_dict = TokenDict()

    # If new_token is set True, create new label, otherwise,
    #   threat all unknown token as 'ukw'
    def fix_vocab(self):
        self.token_dict.new_token = False

    # Each sample is like
    #   "push rbp\nmov rbp,rsp\n ..."
    # In seperate mode, return will be list of labels,
    #   each element corresponds to a sample, like:
    #   [1, 2, 3, 2, 4, ...
    def sample2labels(self, sample: str) -> list:
        sample = sample.split('\n')[:self.max_len]
        return reduce(lambda x, y: x + y,
                      [self.instruction2labels(instruction) for instruction in sample])

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
