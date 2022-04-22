import os
from collections import defaultdict
from functools import reduce
import json

import torch
from torch import nn
import torch.nn.functional as F


class InsOneHotEncoder():
    def __init__(self, seperate_mode=True):
        self.seperate_mode = seperate_mode
        # Even opcode and oprand are both tokens,
        #   seperate them make it easier to detect
        #   and inspect bug
        self.opcode_dict = TokenDict()
        self.oprand_dict = TokenDict()
        # sum of ukw, oprands, opcodes
        self.dim = 1

    def instruction2label(self, instruction: str):
        tokens = instruction.split()
        opcode = [self.opcode_dict[tokens[0]]]
        oprands = [] if len(tokens) == 1 else [self.oprand_dict[oprand]
                                               for oprand in tokens.pop().split(',')]
        return opcode, oprands

    # Each sample is a list of instrustion, like
    #   ["push rbp", "mov rbp,rsp", ...
    # In seperate mode, return will be list of labels,
    #   each element corresponds to a sample, like:
    #   [1, 51, 2, 51, 52, ...
    def __call__(self, samples: list, ukw=False) -> list:
        self.opcode_dict.set_ukw(ukw)
        self.oprand_dict.set_ukw(ukw)
        labels = [[self.instruction2label(instruction) for instruction in instructions]
                  for instructions in samples]
        if not ukw:
            self.dim = (self.opcode_dict.dim + self.oprand_dict.dim - 1 if self.seperate_mode
                        else max(self.opcode_dict.dim, self.oprand_dict.dim))
        if self.seperate_mode:
            opr_offset = self.opcode_dict.dim - 1
            return [reduce(
                lambda x, y: x + y, map(
                    lambda x: x[0] + [opr_label +
                                      opr_offset for opr_label in x[1]],
                    sample_labels)
            ) for sample_labels in labels]
        else:
            raise NotImplementedError

    def save_dict(self, model_dir):
        with open(os.path.join(model_dir, 'opc_dict.json'), 'w') as f:
            json.dump(self.opcode_dict.dict, f)
        with open(os.path.join(model_dir, 'opr_dict.json'), 'w') as f:
            json.dump(self.oprand_dict.dict, f)

    def load_dict(self, model_dir):
        with open(os.path.join(model_dir, 'opc_dict.json'), 'r') as f:
            self.opcode_dict.dict.update(json.load(f))
            self.opcode_dict.dim = len(self.opcode_dict.dict)
        with open(os.path.join(model_dir, 'opr_dict.json'), 'r') as f:
            self.oprand_dict.dict.update(json.load(f))
            self.oprand_dict.dim = len(self.oprand_dict.dict)
        self.dim = (self.opcode_dict.dim + self.oprand_dict.dim - 1 if self.seperate_mode
                    else max(self.opcode_dict.dim, self.oprand_dict.dim))


class TokenDict():

    def __init__(self):
        self.dim = 0
        self.dict = defaultdict(self.incrementer)
        # Unkwon word. When testing, set ukw to True
        # And all ukw will be labeled 0
        self.ukw = False
        # It seems that "self.dict['ukw'] = 0"
        #   will not cause all self.incrementer.
        #   Make sense.
        self.dict['ukw']

    def set_ukw(self, ukw: bool):
        self.ukw = ukw

    def incrementer(self):
        if self.ukw:
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


class Asm2Vec():
    # seperate_mode:
    #   If set, view opcode and oprands as
    #   seperate tokens. Otherwise, token
    #   consists of opc field and opr field
    def __init__(self, seperate_mode=True, model_params={}, train_meta={}):
        self.train_meta = train_meta
        self.model_params = model_params
        self.seperate_mode = seperate_mode
        if not seperate_mode:
            raise NotImplementedError
        self.encoder = InsOneHotEncoder(seperate_mode=seperate_mode)

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        context_size = self.model_params.get('CONTEXT_SIZE', 2)
        self.data = []
        for sample in self.train_samples:
            for i in range(context_size, len(sample) - context_size):
                context = sample[i-context_size: i] + \
                    sample[i+1: i+context_size+1]
                target = sample[i]
                self.data.append((context, target))
        # print("Data example:", self.data[:3])

        def collect_fn(batch_data):
            inputs = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]
            return inputs, targets

        loss_function = nn.NLLLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.train_meta.get('LR', 1e-4))
        train_loader = torch.utils.data.DataLoader(
            dataset=self.data,
            batch_size=self.train_meta.get('BATCH_SIZE', 128),
            shuffle=True,
            collate_fn=collect_fn)

        step_cnts = len(self.data)//self.train_meta.get('BATCH_SIZE', 128) + 1
        print("Start Training")
        for epoch in range(self.train_meta.get('EPOCH', 8)):
            total_loss = 0.0
            print(f"Epoch: {epoch}")
            for i, batch_data in enumerate(train_loader):
                local_loss = 0.0
                self.model.zero_grad()
                inputs, targets = batch_data
                for context, target in zip(inputs, targets):
                    context_idxs = torch.tensor(
                        context, dtype=torch.long).to(device)
                    log_probs = self.model(context_idxs)
                    loss = loss_function(log_probs, torch.tensor(
                        [target], dtype=torch.long).to(device))
                    loss.backward()
                    optimizer.step()
                    local_loss += loss.item()
                    # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += local_loss
                if i % 100 == 99:
                    print(
                        f"Step {i + 1}/{step_cnts}, local avg loss: {local_loss/self.train_meta.get('BATCH_SIZE', 128)}.")
            print(f"Average Loss: {total_loss/len(self.data)}")

    def load_model(self, model_dir):
        # print('Loading model...')
        self.encoder.load_dict(model_dir)
        self.model = CBOW(vocab_size=self.encoder.dim, **self.model_params)
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "asm_cbow_model.ckpt")))

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), os.path.join(
            model_dir, 'asm_cbow_model.ckpt'))
        self.encoder.save_dict(model_dir)

    # load:
    #    Load previous model or generate
    #    a new one with data
    # If refine old model is wanted, init with load
    #   and then call load_train_data + train
    def init_model(self, _dir: str, load=False):
        if load:
            self.load_model(_dir)
        else:
            self.load_train_data(_dir)
            self.model = CBOW(vocab_size=self.encoder.dim, **self.model_params)

    def load_train_data(self, data_dir):
        all_samples = []
        for json_file in filter(lambda x: x.endswith('.json'), os.listdir(data_dir)):
            with open(os.path.join(data_dir, json_file)) as f:
                samples = reduce(lambda x, y: x + y, json.load(f).values())
            all_samples += [sample.split('\n') for sample in samples]
        self.train_samples = self.encoder(all_samples)

    # samples to labels
    def to_labels(self, samples: list, ukw=False):
        return [torch.tensor(labels, dtype=torch.long)
                for labels in self.encoder([sample.split('\n') for sample in samples], ukw)]
    # labels_list to embeddings
    #   pad to langest sample in list

    def __call__(self, labels_list: list):
        tensor_list = [self.model.lookup(labels) for labels in labels_list]
        max_len = max([tensor.shape[0] for tensor in tensor_list])
        return torch.stack([
            F.pad(input=tensor, pad=(0, 0, 0, max_len -
                  tensor.shape[0]), mode='constant', value=0)
            for tensor in tensor_list])

    def inspect_vocab(self):
        print(self.encoder.opcode_dict)
        print(self.encoder.oprand_dict)


class CBOW(nn.Module):

    # context_size: one side
    def __init__(self, vocab_size, embedding_dim=64, context_size=2):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def lookup(self, idxs):
        return self.embeddings(idxs)
