import os
import json

import torch
from torch import nn
import torch.nn.functional as F

from ml_model.models import CBOW, DEVICE
from ml_model.tokenizer import AsmTokenizer


class Asm2Vec():

    def __init__(self, model_dir='', data_dir='', model_params={}):
        self.encoder = AsmTokenizer()
        if model_dir:
            self.load_model(model_dir)
        elif data_dir:
            self.new_model(data_dir, model_params)
        
    def new_model(self, data_dir, model_params={}):
        self.load_train_data(data_dir)
        self.model = CBOW(self.encoder.token_dict.dim, **model_params)

    def train(self, meta_info={}):
        self.model.to(DEVICE)
        context_size = meta_info.get('CONTEXT_SIZE', 2)
        self.data = []
        for sample in self.train_samples:
            for i in range(context_size, len(sample) - context_size):
                context = torch.cat([sample[i-context_size: i], sample[i+1: i+context_size+1]])
                target = sample[i]
                self.data.append((context, target))
        # print("Data example:", self.data[:3])

        def collect_fn(batch_data):
            inputs = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]
            return inputs, targets

        loss_function = nn.NLLLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), meta_info.get('LR', 1e-5))
        train_loader = torch.utils.data.DataLoader(
            dataset=self.data,
            batch_size=meta_info.get('BATCH_SIZE', 128),
            shuffle=True,
            collate_fn=collect_fn)

        step_cnts = len(self.data)//meta_info.get('BATCH_SIZE', 128) + 1
        print("Start Training")
        for epoch in range(meta_info.get('EPOCH', 32)):
            total_loss = 0.0
            print(f"Epoch: {epoch}")
            for i, batch_data in enumerate(train_loader):
                local_loss = 0.0
                self.model.zero_grad()
                inputs, targets = batch_data
                for context, target in zip(inputs, targets):
                    context_idxs = torch.tensor(
                        context, dtype=torch.long).to(DEVICE)
                    log_probs = self.model(context_idxs)
                    loss = loss_function(log_probs, torch.tensor(
                        [target], dtype=torch.long).to(DEVICE))
                    loss.backward()
                    optimizer.step()
                    local_loss += loss.item()
                    # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += local_loss
                if i % 100 == 99:
                    print(
                        f"Step {i + 1}/{step_cnts}, local avg loss: {local_loss/meta_info.get('BATCH_SIZE', 128)}.")
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

    def load_train_data(self, data_dir):
        all_samples = []
        for json_file in filter(lambda x: x.endswith('random_walk.json'), os.listdir(data_dir)):
            with open(os.path.join(data_dir, json_file)) as f:
                samples = json.load(f)
            all_samples += [sample.split('\n') for sample in samples]
        self.train_samples = self.encoder(all_samples)
