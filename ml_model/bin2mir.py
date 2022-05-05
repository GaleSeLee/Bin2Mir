import os
from functools import reduce

import torch
import torch.nn as nn

from ml_model.models import Bin2MirModel, DEVICE
from ml_model.tokenizer import AsmTokenizer, MirCharTokenizer
from ml_model.utils import (
    DataLoader,
    edges2tensor,
)


class Bin2Mir():

    def __init__(self, ml_dir, model_meta={}):
        self.base_dir = ml_dir
        self.asm_tokenizer = AsmTokenizer()
        self.asm_tokenizer.load_dict(os.path.join(self.base_dir, 'models'))
        self.asm_tokenizer.fix_vocab()
        self.mir_encoder = MirCharTokenizer()
        model_params = {
            'cnn_params': {
                'channel_size': model_meta.get('TMP__CNN_CHANNEL_SIZE', 64),
                'input_dim': model_meta.get('TMP__CNN_INPUT_DIN', 128),
                'output_dim': model_meta.get('TMP__CNN_OUTPUT_DIM', 128),
            },
            'rnn_params': {
                'input_size': model_meta.get('RNN_INPUT_DIM', 64),
                'hidden_size': model_meta.get('RNN_HIDDEN_DIM', 64),
                'num_layers': model_meta.get('RNN_LAYER', 2),
                'batch_first': True,
                'dropout': 0.2,
                'bidirectional': True,
            },
            'mir_gnn_params': {
                'state_dim': 128,
                'n_node': 64,
                'n_steps': 8,
                'output_dim': 128,
            },
            'bin_gnn_params': {
                'state_dim': 128,
                'n_node': 64,
                'n_steps': 8,
                'output_dim': 128,
            },
            'hbmp_params': {
                'cells': 2,
                'hidden_dim': 64,
                'lstm_conf': {
                    'input_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'bidirectional': True
                }
            },
            'cbow_params': {
                'vocab_size': self.asm_tokenizer.token_dict.dim,
                'embedding_dim': model_meta.get('ASM_EMBEDDING_SIZE', 64),
            }
        }
        self.model = Bin2MirModel(**model_params)
        self.model.to(DEVICE)

    def load_model(self, model_dir):
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "bin2mir.ckpt")))

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(),
                   os.path.join(model_dir, 'bin2mir.ckpt'))

    def train(self, file_path, train_meta={}):

        train_loader = torch.utils.data.DataLoader(
            dataset=DataLoader(file_path),
            batch_size=train_meta.get('BATCH_SIZE', 64),
            shuffle=True,
            collate_fn=DataLoader.batch_collector)
        criterion = nn.TripletMarginLoss(
            margin=train_meta.get('MARGIN', 0.5),
            p=train_meta.get('MEASURE', 2))
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_meta.get('LR', 1e-4))

        print('Start Training.')
        for epoch in range(train_meta.get('EPOCH', 64)):
            total_loss, batch_cnt = 0.0, 0
            print(f"Epoch {epoch}:")
            for i, batch_content in enumerate(train_loader):
                bin_func_embeddings, mir_func_embeddings, func_label_list = self.deal_batch(batch_content)
                neg_samples = self.norm_weighted_samples(
                    bin_func_embeddings, mir_func_embeddings, func_label_list)
                loss = criterion(
                    bin_func_embeddings,
                    mir_func_embeddings,
                    neg_samples
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                batch_cnt += 1
                if i % 100 == 99:
                    print(f'Step {i+1}, loss: {loss.item()}')
            print(f'Average loss {total_loss/batch_cnt}')

    def test(self, file_path, test_meta, verbose=False):
        raise NotImplementedError
        inner_loader = DataLoader(data_dir)
        test_loader = torch.utils.data.DataLoader(
            dataset=inner_loader,
            batch_size=64,
            collate_fn=DataLoader.batch_collector)
        with torch.no_grad():
            outputs = [self.deal_batch(batch_content)
                       for batch_content in test_loader]
            bin_func_embeddings = torch.cat([x[0] for x in outputs])
            mir_func_embeddings = torch.cat([x[1] for x in outputs])
            labels = reduce(lambda x, y: x + y, [x[2] for x in outputs])
            predict_labels = [labels[self.retrieve(bin_func_embedding, mir_func_embeddings)]
                              for bin_func_embedding in bin_func_embeddings]
        correctness = sum([1 if label == predict else 0 for label, predict in zip(
            labels, predict_labels)])/len(labels)
        print(f'Correctness: {100 * correctness}%')
        if verbose:
            idx2fnname = {v: k for k, v in inner_loader.fn2label.items()}
            for i in range(len(bin_func_embeddings)):
                bin_f = bin_func_embeddings[i]
                label, pre = labels[i], predict_labels[i]
                min_dis, ref_dis = self.verbose_retrive(
                    bin_f, mir_func_embeddings, i)
                func_name, pre_name = idx2fnname[label], idx2fnname[pre]
                print(
                    f'Ref: {ref_dis}({func_name}), Pre: {min_dis}({pre_name})')

    def deal_batch(self, batch_content):
        bin_bbs_list, bin_edges_list, mir_bbs_list, mir_edges_list, func_label_list = batch_content
        bin_bbs_embeddings_list = [
            self.model.cbow.batch_lookup(self.asm_tokenizer(bin_bbs)) for bin_bbs in bin_bbs_list]
        mir_bbs_onehots_list = [
            self.mir_encoder(mir_bbs).unsqueeze(1).to(DEVICE) for mir_bbs in mir_bbs_list]
        raise ValueError
        bin_A_list = [edges2tensor(bin_edges, len(bin_bbs)).to(DEVICE)
                      for bin_edges, bin_bbs in zip(bin_edges_list, bin_bbs_list)]
        mir_A_list = [edges2tensor(mir_edges, len(mir_bbs)).to(DEVICE)
                      for mir_edges, mir_bbs in zip(mir_edges_list, mir_bbs_list)]
        bin_func_embeddings, mir_func_embeddings = self.model(
            bin_bbs_embeddings_list, bin_A_list, mir_bbs_onehots_list, mir_A_list)
        return bin_func_embeddings, mir_func_embeddings, func_label_list

    @staticmethod
    def retrieve(bin_embedding, mir_embeddings):
        dis_vec = mir_embeddings - bin_embedding
        return torch.argmin(torch.sum(dis_vec * dis_vec, dim=-1))

    @staticmethod
    def verbose_retrive(bin_embedding, mir_embeddings, ref_idx):
        dis_vec = mir_embeddings - bin_embedding
        distances = torch.sum(dis_vec * dis_vec, dim=-1)
        min_dis = torch.min(distances)
        # average_dis = torch.mean(distances)
        ref_dis = distances[ref_idx]
        return min_dis, ref_dis

    @staticmethod
    def norm_weighted_samples(bin_embeddings, mir_embeddings, labels, dim=128, s=1.0):
        ret = []
        for bin_embedding, label in zip(bin_embeddings, labels):
            dis_vec = mir_embeddings - bin_embedding
            distances = torch.norm(dis_vec, p=2, dim=-1)
            weighted_log = -(dim - 2) * distances - (dim - 3) / \
                2 * (1 - torch.pow(distances, 2) / 4)
            for i, l in enumerate(labels):
                if l == label:
                    weighted_log[i] = -10000
            m = nn.Softmax()
            weighted = m(s * weighted_log)
            idx = torch.multinomial(weighted, num_samples=1).item()
            ret.append(mir_embeddings[idx])
        return torch.stack(ret)
