import os
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_model.DPCNN import DPCNN
from ml_model.GGNN import GGNN
from ml_model.asm2vec import Asm2Vec
from ml_model.utils import (
    CharOneHotEncoder,
    DataLoader,
    edges2tensor,
)


class Bin2Mir():
    def __init__(self, train_meta={}, data_meta={}):
        self.data_meta = data_meta
        self.train_meta = train_meta
        model_params = {
            'cnn_params': {
                'channel_size': data_meta.get('TMP__CNN_CHANNEL_SIZE', 64),
                'input_dim': data_meta.get('TMP__CNN_INPUT_DIN', 128),
                'output_dim': data_meta.get('TMP__CNN_OUTPUT_DIM', 128),
            },
            'rnn_params': {
                'input_size': data_meta.get('RNN_INPUT_DIM', 64),
                'hidden_size': data_meta.get('RNN_HIDDEN_DIM', 64),
                'num_layers': data_meta.get('RNN_LAYER', 2),
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
        }
        self.model = Bin2MirModel(**model_params)
        asm2vec_params = {
            'model_params': {
                'embedding_dim': data_meta.get('ASM_EMBEDDING_SIZE', 64),
            },
        }
        self.asm2vec = Asm2Vec(**asm2vec_params)
        self.mir2vec = CharOneHotEncoder()

    def load_model(self, model_dir):
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "bin2mir.ckpt")))

    def load_pretrain_model(self, model_dir):
        self.asm2vec.load_model(model_dir)

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(),
                   os.path.join(model_dir, 'bin2mir.ckpt'))

    def train(self, data_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        train_loader = torch.utils.data.DataLoader(
            dataset=DataLoader(data_dir),
            batch_size=self.train_meta.get('BATCH_SIZE', 128),
            shuffle=True,
            collate_fn=DataLoader.batch_collector)
        criterion = nn.TripletMarginLoss(
            margin=self.train_meta.get('MARGIN', 0.5),
            p=self.train_meta.get('MEASURE', 2))
        optimizer = torch.optim.Adam(
            # Not take asm pretrain into consideration yet
            self.model.parameters(),
            lr=self.train_meta.get('LR', 1e-4))
        print('Start Training.')
        for epoch in range(self.train_meta.get('EPOCH', 64) - 1):
            total_loss, batch_cnt = 0.0, 0
            print(f"Epoch {epoch}/{self.train_meta.get('EPOCH', 64)}")
            for i, batch_content in enumerate(train_loader):
                bin_func_embeddings, mir_func_embeddings, func_label_list = self.deal_batch(
                    batch_content, device)
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
                if i % 5 == 4:
                    print(f'Step {i+1}, loss: {loss.item()}')
            print(f'Average loss {total_loss/batch_cnt}')

    def test(self, data_dir, verbose=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        inner_loader = DataLoader(data_dir)
        test_loader = torch.utils.data.DataLoader(
            dataset=inner_loader,
            batch_size=64,
            collate_fn=DataLoader.batch_collector)
        with torch.no_grad():
            outputs = [self.deal_batch(batch_content, device)
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

    def deal_batch(self, batch_content, device):
        bin_bbs_list, bin_edges_list, mir_bbs_list, mir_edges_list, func_label_list = batch_content
        bin_bbs_embeddings_list = [
            self.asm2vec(self.asm2vec.to_labels(bin_bbs, ukw=True)).to(device)
            for bin_bbs in bin_bbs_list]
        mir_bbs_onehots_list = [
            self.mir2vec(mir_bbs).unsqueeze(1).to(device) for mir_bbs in mir_bbs_list]
        bin_A_list = [edges2tensor(bin_edges, len(bin_bbs)).to(device)
                      for bin_edges, bin_bbs in zip(bin_edges_list, bin_bbs_list)]
        mir_A_list = [edges2tensor(mir_edges, len(mir_bbs)).to(device)
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


class Bin2MirModel(nn.Module):

    def __init__(self, cnn_params, bin_gnn_params,
                 mir_gnn_params, rnn_params):
        super().__init__()
        self.dpcnn = DPCNN(**cnn_params)
        self.lstm = nn.LSTM(**rnn_params)
        self.mir_gnn = GGNN(**mir_gnn_params)
        self.bin_gnn = GGNN(**bin_gnn_params)
        self.bn = nn.BatchNorm1d(mir_gnn_params['output_dim'], affine=False)

    def forward(self, bin_bbs_embeddings_list, bin_A_list, mir_bbs_onehots_list, mir_A_list):
        bin_bbs_embedding_list = [self.lstm(bin_bbs_embeddings)[0][:, -1:, ].squeeze(1)
                                  for bin_bbs_embeddings in bin_bbs_embeddings_list]
        mir_bbs_embedding_list = [self.dpcnn(mir_bbs_onehots)
                                  for mir_bbs_onehots in mir_bbs_onehots_list]
        bin_cfg_embedding_list = [
            self.bin_gnn(bin_bbs_embedding, bin_A)[0]
            for bin_bbs_embedding, bin_A in zip(bin_bbs_embedding_list, bin_A_list)
        ]
        mir_cfg_embedding_list = [
            self.mir_gnn(mir_bbs_embedding, mir_A)[0]
            for mir_bbs_embedding, mir_A in zip(mir_bbs_embedding_list, mir_A_list)
        ]
        normalized_mir_embeddings = F.normalize(
            self.bn(torch.stack(mir_cfg_embedding_list)), p=2, dim=-1)
        normalized_bin_embeddings = F.normalize(
            self.bn(torch.stack(bin_cfg_embedding_list)), p=2, dim=-1)
        return normalized_bin_embeddings, normalized_mir_embeddings
