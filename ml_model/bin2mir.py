import os

import torch
import torch.nn as nn

from ml_model.models import Bin2MirModel, DEVICE
from ml_model.tokenizer import AsmTokenizer, MirCharTokenizer
from ml_model.utils import DataLoader


class Bin2Mir():

    def __init__(self, ml_dir, asm_embedding_dim=64, mir_cnn_chan=96,
                 lstm_hidden_dim=64, lstm_layer=1, propo_steps=3,
                 final_embedding_dim=96, with_vnode=True, with_normalize=True):
        self.base_dir = ml_dir
        self.asm_tokenizer = AsmTokenizer()
        self.asm_tokenizer.load_dict(os.path.join(self.base_dir, 'models'))
        self.asm_tokenizer.fix_vocab()
        self.mir_encoder = MirCharTokenizer()
        model_params = {
            'with_vnode': with_vnode,
            'with_normalize': with_normalize,
            'cnn_params': {
                'channel_size': mir_cnn_chan,
                'input_dim': 96,  # printable ascii
                'output_dim': final_embedding_dim,
            },
            'hbmp_params': {
                'out_dim': final_embedding_dim,
                'hidden_size': lstm_hidden_dim,
                'batch_first': True,
                'input_size': asm_embedding_dim,
                'num_layers': lstm_layer,
                # 'dropout': 0.2,
                'bidirectional': True
            },
            'mir_gnn_params': {
                'state_dim': final_embedding_dim,
                'n_steps': propo_steps,
                'output_dim': final_embedding_dim,
            },
            'bin_gnn_params': {
                'state_dim': final_embedding_dim,
                'n_steps': propo_steps,
                'output_dim': final_embedding_dim,
            },
            'cbow_params': {
                'vocab_size': self.asm_tokenizer.token_dict.dim,
                'embedding_dim': asm_embedding_dim,
            }
        }
        self.model = Bin2MirModel(**model_params)
        self.model.cbow.load_state_dict(torch.load(
            os.path.join(self.base_dir, 'models', 'asm_cbow_model.ckpt')))
        self.model.to(DEVICE)

    def load_model(self, model_dir=''):
        if not model_dir:
            model_dir = os.path.join(self.base_dir, 'models')
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "bin2mir.ckpt")))

    def save_model(self, model_dir=''):
        if not model_dir:
            model_dir = os.path.join(self.base_dir, 'models')
        torch.save(self.model.state_dict(),
                   os.path.join(model_dir, 'bin2mir.ckpt'))

    def save_mir_embeddings(self, d, embedding_dir=''):
        if not embedding_dir:
            embedding_dir = os.path.join(self.base_dir, 'db')
        embedding_db_path = os.path.join(embedding_dir, 'mir_funcs.ckpt')
        torch.save(d, embedding_db_path)

    def load_mir_embeddings(self, embedding_dir=''):
        if not embedding_dir:
            embedding_dir = os.path.join(self.base_dir, 'db')
        embedding_db_path = os.path.join(embedding_dir, 'mir_funcs.ckpt')
        return torch.load(embedding_db_path)

    def train(self, file_path, train_meta={}):

        inner_loader = DataLoader(file_path, bin_bb_limit=train_meta.get('bb_limit', 50),
                                  mir_bb_limit=train_meta.get('mir_limit', 100))
        train_loader = torch.utils.data.DataLoader(
            dataset=inner_loader,
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
            for batch_content in train_loader:
                bin_func_embeddings, mir_func_embeddings, _ = self.deal_batch(
                    batch_content)
                neg_samples = self.norm_weighted_samples(
                    bin_func_embeddings, mir_func_embeddings)
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
            print(f'Average loss {total_loss/batch_cnt}')

        d = {}
        with torch.no_grad():
            for batch_content in train_loader:
                _, mir_func_embeddings, labels = self.deal_batch(batch_content)
                for label, mir_func_embedding in zip(labels, mir_func_embeddings):
                    d[label] = mir_func_embedding
        self.save_mir_embeddings(d)

    def dum_test(self, file_path):
        inner_loader = DataLoader(file_path)
        mir_d = self.load_mir_embeddings()
        mir_embeddings, mir_labels = []
        for k, v in mir_d.items():
            mir_embeddings.append(v)
            mir_labels.append(k)
        mir_embeddings = torch.stack(mir_embeddings)
        bin_d = inner_loader.get_bin_funcs()
        correct, tol = 0, 0
        with torch.no_grad():
            for k, v in bin_d.items():
                if k not in mir_d:
                    continue
                bin_embeddings = self.deal_batch_bin(v)
                predict_labels = [mir_labels[self.retrieve(bin_embedding, mir_embeddings)]
                                  for bin_embedding in bin_embeddings]
                tol += len(predict_labels)
                correct += sum([1 if k ==
                               predict else 0 for predict in predict_labels])
        print(f'Correctness: {100 * correct/tol}%')

    def practicle_test(self):
        raise NotImplementedError

    def deal_batch(self, batch_content):
        bin_bbs_list, bin_edges_list, mir_bbs_list, mir_edges_list, func_label_list = batch_content
        bin_bbs_embeddings_list = [
            self.model.cbow.batch_lookup(self.asm_tokenizer(bin_bbs)) for bin_bbs in bin_bbs_list]
        mir_bbs_onehots_list = [
            self.mir_encoder.samples2tensor(mir_bbs).unsqueeze(1).to(DEVICE) for mir_bbs in mir_bbs_list]
        bin_func_embeddings, mir_func_embeddings = self.model(
            bin_bbs_embeddings_list, bin_edges_list, mir_bbs_onehots_list, mir_edges_list)
        return bin_func_embeddings, mir_func_embeddings, func_label_list

    def deal_batch_bin(self, batch_content):
        bin_bbs_list, bin_edges_list = batch_content
        bin_bbs_embeddings_list = [
            self.model.cbow.batch_lookup(self.asm_tokenizer(bin_bbs)) for bin_bbs in bin_bbs_list]
        return self.model.embed_bin(bin_bbs_embeddings_list, bin_edges_list)

    @staticmethod
    def add_vnode(nodes_embedding: torch.tensor) -> torch.tensor:
        vnode = torch.zeros(nodes_embedding.shape)

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

    # Used for train, no dup symbols
    @staticmethod
    def norm_weighted_samples(bin_embeddings, mir_embeddings, dim=128, s=1.0):
        ret = []
        for i, bin_embedding in enumerate(bin_embeddings):
            dis_vec = mir_embeddings - bin_embedding
            distances = torch.norm(dis_vec, p=2, dim=-1)
            weighted_log = -(dim - 2) * distances - (dim - 3)
            weighted_log[i] = -10000
            m = nn.Softmax()
            weighted = m(s * weighted_log)
            idx = torch.multinomial(weighted, num_samples=1).item()
            ret.append(mir_embeddings[idx])
        return torch.stack(ret)
