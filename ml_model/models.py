import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Bin2MirModel(nn.Module):

    def __init__(self, cnn_params, bin_gnn_params,
                 mir_gnn_params, hbmp_params, cbow_params,
                 with_vnode=True, with_normalize=True):
        super().__init__()
        self.dpcnn = DPCNN(**cnn_params)
        self.mir_gnn = GGNN(**mir_gnn_params)
        self.bin_gnn = GGNN(propo_class=EdgePropogator, **bin_gnn_params)
        self.hbmp = HBMP(**hbmp_params)
        self.cbow = CBOW(**cbow_params)
        self.bn = nn.BatchNorm1d(mir_gnn_params['output_dim'], affine=False)
        self.with_vnode = with_vnode
        self.with_normalize = with_normalize

    def forward(self, bin_bbs_embeddings_list, bin_edges_list, mir_bbs_onehots_list, mir_edges_list):
        with_vnode = self.with_vnode
        bin_bbs_embedding_list = [self.hbmp(bin_bbs_embeddings)
                                  for bin_bbs_embeddings in bin_bbs_embeddings_list]
        mir_bbs_embedding_list = [self.dpcnn(mir_bbs_onehots)
                                  for mir_bbs_onehots in mir_bbs_onehots_list]
        bin_A_list = [self.edges2tensor(bin_edges, len(bin_bbs_embedding), with_vnode=with_vnode)
                      for bin_edges, bin_bbs_embedding in zip(bin_edges_list, bin_bbs_embedding_list)]
        mir_A_list = [self.edges2tensor(mir_edges, len(mir_bbs_embedding), with_edge_type=True, with_vnode=with_vnode)
                      for mir_edges, mir_bbs_embedding in zip(mir_edges_list, mir_bbs_embedding_list)]
        bin_cfg_embedding_list = [
            self.bin_gnn(bin_bbs_embedding if not with_vnode else self.add_vnode(
                bin_bbs_embedding), bin_A)[0]
            for bin_bbs_embedding, bin_A in zip(bin_bbs_embedding_list, bin_A_list)
        ]
        mir_cfg_embedding_list = [
            self.mir_gnn(mir_bbs_embedding if not with_vnode else self.add_vnode(
                mir_bbs_embedding), mir_A)[0]
            for mir_bbs_embedding, mir_A in zip(mir_bbs_embedding_list, mir_A_list)
        ]
        mir_embeddings = torch.stack(mir_cfg_embedding_list)
        bin_embeddings = torch.stack(bin_cfg_embedding_list)
        if self.with_normalize:
            return F.normalize(self.bn(bin_embeddings)), F.normalize(self.bn(mir_embeddings))
        return bin_embeddings, mir_embeddings

    @staticmethod
    def add_vnode(nodes_embedding: torch.tensor) -> torch.tensor:
        vnode = torch.zeros((1, nodes_embedding.shape[-1])).to(DEVICE)
        return torch.cat([vnode, nodes_embedding])

    @staticmethod
    # params not take vnode into account
    # support edge_type from 0 to 13
    def edges2tensor(edge_list, max_idx, with_edge_type=False, with_vnode=False):
        offset = 1 if with_vnode else 0
        dims = (max_idx + offset, max_idx + offset)
        ret = torch.zeros(dims, dtype=torch.long)
        for edge in edge_list:
            ret[edge[-2] + offset, edge[-1] + offset] = 2 + (edge[0] if with_edge_type else 0)
        if with_vnode:
            # edge label 1 reserved for vnode
            ret[0, :], ret[:, 0] = 1, 1
        return ret.to(DEVICE)


# Hierarchical Bi-LSTM Max Pooling Encoder
class HBMP(nn.Module):

    # hidden size if equal to 2 * output size
    def __init__(self, out_dim, **kwargs):
        super(HBMP, self).__init__()
        self.cells = (2 if kwargs.get('bidirectional', False)
                      else 1) * kwargs.get('num_layers', 1)
        self.hidden_dim = kwargs.get('hidden_size', 64)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.rnn1 = nn.LSTM(**kwargs)
        self.rnn2 = nn.LSTM(**kwargs)
        self.rnn3 = nn.LSTM(**kwargs)
        self.linear_out = nn.Sequential(
            nn.Linear(6 * kwargs.get('hidden_size', 64), out_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        h_0 = c_0 = Variable(inputs.data.new(self.cells,
                                             batch_size,
                                             self.hidden_dim).zero_())
        out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        emb1 = self.max_pool(out1.permute(0, 2, 1)).permute(0, 2, 1)
        out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        emb2 = self.max_pool(out2.permute(0, 2, 1)).permute(0, 2, 1)
        out3, (_, _) = self.rnn3(inputs, (ht2, ct2))
        emb3 = self.max_pool(out3.permute(0, 2, 1)).permute(0, 2, 1)
        emb = torch.cat([emb1, emb2, emb3], 2)
        emb = emb.squeeze(1)
        return self.linear_out(emb)


# Gated Propogator for GGNN
# Using LSTM gating mechanism
class Propogator(nn.Module):

    def __init__(self, state_dim):
        super(Propogator, self).__init__()
        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid())
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid())
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh())

    def aggregate(self, A: torch.tensor, state: torch.tensor):
        return torch.mm(A.float(), state)

    def forward(self, state_in, state_out, state_cur, A_in):
        A_out = A_in.permute(1, 0)
        a_in = self.aggregate(A_in, state_in)
        a_out = self.aggregate(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), -1)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), -1)
        h_hat = self.tansform(joined_input)
        output = (1 - z) * state_cur + z * h_hat
        return output


class EdgePropogator(Propogator):

    def __init__(self, state_dim):
        super().__init__(state_dim)
        self.edge_lookup = nn.Embedding(
            num_embeddings=16,
            embedding_dim=state_dim)

    def aggregate(self, A, state):
        A = self.edge_lookup(A)
        return torch.sum(torch.mul(A, state), dim=1)


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Take one graph as input and generate nodes' embedding
    """

    def __init__(self, state_dim, n_steps, output_dim, propo_class=Propogator):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_steps = n_steps

        self.in_fc = nn.Linear(self.state_dim, self.state_dim)
        self.out_fc = nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = propo_class(self.state_dim)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, output_dim)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, A):
        for _ in range(self.n_steps):
            in_state = self.in_fc(prop_state)
            out_state = self.out_fc(prop_state)
            prop_state = self.propogator(in_state, out_state, prop_state, A)

        output = self.out(prop_state)

        return output


class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, channel_size, input_dim, output_dim):
        super(DPCNN, self).__init__()
        self.channel_size = channel_size
        self.conv_region_embedding = nn.Conv2d(
            1, channel_size, (3, input_dim), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(channel_size, channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*channel_size, output_dim)

    def forward(self, x):
        batch = x.shape[0]
        # Region embedding
        # [batch_size, channel_size, length, 1]
        x = self.conv_region_embedding(x)
        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        while x.size()[-2] > 2:
            x = self._block(x)
        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)
        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x


class CBOW(nn.Module):

    # context_size: one side
    def __init__(self, vocab_size, embedding_dim=64, context_size=2):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def batch_lookup(self, tensor_list):
        tensor_list = [self.lookup(tensor) for tensor in tensor_list]
        length = max([tensor.shape[0] for tensor in tensor_list])
        return torch.stack([self._pad2d(tensor, length) for tensor in tensor_list])

    def lookup(self, tensor):
        if tensor is None:
            return torch.zeros(1, self.embedding_dim).to(DEVICE)
        return self.embeddings(tensor.to(DEVICE))

    def _pad2d(self, tensor, target_length):
        padding = torch.zeros(
            target_length - tensor.shape[0], tensor.shape[1]).to(DEVICE)
        return torch.cat([tensor, padding])
