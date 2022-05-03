import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

class HBMP(nn.Module):
    """
    Hierarchical Bi-LSTM Max Pooling Encoder
    """
    def __init__(self, config):
        super(HBMP, self).__init__()
        self.config = config
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.cells = config.cells
        self.hidden_dim = config.hidden_dim
        self.rnn1 = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.layers,
                            dropout=config.dropout,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.layers,
                            dropout=config.dropout,
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.layers,
                            dropout=config.dropout,
                            bidirectional=True)


    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.config.hidden_dim).zero_())
        out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        emb1 = self.max_pool(out1.permute(1,2,0)).permute(2,0,1)

        out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        emb2 = self.max_pool(out2.permute(1,2,0)).permute(2,0,1)

        out3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
        emb3 = self.max_pool(out3.permute(1,2,0)).permute(2,0,1)

        emb = torch.cat([emb1, emb2, emb3], 2)
        emb = emb.squeeze(0)

        return emb

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node):
        super(Propogator, self).__init__()

        self.n_node = n_node

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A_in):
        A_out = A_in.permute(1, 0)

        a_in = torch.mm(A_in, state_in)
        a_out = torch.mm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), -1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), -1)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Take one graph as input and generate nodes' embedding
    """
    def __init__(self, state_dim, n_node, n_steps, output_dim):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_node = n_node
        self.n_steps = n_steps

        self.in_fc = nn.Linear(self.state_dim, self.state_dim)
        self.out_fc = nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node)

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
        for i_step in range(self.n_steps):
            in_state = self.in_fc(prop_state)
            out_state = self.out_fc(prop_state)

            prop_state = self.propogator(in_state, out_state, prop_state, A)

        output = self.out(prop_state)

        return output

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, channel_size, input_dim, output_dim=64):
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