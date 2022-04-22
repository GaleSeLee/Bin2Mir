import torch
import torch.nn as nn
import torch.nn.functional as F


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
