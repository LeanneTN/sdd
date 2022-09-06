import torch
from torch.nn import Module, Linear, Sequential, BatchNorm1d, ReLU, Sigmoid, Dropout, Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(Module):

    def __init__(self, input_dim: int):
        super(Attention, self).__init__()
        self.norm = BatchNorm1d(input_dim)
        self.bottleNeck = bottle_neck(input_dim, 32, ratio=0.50, hidden_dim=8, dropout=0.5)
        self.linear = Linear(32, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.bottleNeck(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def bottle_neck(input_dim: int, output_dim: int, ratio: float, hidden_dim: int = 16, dropout: float = 0.0):
    return Sequential(
        Linear(input_dim, output_dim),
        BatchNorm1d(output_dim),
        Swish(),
        Bottleneck(output_dim, ratio=ratio),
        AttentionSimple(output_dim, hidden_dim=hidden_dim),
        Dropout(dropout)
    )


class Bottleneck(Module):

    def __init__(self, input_dim: int, ratio: float):
        super(Bottleneck, self).__init__()
        hidden_dim = max(int(input_dim * ratio), 1)
        self.linear = Linear(input_dim, hidden_dim)
        self.linear_2 = Linear(hidden_dim, input_dim)
        self.norm = BatchNorm1d(input_dim)

    def forward(self, x):
        bottleneck = self.linear(x)
        bottleneck = self.linear_2(bottleneck)
        res = bottleneck + x
        res = self.norm(res)
        return res


class AttentionSimple(Module):

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(AttentionSimple, self).__init__()
        self.hidden_dim = hidden_dim

        self.query_embed = Linear(input_dim, hidden_dim)
        self.key_embed = Linear(input_dim, hidden_dim)
        self.value_embed = Linear(input_dim, hidden_dim)

        self.project = Linear(hidden_dim, input_dim)
        self.norm_and_act = norm_and_activation(input_dim)

    def forward(self, x):
        b, dim = x.size()
        query = self.query_embed(x).view(b, self.hidden_dim)
        key = self.key_embed(x).view(b, self.hidden_dim)
        value = self.value_embed(x).view(b, self.hidden_dim)

        weight = torch.mul(torch.softmax(key, dim=1), value).sum(dim=1, keepdim=True)
        q_sigmoid = torch.sigmoid(query)
        y = torch.mul(weight, q_sigmoid).view(b, self.hidden_dim)
        y = self.project(y)
        out = self.norm_and_act(x + y)
        return out.contiguous()


class Swish(Module):
    """
    Swish激活函数
    """

    def __init__(self, beta: float = 1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        x = x * torch.sigmoid(self.beta * x)
        return x


def norm_and_activation(dim: int, activation: str = 'sigmoid'):
    norm = BatchNorm1d(dim)
    if activation == 'swish':
        activation = Swish()
    elif activation == 'sigmoid':
        activation = Sigmoid()
    elif activation == 'relu':
        activation = ReLU()
    else:
        print('activation %s not supported' % activation)
        exit(1)
    return Sequential(
        norm,
        activation
    )
