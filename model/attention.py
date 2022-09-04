import torch
from torch.nn import Module, Linear, Sequential, BatchNorm1d, ReLU, Sigmoid, Dropout, Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(Module):

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(Attention, self).__init__()
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


def norm_and_activation(dim: int, activation: str = 'swish'):
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
