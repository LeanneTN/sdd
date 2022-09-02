import torch
from torch.nn import Module, Sequential, Linear, BatchNorm1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(Module):

    def __init__(self, input_dim: int, abnormal_rate: float):
        super(VAE, self).__init__()  # module's init function
        hidden_dim = max(input_dim // 2, 16)
        z_dim = max(input_dim // 4, 8)
        self.norm = BatchNorm1d(input_dim)
        self.abnormal_rate = abnormal_rate
        # 编码器
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            Swish(),
            Linear(hidden_dim, z_dim),
            Swish()
        )

        # 隐变量计算的全连接层
        self.project_m = Linear(z_dim, z_dim)
        self.project_lv = Linear(z_dim, z_dim)

        # 解码器
        self.decoder = Sequential(
            Linear(z_dim, z_dim),
            Swish(),
            Linear(z_dim, hidden_dim),
            Swish(),
            Linear(hidden_dim, input_dim),
            BatchNorm1d(input_dim)
        )


class Swish(Module):
    def __init__(self, b: float = 1.0):
        super(Swish, self).__init__()
        self.b = b

    def forward(self, x):
        x = x * torch.sigmoid(self.b * x)
        return x
