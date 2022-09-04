import torch
from torch.nn import Module, Sequential, Linear, BatchNorm1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(Module):

    def __init__(self, input_dim: int, abnormal_rate: float):
        super(VAE, self).__init__()
        hidden_dim = max(input_dim // 2, 16)
        z_dim = max(input_dim // 4, 8)
        self.norm = BatchNorm1d(input_dim)
        self.abnormal_rate = abnormal_rate

        # 构建encoder
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            Swish(),
            Linear(hidden_dim, z_dim),
            Swish()
        )

        # 构建隐变量
        self.mean = Linear(z_dim, z_dim)
        self.log_var = Linear(z_dim, z_dim)

        # 构建decoder
        self.decoder = Sequential(
            Linear(z_dim, z_dim),
            Swish(),
            Linear(z_dim, hidden_dim),
            Swish(),
            Linear(hidden_dim, input_dim),
            BatchNorm1d(input_dim)
        )

    def forward(self, x):
        norm_x = self.norm(x)
        hidden_x = self.encoder(norm_x)
        mean_x = self.mean(hidden_x)
        lv_x = self.log_var(hidden_x)
        z = self.reparameterize(mean_x, lv_x)
        x_hat = self.decoder(z)
        return x_hat, norm_x, mean_x, lv_x

    @staticmethod
    def reparameterize(mean, log_var):
        """
        将encoder得到的隐变量mean和log_var与按照正态分布随机取出的epsilon进行混合
        得到回输入decoder的值
        :param mean: tensor 向量的平均值
        :param log_var: tensor 向量的方差值
        :return: tensor
        """
        epsilon = torch.normal(0, 1, size=mean.size()).to(device)
        std = torch.exp(log_var) ** 0.5
        z = mean + std * epsilon
        return z


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