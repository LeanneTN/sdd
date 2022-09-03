from typing import List

import torch
from torch.nn import Module, Sequential, Linear, BatchNorm2d, LeakyReLU, Conv2d, ConvTranspose2d, Tanh, functional
from torch import tensor as Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs):
        super(VAE, self).__init__()  # module's init function

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter

        # todo: 重设hidden_dims
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # encoder
        for h_dim in hidden_dims:
            modules.append(
                Sequential(
                    Conv2d(in_channels, out_channels=h_dim,
                           kernel_size=3, stride=2, padding=1),
                    BatchNorm2d(h_dim),
                    LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = Sequential(*modules)  # 将modules列表中的值拆开独立作为形参
        # todo:if hidden dims' multi 4 is necessary
        self.mu = Linear(hidden_dims[-1]*4, latent_dim)  # 输入：hidden_dims最后一行数据*4 输出：隐变量维度
        self.var = Linear(hidden_dims[-1]*4, latent_dim)

        # decoder
        modules = []

        self.decoder_input = Linear(latent_dim, hidden_dims[-1]*4)

        # 将层数反转来构建解码器
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                Sequential(
                    # 逆卷积构造器
                    ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i+1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                    BatchNorm2d(hidden_dims[i + 1]),
                    LeakyReLU()
                )
            )

        self.decoder = Sequential(*modules)

        self.final_layer = Sequential(
            ConvTranspose2d(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            BatchNorm2d(hidden_dims[-1]),
            LeakyReLU(),
            Conv2d(hidden_dims[-1],
                   out_channels=3,
                   kernel_size=3,
                   padding=1),
            Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        将输入通过encode模块，返回一个encode过后的编码
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(result)
        log_var = self.var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        将输入通过decode层进行解码，返回一个解码之后的结果
        :param z:
        :return:
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        """
        前向传播，将编码后的μ和σ值在reparameterize函数中进行混合
        返回一个经过解码后的z值、输入值、μ和σ值
        :param input: Tensor
        :param kwargs:
        :return:
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    # 与后面的参考项目有些不同，仿照之前参考项目所写
    @staticmethod
    def reparameterize(mean, log_var):
        epsilon = torch.normal(0, 1, size=mean.size()).to(device)
        std = torch.exp(log_var) ** 0.5
        z = mean + std * epsilon
        return z

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        来自隐藏空间的样本
        返回对应的映射
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        将输入的向量前向传播生成重组的向量
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class Swish(Module):
    def __init__(self, b: float = 1.0):
        super(Swish, self).__init__()
        self.b = b

    def forward(self, x):
        x = x * torch.sigmoid(self.b * x)
        return x
