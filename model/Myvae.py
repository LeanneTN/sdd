from torch import tensor as Tensor
from typing import List,Any
from torch.nn import Module, Sequential, Linear, BatchNorm2d, LeakyReLU, Conv2d, ConvTranspose2d, Tanh, functional
import torch
from model.BaseVae import BaseVAE
from torch import nn
from torch.nn import functional as F


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 abnormal_rate: float,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.abnormal_rate = abnormal_rate
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [80,160,320]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # 平均值
        self.fc_mu = nn.Linear(hidden_dims[-1] , latent_dim)
        # 方差
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # print(modules)
        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] )
        # reverse和下面对应
        hidden_dims.reverse()
        hidden_dims_2 = [320,160,80,40]

        for i in range(len(hidden_dims_2) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims_2[i],
                                       hidden_dims_2[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(hidden_dims_2[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # 最后一层的final_layer，其实可以写死
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims_2[-1],
                               hidden_dims_2[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm1d(hidden_dims_2[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims_2[-1], out_channels=40,
                      kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(40,40,16)
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input.unsqueeze(-1)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 320, 1)
        result = self.decoder(result)
        result = self.final_layer[0](result)
        result = self.final_layer[1](result)
        result = self.final_layer[2](result)
        result = self.final_layer[3](result)
        result = self.final_layer[4](result)
        result = self.final_layer[5](result)
        result = result.squeeze(-1)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # print(self.decode(z))
        # input 原始数据  mu log_var decode返回最终作比较的数据
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    # def sample(self,
    #            num_samples: int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)
    #
    #     z = z.to(current_device)
    #
    #     samples = self.decode(z)
    #     return samples

    # def generate(self, x: Tensor, **kwargs) -> Tensor:
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """
    #
    #     return self.forward(x)[0]
    #

