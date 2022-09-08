from torch.nn import Module, MSELoss, BCELoss
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(Module):

    def __init__(self, binary: bool, gamma: float = 2.0, alpha: float = 0.5, average: bool = True, scale: float = 10.0):
        super(FocalLoss, self).__init__()
        self.binary = binary
        self.average = average
        self.gamma = gamma
        self.alpha = alpha
        self.scale = scale

    def forward(self, y_pred, y_true):
        # 防止log0出现nan
        epsilon = 1e-7
        if self.binary is False:
            # 非binary时需要softmax
            y_pred = F.softmax(y_pred, dim=1)
            # 只保留y_true为1的损失
            index = (y_true == 1.0).nonzero()
            y_pred = y_pred.gather(1, index[:, 1].view(-1, 1))
            y_true = y_true.gather(1, index[:, 1].view(-1, 1))

        loss = y_true * self.alpha * (-torch.log(y_pred + epsilon) * (1 - y_pred) ** self.gamma) + (1 - y_true) * (1 - self.alpha) * (-torch.log(1 - y_pred + epsilon) * y_pred ** self.gamma)
        if self.average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss * self.scale



class VAELoss(Module):

    def __init__(self, rate: float = 0.5):
        super(VAELoss, self).__init__()
        self.rate = rate
        self.mse = MSELoss()
        # reduction参数决定维度要不要缩减
        self.mse_ = MSELoss(reduction='none')

    def forward(self, x_hat, norm_x, mean, log_var, reduction: bool = True):
        # print(x_hat)
        kld = mean ** 2 + torch.exp(log_var) - log_var - 1
        if reduction:
            mse = self.mse(x_hat, norm_x)
            kld = kld.mean()
        else:
            mse = self.mse_(x_hat, norm_x).mean(dim=-1)
            kld = kld.mean(dim=-1)
        return mse + self.rate * kld
