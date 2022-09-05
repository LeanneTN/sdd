import torch
import argparse

from torchsummary import summary
from model.vae_plus import VAE
from model.attention import Attention


# 调用时传入参数的设置
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='select a model to train')
main_args = parser.parse_args()

if torch.cuda.is_available():
    #log("cuda available, using gpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = 'dataset/csv/original/clean.csv'

# 接下来就是根据不同的参数类型选择不同的数据载入和划分方式
model_type = main_args.model
if model_type == 'vae':
    # todo: 获取VAE传入值的参数
    # model = VAE()
    # summary(model=model.float(), (input_dim,))
elif model_type == 'attention':
    model = Attention()
    

