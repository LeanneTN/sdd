import torch
from torchsummary import summary


if torch.cuda.is_available():
    log("cuda available, using gpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = 'dataset/csv/original/clean.csv'

# 接下来就是根据不同的参数类型选择不同的数据载入和划分方式
