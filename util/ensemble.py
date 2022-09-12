from util.CDSDP import CDSDP
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from imblearn.over_sampling import SMOTE
import random


df = pd.read_csv('../dataset/csv/clean/clean.csv')

input_dim = len(df.columns) - 1
dataset_y = np.array(df.pop('label'))
dataset_x = np.array(df)
dataset_x, dataset_y = SMOTE(random_state=random.randint(0, 100)).fit_resample(dataset_x, dataset_y)

dataset = TensorDataset(torch.tensor(dataset_x), torch.tensor(dataset_y))

model = CDSDP(input_dim)
model.test(DataLoader(dataset, batch_size=len(dataset)), dataset_x)
