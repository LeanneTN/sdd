from model.adaboost import AdaBoost
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pickle
from imblearn.over_sampling import SMOTE


model = AdaBoost()

df = pd.read_csv('../dataset/csv/clean/scale.csv')
dataset_y = df.pop('label')
dataset_y = np.array(dataset_y)
dataset_x = df
dataset_x = np.array(dataset_x)
dataset_x, dataset_y = SMOTE(random_state=1).fit_resample(dataset_x, dataset_y)
dataset_x, dataset_y = shuffle(dataset_x, dataset_y)

test_len = 2560
test_x, train_x = dataset_x[0:test_len], dataset_x[test_len:]
test_y, train_y = dataset_y[0:test_len], dataset_y[test_len:]

model.train(train_x, train_y)
model.test(train_x, train_y)
model.test(test_x, test_y)

with open('../model/saved_model/adaboost.pkl', 'wb') as f:
    pickle.dump(model, f)