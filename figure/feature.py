import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from subprocess import check_output

def bar_chart(data):
    ax = sns.countplot(x='label',data=data,label="Count")

def violin_plot(x,y):
    data = x
    data_label = y
    data_n_2 = (data-data.mean())/(data.std())
    for i in range(5):
        data = pd.concat([y, data_n_2.iloc[:, i*8:(i+1)*8]], axis=1)
        data = pd.melt(data, id_vars="label",var_name="features",value_name='value')
        plt.figure(figsize=(10, 10))
        sns.violinplot(x="value", y="features", hue="label", data=data, split=True, inner="quart")

def single_violin_plot(x,y):
    data = x
    data_label = y
    data_n_2 = (data-data.mean())/(data.std())
    data = pd.concat([y, data_n_2.iloc[:, 39]], axis=1)
    data = pd.melt(data, id_vars="label",var_name="features",value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="value", y="features", hue="label", data=data, split=True, inner="quart")


def pair_grid_plot(x):
    sns.set(style="white")
    df = x.loc[:, ['HALSTEAD_EFFORT', 'NUM_OPERATORS', 'HALSTEAD_PROG_TIME','LOC_TOTAL']]
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(plt.scatter)
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=3)

data = pd.read_csv('../dataset/csv/original/clean.csv')#！！！！！！
col = data.columns #存储特征的名字
# print(col)
y = data.label #y存放标签
list = ['label']
x = data.drop(list,axis=1) #x是舍弃标签后、只留下特征的数据集
# bar_chart(data)
# single_violin_plot(x,y)
pair_grid_plot(x)
plt.show()



