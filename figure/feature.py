#数据的特征相关图像
#这个可以正常绘图了
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
        plt.savefig("img/violin_plot_" + str(i) + ".jpg")

#单个小提琴图。该图像横向上表示该类数据的数值分布，而纵向上通过其起伏状态可以看出正样本与负样本的方差。
def single_violin_plot(x,y):
    data = x
    data_label = y
    data_n_2 = (data-data.mean())/(data.std())
    data = pd.concat([y, data_n_2.iloc[:, 39]], axis=1)
    data = pd.melt(data, id_vars="label",var_name="features",value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="value", y="features", hue="label", data=data, split=True, inner="quart")
    plt.savefig("img/single_violin_plot.jpg")

#散点图。对40个属性中的四个属性进行了分布散点图的研究，表示某个属性随着某个属性的变化而变化的情况。
def pair_grid_plot(x):
    sns.set(style="white")
    df = x.loc[:, ['HALSTEAD_EFFORT', 'NUM_OPERATORS', 'HALSTEAD_PROG_TIME','LOC_TOTAL']]
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(plt.scatter)
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=3)
    plt.savefig("img/pair_grid_plot.jpg")

data = pd.read_csv('../dataset/csv/clean/scale.csv')
col = data.columns #存储特征的名字
# print(col)
y = data.label #y存放标签
list = ['label']
x = data.drop(list,axis=1) #x是舍弃标签后、只留下特征的数据集
# bar_chart(data)
violin_plot(x,y) #所有小提琴图
single_violin_plot(x,y) #单个小提琴图
pair_grid_plot(x) #散点图
plt.show()
