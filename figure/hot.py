#数据的特征热力图
#这个可以正常绘图了
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#从csv读取数据
data = pd.read_csv('..\\dataset\\csv\\clean\\clean.csv', encoding='gbk')
data_set = data.corr()  #相关系数矩阵
plt.figure(figsize=(35,30))

ax = sns.heatmap(data_set, center=0)
plt.savefig("hot.jpg")
plt.show()




