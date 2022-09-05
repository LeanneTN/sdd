import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#从csv读取数据
data = pd.read_csv('..\\dataset\\csv\\d2\\clean.csv', encoding='gbk')#！！！！！！
data_set = data.corr()
plt.figure(figsize=(35,30))

ax = sns.heatmap(data_set, center=0)
plt.savefig("hot.jpg")
plt.show()




