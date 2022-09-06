#测试集上不同机器学习模型各指标对比
#数值还要手动改
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

labels = [
    'name',
    'val_acc',          #正确率
    'val_precision',    #查准率
    'val_recall',       #查全率(召回率)
    'val_f1'            #查全率和查准率的调和平均数
    ]

#从json文件中读取“test”。暂时没用到
# def readJSON(path):
#     file = open(path,"rb")
#     fileJson = json.load(file)
#     test = fileJson["test"]
#     return test

def frame(data):
    dic = {}
    for label in labels:
        datas = []
        for j in range(5): #循环5次
            data_detail = data[j][label]
            datas.append(data_detail)
        xx = {label:datas}  #指标名：五个模型值
        dic.update(xx)
    print(dic)
    return pd.DataFrame(dic)

#给每个模型的结果里加f1指标
def get_f1(data):
    for data_i in data:
        val_f1 = (2*data_i['val_precision']*data_i['val_recall'])/(data_i['val_precision']+data_i['val_recall'])
        xx = {'val_f1':val_f1}
        data_i.update(xx)
    return data

#指标的数值是直接给的，估计要先在其他模块运行计算了，再填过来
if __name__ == '__main__':
    XGBoost={
    'name':'XGBoost',
    'val_acc': 0.983984,
    'val_precision': 0.987844,
    'val_recall': 0.979116
    }

    RF={
    'name':'RandomForest',
    'val_acc': 0.939844,
    'val_precision': 0.917871,
    'val_recall': 0.963288
    }

    DT={
    'name':'DecisionTree',
    'val_acc': 0.974219,
    'val_precision': 0.976819,
    'val_recall': 0.970612
    }

    SVM={
    'name':'SVM',
    'val_acc': 0.83955604,
    'val_precision': 0.883672712,
    'val_recall': 0.89912412
    }

    AdaBoost={
    'name':'AdaBoost',
    'val_acc': 0.938672,
    'val_precision': 0.923981,
    'val_recall': 0.951574
    }

    data = [XGBoost,RF,DT,SVM,AdaBoost]
    data = get_f1(data)
    frame = frame(data)
    for label in labels:  #循环，对每个指标都画张模型对比图
        if label is not 'name':
            ax = sns.barplot(x='name',y=label,data=frame)
            plt.show()




