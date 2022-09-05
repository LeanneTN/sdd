import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

labels = [
    'name',
    'val_acc',
    'val_precision',
    'val_recall',
    'val_f1'
    ]

def readJSON(path):
    file = open(path,"rb")
    fileJson = json.load(file)
    test = fileJson["test"]
    return test

def frame(data):
    dic = {}
    for label in labels:
        datas = []
        for j in range(5):
            data_detail = data[j][label]
            datas.append(data_detail)
        xx = {label:datas}
        dic.update(xx)
    print(dic)
    return pd.DataFrame(dic)

def get_f1(data):
    for data_i in data:
        val_f1 = (2*data_i['val_precision']*data_i['val_recall'])/(data_i['val_precision']+data_i['val_recall'])
        xx = {'val_f1':val_f1}
        data_i.update(xx)
    return data

#数值是直接给的
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
    for label in labels:
        if label is not 'name':
            ax = sns.barplot(x='name',y=label,data=frame)
            plt.show()




