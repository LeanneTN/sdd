import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import glob
import numpy as np
from scipy.io import arff
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from util.index import log
import random
#glob用于查找文件目录
import glob


def arff_to_csv(arff_path, csv_path) -> None:
    """
    将arff转化为csv,并且对于最后一列的数据进行处理
    :param arff_path:arff文件路径
    :param csv_path:csv文件路径
    :return:None
    """
    data, meta = arff.loadarff(arff_path)
    # print(data)
    csv_data = pd.DataFrame(data)
    # print(csv_data)
    # print(csv_data['Defective'])
    # csv_data['label'] = -1.0
    # print(csv_data['label'])

    if 'label' in csv_data.columns:
        csv_data.loc[csv_data['label'] == b'N', 'label'] = 0.0
        csv_data.loc[csv_data['label'] == b'Y', 'label'] = 1.0
    elif 'Defective' in csv_data.columns:
        # 添加label以便于后面来处理，并且将defective替换成label
        csv_data['label'] = -1.0
        csv_data.loc[csv_data['Defective'] == b'N', 'label'] = 0.0
        csv_data.loc[csv_data['Defective'] == b'Y', 'label'] = 1.0
        csv_data.drop(['Defective'], axis=1, inplace=True)
        # 不需要添加索引
        csv_data.to_csv(csv_path, index=False)

def data_devide(datapath, batch_size, val_rate, test_rate, shuffle: bool=True):
    df = pd.read_csv(datapath)
    input_dim = len(df.columns) - 1
    # 0为无缺陷样本，为负样本
    df_negative = df[df['label'] == 0]
    df_positive = df[df['label'] == 1]
    dataset_y_negative = df_negative.pop('label')
    dataset_y_positive = df_positive.pop('label')
    dataset_x_negative = df_negative
    dataset_x_positive = df_positive
    dataset_negative = TensorDataset(torch.tensor(np.array(dataset_x_negative)),
                                     torch.tensor(np.array(dataset_y_negative)))
    dataset_positive = TensorDataset(torch.tensor(np.array(dataset_x_positive)),
                                     torch.tensor(np.array(dataset_y_positive)))

    # 负样本
    negative_length = len(dataset_negative)
    log('negative length: %d' % negative_length)
    val_size_negative = int(val_rate * negative_length)
    test_size_negative = int(test_rate * negative_length)
    train_size = negative_length - val_size_negative - test_size_negative
    # 这里有点问题，只使用负样本进行了训练
    # 负样本按照8：1：1的比例进行划分
    train_dataset, val_dataset_negative, test_dataset_negative = random_split(dataset_negative,
                                                                              [train_size, val_size_negative,
                                                                               test_size_negative])
    # 正样本
    positive_length = len(dataset_positive)
    log('positive length: %d' % positive_length)
    # 正样本按照1：1的比例进行划分
    val_size_positive = int(val_rate * positive_length / (val_rate + test_rate))
    test_size_positive = positive_length - val_size_positive
    val_dataset_positive, test_dataset_positive = random_split(dataset_positive,
                                                               [val_size_positive, test_size_positive])
    # 一批64个数据
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), \
           DataLoader(val_dataset_negative, batch_size=batch_size), \
           DataLoader(val_dataset_positive, batch_size=batch_size), \
           DataLoader(test_dataset_negative, batch_size=batch_size), \
           DataLoader(test_dataset_positive, batch_size=batch_size), input_dim

# 不加float=0.0会报错
def data_pro_unbalance(datapath, batch_size, shuffle: bool = True, val_rate:float=0.0, test_rate:float=0.0):
    """
    解决数据不平衡问题
    :param datapath: 数据集文件路径
    :param batch_size: 一批次的数据大小
    :param shuffle: 是否打乱，默认True
    :param val_rate: 验证集比率
    :param test_rate: 测试集比率
    :return: None
    """
    dataframe = pd.read_csv(datapath, low_memory=False, dtype='float')
    # 放入的向量的维度是columns-1
    input_dim = len(dataframe.columns) - 1
    # dataset_y是不带label的那一列的数据
    dataset_y = dataframe.pop('label')
    # 数据格式的转换，y是label对应的，x是前面对应的
    dataset_y = np.array(dataset_y)
    dataset_x = dataframe
    dataset_x = np.array(dataset_x)
    # 过采样
    log('using smote to over sample')
    # float is only available for binary classification. An error is raised for multi-class classification.
    # data_x作为数据，data_y作为标签
    dataset_x, dataset_y = SMOTE(sampling_strategy='float',random_state=random.randint(0, 100)).fit_resample(dataset_x, dataset_y)

    # tensorDataset是对于给定的tensor数据（样本和标签），将他们包装成dataset，如果是numpy的array，或者pandas的dataframe需要先转化成tensor
    dataset = TensorDataset(torch.tensor(dataset_x), torch.tensor(dataset_y.reshape(-1, 1)))
    # 数据切割
    dataset_length = len(dataset)
    log('read %d data' % dataset_length)
    val_size = int(val_rate * dataset_length)
    test_size = int(test_rate * dataset_length)
    train_size = dataset_length - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # if val_size <= 0:
    #     return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), input_dim
    # elif test_size <= 0:
    #     return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset,
    #                                                                                          batch_size=batch_size), input_dim
    # else:
    # dataloader,bath_size：每个batch加载多少个样本，drop_last设置是否删除最后一个不完整的batch
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset,batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size), input_dim


def csv_merge(file_reg: str, merge_path: str) -> None:
    """
    csv文件合并

    :param file_reg: csv文件路径的正则匹配
    :param merge_path: 合并之后的csv文件路径
    :return: None
    """
    data_frame = pd.DataFrame()
    index = 0
    for file in glob.glob(file_reg):
        data = pd.read_csv(file)
        #对csv文件逐个合并
        data_frame = pd.concat([data_frame, data], ignore_index=True)
        #对已经完成合并的文件打印信息
        print(file + ' finished')
        print('%d rows, %d columns' % (data.shape[0], data.shape[1]))
        index += 1
    #打印总行数
    print('%d rows total' % data_frame.shape[0])
    #将DataFrame写入提供路径下的csv文件中，且不需要添加索引
    data_frame.to_csv(merge_path, index=False)

def data_clean(csv_path, clean_path, fill_num)->None:
    """
    对于csv数据进行清洗
    :param csv_path:清洗之前的路径
    :param clean_path:清洗后保留的路径
    :param fill_num:填充的数字
    :return:None
    """
    data = pd.read_csv(csv_path, low_memory=False, dtype='float')
    # dropna 是去掉缺失数据NAN,如果缺失的数据大于8个，那么这一行就不用保存了
    data = data.dropna(thresh=32)
    # 用fill_num来填充
    data.fillna(value=fill_num, inplace=True)

    data.to_csv(clean_path, index=False)

def data_scale(clean_path, scale_path)->None:
    """
        对于清洗后的数据进行等比例缩放至0-1
        :param csv_path:缩放之前的路径
        :param clean_path:缩放后保留的路径
        :return:None
        """
    data = pd.read_csv(clean_path, low_memory=False, dtype='float')
    for i in data.columns:
        print(i)
        x_max = data[i].max()
        print(x_max)
        x_min = data[i].min()
        print(x_min)
        if x_max > 0.0:
            # 对i列数据的每一行按比例缩放
            for j in range(len(data)):
                print(j)
                print(data.loc[j, i])
                data.loc[j, i] = data.loc[j, i] / x_max
                print(data.loc[j, i])

    data.to_csv(scale_path, index=False)