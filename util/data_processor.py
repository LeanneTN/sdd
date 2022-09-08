import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from util.index import log
import glob
import numpy as np
from scipy.io import arff
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from util.feature_processor import select_features
import random


def get_classification_dataset(dataset_path: str, batch_size: int, shuffle: bool = True, val_rate: float = 0.0, test_rate: float = 0.0, over_sampling: str = None, under_sampling: str = None, feature_num: int = None):
    """
    获取训练、验证、测试用的数据集

    :param dataset_path: 数据集文件路径
    :param batch_size: batch大小
    :param shuffle: 是否打乱
    :param val_rate: 验证集比率
    :param test_rate: 测试集比率
    :param over_sampling: 是否过采样
    :param under_sampling: 是否欠采样
    :param feature_num: 特征选择
    :return: DataLoader
    """
    dataframe = pd.read_csv(dataset_path, low_memory=False, dtype='float')
    if feature_num is not None:
        assert feature_num > 0
        selected_features = select_features(dataframe, feature_num)
        selected_features.append('label')
        dataframe = pd.DataFrame(dataframe, columns=selected_features)
    input_dim = len(dataframe.columns) - 1
    dataset_y = dataframe.pop('label')
    dataset_y = np.array(dataset_y)
    dataset_x = dataframe
    dataset_x = np.array(dataset_x)
    # 过采样
    if over_sampling == 'smote':
        log('using smote to over sample')
        dataset_x, dataset_y = SMOTE(random_state=random.randint(0, 100)).fit_resample(dataset_x, dataset_y)
    # 欠采样
    if under_sampling == 'tl':
        log('using tomek links to under sample')
        dataset_x, dataset_y = TomekLinks().fit_resample(dataset_x, dataset_y)
    # 生成Tensor数据集
    dataset = TensorDataset(torch.tensor(dataset_x), torch.tensor(dataset_y.reshape(-1, 1)))
    # 数据切割
    dataset_length = len(dataset)
    log('read %d data' % dataset_length)
    val_size = int(val_rate * dataset_length)
    test_size = int(test_rate * dataset_length)
    train_size = dataset_length - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    if val_size <= 0:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), input_dim
    elif test_size <= 0:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset, batch_size=batch_size), input_dim
    else:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size), input_dim


def get_ae_dataset(dataset_path: str, batch_size: int, val_rate: float, test_rate: float, shuffle: bool = True, feature_num: int = None):
    assert val_rate > 0
    assert test_rate > 0
    df = pd.read_csv(dataset_path)
    if feature_num is not None:
        assert feature_num > 0
        selected_features = select_features(df, feature_num)
        selected_features.append('label')
        df = pd.DataFrame(df, columns=selected_features)
    input_dim = len(df.columns) - 1
    df_negative = df[df['label'] == 0]
    df_positive = df[df['label'] == 1]
    dataset_y_negative = df_negative.pop('label')
    dataset_y_positive = df_positive.pop('label')
    dataset_x_negative = df_negative
    dataset_x_positive = df_positive
    dataset_negative = TensorDataset(torch.tensor(np.array(dataset_x_negative)), torch.tensor(np.array(dataset_y_negative)))
    dataset_positive = TensorDataset(torch.tensor(np.array(dataset_x_positive)), torch.tensor(np.array(dataset_y_positive)))
    # 负样本
    negative_length = len(dataset_negative)
    log('negative length: %d' % negative_length)
    val_size_negative = int(val_rate * negative_length)
    test_size_negative = int(test_rate * negative_length)
    train_size = negative_length - val_size_negative - test_size_negative
    train_dataset, val_dataset_negative, test_dataset_negative = random_split(dataset_negative, [train_size, val_size_negative, test_size_negative])
    # 正样本
    positive_length = len(dataset_positive)
    log('positive length: %d' % positive_length)
    val_size_positive = int(val_rate * positive_length / (val_rate + test_rate))
    test_size_positive = positive_length - val_size_positive
    val_dataset_positive, test_dataset_positive = random_split(dataset_positive, [val_size_positive, test_size_positive])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset_negative, batch_size=batch_size), DataLoader(val_dataset_positive, batch_size=batch_size), DataLoader(test_dataset_negative, batch_size=batch_size), DataLoader(test_dataset_positive, batch_size=batch_size), input_dim


def arff_to_csv(arff_path: str, csv_path: str) -> None:
    """
    arff文件格式转换为csv文件

    :param arff_path: arff文件路径
    :param csv_path: csv文件路径
    :return: None
    """
    data, meta = arff.loadarff(arff_path)
    csv_data = pd.DataFrame(data)
    if 'label' in csv_data.columns:
        csv_data.loc[csv_data['label'] == b'N', 'label'] = 0.0
        csv_data.loc[csv_data['label'] == b'Y', 'label'] = 1.0
    elif 'Defective' in csv_data.columns:
        csv_data['label'] = -1.0
        csv_data.loc[csv_data['Defective'] == b'N', 'label'] = 0.0
        csv_data.loc[csv_data['Defective'] == b'Y', 'label'] = 1.0
        csv_data.drop(['Defective'], axis=1, inplace=True)
    log(arff_path + ' %d rows, %d columns' % (csv_data.shape[0], csv_data.shape[1]))
    csv_data.to_csv(csv_path, index=False)


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
        data_frame = pd.concat([data_frame, data], ignore_index=True)
        log(file + ' finished')
        log('%d rows, %d columns' % (data.shape[0], data.shape[1]))
        index += 1

    log('%d rows total' % data_frame.shape[0])
    data_frame.to_csv(merge_path, index=False)


def data_clean(csv_path: str, clean_path: str, use_noise: bool = False, fill_na: float = -1.0) -> None:
    """
    csv数据清洗

    :param csv_path: csv文件路径
    :param clean_path: 清洗后csv文件路径
    :param use_noise: 是否使用噪声来填补缺省值
    :param fill_na: 缺省填充值
    :return: None
    """
    data = pd.read_csv(csv_path, low_memory=False, dtype='float')
    data = data.dropna(thresh=32)
    if use_noise:
        log('user noise not supported')
        exit(1)
    else:
        data.fillna(value=fill_na, inplace=True)
    data.to_csv(clean_path, index=False)
