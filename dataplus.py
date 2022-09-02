from scipy.io import arff
import pandas as pd


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


def csv_merge()

# if __name__ == '__main__':
#   arff_to_csv('../dataset/CleanedData/MDP/D1/CM1.arff')