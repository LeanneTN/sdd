from scipy.io import arff
import pandas as pd
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

# if __name__ == '__main__':
#   arff_to_csv('../dataset/CleanedData/MDP/D1/CM1.arff')