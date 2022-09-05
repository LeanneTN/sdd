# 本文件调用dataplus.py进行数据的处理
# import dataplus是没有引入函数的，没有用，应该和下面的方式一样引入函数
import os.path
import sys
ROOT_DIR = os.path.abspath('')
sys.path.append('D:/good_memory/大三上/实训/MyPro')
sys.path.append(ROOT_DIR)
from data_pro.dataplus import arff_to_csv, csv_merge, data_clean
import glob
import argparse

# 通过命令行参数来实现函数的调用
parser = argparse.ArgumentParser(description='SDP script')
parser.add_argument('--process', type=str, required=True, help='choose a process: arff_to_csv, csv_merge')
args = parser.parse_args()

if __name__ == '__main__':
    """
    Args: --process
        arff_to_csv: arff转换为csv
        csv_merge: csv文件合并
    """
    arff_sub_path = 'CleanedData'
    csv_sub_path = 'original'
    csv_merge_path = 'merge'
    csv_clean_path = 'clean'

    if args.process == 'arff_to_csv':
        print("你进来了吗")
        # 遍历所有的arff文件
        for file in glob.glob('../dataset/%s/MDP/D1/*.arff' % arff_sub_path):
            print("really")
            csv_path = file.replace('arff', 'csv').replace('%s/MDP/D1'% arff_sub_path, 'csv/%s' % csv_sub_path)
            print(csv_path)
            arff_to_csv(file, csv_path)
            print(file, ' finished')

    elif args.process == 'csv_merge':
        csv_merge('../dataset/csv/%s/*.csv' % csv_sub_path, '../dataset/csv/%s/merge.csv' % csv_merge_path)

    elif args.process == 'data_clean':
        data_clean('../dataset/csv/%s/merge.csv' % csv_merge_path, '../dataset/csv/%s/clean.csv' % csv_clean_path, fill_num=0.0)

    else:
        print('parameter error: --process ', args.process)
        exit(1)

    # cleaned_data_path='../dataset/csv/original/*.csv'
    # merge_path='../dataset/csv/merge/merge.csv'
    # csv_merge(cleaned_data_path,merge_path)
