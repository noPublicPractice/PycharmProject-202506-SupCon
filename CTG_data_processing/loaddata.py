import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class const:
    random_num = 0
    model_input_len = 56100  # 最大截断长度 20hz 36000对应30分钟
    now_col = 'TOCO'
    now_col_name = 'CTG:%s' % now_col
def count_record(csv_data):
    for i in range(len(csv_data)):
        fhr_len = len(csv_data["CTG:FHR"][i].split(','))
        toco_len = len(csv_data["CTG:TOCO"][i].split(','))
        fm_len = len(csv_data["CTG:FM"][i].split(','))
        print("%s\t%d\t%d\t%d\t%s" % (csv_data["病案号+档案号"][i], fhr_len, toco_len, fm_len, csv_data["classify"][i]))
def do_adjust(temp_list):
    temp_len = len(temp_list)
    mul = const.model_input_len // temp_len  # 倍数  5610/100=56
    ext = const.model_input_len % temp_len  # 不足1倍时补的个数
    temp_list_2 = temp_list * mul
    temp_list_2.extend(temp_list[:ext])
    return temp_list_2
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)  # 范数
    matrix = matrix / norm  # 归一化
    return matrix
def do_data(x_train):
    for i in range(len(x_train)):
        temp_arr = np.array(x_train[i].split(','))
        temp_list = list(map(int, temp_arr))
        if len(temp_list) < const.model_input_len:
            temp_list = do_adjust(temp_list)
        else:
            temp_list = temp_list[:const.model_input_len]
        if i == 0:
            res_arr = np.array(temp_list)
            res_arr_2d = res_arr[np.newaxis, :]  # 数组扩展维度
        else:
            res_arr = np.array(temp_list)
            res_arr_temp = res_arr[np.newaxis, :]
            res_arr_2d = np.vstack([res_arr_2d, res_arr_temp])
    normalize_arr_2d = normalize_2d(res_arr_2d)  # 数据归一化
    res_list = normalize_arr_2d.tolist()
    return res_list
def prepare_data():
    # 导入数据集
    csv_data = pd.read_csv('ctg_data-%s-score.csv' % const.now_col, delimiter=';')  # 原始数据导入
    csv_data.columns = [x.strip() for x in csv_data.columns]  # 如果没有这一行，下一行会报错
    sample_space = list(csv_data[const.now_col_name])  # 样本空间
    label_space = list(csv_data['score'])  # 标记空间
    # 笔记：在HAR Dataset中，没有id列，所以此处暂时不处理id。详见：E:\Python\PycharmProject-202307-深度学习\202505-半监督分类CATCC-其它文件\dataset-HAR-2\X_test.txt
    # 笔记：关于数据平衡：0-3分：190条记录   4-7分：485条记录     差不多呈3:7的比例
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(sample_space, label_space, test_size=0.3, random_state=const.random_num)  # test_size: 划分比例
    x_train_list = do_data(x_train)
    x_test_list = do_data(x_test)
    return x_train_list, x_test_list, y_train, y_test
def txt_create_data_frame_to_pt(mode_num, x_list, y_list, file_name):
    point_mid = 99
    point_1p = 46
    if mode_num == 1:
        train_dataset = pd.DataFrame({'samples': x_list, 'labels': y_list})  # 构建数据框
    elif mode_num == 2:
        train_dataset = pd.DataFrame({'samples': x_list[:point_mid], 'labels': y_list[:point_mid]})  # 构建数据框
    elif mode_num == 3:
        train_dataset = pd.DataFrame({'samples': x_list[point_mid:], 'labels': y_list[point_mid:]})  # 构建数据框
    elif mode_num == 4:
        train_dataset = pd.DataFrame({'samples': x_list[:point_1p], 'labels': y_list[:point_1p]})  # 构建数据框
    torch.save(train_dataset, file_name)  # 保存变量到文件.注意:二进制模式不采用编码参数
def do_work():
    x_train_list, x_test_list, y_train, y_test = prepare_data()
    txt_create_data_frame_to_pt(1, x_train_list, y_train, 'train.pt')
    txt_create_data_frame_to_pt(2, x_test_list, y_test, 'val.pt')
    txt_create_data_frame_to_pt(3, x_test_list, y_test, 'test.pt')
    txt_create_data_frame_to_pt(4, x_train_list, y_train, 'train_1perc.pt')
if __name__ == '__main__':
    do_work()
