import os
import pandas as pd

# import openpyxl  # 使用pd.read_excel中需要保证openpyxl库已安装，但可以不导入。

class const:
    # col=['病案号','档案号','开始时间','CTG:FHR','CTG:TOCO','CTG:FM','报告结果:监护方法','报告结果:图形特征','报告结果:监护结果','classify']
    score_dict = {}  # 四个'病案号+档案号'出现了两次，分别是：05863798+178_1_230822100036、05863798+179_1_230625123341、05863798+202_1_230822233827、05967700+177_1_230605181029
    classifies = [r'0-3分\\', r'4-7分\\']
    id_1 = '病案号'
    id_2 = '档案号'
    index_col_name = '%s+%s' % (id_1, id_2)
    now_col = 'TOCO'
    now_col_name = 'CTG:%s' % now_col
def prepare_score_dict():
    excel_data = pd.read_excel('ctg_data-病案号+档案号、分数.xlsx')
    for i in range(len(excel_data)):
        const.score_dict[excel_data[const.index_col_name][i]] = excel_data['分数'][i]
def load_data():
    record_data_list = []
    for classify_index in range(len(const.classifies)):
        excel_list = os.listdir(const.classifies[classify_index])
        for excel in excel_list:
            try:
                excel_data = pd.read_excel(const.classifies[classify_index] + excel, header=None)
            except PermissionError:
                print(const.classifies[classify_index] + excel + '权限问题')
                continue
            temp_dict = {}
            for i in range(len(excel_data[0])):
                temp_dict[excel_data[0][i]] = excel_data[1][i]
            record_data = {
                const.index_col_name: '%s+%s' % (temp_dict[const.id_1], temp_dict[const.id_2]),
                const.now_col_name:   temp_dict[const.now_col_name]
            }
            record_data_list.append(record_data)
    return record_data_list
def write_ctg_data(record_data_list):
    fp = open('ctg_data-%s-score.csv' % const.now_col, 'w', encoding='utf-8')
    fp.write(';'.join([const.index_col_name, const.now_col_name, 'score\n']))
    for record in record_data_list:
        try:
            fp.write(';'.join([record[const.index_col_name], record[const.now_col_name], str(const.score_dict[record[const.index_col_name]])]) + '\n')
        except TypeError:
            print(record[const.index_col_name] + '数据问题')
    fp.close()
def do_work():
    prepare_score_dict()
    record_data_list = load_data()
    write_ctg_data(record_data_list)
if __name__ == '__main__':
    do_work()
