import os
import time

import joblib  # jbolib模块
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae


def hitrate(y_test, y_predict, level1=0.39, level2=0.78, level3=1.17):
    '''
    输入：
    y_test,实际值
    y_predict,预测值
    '''
    num1 = 0
    num2 = 0
    num3 = 0
    total = 0
    for i in range(len(y_predict)):
        error = abs(y_predict[i] - y_test[i])
        total += 1
        if error <= level1:
            num1 += 1
        elif level1 < error <= level2:
            num2 += 1
        elif level2 < error <= level3:
            num3 += 1

    hit_rate1 = num1 / total
    hit_rate2 = (num1 + num2) / total
    hit_rate3 = (num1 + num2 + num3) / total
    MAE = mae(y_test, y_predict)
    return hit_rate1, hit_rate2, hit_rate3, MAE


def dataScaler(data_X):
    # 标准化
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data_X)
    # print(data_std.shape)
    return data_std


# 石灰预测模型
def cal_lime(name, mean, scale):
    # 模型需要的数据项
    predict_SH = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                  '8ZCST', '5G11']
    y_SH = ['8ZCSSH']
    # 读取训练后的模型
    model = joblib.load(f'{name[:-5]}.pkl')
    # 读取用于预测的数据
    data1 = pd.read_excel(name, header=0)
    X_test = data1[predict_SH].values
    y_test = data1[y_SH].values
    # 输入数据标准化
    X_test = dataScaler(X_test)
    # 预测
    y_predict = model.predict(X_test)
    # print(y_predict)

    # 输出数据处理
    answer = y_predict * scale + mean

    hit_rate1, hit_rate2, hit_rate3, MAE = hitrate(y_test, answer, level1=0.39, level2=0.78, level3=1.17)
    print('石灰命中率:', hit_rate1, hit_rate2, hit_rate3, MAE)

    return answer


# 白云石预测模型
def cal_dolomite(name, mean, scale):
    predict_BYS = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn',
                   'iron_p', 'iron_s', '8ZCST', '5G11']
    y_BYS = ['QSBYS']
    # 读取训练后的模型
    model = joblib.load(f'{name[:-5]}.pkl')
    # 读取用于预测的数据
    data1 = pd.read_excel(name, header=0)
    X_test = data1[predict_BYS].values
    y_test = data1[y_BYS].values
    # 输入数据标准化
    X_test = dataScaler(X_test)
    # 预测
    y_predict = model.predict(X_test)
    # print(y_predict)
    # 输出数据处理
    answer = y_predict * scale + mean
    hit_rate1, hit_rate2, hit_rate3, MAE = hitrate(y_test, answer, level1=0.38, level2=0.76, level3=1.14)
    print('白云石命中率:', hit_rate1, hit_rate2, hit_rate3, MAE)

    return answer


def main():
    '''
    根据判断条件选择读取的模型文件和
    '''
    df_std = pd.read_excel('不分钢种std.xlsx', index_col=0)
    for fname in os.listdir('.'):
        if fname.find(f'石灰') >= 0 and fname.find(f'xlsx') > 0:
            y_predict = cal_lime(fname, df_std.loc[fname[:-5], 'outmean'], df_std.loc[fname[:-5], 'outscale'])
            # print('预测结果:', y_predict)

    for fname in os.listdir('.'):
        if fname.find(f'白云石') >= 0 and fname.find(f'xlsx') > 0:
            y_predict = cal_dolomite(fname, df_std.loc[fname[:-5], 'outmean'], df_std.loc[fname[:-5], 'outscale'])
            # print('预测结果:', y_predict)


if __name__ == '__main__':
    tbg = time.time()
    main()
    ted = time.time()
    print('总用时：', ted - tbg)
