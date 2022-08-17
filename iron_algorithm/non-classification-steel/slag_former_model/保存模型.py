import os

import joblib  # jbolib模块
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def dataScaler(data_X):
    # 标准化
    scaler = StandardScaler()
    # MinMaX
    data_std = scaler.fit_transform(data_X)
    # print(data_std.shape)
    return data_std, scaler


# <editor-fold desc=" 4.预测评价之命中率 绝对误差，石灰">
def hitrate_SH(y_test, y_predict, level1=0.39, level2=0.78, level3=1.17):
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


# </editor-fold>

# <editor-fold desc=" 5.预测评价之命中率 绝对误差，白云石">
def hitrate_BYS(y_test, y_predict, level1=0.38, level2=0.76, level3=1.14):
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


# </editor-fold>


def prdt_bpnn(data, X, y, type, name='CB112_石灰未分类'):
    """
    进行预测并作图。
    输入：
    data,用于预测的数据总集，DataFrame
    X,输入项的标签，[]
    y,输出项的标签，[]
    type，预测目标，'SH'或‘BYS’
    输出：
    三个命中率
    """
    hit_rate1, hit_rate2, hit_rate3, MAE, answer = 0, 0, 0, 0, 0
    if type == 'SH':
        data_X = data[X].values
        data_y_endSH = data[y].values.ravel()
        y_predict_SH, y_test_SH = BPNN_SH_bcb(data_X, data_y_endSH, f'{name}.pkl')
        hit_rate1, hit_rate2, hit_rate3, MAE = hitrate_SH(y_test_SH, y_predict_SH)
        # print('石灰命中率:',hit_rate1, hit_rate2, hit_rate3,MAE)
        answer = y_predict_SH

    elif type == 'BYS':
        data_X = data[X].values
        data_y_endBYS = data[y].values.ravel()
        y_predict_BYS, y_test_BYS = BPNN_BYS_bcb(data_X, data_y_endBYS, f'{name}.pkl')
        hit_rate1, hit_rate2, hit_rate3, MAE = hitrate_BYS(y_test_BYS, y_predict_BYS)
        # print('白云石命中率:',hit_rate1, hit_rate2, hit_rate3,MAE)
        answer = y_predict_BYS
    return hit_rate1, hit_rate2, hit_rate3, MAE, answer


def BPNN_SH_bcb(feature, target, name=r'CB112未分类石灰预测模型.pkl'):  # *data 为全部数据
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.33,
                                                                              random_state=1)

    feature_train, fea_trainscl = dataScaler(feature_train)
    feature_test, fea_testscl = dataScaler(feature_test)
    target_train, tarscl = dataScaler(target_train.reshape(-1, 1))

    # 创建BPNN模型
    bpnn = MLPRegressor(solver='adam',  # 随机梯度下降
                        activation='relu',  # 以relu函数作为激活函数
                        hidden_layer_sizes=(10),  # 设置一个有10个节点的隐藏层。
                        random_state=1,
                        learning_rate_init=0.1,  # 初始学习率，设为0.1
                        max_iter=100000)  # 最大迭代次数，设为10000
    # print(feature_train,target_train)
    bpnn.fit(feature_train, target_train.ravel())
    joblib.dump(bpnn, name)
    tarstd = tarscl.scale_
    tarmean = tarscl.mean_
    # print(f'输出值标准差{tarstd}平均值{tarmean}')
    model = joblib.load(name)
    target_predict = model.predict(feature_test)
    y_predict = tarscl.inverse_transform(target_predict)
    # tarscl.inverse_transform(target_predict)
    return y_predict, target_test


def BPNN_BYS_bcb(feature, target, name=r'CB112未分类白云石预测模型.pkl'):  # *data 为全部数据

    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.33,
                                                                              random_state=1)

    feature_train, fea_trainscl = dataScaler(feature_train)
    feature_test, fea_testscl = dataScaler(feature_test)
    target_train, tarscl = dataScaler(target_train.reshape(-1, 1))

    # 创建BPNN模型
    bpnn = MLPRegressor(solver='adam',  # 随机梯度下降
                        activation='relu',  # 以relu函数作为激活函数
                        hidden_layer_sizes=(10),  # 设置一个有5个节点的隐藏层。
                        random_state=1,
                        learning_rate_init=0.1,  # 初始学习率，设为0.1
                        max_iter=100000)  # 最大迭代次数，设为10000
    # print(feature_train,target_train)
    bpnn.fit(feature_train, target_train.ravel())
    joblib.dump(bpnn, name)
    tarstd = tarscl.scale_
    tarmean = tarscl.mean_
    # print(f'输出值标准差{tarstd}平均值{tarmean}')
    model = joblib.load(name)
    target_predict = model.predict(feature_test)
    y_predict = tarscl.inverse_transform(target_predict)
    # tarscl.inverse_transform(target_predict)
    return y_predict, target_test


# 三类石灰预测模型 标注数据范围、判断规则，给出打包好的模型。在线逐条处理，离线批量处理
def cal_lime0(name='CB112_12_27_氧气石灰优化.xlsx'):
    """
    用于进行主流程的函数。需选择不同的聚类模型、预测模型函数。
    输入(?):
    读取数据文件名

    """
    predict_SH = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                  '8ZCST', '5G11']
    y_SH = ['8ZCSSH']
    df_data = pd.read_excel(name, header=0)
    hit_rate1, hit_rate2, hit_rate3, MAE, y_predict = prdt_bpnn(df_data, predict_SH, y_SH, 'SH', 'CB112石灰未分类')
    print('CB112未分类石灰命中率:', hit_rate1, hit_rate2, hit_rate3, MAE)
    # print(y_predict)


def cal_lime1(name='CB112_No0.xlsx'):
    """
    用于进行主流程的函数。需选择不同的聚类模型、预测模型函数。
    输入(?):
    读取数据文件名

    """
    predict_SH = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                  '8ZCST', '5G11']
    y_SH = ['8ZCSSH']
    df_data = pd.read_excel(name, header=0)
    hit_rate1, hit_rate2, hit_rate3, MAE, y_predict = prdt_bpnn(df_data, predict_SH, y_SH, 'SH', f'{name[:-5]}石灰')
    print(f'{name[6:-5]}类石灰命中率:', hit_rate1, hit_rate2, hit_rate3, MAE)
    # print(y_predict)


# 三类白云石预测模型
def cal_dolomite0(name='CB112_12_27_氧气石灰优化.xlsx'):
    """
    用于进行主流程的函数。需选择不同的聚类模型、预测模型函数。
    输入(?):
    读取数据文件名
    """
    predict_BYS = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                   '8ZCST', '5G11']
    y_BYS = ['QSBYS']
    df_data = pd.read_excel(name, header=0)

    hit_rate1, hit_rate2, hit_rate3, MAE, y_predict = prdt_bpnn(df_data, predict_BYS, y_BYS, 'BYS', 'CB112白云石未分类')
    print('CB112未分类白云石命中率:', hit_rate1, hit_rate2, hit_rate3, MAE)
    # print(y_predict)
    return y_predict


def cal_dolomite1(name='CB112_No0.xlsx'):
    """
    用于进行主流程的函数。需选择不同的聚类模型、预测模型函数。
    输入(?):
    读取数据文件名
    """
    predict_BYS = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                   '8ZCST', '5G11']
    y_BYS = ['QSBYS']
    df_data = pd.read_excel(name, header=0)

    hit_rate1, hit_rate2, hit_rate3, MAE, y_predict = prdt_bpnn(df_data, predict_BYS, y_BYS, 'BYS',
                                                                f'{name[:-5]}白云石')
    print(f'{name[6:-5]}类白云石命中率:', hit_rate1, hit_rate2, hit_rate3, MAE)
    # print(y_predict)
    return y_predict


def find_bst_model():
    activation = ['identity', 'logistic', 'tanh', 'relu']
    hidden_layer_sizes = [(5), (10), (15), (20),
                          (25), (30), (35), (40), (45), (50), (20, 20), (20, 30), (20, 40),
                          (20, 50), (30, 20), (30, 30), (30, 40),
                          (30, 50), (40, 20), (40, 30), (40, 40), (40, 50)
                          ]
    learning_rate_init = [0.005, 0.01, 0.015, 0.02, 0.025,
                          0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085,
                          0.09, 0.095, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5
                          ]
    df = pd.DataFrame(np.zeros((1, 6)))
    for fname in os.listdir('.'):
        p = 0
        if fname.find(f'300, -30, 0.96)分类') > 0:
            for i in activation:
                for j in hidden_layer_sizes:
                    for k in learning_rate_init:
                        p += 1
                        hit_rate1, hit_rate2, hit_rate3 = main0(fname, i, j, k)
                        if hit_rate1 > 0.3089 and hit_rate2 > 0.5655 and hit_rate3 > 0.755:
                            # print(f'{fname[:-5]}最佳结果 {i,j,k} C含量命中率: ', hit_rate)
                            df.loc[f'{fname[:-5], p}'] = [i, str(j), k, hit_rate1, hit_rate2, hit_rate3]


if __name__ == '__main__':
    # cal_lime0()
    for fname in os.listdir('.'):
        p = 0
        # print(fname.find(f'CB112'))
        if fname.find(f'辅料优化') >= 0 and fname.find(f'pkl') < 0:
            cal_lime1(fname)

    # cal_dolomite0()

    for fname in os.listdir('.'):
        p = 0
        # print(fname.find(f'CB112'))
        if fname.find(f'辅料优化') >= 0 and fname.find(f'pkl') < 0:
            cal_dolomite1(fname)
