import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib  # jbolib模块
import os


# 读取excel表中数据
def load_data(df):
    # StSi<0按0处理
    # df.loc[df['ST_SI_Q'] <= 0, 'ST_SI_Q'] = 0

    # 导入数据中的影响因素
    predict_T = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                 'O2_SUM_COMSUME', '8ZCQSBYS', '8ZCSSH', '91LTFK', '8ZCST', '5G11', 'FURNACE_AGE']
    feature = df[predict_T]

    # 导入目标值，本模型中是转炉终点碳含量
    y_T = ['OUT_STEEL_PRE_TEMP']
    target = df[y_T]
    return feature, target


def End_T():
    for fname in os.listdir('.'):
        # print(df1)
        if fname.find(f'温度') >= 0 and fname.find(f'.xlsx') >= 0:
            # print(fname[0:-5])
            data1 = pd.read_excel(fname, header=0)

            feature, target = load_data(data1)

            x_scale = StandardScaler()
            y_scale = StandardScaler()

            feature = feature.values
            target = (target.values.ravel()).reshape((-1, 1))

            # 切分训练集和测试集
            feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.33,
                                                                                      random_state=1)

            # 数据标准化处理
            feature_train = x_scale.fit_transform(feature_train)
            feature_test = x_scale.transform(feature_test)
            target_train = y_scale.fit_transform(target_train)
            # 创建模型
            bpnn = MLPRegressor(solver='adam',  # 随机梯度下降
                                activation='relu',  # 以relu函数作为激活函数
                                hidden_layer_sizes=(5,),  # 设置一个有5个节点的隐藏层。
                                random_state=1,
                                learning_rate_init=0.1,  # 初始学习率，设为0.1
                                max_iter=100000)  # 最大迭代次数，设为10000
            # 实例化
            bpnn.fit(feature_train, target_train)

            joblib.dump(bpnn, fname[0:-5] + '.pkl')


if __name__ == '__main__':
    End_T()
