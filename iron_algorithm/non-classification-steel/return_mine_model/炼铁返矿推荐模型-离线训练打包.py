import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib # jbolib模块
import os


# 读取excel表中数据
def load_data(df):

    #导入数据中的影响因素
    predict_LTFK = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                    '8ZCST', '5G11']
    feature=df[predict_LTFK]

    #导入目标值，本模型中是转炉氧气耗量
    y_LTFK = ['91LTFK']
    target = df[y_LTFK]
    return feature,target




def BPNN_LTFK():
    df1 = pd.DataFrame()
    for fname in os.listdir('.'):
        # print(df1)
        if fname.find(f'返矿') >= 0 and fname.find(f'xlsx') > 0:
            # print(fname[0:-5])
            data1 = pd.read_excel(fname, header=0)
            feature, target=load_data(data1)

            x_scale = StandardScaler()
            y_scale = StandardScaler()

            feature = feature.values
            target = (target.values.ravel()).reshape((-1, 1))


            #切分训练集和测试集
            feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.33,
                                                                                    random_state=1)

            # 数据标准化处理
            feature_train = x_scale.fit_transform(feature_train)
            feature_test  =  x_scale.transform(feature_test)
            target_train = y_scale.fit_transform(target_train)

            # 创建BPNN模型
            bpnn = MLPRegressor(solver='adam',  # 随机梯度下降
                                activation='relu',  # 以relu函数作为激活函数
                                hidden_layer_sizes=(6,),  # 设置一个有5个节点的隐藏层。
                                random_state=1,
                                learning_rate_init=0.1,  # 初始学习率，设为0.1
                                max_iter=10000)  # 最大迭代次数，设为10000
            #实例化
            bpnn.fit(feature_train, target_train)

            joblib.dump(bpnn, fname[0:-5]+'.pkl')



if __name__ == '__main__':
    BPNN_LTFK()






