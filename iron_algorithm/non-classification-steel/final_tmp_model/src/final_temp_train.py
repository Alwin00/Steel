import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import joblib  # jbolib模块
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


class FinalTempModel:
    def __init__(self, train_df, save_path):
        self.feature = train_df[0]
        self.target = train_df[1]
        self.save_path = save_path

    def train(self):
        x_scale = StandardScaler()
        y_scale = StandardScaler()

        # 切分训练集和测试集
        feature_train, feature_test, target_train, target_test = train_test_split(self.feature, self.target,
                                                                                  test_size=0.2, random_state=1)

        # 数据标准化处理
        feature_train = x_scale.fit_transform(feature_train)
        feature_test = x_scale.transform(feature_test)
        target_train = y_scale.fit_transform(target_train)
        target_test = y_scale.fit_transform(target_test)

        # 创建模型
        bpnn = MLPRegressor(solver='adam',  # 随机梯度下降
                            activation='relu',  # 以relu函数作为激活函数
                            hidden_layer_sizes=(10, 5),  # 设置一个有5个节点的隐藏层。
                            random_state=1, early_stopping=True,
                            learning_rate_init=0.1,  # 初始学习率，设为0.1
                            max_iter=100000, verbose=1)  # 最大迭代次数，设为10000
        # 实例化
        bpnn.fit(feature_train, target_train)

        joblib.dump(bpnn, self.save_path)
        logging.info('model has been saved successfully')
