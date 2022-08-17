# -*- coding: utf-8 -*-
import joblib
import lightgbm as lgb
from conf.lgb_conf import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


class OxygenPredict:
    def __init__(self, model_path):
        """
        :param model_conf:  模型配置模板内容
        :param context_constant: 运行环境常量
        :param default_key_cols: Pacos
        """
        self.model_path = model_path

    def predict(self, data):
        """
        这是外部调用的入口，训练
        :return: model
        """
        train_df = data.copy()

        X = train_df[category_features + integer_features + double_features]
        y = train_df.O2_SUM_COMSUME

        # 模型加载
        model = joblib.load(self.model_path)

        # 模型预测
        pred = model.predict(X)
        mape = mean_absolute_percentage_error(pred, y)
        mae = mean_absolute_error(pred, y)
        print('pred mape: {}'.format(mape))
        print('pred mae: {}'.format(mae))

        hit_rate1, hit_rate2, hit_rate3 = self.hitrate(y, pred)
        print('pred hit rate 150: {}'.format(hit_rate1))
        print('pred hit rate 300: {}'.format(hit_rate2))
        print('pred hit rate 450: {}'.format(hit_rate3))

    def hitrate(self, y_test, y_predict, level1=150, level2=300, level3=450):
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
        return hit_rate1, hit_rate2, hit_rate3