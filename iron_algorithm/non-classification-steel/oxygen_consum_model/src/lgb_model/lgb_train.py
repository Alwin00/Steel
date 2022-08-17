# -*- coding: utf-8 -*-
import os
import sys
import joblib
import lightgbm as lgb
from conf.lgb_conf import *
import pickle
import numpy as np
import pandas as pd
import base64
import time
import zlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('{}/../'.format(__file__))))
sys.path.append(BASE_DIR)


class OxygenModel:
    def __init__(self, model_conf):
        """
        :param model_conf:  模型配置模板内容
        :param context_constant: 运行环境常量
        :param default_key_cols: Pacos
        """
        self.model_conf = model_conf

    def fit(self, data):
        """
        这是外部调用的入口，训练
        :param data: DataCube
        :param item_id: 每条观测id组成字段
        :param ref_df:  算法日历参考data frame
        :return: model
        """
        train_df = data.copy()

        y = train_df.O2_SUM_COMSUME
        X = train_df[category_features + integer_features + double_features]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        # 模型训练
        parameters = self.model_conf['model_conf'].get('model_parameter')

        try:
            model = lgb.LGBMRegressor(**parameters)
            test_param = {'max_depth': [5, 7, 9], 'n_estimators': [100, 250], 'num_leaves': [31, 51], }
            grid_search = GridSearchCV(estimator=model, param_grid=test_param, cv=3, n_jobs=-1,
                                       scoring='neg_mean_squared_error')
            grid_search.fit(x_train, y_train)
            print("best parameter==========================：{}".format(grid_search.best_params_))
            self.model = grid_search.best_estimator_
        except:
            lgb_model = lgb.LGBMRegressor(**parameters)
            self.model = lgb_model.fit(x_train, y_train, categorical_feature=category_features,
                                       eval_set=[(x_test, y_test)],
                                       eval_metric=['mape'], early_stopping_rounds=100)

        # 查看特征重要性
        feature_importance_df = pd.DataFrame(self.model.feature_importances_, columns=['importance'])
        feature_importance_df['feature_name'] = category_features + integer_features + double_features
        feature_importance_df.sort_values('importance', ascending=False, inplace=True)
        feature_importance_df['rank'] = feature_importance_df['importance'].rank(method='first', ascending=False)

        feature_importance_df.to_csv('{}/data/fea_importance.csv'.format(BASE_DIR), index=False)

        # 验证模型
        pred = self.model.predict(x_test)
        mape = mean_absolute_percentage_error(pred, y_test)
        mae = mean_absolute_error(pred, y_test)
        print('eval mape: {}'.format(mape))
        print('eval mae: {}'.format(mae))

        print("save the model")
        self.save_model(self.model, '{}/model/lgb_model.pkl'.format(BASE_DIR))

    def save_model(self, model, path):
        joblib.dump(model, path)
