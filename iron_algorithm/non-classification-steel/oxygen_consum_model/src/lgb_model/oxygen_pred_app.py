import os
import sys
from lgb_train import OxygenModel
from lgb_predict import OxygenPredict
from process_data import *
from conf.lgb_conf import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('{}/../'.format(__file__))))
sys.path.append(BASE_DIR)

if __name__ == '__main__':
    train_df = process_data('{}/data/2021_data.xlsx'.format(BASE_DIR))
    om = OxygenModel(model_confs)
    om.fit(train_df)

    # 模型预测
    predict_df = process_data('{}/data/2022_data.xlsx'.format(BASE_DIR))
    op = OxygenPredict('{}/model/lgb_model.pkl'.format(BASE_DIR))
    op.predict(predict_df)

