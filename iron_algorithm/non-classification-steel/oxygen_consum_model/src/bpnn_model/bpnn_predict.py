import os
import sys
import time
import joblib  # jbolib模块
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('{}/../'.format(__file__))))
sys.path.append(BASE_DIR)


# 读取excel表中数据
def load_data(df):
    # 导入数据中的影响因素
    predict_O2 = ['MOLTIRON_WT', 'IRON_TEMP', 'IRON_C', 'IRON_SI', 'IRON_MN', 'IRON_P', 'IRON_S',
                  '8ZCST', '5G11']
    feature = df[predict_O2]

    # 导入目标值，本模型中是转炉氧气耗量
    target = df.O2_SUM_COMSUME
    return feature, target


def predict(pred_df, model_path, standard_flag):
    feature, target = load_data(pred_df)

    x_scale = StandardScaler()
    y_scale = StandardScaler()

    feature = feature.values
    target = (target.values.ravel()).reshape((-1, 1))

    # 数据标准化处理
    feature = x_scale.fit_transform(feature)
    if standard_flag:
        target = y_scale.fit_transform(target)

    # 模型加载
    model = joblib.load(model_path)

    # 模型预测
    pred = model.predict(feature)
    pred_test = pd.DataFrame(pred)
    pred_test.to_csv('result.csv')
    mape = mean_absolute_percentage_error(pred, target)
    mae = mean_absolute_error(pred, target)
    print('pred mape: {}'.format(mape))
    print('pred mae: {}'.format(mae))

    hit_rate1, hit_rate2, hit_rate3 = hitrate(target, pred)
    print('pred hit rate 150: {}'.format(hit_rate1))
    print('pred hit rate 300: {}'.format(hit_rate2))
    print('pred hit rate 450: {}'.format(hit_rate3))


def hitrate(y_test, y_predict, level1=150, level2=300, level3=450):
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
