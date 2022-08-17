import os
import time
import numpy as np
import joblib  # jbolib模块
import pandas as pd
from sklearn.preprocessing import StandardScaler


def BOF_Slag_Recommend(w_Si_hm, Weight_hm, w_Si_scrap=0.598, Weight_scrap=10, w_Si_pi=0.46, Weight_pi=3,
                       w_CaO_lime=91.1, w_SiO2_lime=1.12, w_MgO_lime=0.95,
                       w_CaO_dolomite=51.46, w_MgO_dolomite=33.94,
                       End_Si=0, w_MgO_slag=6, R=3.47,
                       w_MgO_lc=64, per_lc=0.8, per_slag=4.417, lyl_Lime=100, lyl_Dolomite=100
                       ):
    '''
    w_MgO_slag      渣中MgO要求，%
    w_CaO_dolomite  轻烧白云石中CaO含量，%
    w_MgO_dolomite  轻烧白云石中MgO含量，%
    # lcMgO = 2.5  # 炉衬中MgO含量，%
    '''
    Weight_lc = per_lc / 100 * (Weight_hm + Weight_scrap + Weight_pi)
    Weight_slag = per_slag / 100 * (Weight_hm + Weight_scrap + Weight_pi)  # 渣量，t

    # 所需石灰初算
    Weight_Lime = (2.14 * (w_Si_hm - End_Si)) / (w_CaO_lime - R * w_SiO2_lime) * R * (
            Weight_hm + w_Si_scrap / w_Si_hm * Weight_scrap + w_Si_pi / w_Si_hm * Weight_pi) / (lyl_Lime / 100)

    # 所需轻烧白云石计算
    Weight_dolomite = Weight_slag * w_MgO_slag / w_MgO_dolomite
    Weight_Limemgo = Weight_Lime * w_MgO_lime / w_MgO_dolomite
    Weight_lcmgo = Weight_lc * w_MgO_lc / w_MgO_dolomite
    Weight_dolomite_final = (Weight_dolomite - Weight_Limemgo - Weight_lcmgo) / (lyl_Dolomite / 100)

    # 石灰修正
    Weight_Limezhe = Weight_dolomite_final * w_CaO_dolomite / (w_CaO_lime - R * w_SiO2_lime)
    Weight_Lime_final = Weight_dolomite - Weight_Limezhe

    if Weight_Lime_final < 1.428:
        Weight_Lime_final = 1.428
    elif Weight_Lime_final > 5.773:
        Weight_Lime_final = 5.773

    if Weight_dolomite_final < 0.386:
        Weight_dolomite_final = 0.386
    elif Weight_dolomite_final > 5.246:
        Weight_dolomite_final = 5.246

    return Weight_Lime_final, Weight_dolomite_final


# CB004数据模型使用范围
def JUDGE(data):
    if 68.69 <= data['MOLTIRON_WT'] <= 91.52 and 1225 <= data['IRON_TEMP'] <= 1412 and 3.56 <= data[
        'iron_c'] <= 5.21 and 0.12 <= data['iron_si'] <= 0.82 and 10.0 <= data['5G11'] <= 29.0:
        judge = True
        return judge
    else:
        judge = False
        return judge


def run(data, name):
    predict = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn',
               'iron_p', 'iron_s', '8ZCST', '5G11']
    df_std = pd.read_excel('不分钢种std.xlsx', index_col=0)
    outmean = df_std.loc[name, 'outmean']
    outscale = df_std.loc[name, 'outscale']
    # 读取训练后的模型
    model = joblib.load(f'{name}.pkl')
    # 读取用于预测的数据
    X_test = data[predict]
    # 输入数据标准化

    X_test_final = np.empty(X_test[predict].values.shape)
    for i in range(len(predict)):
        mean = df_std.loc[name, f'inmean{i}']
        scale = df_std.loc[name, f'inscale{i}']
        x = (X_test[predict[i]] - mean) / scale
        # print(x)
        X_test_final[i] = x

    X_test_final = X_test_final.reshape(1, len(X_test_final))
    # 预测
    y_predict = model.predict(X_test_final)
    # 输出数据处理
    answer = y_predict * outscale + outmean
    return answer


# 石灰、白云石预测过程
def main(data, type):
    '''
    data:dataframe数据，用于预测的数据,(1,n)
    judge:bool，判断数据是否在正常范围内的判断结果
    type:str,预测目标，'SH'或'BYS'
    '''
    judge = JUDGE(data)

    if type == 'SH':
        if judge:
            # 调用运行函数，得到结果
            answer = run(data, '不分钢种辅料优化石灰')
            return answer



        else:
            arrival_Si = data['iron_si']
            weight_Iron = data['MOLTIRON_WT']
            weight_scrap = data['5G11']
            weight_pi = data['8ZCST']
            Weight_lime, Weight_dolomite = BOF_Slag_Recommend(w_Si_hm=arrival_Si, Weight_hm=weight_Iron,
                                                              Weight_scrap=weight_scrap, Weight_pi=weight_pi,
                                                              w_CaO_dolomite=55.4, w_MgO_dolomite=35.26,
                                                              )
            return Weight_dolomite


    elif type == 'BYS':
        if judge:

            # 调用运行函数，得到结果
            answer = run(data, '不分钢种辅料优化白云石')
            return answer
        else:
            arrival_Si = data['iron_si']
            weight_Iron = data['MOLTIRON_WT']
            weight_scrap = data['5G11']
            weight_pi = data['8ZCST']
            Weight_lime, Weight_dolomite = BOF_Slag_Recommend(w_Si_hm=arrival_Si, Weight_hm=weight_Iron,
                                                              Weight_scrap=weight_scrap, Weight_pi=weight_pi,
                                                              w_CaO_dolomite=55.4, w_MgO_dolomite=35.26,
                                                              )
            return Weight_dolomite


if __name__ == '__main__':
    examples = pd.read_excel('不分钢种辅料优化白云石.xlsx')
    for i in range(len(examples)):
        exm = examples.iloc[i]
        Weight_dolomite = main(exm, 'SH')
        print(Weight_dolomite)
