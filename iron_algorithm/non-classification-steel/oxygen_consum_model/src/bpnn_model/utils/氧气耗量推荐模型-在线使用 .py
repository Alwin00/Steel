import os
import time
import joblib  # jbolib模块
import pandas as pd
import numpy as np


# 定义机理模型
def BOF_OXYGEN_CACL(data):
    per_C = 0.957  # 碳氧反应生成的co和co2的比例
    per_S = 1 / 3  # 炉内脱硫1/3
    Slag_TFe = 0.16  # 渣中全铁
    Slag_ratio = 0.07  # 炉渣量和金属料的比值
    O2_ratio = 0.87  # 氧气利用率
    FG_C = 0.23  # 废钢中C含量
    FG_Si = 0.55  # 废钢中Si含量
    FG_Mn = 0.5  # 废钢中Mn含量
    FG_P = 0.035  # 废钢中P含量
    FG_S = 0.011  # 废钢中S含量
    ZCST_C = 4.42  # 自产生铁中C含量
    ZCST_Si = 0.46  # 自产生铁中Si含量
    ZCST_Mn = 0.36  # 自产生铁中Mn含量
    ZCST_P = 0.12  # 自产生铁中P含量
    ZCST_S = 0.03  # 自产生铁中S含量
    Arrival_C = data['iron_c']  # 铁水C含量
    Arrival_Si = data['iron_si']  # 铁水Si含量
    Arrival_Mn = data['iron_mn']  # 铁水Mn含量
    Arrival_P = data['iron_p']  # 铁水P含量
    Arrival_S = data['iron_s']  # 铁水S含量
    Weight_Iron = data['MOLTIRON_WT']  # 铁水重量
    Weight_FG = data['5G11']  # 废钢重量
    Weight_ZCST = data['8ZCST']  # 自产生铁重量

    End_C = 0.24  # 目标C含量
    End_Si = 0.55  # 目标Si含量
    End_Mn = 1.48  # 目标Mn含量
    End_P = 0  # 目标P含量
    End_S = 0  # 目标S含量
    Weight_Slag = (Weight_Iron + Weight_FG + Weight_ZCST) * Slag_ratio  # 炉渣重量
    '''

    CO:CO2生成比例为9.57：0.43
    氧气纯度取99%，氧气利用率取87%
    假设渣量为金属料的7%，渣中TFe=16%，渣中生成FeO
    烟尘铁损量为金属料的1.6%
    炉衬侵蚀量为金属料的0.5%
    '''
    Weight_YQ = (((
                      (
                              (Arrival_C) * per_C * 16 / 12 + (Arrival_C) * (1 - per_C) * 32 / 12 +
                              (Arrival_Si) * 32 / 28 +
                              (Arrival_Mn) * 16 / 55 +
                              (Arrival_P) * 80 / 62 +
                              (Arrival_S) * per_S * (32 / 32) + (Arrival_S) * (1 - per_S) * ((-16) / 32)
                      )

                  ) / 100 * Weight_Iron * 1000
                  + (
                          (FG_C) * per_C * 16 / 12 + (FG_C) * (1 - per_C) * 32 / 12 +
                          (FG_Si) * 32 / 28 +
                          (FG_Mn) * 16 / 55 +
                          (FG_P) * 80 / 62 +
                          (FG_S) * per_S * (32 / 32) + (FG_S) * (1 - per_S) * ((-16) / 32)
                  ) / 100 * Weight_FG * 1000
                  + (
                          (ZCST_C) * per_C * 16 / 12 + (FG_C) * (1 - per_C) * 32 / 12 +
                          (ZCST_Si) * 32 / 28 +
                          (ZCST_Mn) * 16 / 55 +
                          (ZCST_P) * 80 / 62 +
                          (ZCST_S) * per_S * (32 / 32) + (FG_S) * (1 - per_S) * ((-16) / 32)
                  ) / 100 * Weight_ZCST * 1000

                  + Weight_Slag * Slag_TFe * 16 / 46

                  - (
                          (End_C) * per_C * 16 / 12 + (End_C) * (1 - per_C) * 32 / 12 +
                          (End_Si) * 32 / 28 +
                          (End_Mn) * 16 / 55 +
                          (End_P) * 80 / 62 +
                          (End_S) * per_S * (32 / 32) + (End_S) * (1 - per_S) * ((-16) / 32)
                  ) / 100 * Weight_Iron * 1000
                  + (Weight_Iron + Weight_FG + Weight_ZCST) * 0.016 * (0.77 * 16 / 72 + 0.2 * 48 / 160)  # 烟尘铁氧化
                  + (Weight_Iron + Weight_FG + Weight_ZCST) * 0.005 * 0.13 * (per_C * 16 / 12 + (1 - per_C) * 32 / 12)
                  # 炉衬侵蚀碳氧化
                  )
                ) / 0.9 / 0.99 * O2_ratio

    # 质量、体积转化
    v_O2 = Weight_YQ * 1000 / 32 * 22.4 * 0.001

    V_O2 = np.array(v_O2)

    if V_O2 >= 9964:
        answer = 9964
        return answer
    else:
        answer = V_O2
    return answer


# 数据模型
def BOF_OXYGEN_DATA(name, data):
    # 模型需要的数据项
    predict_O2 = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                  '8ZCST', '5G11']

    # 读取标准化规则数据表
    df_std = pd.read_excel('std.xlsx', index_col=0)
    mean = df_std.loc[name, 'outmean']
    scale = df_std.loc[name, 'outscale']

    # 读取训练后的模型
    model = joblib.load(name + '.pkl')

    # 读取用于预测的数据
    X = data[predict_O2]

    #  输入数据标准化
    X_test = np.empty(X[predict_O2].values.shape)
    for i in range(len(predict_O2)):
        x = (X[predict_O2[i]].values - df_std.loc[name, 'inmean' + str(i)]) / df_std.loc[
            name, 'inscale' + str(i)]
        X_test[:, i] = x

    # 预测
    y_predict = model.predict(X_test)

    # 输出数据处理
    answer = y_predict * scale + mean

    return answer


# 不分钢种氧气数据模型使用范围
def JUDGE1(data):
    if 69.05 <= data['MOLTIRON_WT'].values <= 100.47 and 1227 <= data['IRON_TEMP'].values <= 1413 and 3.54 <= data[
        'iron_c'].values <= 5.27 and 0.1 <= data['iron_si'].values <= 0.85 and 6.0 <= data['5G11'].values <= 28.0:
        judge = True
        return judge
    else:
        judge = False
        return judge


# 氧气预测过程
def mainpredict(data):
    '''
    data:dataframe数据，用于预测的数据,(1,n)
    judge:bool，判断数据是否在正常范围内的判断结果
    '''

    judge = JUDGE1(data)
    if judge == True:
        answer = BOF_OXYGEN_DATA('不分钢种氧气优化', data)
        return answer
    else:
        answer = BOF_OXYGEN_CACL(data)
        return answer


if __name__ == '__main__':
    data = pd.read_excel('try.xlsx')
    tbg = time.time()
    print('氧气耗量：', mainpredict(data))
    ted = time.time()
    print('总用时：', ted - tbg)
