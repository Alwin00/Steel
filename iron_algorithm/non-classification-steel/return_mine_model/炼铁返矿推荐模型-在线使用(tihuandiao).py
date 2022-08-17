import os
import time
import joblib  # jbolib模块
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 定义函数 机理模型
def BOF_LTFK_CACL(data):
    per_C = 0.957   #碳氧反应生成的co和co2的比例
    Slag_TFe = 0.16   #渣中全铁
    Slag_ratio = 0.07   #炉渣量和金属料的比值
    FG_C = 0.23   #废钢中C含量
    FG_Si = 0.55  #废钢中Si含量
    FG_Mn = 0.5   #废钢中Mn含量
    FG_P = 0.035  #废钢中P含量
    ZCST_C = 4.42  #自产生铁中C含量
    ZCST_Si = 0.46 #自产生铁中Si含量
    ZCST_Mn = 0.36 #自产生铁中Mn含量
    ZCST_P = 0.12  #自产生铁中P含量
    Arrival_C = data['iron_c'] #铁水C含量
    Arrival_Si = Idata['iron_si'] #铁水Si含量
    Arrival_Mn = data['iron_mn'] #铁水Mn含量
    Arrival_P = data['iron_p']  #铁水P含量
    Arrival_S = data['iron_s']  #铁水S含量
    Weight_Iron = data['MOLTIRON_WT']   #铁水重量
    Weight_FG = data['5G11']    #废钢重量
    Weight_ZCST = data['8ZCST'] #自产生铁重量

    End_C =0.24 #目标C含量
    End_Si = 0.55   #目标Si含量
    End_Mn = 1.48   #目标Mn含量
    End_P = 0       #目标P含量
    End_S = 0       #目标S含量
    Weight_Slag = (Weight_Iron + Weight_FG + Weight_ZCST) * Slag_ratio  # 炉渣重量


    #热收入
    #铁水熔点  其中1539℃为纯铁的熔点,元素前的数字代表对应元素的温度系数，w代表对应元素的质量分数，7代表气体O、H、N共降低铁水熔点值℃
    T_rd=1539-(100*Arrival_C+8*Arrival_Si+5*Arrival_Mn+30*Arrival_P+25*Arrival_S)-7
    # print(T_rd)
    #铁水物理热 取加入炉内的炉料温度均为25℃
    Iron_Physical_Heat=Weight_Iron*(0.744*(T_rd-25)+217.486+0.8368*(IRON_TEMP-T_rd))

    #元素氧化以及成渣热
    Oxidation_Slagging_Heat=(
                                ((Weight_Iron*Arrival_C+Weight_FG+FG_C+Weight_ZCST*ZCST_C-Weight_Iron*End_C)*(per_C*11637+(1-per_C)*34824))+
                                ((Weight_Iron*Arrival_Si+Weight_FG+FG_Si+Weight_ZCST*ZCST_Si-Weight_Iron*End_Si)*29177)+
                                ((Weight_Iron*Arrival_Mn+Weight_FG+FG_Mn+Weight_ZCST*ZCST_Mn-Weight_Iron*End_Mn)*6593)+
                                ((Weight_Iron*Arrival_P+Weight_FG+FG_P+Weight_ZCST*ZCST_P-Weight_Iron*End_P)*18923)+
                                (Weight_Slag * Slag_TFe) * 4249+
                                (((Weight_Iron*Arrival_Si+Weight_FG+FG_Si+Weight_ZCST*ZCST_Si-Weight_Iron*End_Si)* 60 / 28+(Weight_Iron+Weight_FG+Weight_ZCST)*0.005*0.3)*1620)+
                                (Weight_Iron*Arrival_P+Weight_FG+FG_P+Weight_ZCST*ZCST_P-Weight_Iron*End_P)*142/62*4020
                            )/100
    #烟尘氧化放热
    Soot_Oxidation_Heat=(Weight_Iron+Weight_FG+Weight_ZCST)*0.016*(0.77*56/72*4020+0.2*112/160*6670)
    #炉衬中碳氧化放热
    LiningQS_Oxidation_Heat=(Weight_Iron + Weight_FG + Weight_ZCST) * 0.005 * 0.05* (per_C * 10940 + (1 - per_C) * 34420)
    #总热收入
    in_heat=(Iron_Physical_Heat+Oxidation_Slagging_Heat+Soot_Oxidation_Heat+LiningQS_Oxidation_Heat)


    #热支出
    #钢水物理热
    T_rd_steel = 1539 - (65 * End_C + 8 *End_Si + 5 * End_Mn + 30 * End_P + 25 * Arrival_S) - 7
    # print(T_rd_steel)
    Steel_Physical_Heat=Weight_Iron *(0.699*(T_rd_steel-25)+271.96+0.8386*(End_T-T_rd_steel))
    #废钢物理热
    FG_Steel_Physical_Heat=(Weight_FG)*(0.699*(T_rd_steel-25)+271.96+0.8386*(End_T-T_rd_steel))
    #生铁物理热
    ZCST_Steel_Physical_Heat=(Weight_ZCST)*(0.745*(T_rd_steel-25)+218+0.8386*(End_T-T_rd_steel))
    #炉渣物理热
    Slag_Physical_Heat=Weight_Slag*(1.247*(End_T-25)+209.2)
    #烟尘物理热 为为铁水量的1.6%
    Soot_Physical_Heat=(Weight_Iron + Weight_FG + Weight_ZCST) * 0.016*(1*(1450-25)+209.2)
    #炉气物理热
    Gas_Physical_Heat = (
        (Weight_Iron * Arrival_C + Weight_FG + FG_C + Weight_ZCST * ZCST_C - Weight_Steel * End_C)/100
        +(Weight_Iron + Weight_FG + Weight_ZCST) * 0.005 * 0.0013
         )*(per_C*28/12+(1-per_C)*44/12)*1.136*(1450-25)
    #渣中铁珠物理热  渣中铁珠量为渣量的8%
    Slag_Iron_bead_Physical_Heat=Weight_Slag*0.08*(0.744*(T_rd_steel-25)+271.468+0.8386*(End_T-T_rd_steel))
    #喷溅金属物理热 为铁水量的1%
    Splashing_metal_Physical_Heat=(Weight_Iron + Weight_FG + Weight_ZCST)*0.01*(0.744*(T_rd_steel-25)+271.468+0.8386*(End_T-T_rd_steel))
    #吹炼过程热损失 一般为热量总收入的3~8%，本次取5%
    Loss_Heat = in_heat*0.05
    #总热支出
    out_heat=Steel_Physical_Heat + FG_Steel_Physical_Heat+ZCST_Steel_Physical_Heat + Slag_Physical_Heat + Soot_Physical_Heat + Gas_Physical_Heat + Slag_Iron_bead_Physical_Heat + Splashing_metal_Physical_Heat+Loss_Heat

    #1kg炼铁返矿的冷却效应
    per_LTFK_Heat=1*(1.016*(1450-25)+209.20+0.8368*(End_T-25)+LTFK_Fe2O3*112/160/100*6670+LTFK_FeO*56/72*4020/100)
    #炼铁返矿消耗量
    Weight_LTFK=(in_heat-out_heat)/per_LTFK_Heat/1000

    Weight_LTFK = np.array(Weight_LTFK)

    if Weight_LTFK >= 6.421:
        answer = 6.421
        return answer
    else:
        answer = Weight_LTFK
    return answer


#数据模型
def  BOF_LTFK_DATA(name,data):
    # 模型需要的数据项
    predict_O2 = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                  '8ZCST', '5G11']


    #读取标准化规则数据表
    df_std = pd.read_excel('std.xlsx',  index_col=0)
    mean = df_std.loc[name, 'outmean']
    scale = df_std.loc[name, 'outscale']

    # 读取训练后的模型
    model = joblib.load(name+'.pkl')

    # 读取用于预测的数据
    X = data[predict_O2]

    #  输入数据标准化
    X_test = np.empty(X[predict_O2].values.shape)
    for i in range(len(predict_O2) ):
        x = (X[predict_O2[i]].values - df_std.loc[name, 'inmean' + str(i)]) / df_std.loc[
            name, 'inscale' + str(i)]
        X_test[:, i] = x

    # 预测
    y_predict = model.predict(X_test)

    # 输出数据处理
    answer = y_predict * scale + mean

    return answer

#不分钢种炼铁返矿数据模型使用范围
def JUDGE1(data):
    if 69.05<=data['MOLTIRON_WT'].values<=100.47 and 1227<=data['IRON_TEMP'].values<=1413 and 3.54<=data['iron_c'].values<=5.27 and 0.1<=data['iron_si'].values<=0.85and 6.0<=data['5G11'].values<=28.0:
        judge = True
        return judge
    else:
        judge = False
        return judge

# 炼铁返矿预测过程
def mainpredict(data):
    '''
    data:dataframe数据，用于预测的数据,(1,n)
    judge:bool，判断数据是否在正常范围内的判断结果
    '''

    judge = JUDGE1(data)
    if judge == True:

       answer=BOF_LTFK_DATA('不分钢种炼铁返矿',data)
       return answer

    else:
        answer = BOF_LTFK_CACL(data)
        return answer

if __name__ == '__main__':

    data=pd.read_excel('try.xlsx')
    tbg = time.time()
    answer=mainpredict(data)
    print(answer)
    ted = time.time()
    print('总用时：', ted - tbg)
