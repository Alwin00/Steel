import os
import time
import joblib  # jbolib模块
import pandas as pd
import numpy as np




#数据模型
def  BOF_END_C_DATA(name,data):
    # 模型需要的数据项
    predict_C = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                 'O2_SUM_COMSUME', '8ZCQSBYS', '8ZCSSH', '91LTFK', '8ZCST', '5G11']

    #读取标准化规则数据表
    df_std = pd.read_excel('std.xlsx',  index_col=0)
    mean = df_std.loc[name, 'outmean']
    scale = df_std.loc[name, 'outscale']

    # 读取训练后的模型
    model = joblib.load(name+'.pkl')

    # 读取用于预测的数据
    X = data[predict_C]

    #  输入数据标准化
    X_test = np.empty(X[predict_C].values.shape)
    for i in range(len(predict_C) ):
        x = (X[predict_C[i]].values - df_std.loc[name, 'inmean' + str(i)]) / df_std.loc[
            name, 'inscale' + str(i)]
        X_test[:, i] = x

    # 预测
    y_predict = model.predict(X_test)

    # 输出数据处理
    answer = y_predict * scale + mean

    return answer

#不分钢种数据模型使用范围
def JUDGE(data):
    if 69.89<=data['MOLTIRON_WT'].values<=100.35 and 1224<=data['IRON_TEMP'].values<=1413 and 3.55<=data['iron_c'].values<=5.25 and\
            0.07<=data['iron_si'].values<=0.83 and 6.0<=data['5G11'].values<=28.0 and 3289<=data['O2_SUM_COMSUME'].values<=5095 and 0.91<=data['8ZCSSH'].values<=5.42 :
        judge = True
        return judge
    else:
        judge = False
        return judge

# 终点温度预测过程
def mainpredict(data):
    '''
    data:dataframe数据，用于预测的数据,(1,n)
    judge:bool，判断数据是否在正常范围内的判断结果
    '''

    judge = JUDGE(data)
    if judge == True:
       answer=BOF_END_C_DATA('不分钢种终点碳含量',data)
       return answer



if __name__ == '__main__':
    data=pd.read_excel('try.xlsx')
    tbg = time.time()
    print('终点碳含量：',mainpredict(data))
    ted = time.time()
    print('总用时：', ted - tbg)

