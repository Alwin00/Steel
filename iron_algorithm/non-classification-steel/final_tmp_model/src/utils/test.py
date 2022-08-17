import os
import time
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",None)


if __name__ == '__main__':
    name = "不分钢种终点温度_1630-1700"
    data=pd.read_excel('try.xlsx')
    df_std = pd.read_excel('std.xlsx', index_col=0)
    predict_T = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                 'O2_SUM_COMSUME', '8ZCQSBYS', '8ZCSSH', '91LTFK', '8ZCST', '5G11', 'FURNACE_AGE']

    # 读取用于预测的数据
    X1 = np.array(data[predict_T])
    X = pd.DataFrame(X1,columns=predict_T)
    #print(X)
    print(df_std)

    # print(X)
    # print(X[predict_T])
    # print(X[predict_T].values.shape)

    #  输入数据标准化
    X_test = np.empty(X[predict_T].values.shape)
    for i in range(len(predict_T) ):
        x = (X[predict_T[i]].values - df_std.loc[name, 'inmean' + str(i)]) / df_std.loc[
            name, 'inscale' + str(i)]
        X_test[:, i] = x

    print(X_test)
