import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib # jbolib模块
import os


# 读取excel表中数据
def load_data(df):

    #导入数据中的影响因素
    predict_T = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                 'O2_SUM_COMSUME', '8ZCQSBYS', '8ZCSSH', '91LTFK', '8ZCST', '5G11','FURNACE_AGE']
    feature=df[predict_T]
    #导入目标值，本模型中是转炉终点碳含量
    y_T = ['OUT_STEEL_PRE_TEMP']
    target = df[y_T]
    return feature,target




def std():
    df1 = pd.DataFrame()
    for fname in os.listdir('../..'):
        # print(df1)
        if fname.find(f'温度') >= 0 and fname.find(f'xlsx') > 0:

            data1 = pd.read_excel(fname, header=0)

            feature, target=load_data(data1)

            x_scale = StandardScaler()
            y_scale = StandardScaler()

            feature = feature.values
            target = (target.values.ravel()).reshape((-1, 1))


            #切分训练集和测试集
            feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.33,
                                                                                    random_state=1)

            # 数据标准化处理
            feature_train = x_scale.fit_transform(feature_train)
            target_train = y_scale.fit_transform(target_train)

            for i in range(len(x_scale.mean_)):
                df1.loc[fname[0:-5], 'inmean'+str(i)] = x_scale.mean_[i]
                df1.loc[fname[0:-5], 'inscale'+str(i)] = x_scale.scale_[i]
            df1.loc[fname[0:-5], 'outmean'] = y_scale.mean_[0]
            df1.loc[fname[0:-5], 'outscale'] = y_scale.scale_[0]
    return df1


if __name__ == '__main__':
    df1=std()

    with pd.ExcelWriter('../../data/std.xlsx') as writer:
        df1.to_excel(writer, sheet_name='std')
