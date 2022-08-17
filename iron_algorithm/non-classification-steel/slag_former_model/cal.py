import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cal():
    predict_BYS = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                   '8ZCST', '5G11']
    y_BYS = ['QSBYS']

    predict_SH = ['MOLTIRON_WT', 'IRON_TEMP', 'iron_c', 'iron_si', 'iron_mn', 'iron_p', 'iron_s',
                  '8ZCST', '5G11']
    y_SH = ['8ZCSSH']

    df1 = pd.DataFrame()
    for fname in os.listdir('.'):
        p = 0
        # print(fname.find(f'CB004'))
        if fname.find(f'辅料优化') >= 0 and fname.find(f'xlsx') > 0:
            if fname.find(f'石灰') >= 0:
                print(fname)
                data1 = pd.read_excel(fname, header=0)
                X_test = data1[predict_SH].values
                y_test = data1[y_SH].values
                # 读取用于反标准化的数据
                feature_train, feature_test, target_train, target_test = train_test_split(X_test, y_test,
                                                                                          test_size=0.33,
                                                                                          random_state=1)

                x_scale = StandardScaler()
                feature_train = x_scale.fit_transform(feature_train)

                y_scale = StandardScaler()
                target_train = y_scale.fit_transform(target_train)

                for i in range(len(x_scale.mean_)):
                    df1.loc[fname[:-5], f'inmean{i}'] = x_scale.mean_[i]
                    df1.loc[fname[:-5], f'inscale{i}'] = x_scale.scale_[i]
                df1.loc[fname[:-5], 'outmean'] = y_scale.mean_[0]
                df1.loc[fname[:-5], 'outscale'] = y_scale.scale_[0]
            elif fname.find(f'白云石') >= 0:
                print(fname)
                data1 = pd.read_excel(fname, header=0)
                X_test = data1[predict_BYS].values
                y_test = data1[y_BYS].values
                # 读取用于反标准化的数据
                feature_train, feature_test, target_train, target_test = train_test_split(X_test, y_test,
                                                                                          test_size=0.33,
                                                                                          random_state=1)

                x_scale = StandardScaler()
                feature_train = x_scale.fit_transform(feature_train)

                y_scale = StandardScaler()
                target_train = y_scale.fit_transform(target_train)

                for i in range(len(x_scale.mean_)):
                    df1.loc[fname[:-5], 'inmean' + str(i)] = x_scale.mean_[i]
                    df1.loc[fname[:-5], 'inscale' + str(i)] = x_scale.scale_[i]
                df1.loc[fname[:-5], 'outmean'] = y_scale.mean_[0]
                df1.loc[fname[:-5], 'outscale'] = y_scale.scale_[0]

    df1.to_excel('不分钢种std.xlsx')


if __name__ == '__main__':
    cal()
