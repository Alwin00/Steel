import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

feature_col = ['MOLTIRON_WT', 'IRON_TEMP', 'IRON_C', 'IRON_SI', 'IRON_MN', 'IRON_P', 'IRON_S',
               'O2_SUM_COMSUME', '8ZCQSBYS', '8ZCSSH', '91LTFK', '8ZCST', '5G11', 'FURNACE_AGE']
target_col = ['OUT_STEEL_PRE_TEMP']


def __remove_missing_val(df):
    logging.info(len(df))
    df.dropna(inplace=True, subset=feature_col)
    logging.info(len(df))
    return df


def load_data(df_path):
    df = pd.read_excel(df_path, index_col=None)
    df = __remove_missing_val(df)
    # 导入数据中的影响因素
    feature = df[feature_col]
    # 导入目标值，本模型中是转炉终点碳含量
    target = df[target_col]

    feature = feature.values
    target = (target.values.ravel()).reshape((-1, 1))

    return feature, target


def write_std_mean(target_df, df_path):
    df1 = pd.DataFrame()

    y_scale = StandardScaler()
    target = y_scale.fit_transform(target_df)

    df1['outmean'] = y_scale.mean_
    df1['outscale'] = y_scale.scale_

    df1.to_excel(df_path, sheet_name='std')


def read_std_mean(df_path):
    mean_std = pd.read_excel(df_path, index_col=0)
    mean = mean_std['outmean'].values[0]
    std = mean_std['outscale'].values[0]
    return std, mean
