import os
from datetime import datetime, date
import pandas as pd
from conf.lgb_conf import *
import numpy as np


def __time_diff(time1, time2):
    if not isinstance(time1, int) and not isinstance(time1, float):
        return np.nan
    if not isinstance(time2, int) and not isinstance(time2, float):
        return np.nan
    if np.isnan(time1) or np.isnan(time2):
        return np.nan
    try:
        t1 = str(int(time1))
        t2 = str(int(time2))
        hour1 = t1[8:10]
        hour2 = t2[8:10]
        min1 = t1[10:12]
        min2 = t2[10:12]
        sec1 = t1[12:]
        sec2 = t2[12:]

        t1 = ':'.join([hour1, min1, sec1])
        t2 = ':'.join([hour2, min2, sec2])
        time_1_struct = datetime.strptime(t1, "%H:%M:%S")
        time_2_struct = datetime.strptime(t2, "%H:%M:%S")
        seconds = (time_2_struct - time_1_struct).seconds
        return seconds
    except Exception as ex:
        print(time1)
        print(time2)
        return np.nan


def __lance_diff(pos1, pos2):
    if np.isnan(pos1) or np.isnan(pos2):
        return np.nan
    return pos2 - pos1


def process_data(data_path):
    df = pd.read_excel(data_path, index_col=None)

    # 构造时间差值特征
    df['TOTAL_COST_SEC'] = df.apply(lambda row: __time_diff(row['START_TIME'], row['END_TIME']), axis=1)
    df['IRON_BLOW_COST_SEC'] = df.apply(lambda row: __time_diff(row['IRON_LOAD_TIME'], row['BLOW_START_TIME']), axis=1)
    df['BLOW_SCRAP_COST_SEC'] = df.apply(lambda row: __time_diff(row['BLOW_START_TIME'], row['SCRAP_LOAD_TIME']),
                                         axis=1)
    df['BLOW_COST_SEC'] = df.apply(lambda row: __time_diff(row['BLOW_START_TIME'], row['BLOW_END_TIME']), axis=1)
    df['SEC_BLOW_SEC'] = df.apply(lambda row: __time_diff(row['SEC_BLOW_START_TIME'], row['SEC_BLOW_END_TIME']), axis=1)
    df['SEC_BLOW_CATCH_COST_SEC'] = df.apply(lambda row: __time_diff(row['SEC_BLOW_START_TIME'], row['CATCH_C_TIME']),
                                             axis=1)
    df['TAP_COST_SEC'] = df.apply(lambda row: __time_diff(row['TAP_START_TIME'], row['TAP_END_TIME']), axis=1)

    # 枪位高度特征
    df['START_MID_LANCE_POS_DIFF'] = df.apply(lambda row: __lance_diff(row['START_LANCE_POS'], row['MID_LANCE_POS']),
                                              axis=1)
    df['MID_END_LANCE_POS_DIFF'] = df.apply(lambda row: __lance_diff(row['MID_LANCE_POS'], row['END_LANCE_POS']),
                                            axis=1)
    df['START_END_LANCE_POS_DIFF'] = df.apply(lambda row: __lance_diff(row['START_LANCE_POS'], row['END_LANCE_POS']),
                                              axis=1)

    # 规范类型
    df[category_features] = df[category_features].astype('category')
    # df[integer_features] = df[integer_features].astype('int')
    # df[double_features] = df[double_features].astype('float')
    return df


if __name__ == '__main__':
    pass