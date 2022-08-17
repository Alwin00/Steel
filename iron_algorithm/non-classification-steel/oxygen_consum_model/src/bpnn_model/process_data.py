import os
import pandas as pd
import logging


def remove_missing_val(data_path):
    df = pd.read_excel(data_path, index_col=None)
    cols = ['MOLTIRON_WT', 'IRON_TEMP', 'IRON_C', 'IRON_SI', 'IRON_MN', 'IRON_P', 'IRON_S', '8ZCST', '5G11']
    logging.info(len(df))
    df.dropna(inplace=True, subset=cols)
    logging.info(len(df))

    return df
