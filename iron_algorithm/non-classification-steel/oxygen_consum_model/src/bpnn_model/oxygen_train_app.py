import os
import sys
from bpnn_train import *
import argparse
import logging
from process_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--df_path', type=str,
                        default='../../data/data2020-202111.xlsx', help='set train data path')
    parser.add_argument('--save_path', type=str,
                        default='../../model/bpnn.pkl', help='set train data path')
    parser.add_argument('--standard_flag', type=bool, default=False, help='whether do standardization for label')

    args = parser.parse_args()

    train_df = remove_missing_val(args.df_path)
    BPNN_O2(train_df, args.standard_flag, args.save_path)


