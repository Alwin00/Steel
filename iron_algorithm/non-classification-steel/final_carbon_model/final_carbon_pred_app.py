import os
import sys
from src.final_carbon_pred import *
import argparse
import logging
from src.process_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--df_path', type=str,
                        default='./data/2022_data.xlsx', help='set predict data path')
    parser.add_argument('--model_path', type=str,
                        default='./model/bpnn.pkl', help='set model path')
    parser.add_argument('--std_mean_path', type=str,
                        default='./data/std.xlsx', help='set model path')

    args = parser.parse_args()

    pred_df = load_data(args.df_path)
    std, mean = read_std_mean(args.std_mean_path)
    pred_model = FinalTempPred(pred_df, args.model_path, std, mean)
    pred_model.predict()
