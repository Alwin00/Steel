import os
import sys
from src.final_carbon_train import *
import argparse
import logging
from src.process_data import *

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--df_path', type=str,
                        default='./data/2021_data.xlsx', help='set train data path')
    parser.add_argument('--save_path', type=str,
                        default='./model/bpnn.pkl', help='set train data path')
    parser.add_argument('--std_mean_path', type=str,
                        default='./data/std.xlsx', help='set model path')

    args = parser.parse_args()

    train_df = load_data(args.df_path)
    write_std_mean(train_df[1], args.std_mean_path)

    train_model = FinalTempModel(train_df, args.save_path)
    train_model.train()
