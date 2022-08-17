import argparse
from bpnn_predict import *
from process_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--df_path', type=str,
                        default='../../data/2022年全部数据.xlsx', help='set train data path')
    parser.add_argument('--model_path', type=str,
                        default='../../model/bpnn.pkl', help='set train data path')
    parser.add_argument('--standard_flag', type=bool, default=False, help='whether do standardization for label')

    args = parser.parse_args()

    pred_df = remove_missing_val(args.df_path)

    # 模型预测
    predict(pred_df, args.model_path, args.standard_flag)
