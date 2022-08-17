import joblib  # jbolib模块
from src.metrics import *
from sklearn.preprocessing import StandardScaler

class FinalTempPred:
    def __init__(self, pred_df, model_path, std, mean):
        self.feature = pred_df[0]
        self.target = pred_df[1]
        self.model_path = model_path
        self.std = std
        self.mean = mean

    def predict(self):
        x_scale = StandardScaler()

        # 数据标准化处理
        feature = x_scale.fit_transform(self.feature)

        # 模型加载
        model = joblib.load(self.model_path)

        # 模型预测
        pred = model.predict(feature)
        pred = pred * self.std + self.mean

        mape = mean_absolute_percentage_error(self.target, pred)
        mae = mean_absolute_error(self.target, pred)
        print('pred target: {}'.format(np.mean(self.target)))
        print('pred pred: {}'.format(np.mean(pred)))
        print('pred mape: {}'.format(mape))
        print('pred mae: {}'.format(mae))

        hit_rate1, hit_rate2, hit_rate3 = hitrate(self.target, pred)
        print('pred hit rate 8: {}'.format(hit_rate1))
        print('pred hit rate 10: {}'.format(hit_rate2))
        print('pred hit rate 15: {}'.format(hit_rate3))

# if __name__ == '__main__':
#     predict('{}/data/2022_data_process.xlsx'.format(BASE_DIR), '{}/model/bpnn.pkl'.format(BASE_DIR), False)
