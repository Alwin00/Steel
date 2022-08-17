from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import joblib  # jbolib模块
import logging


# 读取excel表中数据
def load_data(df):
    # 导入数据中的影响因素
    predict_O2 = ['MOLTIRON_WT', 'IRON_TEMP', 'IRON_C', 'IRON_SI', 'IRON_MN', 'IRON_P', 'IRON_S',
                  '8ZCST', '5G11']
    feature = df[predict_O2]

    # 导入目标值，本模型中是转炉氧气耗量
    y_O2 = ['O2_SUM_COMSUME']
    target = df[y_O2]
    return feature, target


def BPNN_O2(train_df, standard_flag, save_path):
    feature, target = load_data(train_df)

    x_scale = StandardScaler()
    y_scale = StandardScaler()

    feature = feature.values
    target = (target.values.ravel()).reshape((-1, 1))

    # 切分训练集和测试集
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.33,
                                                                              random_state=1)

    # 数据标准化处理
    logging.info('standardize data')
    feature_train = x_scale.fit_transform(feature_train)
    feature_test = x_scale.transform(feature_test)
    if standard_flag:
        target_train = y_scale.fit_transform(target_train)
        target_test = y_scale.fit_transform(target_test)

    # 创建BPNN模型
    bpnn = MLPRegressor(solver='adam',  # 随机梯度下降
                        activation='relu',  # 以relu函数作为激活函数
                        hidden_layer_sizes=(10, 5),  # 设置一个有5个节点的隐藏层。
                        random_state=1,
                        early_stopping=True,
                        learning_rate_init=0.1,  # 初始学习率，设为0.1
                        max_iter=2000, verbose=1)  # 最大迭代次数，设为10000
    # 实例化
    logging.info('start training')
    bpnn.fit(feature_train, target_train)

    # 验证
    pred = bpnn.predict(feature_test)
    mape = mean_absolute_percentage_error(pred, target_test)
    mae = mean_absolute_error(pred, target_test)
    logging.info('mape: {}'.format(mape))
    logging.info('mae: {}'.format(mae))

    joblib.dump(bpnn, save_path)
    logging.info('model has been saved in: {}'.format(save_path))
