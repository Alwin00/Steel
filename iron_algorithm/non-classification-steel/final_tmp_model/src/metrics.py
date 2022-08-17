import numpy as np


def hitrate(y_test, y_predict, level1=8, level2=10, level3=15):
    '''
    输入：
    y_test,实际值
    y_predict,预测值
    '''
    num1 = 0
    num2 = 0
    num3 = 0
    total = 0
    for i in range(len(y_predict)):
        error = abs(y_predict[i] - y_test[i])
        total += 1
        if error <= level1:
            num1 += 1
        elif level1 < error <= level2:
            num2 += 1
        elif level2 < error <= level3:
            num3 += 1

    hit_rate1 = num1 / total
    hit_rate2 = (num1 + num2) / total
    hit_rate3 = (num1 + num2 + num3) / total
    return hit_rate1, hit_rate2, hit_rate3


def mean_absolute_percentage_error(target, pred):
    res = 0
    for i in range(len(target)):
        tmp = abs(target[i] - pred[i])
        res += tmp
    mape = res / np.sum(target)
    return mape


def mean_absolute_error(target, pred):
    res = 0
    for i in range(len(target)):
        res += np.abs(target[i] - pred[i])

    res /= len(target)
    return res
