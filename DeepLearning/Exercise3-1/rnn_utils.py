import numpy as np

def softmax(x):
    # 此处每个元素减去矩阵中的最大值，目的是防止，矩阵中元素过大(100000)或者过小(-10000)时，计算结果特别大造成数据上溢
    e_x = np.exp(x - np.max(x))
    # axis=0，则沿着纵轴进行操作，若axis=1则沿着横轴进行操作
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))