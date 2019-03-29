import numpy as np
from rnn_utils import *
'''
    遗忘门 更新门 待选择的记忆细胞
    gf = sigmoid(np.dot(Wf,[a_prev, xt]) + bf)
    gu = sigmoid(np.dot(Wu,[a_prev, xt]) + bu)
    cct = tanh(np.dot(Wc,[a_prev, xt]) + bc)
    计算传递给下一个时间步的记忆细胞
    遗忘门-c_prev  当然是遗忘之前的
    最重要：c_next = np.dot(gf, c_prev) + np.dot(gu, cct)

    输出a_next需要输出门
    go = sigmoid(np.dot(Wo,[a_prev, xt]) + bo)
    a_next = np.dot(go, c_next)

    计算y_pred
    y_pred = softmoax(np.dot(Wy, a_next) + by)

'''

# ======================================================================================================================
"""
    构建lstm前向单元运算： 
        1、合并a_prev和xt为一个向量矩阵 上下排布
"""
# Graded Function:lstm_cell_forward
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    # 获取计算t时步的记忆细胞的参数
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]

    # 将a_prev和xt矩阵合并进行遗忘门的计算
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    # a_prev.shape = (n_a, m)  xt.shape = (n_x, m)
    # 所以合并后concat.shape = (n_a+n_x, m)
    concat = np.zeros((n_a+n_x, m))
    # python 的语法结构
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt
    # 计算遗忘门
    ft = sigmoid(np.dot(Wf, concat) + bf)
    # 计算更新门
    it = sigmoid(np.dot(Wi, concat) + bi)
    # 计算候选记忆细胞
    cct = np.tanh(np.dot(Wc, concat) + bc)
    # 计算t时步的记忆细胞
    # 注意 更新记忆细胞 矩阵点乘 不是 交叉相乘
    c_next = ft*c_prev + it*cct
    # 计算输出门和t时步的激活值
    # 注意 计算激活值 矩阵点乘 不是 交叉相乘
    ot = sigmoid(np.dot(Wo, concat) + bo)
    # 这个计算t时步的激活值 这个计算方式有点奇怪
    a_next = ot * np.tanh(c_next)
    # 计算预测值y
    yt_pred = np.tanh(np.dot(Wy, a_next) + by)

    # 老规矩 存储 当前时步的 a_prev a_next c_prev c_next 还有门 ft it ot 候选记忆细胞cct 输入xt 及参数 parameters
    # 只要是计算出来的值 都必须存储 反向传播的时候计算需要
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache

# 测试
# np.random.seed(1)
# xt = np.random.randn(3, 10)
# a_prev = np.random.randn(5, 10)       # 暂且理解为 a 和 c 矩阵向量大小相等
# c_prev = np.random.randn(5, 10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5, 1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5, 1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5, 1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5, 1)
# Wy = np.random.randn(2, 5)
# by = np.random.randn(2, 1)
# parameters = {"Wf": Wf, "bf": bf, "Wi": Wi, "bi": bi, "Wo": Wo, "bo": bo, "Wc": Wc, "bc": bc, "Wy": Wy, "by": by}
# a_next, c_next, yt_pred, cache = lstm_cell_forward(xt, a_prev, c_prev ,parameters)
# print("a_next[4]=", a_next[4])

# ======================================================================================================================
'''
    lstm 整体前向计算
'''
# Graded Funcion;lstm_forward
def lstm_forward(x, a0, parameters):
    # 最后需要保存 c a y caches
    Wy = parameters["Wy"]
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    a = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    c = np.zeros((n_a, m, T_x))
    caches = []
    a_next = a0
    # 第一个记忆细胞c 用0向量进行初始化
    c_next = np.zeros((n_a, m))
    # 循环计算lstm单元
    for t in range(T_x):
        # 获取第一个步时的序列
        x_t = x[:, :, t]
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x_t, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt_pred
        caches.append(cache)

    # 汇总
    caches = (caches, x)
    return a, y, c, caches

# 测试
# 初始化位置定义与参考案例不同 最终结果也是有差别的 但是函数是没问题的
# np.random.seed(1)
# x = np.random.randn(3, 10, 7)
# a0 = np.random.randn(5, 10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5, 1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5, 1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.rand(5, 1)
# Wc = np.random.rand(5, 5+3)
# bc = np.random.randn(5, 1)
# Wy = np.random.randn(2, 5)
# by = np.random.randn(2, 1)
#
# parameters = {"Wf": Wf, "bf": bf, "Wi": Wi, "bi": bi, "Wo": Wo, "bo": bo, "Wc": Wc, "bc": bc, "Wy": Wy, "by": by}
# a, y, c, caches = lstm_forward(x, a0, parameters)
# print("a[4][3][6]=", a[4][3][6])
# print(a.shape)


# ======================================================================================================================
# lstm 单元反向传播
def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    n_x, m = xt.shape
    n_a, m = a_next.shape
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    ## ？？？？公式有问题？
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)
    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis=0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis=0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis=0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis=0).T)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(
        parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(
        parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

# 测试
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
#
# da_next = np.random.randn(5,10)
# dc_next = np.random.randn(5,10)
# gradients = lstm_cell_backward(da_next, dc_next, cache)
# print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])

# ======================================================================================================================
# lstm反向传播
def lstm_backward(da, caches):
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
        # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


# 测试
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y, c, caches = lstm_forward(x, a0, parameters)
#
# da = np.random.randn(5, 10, 4)
# gradients = lstm_backward(da, caches)
#
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)









