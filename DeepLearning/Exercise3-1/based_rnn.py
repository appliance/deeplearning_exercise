"""
    注：
        1、l代表第l层
        2、a[4]代表第4层的激活值 W[5] b[5]代表第五层的参数
        3、x(i)代表第i个训练输入序列
        4、x<t>代表x的第t个时间步        x(i)<t> 代表第i个训练输入序列的第t个时间步
        5、深层Rnn中 ai[l]代表第l层的第i个输出激活值
"""
import numpy as np
from rnn_utils import *

# Based_Rnn 前向计算
# step1: 计算一个时间步的Rnn单元计算
# step2: 利用一个时间步的Rnn单元计算，迭代循环Tx次，完成Based_Rnn的前向计算

# ======================================================================================================================

# Graded Function:rnn_cell_forward
def rnn_cell_forward(xt, a_prev, parameters):
    # 获取计算参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba = parameters["ba"]
    Wya = parameters["Wya"]
    by = parameters["by"]

    # 计算激活值
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # 计算当前步时的预测值
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    # 保存 a_next a_prev xt parameters 到cache中 供反向传播时使用
    cache = (a_next, a_prev, xt, parameters)
    # a_next 供下一个时间步计算 yt_pred获取当前时步的预测情况
    return a_next, yt_pred, cache

# 测试
# np.random.seed(1)
# xt = np.random.randn(3, 10)     # 3代表3个元素组成一个向量表示当前序列元素  10代表10组数据
# a_prev = np.random.randn(5, 10) # 激活值矩阵列的值 = 输入序列的个数  （竖着排列）
# Waa = np.random.randn(5, 5)
# Wax = np.random.randn(5, 3)
# ba = np.random.randn(5, 1)
# Wya = np.random.randn(2, 5)    # 2代表最终预测结果由两个向量组成，共十组
# by = np.random.randn(2, 1)
# parameters = {"Waa": Waa, "Wax": Wax, "ba": ba, "Wya": Wya, "by": by}
# a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
# print("a_next[4]", a_next[4])
# print("a_next.shape",a_next.shape)
# print("yt_pred[1]", yt_pred[1])


"""
    疑惑：
        1、 如果序列组中的序列不等长，批量计算时，如何处理？
"""

# ======================================================================================================================

# Based Rnn 前向计算
"""
    构建矩阵向量a(初始化为0向量) 来保存所有隐含层的状态即激活值
    第一个激活值设置为a0
    循环计算时，t代表迭代下标
        存储所有的激活值到a矩阵向量中（保存在第t个位置）
        存储所有的预测值到y矩阵向量中
        将计算出的cache(xt, a_prev, a_next, parameters) 保存到caches中
    最终返回 a y_pred 以及 caches
"""
# Graded Function: rnn_forward
def rnn_forward(x, a0, parameters):
    caches = []
    # 获取计算的参数
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]
    ba = parameters["ba"]
    Wya = parameters["Wya"]
    by = parameters["by"]
    # 统计需要迭代的次数 序列长度
    n_x, m, T_x = x.shape       # 注 记住x的矩阵分布
    # 创建矩阵向量a，y
    a = np.zeros((Wya.shape[1], m, T_x))    # 注 np.zeros((m,n)) 中还要再添加括号
    y_pred = np.zeros((Wya.shape[0], m, T_x))
    # 开始迭代计算
    for t in range(T_x):
        # 取出序列组的第一个步时数据
        x_t = x[:, :, t]
        a_next, yt_pred, cache = rnn_cell_forward(x_t, a0, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)
    caches = (caches, x)
    return a, y_pred, caches

# 测试
# np.random.seed(1)
# x = np.random.randn(3, 10, 4)       # 10组序列 每个序列总步长为4 每个步长列向量长度都为3
# a0 = np.random.randn(5, 10)
# Waa = np.random.randn(5, 5)
# Wax = np.random.randn(5, 3)
# ba = np.random.randn(5, 1)
# Wya = np.random.randn(2, 5)
# by = np.random.randn(2, 1)
# parameters = {"Waa": Waa, "Wax": Wax, "ba": ba, "Wya": Wya, "by": by}
# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1]=", a[4][1])
# print("len(caches)=", len(caches))

# ======================================================================================================================
# based_rnn反向传播
def rnn_cell_backward(da_next, cache):
    # 从cache存储中获取求导所需的数值
    (a_next, a_prev, xt, parameters) = cache
    # 获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    # 利用导函数代入参数值求导函数值
    dtanh = (1-a_next**2)*da_next
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, keepdims=True, axis=-1)
    gradients = {"dxt":dxt, "da_prev":da_prev, "dWax":dWax, "dWaa":dWaa, "dba": dba}
    return gradients

# 测试
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Wax = np.random.randn(5,3)
# Waa = np.random.randn(5,5)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
# a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)
# da_next = np.random.randn(5,10)
# gradients = rnn_cell_backward(da_next, cache)
#
# print("gradients[dxt][1][2]=", gradients["dxt"][1][2])
# print("gradients[dba][4]=", gradients["dba"][4])

# ======================================================================================================================
# base_rnn 反向传播
def rnn_backward(da, caches):
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    for t in reversed(range(T_x)):
        # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        # Retrieve derivatives from gradients (≈ 1 line)
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
    da0 = da_prevt
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients

# 测试
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Wax = np.random.randn(5,3)
# Waa = np.random.randn(5,5)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
# a, y, caches = rnn_forward(x, a0, parameters)
# da = np.random.randn(5, 10, 4)
# gradients = rnn_backward(da, caches)
#
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)






