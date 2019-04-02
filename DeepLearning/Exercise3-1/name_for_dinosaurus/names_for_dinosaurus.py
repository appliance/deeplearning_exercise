'''
    搭建字符级的rnn网络
    从姓名数据集合中学习隐含序列顺序，在自动生成新的姓名

    掌握技能：
        1、使用RNN从text数据集中获取数据进行处理
        2、如何综合数据，通过每一步时的抽样预测，并传递给下一个单元进行预测（模型训练后用于预测时）
        3、利用Rnn穿件一个字符级的文本生成
        4、理解为什么梯度裁剪比较重要

    疑惑：
        1、自动生成姓名时 姓名的字符长度是如何限定的？
        2、输入的第一个字符是如何选择的？
'''
import numpy as np
from rnn_utils import *
# from based_rnn import *
import random

'''
    本来打算直接利用 based_rnn里面的已经写好的代码直接来实现此demo
    结果发现 based_rnn练习中的函数是阉割版的 前向计算里没有包含cost函数计算
    因此 在此项目中重新写过函数 同时加深理解
'''
# ======================================================================================================================
# 参数初始化
# 作者在练习中设置n_a大小为100
# 疑惑为什么要乘以0.01？？？
def initialize_parameters(n_a, n_x, n_y):
    Waa = np.random.randn(n_a, n_a)*0.01
    Wax = np.random.randn(n_a, n_x)*0.01
    ba = np.zeros((n_a, 1))
    Wya = np.random.randn(n_y, n_a)*0.01
    by = np.zeros((n_x, 1))
    parameters = {"Wax": Wax, "Waa": Waa, "ba": ba, "Wya": Wya, "by": by}
    return parameters


# 前向传播计算
def rnn_step_forward(parameters, a_prev, x_t):
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]
    ba = parameters["ba"]
    Wya = parameters["Wya"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x_t) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    return a_next, yt_pred

# y_t = x_t+1 (x_t 预测的其实就是x_t+1)
# Y = [index1, index2, index3...] 字符对应的下标值
# X = [index1, index2, index3...]
# X, Y 传入的都是姓名字符串转的下标
def rnn_forward(X, Y, a0, parameters, vocab_size = 27):
    # 如果X 代表一个训练样本
    # 定义损失函数值
    loss = 0
    X_matrix = {}
    a = {}
    a[-1] = np.copy(a0)
    y_pred = {}
    # 对X文本序列进行向量化处理 并进行前向计算和损失值计算
    a_next = a0
    for t in range(len(X)):
        x_t = np.zeros((vocab_size, 1))
        # 如果有一个字符在字典中不存在 X[t]就有可能为空
        if(X[t] != None):
            x_t[X[t]] = 1
        X_matrix[t] = x_t
        # 进行当前t步时rnn_step_forward计算
        a_next, yt_pred = rnn_step_forward(parameters, a_next, x_t)
        a[t] = a_next
        y_pred[t] = yt_pred
        loss -= np.log(yt_pred[Y[t], :])
    cache = (y_pred, a, X_matrix)
    return loss, cache


# 反向传播计算
def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients["dWya"] += np.dot(dy, a.T)
    gradients["dby"] += dy
    da = np.dot(parameters["Wya"].T, dy) + gradients["da_next"]
    daraw = (1 - a * a)*da
    gradients["dba"] += daraw
    gradients["dWax"] += np.dot(daraw, x.T)
    gradients["dWaa"] += np.dot(daraw, a_prev.T)
    gradients["da_next"] = np.dot(parameters["Waa"].T, daraw)
    return gradients


def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    (y_pred, a, X_matrix) = cache
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    # 对梯度值矩阵进行初始化
    gradients["dWax"] = np.zeros_like(Wax)
    gradients["dWaa"] = np.zeros_like(Waa)
    gradients["dWya"] = np.zeros_like(Wya)
    gradients["dba"] = np.zeros_like(ba)
    gradients["dby"] = np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    for t in reversed(range(len(X))):
        dy = np.copy(y_pred[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, X_matrix[t], a[t], a[t-1])
    return gradients, a


# 参数更新
def update_parameters(parameters, gradients, lr):
    parameters["Waa"] += -lr * gradients["dWaa"]
    parameters["Wax"] += -lr * gradients["dWax"]
    parameters["ba"] += -lr * gradients["dba"]
    parameters["Wya"] += -lr * gradients["dWya"]
    parameters["by"] += -lr * gradients["dby"]
    return parameters


'''
    梯度裁剪函数
        在进行参数跟新之前，为了防止梯度爆炸，需要进行梯度裁剪处理
        此处利用梯度剪切的方法实现梯度裁剪
            每一个梯度值保持在[-N, N]区域中
'''
# 只试用Based_Rnn
def clip(gradients, maxValue):
    dWaa = gradients["dWaa"]
    dWax = gradients["dWax"]
    dWya = gradients["dWya"]
    dba = gradients["dba"]
    dby = gradients["dby"]
    for gradient in [dWaa, dWax, dWya, dba, dby]:
        # numpy 自带的梯度裁剪函数
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
    return gradients

# 测试
# np.random.seed(3)
# dWax = np.random.randn(5, 3)*10
# dWaa = np.random.randn(5, 5)*10
# dWya = np.random.randn(2, 5)*10
# dba = np.random.randn(5, 1)*10
# dby = np.random.randn(2, 1)*10
# gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
# gradients = clip(gradients, 10)
# print(gradients["dWaa"][1][2])


# ======================================================================================================================
'''
    训练数据集合预处理
        1、构建一个字符字典
        2、统计一下 恐龙姓名个数(一个恐龙姓名算一个训练样本)
        3、统计一下字符字典中的字符个数
'''
charset = []
dataset = []
with open("./dataset/dinos.txt") as f:
    for item in f.readlines():
        item = item.lower()
        dataset.append(item)
        # extend内容追加
        charset.extend(item)
# 统计dataset中不重复字符的个数
# 学习： test = ['a', 'f', 'd', 'b'] 集合 set(test) = {'a', 'f', 'd', 'b'} 需要list(set(test))转化为list集合后才方便操作 [] {}
charset = list(set(charset))
data_size, vocab_size = len(dataset), len(charset)
print("训练集和一共有恐龙姓名 %d 个，字符字典一共有 %d 个字符" % (data_size, vocab_size))
# 字符字典有26个小写英文字母+'/n'组成，其中'/n'扮演着'eos'的角色（句子结束）

# ======================================================================================================================

# 实现字典 字符到下标的映射 以及下标到字符的反映射
'''
    python 语法学习
        1、排序sorted
            sorted(charset) 对其进行排序
            print(charset)
            print(sorted(charset))
        2、枚举enumerate
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']  list(enumerate(seasons))
            输出[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        3、字典中的for循环
'''
char_to_ix = {ch: i for i, ch in enumerate(sorted(charset))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(charset))}

# ======================================================================================================================

'''
    Based_Rnn 模型的构建
        初始化参数
        循环
            前向计算统计算出损失函数
            反向计算计算出损失函数各个参数的梯度值
            梯度裁剪避免梯度爆炸
            跟新参数值
        返学最终学习到的参数
'''
# ======================================================================================================================
# 循环中一个整体的操作从前向计算到参数更新
def optimize(X, Y, a0, parameters, vocab_size, learning_rate = 0.01):
    loss, cache = rnn_forward(X, Y, a0, parameters, vocab_size)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    # 返回最后一个激活值 可以作为下一次循环的a0
    return loss, gradients, a[len(X)-1]


# 测试
# np.random.seed(1)
# vocab_size, n_a = 27, 100
# a_prev = np.random.randn(n_a, 1)
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# ba, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
# X = [12,3,5,11,22,3]
# Y = [4, 14,11,22,25,26]
# loss, gradients, a_last = optimize(X, Y, a_prev, parameters,27, learning_rate=0.01)
# print(loss)

# ======================================================================================================================
'''
    测试取样
        第一个x向量 设为0向量输入
        前向计算获取a1 和 y1_pred
        通过y1_pred 去构造输入的x2(np.random.choice())
        直到获取\n 就停止预测
'''
def sample(parameters, char_to_ix, seed):
    # 获取参数
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]
    ba = parameters["ba"]
    Wya = parameters["Wya"]
    by = parameters["by"]
    # 做一次前向计算
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_charcter = char_to_ix['\n']
    while(idx != newline_charcter and counter !=50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        np.random.seed(counter + seed)
        idx = np.random.choice(range(len(y)), p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        seed += 1
        counter += 1
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    return indices





    # ======================================================================================================================
# 模型训练
def model(dataset, ix_to_char, char_to_ix, num_iterations=35000, n_a=100, dino_names=7, vocab_size=27):
    # 参数初始化
    n_x, n_y = vocab_size, vocab_size
    loss = get_initial_loss(vocab_size, dino_names)
    paremeters = initialize_parameters(n_a, n_x, n_y)
    # 打乱训练样本
    np.random.shuffle(dataset)
    a0 = np.zeros((n_a, 1))
    for j in range(num_iterations):
        index = j % len(dataset)
        # 构造X，Y的下标序列
        X = [None] + [char_to_ix[ch] for ch in dataset[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        # 一次循环
        curr_loss, gradients, a_prev = optimize(X, Y, a0, paremeters, vocab_size, learning_rate=0.01)
        loss = smooth(loss, curr_loss)
        if j % 2000 == 0:
            print("Iteration:%d,Loss:%f" % (j, loss) + '\n')
            seed = 0
            for name in range(dino_names):
                sample_indices = sample(paremeters, char_to_ix, seed)
                print_sample(sample_indices, ix_to_char)
                seed += 1
            print('\n')
    return paremeters


parameters = model(dataset, ix_to_char, char_to_ix)

