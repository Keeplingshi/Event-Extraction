# coding:utf-8

'''
Created on 2017年2月6日

@author: chenbin
'''

import pprint
import pickle
import tensorflow as tf
import numpy as np
import sys

import logging

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     filename='lstm_log.log',
#                     filemode='w')


# 数据读取，训练集和测试集
ace_data_train_file = open('./corpus_deal/ace_data/ace_data_train.pkl', 'rb')
ace_data_train = pickle.load(ace_data_train_file)

ace_data_train_labels_file = open(
    './corpus_deal/ace_data/ace_data_train_labels.pkl', 'rb')
ace_data_train_labels = pickle.load(ace_data_train_labels_file)

ace_data_test_file = open('./corpus_deal/ace_data/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)

ace_data_test_labels_file = open(
    './corpus_deal/ace_data/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

ace_data_train_file.close()
ace_data_train_labels_file.close()
ace_data_test_file.close()
ace_data_test_labels_file.close()


# RNN学习时使用的参数
learning_rate = 0.001  # 1
training_iters = 16000
batch_size = 1
display_step = 10

# 神经网络的参数
n_input = 200  # 输入层的n
n_steps = 1  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 34  # 输出的数量，因为是分类问题，这里一共有34个

# 构建tensorflow的输入X的placeholder
x = tf.placeholder("float", [None, n_steps, n_input])
# tensorflow里的LSTM需要两倍于n_hidden的长度的状态，一个state和一个cell
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2 * n_hidden])
# 输出Y
y = tf.placeholder("float", [None, n_classes])  # 2

# 随机初始化每一层的权值和偏置
weights = {
    # Hidden layer weights
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

'''
构建RNN
'''


def RNN(_X, _istate, _weights, _biases):
    # 规整输入的数据
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size    #3

    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # 输入层到隐含层，第一次是直接运算
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    # 之后使用LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        n_hidden, forget_bias=1.0, state_is_tuple=False)

    #gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden)

#     lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=False)
    # 28长度的sequence，所以是需要分解位28次
    _X = tf.split(0, n_steps, _X)  # n_steps * (batch_size, n_hidden)    #4
    # 开始跑RNN那部分
    outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=_istate)
    #outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=_istate)

    results = tf.matmul(outputs[-1], _weights['out']) + _biases['out']
    # 输出层
    return results


pred = RNN(x, istate, weights, biases)

# 定义损失和优化方法，其中算是为softmax交叉熵，优化方法为Adam
# Softmax loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 进行模型的评估，argmax是取出取值最大的那一个的标签作为输出
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()
# init = tf.initialize_all_variables()  # 6

# 开始运行
with tf.Session() as sess:
    sess.run(init)
    k = 0
    # 持续迭代
    while k < training_iters:

        step = k % 1600

        batch_xs = ace_data_train[step]
        batch_ys = ace_data_train_labels[step]
        batch_size = len(batch_xs)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
        # 迭代
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        # 在特定的迭代回合进行数据的输出
        if k % 100 == 0:
            print("Iter " + str(k))

        k += 1

    #logging.info("Optimization Finished!")
    # 载入测试集进行测试
    length = len(ace_data_test)
    test_accuracy = 0.0
    p_s = 0  # 识别的个体总数
    r_s = 0  # 测试集中存在个个体总数
    pr_acc = 0  # 正确识别的个数
#     s = 0
#     acc = 0
    for i in range(length):
        test_len = len(ace_data_test[i])
        test_data = ace_data_test[i].reshape(
            (-1, n_steps, n_input))  # 8
        test_label = ace_data_test_labels[i]
        # prediction识别出的结果，y_测试集中的正确结果
        prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label,
                                                                                    istate: np.zeros((test_len, 2 * n_hidden))})
        for t in range(len(y_)):
            if prediction[t] != 33:
                p_s = p_s + 1

            if y_[t] != 33:
                r_s = r_s + 1
                if y_[t] == prediction[t]:
                    pr_acc = pr_acc + 1

    print('----------------------------------------------------')
    print(str(pr_acc) + '------------' + str(r_s))
    p = pr_acc / p_s
    r = pr_acc / r_s
    f = 2 * p * r / (p + r)
    print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))


# 240------------579
# P=0.46332046332046334    R=0.41450777202072536    F=0.4375569735642662
