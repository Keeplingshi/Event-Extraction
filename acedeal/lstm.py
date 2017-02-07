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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='lstm_log.log',
                    filemode='w')


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
training_iters = 10000000
batch_size = 1
display_step = 10

# 神经网络的参数
n_input = 200  # 输入层的n
n_steps = 1  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 34  # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

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
            sk = 0
            acck = 0
            predictionk, y_k = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: batch_xs, y: batch_ys,
                                                                                          istate: np.zeros((batch_size, 2 * n_hidden))})
            for t in range(len(y_k)):
                if y_k[t] != 33:
                    logging.info(
                        'actual:' + str(y_k[t]) + '\t predict:' + str(predictionk[t]))
                    #print(str(y_[t]) + '\t' + str(prediction[t]))
                    sk = sk + 1
                    if y_k[t] == predictionk[t]:
                        acck = acck + 1

            if sk != 0:
                logging.info(
                    "Iter " + str(k) + '-----------acc=' + str(acck / sk))
            #print("Iter " + str(k))
            #             # Calculate batch accuracy
            #             acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
            #                                                 istate: np.zeros((batch_size, 2 * n_hidden))})
            #             # Calculate batch loss
            #             loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
            #                                              istate: np.zeros((batch_size, 2 * n_hidden))})
            #             print(batch_ys)
            #             for i in range(len(batch_ys)):
            #                 ys = batch_ys[i]
            #                 print(ys.index(1.0))

#             prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: batch_xs, y: batch_ys,
#                                                                                         istate: np.zeros((batch_size, 2 * n_hidden))})
#             print(prediction)
#             print(y_)


#             print("Iter " + str(k) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
#                   ", Training Accuracy= " + "{:.5f}".format(acc))
        k += 1

    logging.info("Optimization Finished!")
    # 载入测试集进行测试
    length = len(ace_data_test)
    test_accuracy = 0.0
    s = 0
    acc = 0
    for i in range(length):
        test_len = len(ace_data_test[i])
        test_data = ace_data_test[i].reshape(
            (-1, n_steps, n_input))  # 8
        test_label = ace_data_test_labels[i]
        prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label,
                                                                                    istate: np.zeros((test_len, 2 * n_hidden))})
        for t in range(len(y_)):
            if y_[t] != 33:
                logging.info(
                    'actual:' + str(y_[t]) + '\t predict:' + str(prediction[t]))
                #print(str(y_[t]) + '\t' + str(prediction[t]))
                s = s + 1
                if y_[t] == prediction[t]:
                    acc = acc + 1

    logging.info('----------------------------------------------------')
    logging.info(str(acc) + '------------' + str(s))
    logging.info(acc / s)
    print('----------------------------------------------------')
    print(str(acc) + '------------' + str(s))
    print(acc / s)
#         print(prediction)
#         print(y_)
#         print('-------------------------------------------------------------')
    # 两种计算精确度的方式
#         print(accuracy.eval(
#             {x: test_data, y: test_label, istate: np.zeros((test_len, 2 * n_hidden))}))

#         print(sess.run(accuracy, feed_dict={
# x: test_data, y: test_label, istate: np.zeros((test_len, 2*n_hidden))}))
#
#         test_accuracy += sess.run(accuracy, feed_dict={
#             x: test_data, y: test_label, istate: np.zeros((test_len, 2 * n_hidden))})
#         print("Testing Accuracy:", sess.run(accuracy, feed_dict={
#             x: test_data, y: test_label, istate: np.zeros((test_len, 2 *
#                                                            n_hidden))}))
    #print(test_accuracy / length)

# # print(ace_data_train[0])
# # print(ace_data_train_labels)
#
# # sys.exit(0)
#
# # set random seed for comparing the two result calculations
# tf.set_random_seed(1)
#
# # hyperparameters
# lr = 0.001
# training_iters = len(ace_data_train)
# batch_size = 1
#
# n_inputs = 200   # MNIST data input (img shape: 28*28)
# n_steps = 1    # time steps
# n_hidden_units = 128   # neurons in hidden layer
# n_classes = 34      # MNIST classes (0-9 digits)
#
# # tf Graph input
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# istate = tf.placeholder("float", [None, 2 * n_hidden_units])
# y = tf.placeholder(tf.float32, [None, n_classes])
#
# # Define weights
# weights = {
#     # (28, 128)
#     'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
#     # (128, 10)
#     'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
# }
# biases = {
#     # (128, )
#     'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
#     # (10, )
#     'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
# }
#
#
# def RNN(X, istate, weights, biases):
#     # hidden layer for input to cell
#     ########################################
#
#     # transpose the inputs shape from
#     # X ==> (128 batch * 28 steps, 28 inputs)
#     X = tf.reshape(X, [-1, n_inputs])
#
#     # into hidden
#     # X_in = (128 batch * 28 steps, 128 hidden)
#     X_in = tf.matmul(X, weights['in']) + biases['in']
#     # X_in ==> (128 batch, 28 steps, 128 hidden)
#     X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
#
#     # cell
#     ##########################################
#
#     # basic LSTM Cell.
#     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
#         n_hidden_units, forget_bias=1.0, state_is_tuple=False)
#     # lstm cell is divided into two parts (c_state, h_state)
#     #init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
#
#     # You have 2 options for following step.
#     # 1: tf.nn.rnn(cell, inputs);
#     # 2: tf.nn.dynamic_rnn(cell, inputs).
#     # If use option 1, you have to modified the shape of X_in, go and check out this:
#     # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
#     # In here, we go for option 2.
#     # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
#     # Make sure the time_major is changed accordingly.
#     outputs, final_state = tf.nn.dynamic_rnn(
#         lstm_cell, X_in, initial_state=istate, time_major=False)
#
#     # hidden layer for output as the final results
#     #############################################
#     results = tf.matmul(final_state[1], weights['out']) + biases['out']
#
#     # # or
#     # unpack to list [(batch, outputs)..] * steps
#     # states is the last outputs
# #     outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
# #     results = tf.matmul(outputs[-1], weights['out']) + biases['out']
#
#     return results
#
#
# pred = RNN(x, istate, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)
#
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# with tf.Session() as sess:
#     # tf.initialize_all_variables() no long valid from
#     # 2017-03-02 if using tensorflow >= 0.12
#     #     if int((tf.__version__).split('.')[1]) < 12:
#     #         init = tf.initialize_all_variables()
#     #     else:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     step = 0
#     while step < training_iters:
#
#         batch_xs = ace_data_train[step]
#         batch_ys = ace_data_train_labels[step]
#         batch_size = len(batch_xs)
#         print(batch_size)
#         batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#         #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         #         batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#         sess.run([train_op], feed_dict={
#             x: batch_xs,
#             y: batch_ys,
#             istate: np.zeros((batch_size, 2 * n_hidden_units))
#         })
#         if step % 20 == 0:
#             print(sess.run(accuracy, feed_dict={
#                 x: batch_xs,
#                 y: batch_ys,
#                 istate: np.zeros((batch_size, 2 * n_hidden_units))
#             }))
#         step += 1
#
#     print("Optimization Finished!")
#     # 载入测试集进行测试
# #     test_len = 128
# #     test_data = mnist.test.images[:test_len].reshape(
# #         (-1, n_steps, n_inputs))  # 8
# #     test_label = mnist.test.labels[:test_len]
# #     print("Testing Accuracy:", sess.run(accuracy, feed_dict={
# #           x: test_data, y: test_label}))
