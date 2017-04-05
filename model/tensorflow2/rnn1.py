# coding:utf-8
"""
Created on 2017年4月4日
事件识别  34分类
rnn 测试提交
@author: chenbin
"""

import tensorflow as tf
import numpy as np
import pickle
import nltk
import itertools
import json
import sys
import time

data_f = open('./data/1/train_data_form34.data', 'rb')
X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
data_f.close()

print(np.array(X_train).shape)
# print(np.array(Y_train).shape)
# print(np.array(W_train).shape)
# print(np.array(X_test).shape)
# print(np.array(Y_test).shape)
# print(np.array(W_test).shape)
# print(np.array(X_dev).shape)
# print(np.array(Y_dev).shape)
# print(np.array(W_dev).shape)
sys.exit()
event_num=len(X_train)

# RNN学习时使用的参数
learning_rate = 0.03
training_iters = 10000
batch_size = 128
display_step = 10

# 神经网络的参数
n_input = 300  # 输入层的n
n_steps = 30  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 34  # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

x = tf.placeholder("float", [None, n_steps, n_input])
# istate = tf.placeholder("float", [None, 2 * n_hidden])
y = tf.placeholder("float", [None, n_classes])

# 随机初始化每一层的权值和偏置
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def lstm_pred(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)

    outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return outputs,x

pred = lstm_pred(x, weights, biases)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# 开始运行
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # 持续迭代
    while step < training_iters:

        start=step*batch_size%event_num
        end=(step+1)*batch_size

        if start<event_num and end<=event_num:
            pass
        else:
            start%=event_num
            end%=event_num

        if start>=end:
            end=event_num-1

        # 随机抽出这一次迭代训练时用的数据
        batch_xs=X_train[start:end]
        batch_ys=Y_train[start:end]

        batch_xs = np.array(batch_xs).reshape([batch_size, n_steps, n_input])

        output,x=sess.run(pred, feed_dict={x: batch_xs})
        print(output[0])
        print(output[1])
        print(np.array(output).shape)

        print(np.array(x).shape)
        sys.exit()

        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))

        # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        #
        # if step % display_step == 0:
        #     pass
        step += 1
    print("Optimization Finished!")


# # 参数
# event_num=12524
# learningRate = 0.001
# training_iters = event_num*100
# batch_size = 1
#
# nInput = 300
# nSteps = 1
# nHidden = 100
# nClasses = 2
#
# x = tf.placeholder('float', [None, nSteps, nInput])
# y = tf.placeholder('float', [None, nClasses])
#
# weights = tf.Variable(tf.random_normal([2 * nHidden, nClasses]))
#
# biases = tf.Variable(tf.random_normal([nClasses]))
#
#
# def gru_RNN(x, weights, biases):
#     x = tf.transpose(x, [1, 0, 2])
#     x = tf.reshape(x, [-1, nInput])
#     # x = tf.reshape(x, [-1, nSteps, nInput])
#     x = tf.split(0, nSteps, x)
#
#     # Define gru cells with tensorflow
#     # Forward direction cell
#     gru_fw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
#     # Backward direction cell
#     gru_bw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
#
#     # Get gru cell output
#     outputs, _, _ = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
#     #outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, x,sequence_length=batch_size, dtype=tf.float32)
#     results = tf.tanh(tf.matmul(outputs[-1], weights) + biases)
#
#     return results
#
# pred = gru_RNN(x, weights, biases)
#
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
#
# # correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# # accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#
#     k = 0
#
#     while k < training_iters:
#         step = k % event_num
#
#         batch_xs = X_train[step]
#         batch_ys = Y_train[step]
#         batch_size = len(batch_xs)
#
#         batch_xs = np.array(batch_xs).reshape([batch_size, nSteps, nInput])
#
#         sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
#
#         if k%event_num==0:
#             # 载入测试集进行测试
#             length = len(X_test)
#             test_accuracy = 0.0
#             p_s = 0  # 识别的个体总数
#             r_s = 0  # 测试集中存在个个体总数
#             pr_acc = 0  # 正确识别的个数
#             for i in range(length):
#                 test_len = len(X_test[i])
#                 test_data = np.array(X_test[i]).reshape(
#                     (-1, nSteps, nInput))  # 8
#                 test_label = Y_test[i]
#                 # prediction识别出的结果，y_测试集中的正确结果
#                 prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
#                 for t in range(len(y_)):
#                     if prediction[t] != 1:
#                         p_s +=1
#
#                     if y_[t] != 1:
#                         r_s +=1
#                         if y_[t] == prediction[t]:
#                             pr_acc +=1
#
#             print('----------------------------------------------------')
#             print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
#             p = pr_acc / p_s
#             r = pr_acc / r_s
#             f = 2 * p * r / (p + r)
#             print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
#             print('\n')
#
#
#
#         k += 1
#     print('Optimization finished')
#
#     # 载入测试集进行测试
#     length = len(X_test)
#     test_accuracy = 0.0
#     p_s = 0  # 识别的个体总数
#     r_s = 0  # 测试集中存在个个体总数
#     pr_acc = 0  # 正确识别的个数
# #     s = 0
# #     acc = 0
#     for i in range(length):
#         test_len = len(X_test[i])
#         test_data = np.array(X_test[i]).reshape(
#             (-1, nSteps, nInput))  # 8
#         test_label = Y_test[i]
#         # prediction识别出的结果，y_测试集中的正确结果
#         prediction, y_ = sess.run(
#             [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
#         for t in range(len(y_)):
#             if prediction[t] != 1:
#                 p_s+=1
#
#             if y_[t] != 1:
#                 r_s +=1
#                 if y_[t] == prediction[t]:
#                     pr_acc +=1
#
#     print('----------------------------------------------------')
#     print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
#     p = pr_acc / p_s
#     r = pr_acc / r_s
#     f = 2 * p * r / (p + r)
#     print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
#     print('\n')
#
# """
# 178------326------289
# P=0.5460122699386503	R=0.615916955017301	F=0.5788617886178862
#
# 218------374------337
# P=0.5828877005347594	R=0.6468842729970327	F=0.6132208157524613
# """
