# coding:utf-8
"""
Created on 2017年3月20日
事件识别  二分类
rnn
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

data_f = open('./enACEdata/train_data2.data', 'rb')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = pickle.load(data_f)
data_f.close()


# 参数
learningRate = 0.001
training_iters = 29830
batch_size = 1

nInput = 200
nSteps = 1
nHidden = 100
nClasses = 2

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = tf.Variable(tf.random_normal([2 * nHidden, nClasses]))

biases = tf.Variable(tf.random_normal([nClasses]))


def gru_RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    # x = tf.reshape(x, [-1, nSteps, nInput])
    x = tf.split(0, nSteps, x)

    # Define gru cells with tensorflow
    # Forward direction cell
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
    # Backward direction cell
    gru_bw_cell = tf.nn.rnn_cell.GRUCell(nHidden)

    # Get gru cell output
    outputs, _, _ = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
    #outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, x,sequence_length=batch_size, dtype=tf.float32)
    results = tf.tanh(tf.matmul(outputs[-1], weights) + biases)

    return results

pred = gru_RNN(x, weights, biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    k = 0

    while k < training_iters:
        step = k % 2983

        batch_xs = X_train[step]
        batch_ys = Y_train[step]
        batch_size = len(batch_xs)
        batch_xs = batch_xs.reshape([batch_size, nSteps, nInput])

        # output = sess.run(test, feed_dict={x: batch_xs})
        # print(output)
        # print(len(output))
        # print(len(output[0]))
        # print(len(output[0][0]))
        # sys.exit()

        # train_seq_len = np.ones(batch_size) * batch_size

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})


        k += 1
    print('Optimization finished')

    # 载入测试集进行测试
    length = len(X_test)
    test_accuracy = 0.0
    p_s = 0  # 识别的个体总数
    r_s = 0  # 测试集中存在个个体总数
    pr_acc = 0  # 正确识别的个数
#     s = 0
#     acc = 0
    for i in range(length):
        test_len = len(X_test[i])
        test_data = X_test[i].reshape(
            (-1, nSteps, nInput))  # 8
        test_label = Y_test[i]
        # prediction识别出的结果，y_测试集中的正确结果
        prediction, y_ = sess.run(
            [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
        for t in range(len(y_)):
            if prediction[t] != 1:
                p_s = p_s + 1

            if y_[t] != 1:
                r_s = r_s + 1
                if y_[t] == prediction[t]:
                    pr_acc = pr_acc + 1

    print('----------------------------------------------------')
    print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
    p = pr_acc / p_s
    r = pr_acc / r_s
    f = 2 * p * r / (p + r)
    print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))

"""
178------326------289
P=0.5460122699386503	R=0.615916955017301	F=0.5788617886178862
"""
