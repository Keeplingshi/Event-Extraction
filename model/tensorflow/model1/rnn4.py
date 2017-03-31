# coding:utf-8
"""
Created on 2017年3月20日
事件识别  34分类
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

data_f = open('../enACEdata/data2/train_data34.data', 'rb')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = pickle.load(data_f)
data_f.close()

saver_path="../enACEdata/saver/checkpoint3.data"

# 参数
event_num=12524
learningRate = 0.03
training_iters = event_num*10
batch_size = 1

nInput = 300
nSteps = 1
nHidden = 100
nClasses = 34

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

seq_len = tf.placeholder(tf.int32, [None])

weights = tf.Variable(tf.random_normal([2 * nHidden, nClasses]))

biases = tf.Variable(tf.random_normal([nClasses]))


def gru_RNN(x, weights, biases,seq_len):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.reshape(x, [-1, nSteps, nInput])
    #x = tf.split(0, nSteps, x)

    # Define gru cells with tensorflow
    # Forward direction cell
    # gru_fw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
    # # Backward direction cell
    # gru_bw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
    lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(nHidden,forget_bias=1.0)
    lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(nHidden,forget_bias=1.0)

    # lstm_fw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
    # lstm_bw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

    lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * 2)
    lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * 2)

    # Get gru cell output
    #outputs, _, _ = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,sequence_length=seq_len, dtype=tf.float32)

    outputs = tf.concat(2, outputs)
    # As we want do classification, we only need the last output from LSTM.
    last_output = outputs[:, 0, :]
    results =tf.matmul(last_output, weights) + biases

    return results

pred = gru_RNN(x, weights, biases,seq_len)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)


def compute_accuracy():
    # saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, "../enACEdata/saver/checkpoint.data")
    # 载入测试集进行测试
    length = len(X_test)
    p_s = 0  # 识别的个体总数
    r_s = 0  # 测试集中存在个个体总数
    pr_acc = 0  # 正确识别的个数
    for i in range(length):
        test_len = len(X_test[i])
        test_data = np.array(X_test[i]).reshape((-1, nSteps, nInput))  # 8
        train_seq_len = np.ones(test_len) * nSteps
        test_label = Y_test[i]
        # prediction识别出的结果，y_测试集中的正确结果
        prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label,seq_len:train_seq_len})
        for t in range(len(y_)):
            if prediction[t] != 33:
                p_s +=1

            if y_[t] != 33:
                r_s +=1
                if y_[t] == prediction[t]:
                    pr_acc +=1

    print('----------------------------------------------------')
    print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
    p = pr_acc / p_s
    r = pr_acc / r_s
    if p+r!=0:
        f = 2 * p * r / (p + r)
        print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
        return f

    return 0


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, saver_path)
    # compute_accuracy()
    # sys.exit()

    k = 0
    max_f=0
    while k < training_iters:
        step = k % event_num

        batch_xs = X_train[step]
        batch_ys = Y_train[step]
        batch_size = len(batch_xs)
        batch_xs = np.array(batch_xs).reshape([batch_size, nSteps, nInput])

        train_seq_len = np.ones(batch_size) * nSteps

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,seq_len:train_seq_len})

        if k!=0 and step==0:

            f=compute_accuracy()
            if f>max_f:
                saver = tf.train.Saver(tf.global_variables())
                saver.save(sess,saver_path)
                max_f=f


        k += 1
    print('Optimization finished')

"""
----------------------------------------------------
294------434------497
P=0.6774193548387096	R=0.5915492957746479	F=0.631578947368421

264------356------497
P=0.7415730337078652	R=0.5311871227364185	F=0.6189917936694023

277------378------497
P=0.7328042328042328	R=0.5573440643863179	F=0.6331428571428571

----------------------------------------------------
284------384------497
P=0.7395833333333334	R=0.5714285714285714	F=0.6447219069239501

----------------------------------------------------
292------398------497
P=0.7336683417085427	R=0.5875251509054326	F=0.6525139664804469

----------------------------------------------------
296------403------497
P=0.7344913151364765	R=0.5955734406438632	F=0.6577777777777778
"""