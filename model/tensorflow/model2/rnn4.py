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

data_f = open('../enACEdata/data2/train_data34.data', 'rb')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = pickle.load(data_f)
data_f.close()


# 参数
event_num=12524
learningRate = 0.03
training_iters = event_num*1
batch_size = 1

nInput = 300
nSteps = 1
nHidden = 128
nClasses = 34

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

seq_len = tf.placeholder(tf.int32, [None],name="seq_len")

weights = tf.Variable(tf.random_normal([2 * nHidden, nClasses]),name="W1")

biases = tf.Variable(tf.random_normal([nClasses]),name="b1")

log_f = open('../enACEdata/rnn4.log', 'a', encoding='utf8')
log_f.write('\n')
log_f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\t参数记录与实验结果\n")


def gru_RNN(x, weights, biases,seq_len):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.reshape(x, [-1, nSteps, nInput])
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)

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
    results = tf.matmul(last_output, weights) + biases

    return results

pred = gru_RNN(x, weights, biases,seq_len)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    k = 0

    while k < training_iters:
        step = k % event_num

        batch_xs = X_train[step]
        batch_ys = Y_train[step]
        batch_size = len(batch_xs)
        batch_xs = np.array(batch_xs).reshape([batch_size, nSteps, nInput])

        train_seq_len = np.ones(batch_size) * nSteps

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,seq_len:train_seq_len})
        # saver=tf.train.Saver(tf.all_variables())
        # saver.save(sess,"../enACEdata/rnn4_checkpoint.data")

        if k!=0 and step==0:
            # 载入测试集进行测试
            length = len(X_test)
            test_accuracy = 0.0
            p_s = 0  # 识别的个体总数
            r_s = 0  # 测试集中存在个个体总数
            pr_acc = 0  # 正确识别的个数
            for i in range(length):
                test_len = len(X_test[i])
                test_data = np.array(X_test[i]).reshape((-1, nSteps, nInput))  # 8
                train_seq_len = np.ones(test_len) * nSteps
                test_label = Y_test[i]
                # prediction识别出的结果，y_测试集中的正确结果
                prediction, y_ = sess.run(
                    [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label,seq_len:train_seq_len})
                for t in range(len(y_)):
                    if prediction[t] != 33:
                        p_s +=1

                    if y_[t] != 33:
                        r_s +=1
                        if y_[t] == prediction[t]:
                            pr_acc +=1

            print('----------------------------------------------------')
            print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
            log_f.write('---------------------------------------------------------------------------\n')
            log_f.write('实验结果：' + str(pr_acc) + '\t' + str(p_s) + '\t' + str(r_s) + '\t' + "\n")
            p = pr_acc / p_s
            r = pr_acc / r_s
            if p+r!=0:
                f = 2 * p * r / (p + r)
                log_f.write('P,R,F：' + str(p) + '\t' + str(r) + '\t' + str(f) + "\n")
                print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))

        k += 1
    print('Optimization finished')
    log_f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\tOptimization finished\n")

    # # 载入测试集进行测试
    # length = len(X_test)
    # test_accuracy = 0.0
    # p_s = 0  # 识别的个体总数
    # r_s = 0  # 测试集中存在个个体总数
    # pr_acc = 0  # 正确识别的个数
    # for i in range(length):
    #     test_len = len(X_test[i])
    #     test_data = np.array(X_test[i]).reshape((-1, nSteps, nInput))  # 8
    #     train_seq_len = np.ones(test_len) * nSteps
    #     test_label = Y_test[i]
    #     # prediction识别出的结果，y_测试集中的正确结果
    #     prediction, y_ = sess.run(
    #         [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label,seq_len:train_seq_len})
    #     for t in range(len(y_)):
    #         if prediction[t] != 33:
    #             p_s +=1
    #
    #         if y_[t] != 33:
    #             r_s +=1
    #             if y_[t] == prediction[t]:
    #                 pr_acc +=1
    #
    # print('----------------------------------------------------')
    # print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
    # p = pr_acc / p_s
    # r = pr_acc / r_s
    # if p+r!=0:
    #     f = 2 * p * r / (p + r)
    #     print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))

"""
232------329------414
P=0.7051671732522796	R=0.5603864734299517	F=0.6244952893674294

"""