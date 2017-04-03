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

data_f = open('../enACEdata/data3/train_data34.data', 'rb')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = pickle.load(data_f)
data_f.close()

saver_path="../enACEdata/saver/checkpointrnn56.data"

# 参数
event_num=len(X_train)
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

weights = tf.Variable(tf.random_normal([2 * nHidden+nInput-4, nClasses]))
biases = tf.Variable(tf.random_normal([nClasses]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积与最大池化
def con_max_pool_3x3(x):
    #cnn 卷积和池化
    width=tf.shape(x)[0]
    height=tf.shape(x)[2]
    x=tf.reshape(x,[1,width, height,1])

    W_conv1 = weight_variable([3,3,1,1])
    b_conv1 = bias_variable([1])
    #Convolution  Stride & Padding
    con2d_result = tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], 'VALID')
    con2d_result=tf.reshape(con2d_result,[1,width-2,height-2,1])

    h_conv1=tf.nn.relu(con2d_result+b_conv1)
    max_pool_result = tf.nn.max_pool(h_conv1, [1,500,1,1], [1,1,1,1], 'SAME')

    max_pool_result=tf.reshape(max_pool_result,[width-2,height-2])
    max_pool_one=[max_pool_result[0]]

    contact_pool_result=tf.concat(0,[max_pool_result,max_pool_one,max_pool_one])

    return contact_pool_result

def con_max_pool_2x2(x):
    #cnn 卷积和池化
    width=tf.shape(x)[0]
    height=tf.shape(x)[2]
    x=tf.reshape(x,[1,width, height,1])

    W_conv1 = weight_variable([2,2,1,1])
    b_conv1 = bias_variable([1])
    #Convolution  Stride & Padding
    con2d_result = tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], 'VALID')
    con2d_result=tf.reshape(con2d_result,[1,width-1,height-1,1])

    h_conv1=tf.nn.relu(con2d_result+b_conv1)
    max_pool_result = tf.nn.max_pool(h_conv1, [1,500,1,1], [1,1,1,1], 'SAME')

    max_pool_result=tf.reshape(max_pool_result,[width-1,height-1])
    max_pool_one=[max_pool_result[0]]

    contact_pool_result=tf.concat(0,[max_pool_result,max_pool_one])
    return contact_pool_result


def con_max_pool_5x5(x):
    #cnn 卷积和池化
    width=tf.shape(x)[0]
    height=tf.shape(x)[2]
    x=tf.reshape(x,[1,width, height,1])

    W_conv1 = weight_variable([5,5,1,1])
    b_conv1 = bias_variable([1])
    #Convolution  Stride & Padding
    con2d_result = tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], 'VALID')
    con2d_result=tf.reshape(con2d_result,[1,width-4,height-4,1])

    h_conv1=tf.nn.relu(con2d_result+b_conv1)
    max_pool_result = tf.nn.max_pool(h_conv1, [1,500,1,1], [1,1,1,1], 'SAME')

    max_pool_result=tf.reshape(max_pool_result,[width-4,height-4])
    max_pool_one=[max_pool_result[0]]

    contact_pool_result=tf.concat(0,[max_pool_result,max_pool_one,max_pool_one,max_pool_one,max_pool_one])
    return contact_pool_result

def gru_RNN(x, weights, biases,seq_len):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.reshape(x, [-1, nSteps, nInput])
    #x = tf.split(0, nSteps, x)

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
    lstm_output = outputs[:, 0, :]

    # contact_pool_3x3_result=con_max_pool_3x3(x)
    # contact_pool_2x2_result=con_max_pool_2x2(x)
    contact_pool_5x5_result=con_max_pool_5x5(x)

    # lstm_output=tf.split(1,2,lstm_output)
    # x=tf.reshape(x, [-1, nInput])
    lstm_cnn_output=tf.concat(1,[lstm_output,contact_pool_5x5_result])

    results = tf.matmul(lstm_cnn_output, weights) + biases
    return results     #,lstm_output,x,con2d_result,max_pool_result,contact_pool_result,lstm_cnn_output

pred = gru_RNN(x, weights, biases,seq_len)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)


def compute_accuracy():
    # saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, "../enACEdata/saver/checkpoint.data")
    # 载入测试集进行测试
    length = len(X_test)
    iden_p=0   # 识别的个体总数
    iden_r=0    # 测试集中存在个个体总数
    iden_acc=0  # 正确识别的个数

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
                iden_p+=1

            if y_[t] != 33:
                r_s +=1
                iden_r+=1
                if prediction[t]!=33:
                    iden_acc+=1
                if y_[t] == prediction[t]:
                    pr_acc +=1

    print('----------------------------------------------------')
    print('Trigger Identification:')
    print(str(iden_acc) + '------'+str(iden_p)+'------' + str(iden_r))
    p = iden_acc / iden_p
    r = iden_acc / iden_r
    if p+r!=0:
        f = 2 * p * r / (p + r)
        print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
    print('Trigger Classification:')
    print(str(pr_acc) + '------'+str(p_s)+'------' + str(r_s))
    p = pr_acc / p_s
    r = pr_acc / r_s
    if p+r!=0:
        f = 2 * p * r / (p + r)
        print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
        print('----------------------------------------------------')
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

        # print(np.array(batch_xs).shape)
        #
        # a,b,c=sess.run(pred, feed_dict={x: batch_xs, y: batch_ys,seq_len:train_seq_len})
        # print(np.array(a).shape)
        # print(np.array(b).shape)
        # print(np.array(c).shape)
        # sys.exit()
        # print(b)
        # print(np.array(xx).shape)
        # print(np.array(con2d_result).shape)
        # print(np.array(max_pool_result).shape)
        # print(np.array(contact_pool_result).shape)
        # print(np.array(lstm_cnn_output).shape)
        # print(last_output)
        # print(max_pool_result)
        # print(contact_pool_result)
        # print(lstm_cnn_output)
        #
        # sys.exit()

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,seq_len:train_seq_len})

        if k!=0 and step==0:

            f=compute_accuracy()
            if f>max_f:
                saver = tf.train.Saver(tf.global_variables())
                saver.save(sess,saver_path)
                max_f=f


        k += 1
    print('Optimization finished')
