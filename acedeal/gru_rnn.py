'''
Created on 2017年2月17日
对含有词性，位置信息的事件特征，进行双向gru操作
@author: chenbin
'''
import pickle
import tensorflow as tf
# from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sys

# 数据读取，训练集和测试集
ace_data_train_file = open('./corpus_deal/ace_data2/ace_data_train.pkl', 'rb')
ace_data_train = pickle.load(ace_data_train_file)

ace_data_train_labels_file = open('./corpus_deal/ace_data2/ace_data_train_labels.pkl', 'rb')
ace_data_train_labels = pickle.load(ace_data_train_labels_file)

ace_data_test_file = open('./corpus_deal/ace_data2/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)

ace_data_test_labels_file = open('./corpus_deal/ace_data2/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

ace_data_train_file.close()
ace_data_train_labels_file.close()
ace_data_test_file.close()
ace_data_test_labels_file.close()


# 参数
learningRate = 0.001
training_iters = 16000
batchSize = 1

nInput = 223
nSteps = 1 
nHidden = 128
nClasses = 34

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = tf.Variable(tf.random_normal([2 * nHidden+23, nClasses]))

biases = tf.Variable(tf.random_normal([nClasses]))


def gru_RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(0, nSteps, x)
    x_front=tf.slice(x, [0,0,0],[-1,-1,200])[0]
    x_front=tf.split(0, nSteps, x_front)
 
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
    gru_bw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
   
    outputs, _, _ = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x_front,
                                            dtype=tf.float32)
    
    x_back=tf.slice(x, [0,0,200],[-1,-1,23])[0]
    x_back=tf.split(0, nSteps, x_back)
    
    #concat_dim：0表示纵向，1表示行，2表示列
    x_all=tf.concat(2, [outputs,x_back])
    
    results=tf.matmul(x_all[-1], weights) + biases
    return results

pred = gru_RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    k = 0

    while k < training_iters:
        step = k % 1600

        batch_xs = ace_data_train[step]
        batch_ys = ace_data_train_labels[step]
        batch_size = len(batch_xs)
        batch_xs = batch_xs.reshape([batch_size, nSteps, nInput])

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        if step % 100 == 0:
            sk = 0
            acck = 0
            predictionk, y_k = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: batch_xs, y: batch_ys})
            for t in range(len(y_k)):
                if y_k[t] != 33:
                    sk = sk + 1
                    if y_k[t] == predictionk[t]:
                        acck = acck + 1
 
            if sk != 0:
                print("Iter " + str(k) + '-----------acc=' + str(acck / sk))
        k += 1
    print('Optimization finished')

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
        test_data = ace_data_test[i].reshape((-1, nSteps, nInput))  # 8
        test_label = ace_data_test_labels[i]
        # prediction识别出的结果，y_测试集中的正确结果
        prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
        for t in range(len(y_)):
            if prediction[t] != 33:
                p_s = p_s + 1

            if y_[t] != 33:
                r_s = r_s + 1
                if y_[t] == prediction[t]:
                    pr_acc = pr_acc + 1

    print('----------------------------------------------------')
    print(str(p_s)+'------' + str(r_s)+ '------'+str(pr_acc))
    p = pr_acc / p_s
    r = pr_acc / r_s
    f = 2 * p * r / (p + r)
    print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))


# 299------------579
# P=0.5567970204841713    R=0.5164075993091537    F=0.5358422939068099


# 312------------579
# P=0.5397923875432526    R=0.538860103626943    F=0.5393258426966293
