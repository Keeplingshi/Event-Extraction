# coding:utf-8

'''
Created on 2017年2月6日
双向lstm进行事件抽取
测试提交
@author: chenbin
'''

import pickle
import tensorflow as tf
import numpy as np

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
training_iters = 1600
batch_size = 1
display_step = 10

# 神经网络的参数
n_input = 200  # 输入层的n
n_steps = 1  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 34  # 输出的数量，因为是分类问题，这里一共有34个


sess = tf.InteractiveSession()
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
# weights = {
#     # Hidden layer weights => 2*n_hidden because of foward + backward cells
#     'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

W1 = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]), name="W1")
b1 = tf.Variable(tf.random_normal([n_classes]), name="b1")

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()
#saver.restore(sess, "./ckpt_file/ace_bl.ckpt")


def BiRNN(x, weights, biases):

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, W1, b1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

sess.run(init)
k = 0
# 持续迭代
while k < training_iters:

    step = k % 1600

    batch_xs = ace_data_train[step]
    batch_ys = ace_data_train_labels[step]
    batch_size = len(batch_xs)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
    
    print(batch_ys)
    print(len(batch_ys))
    print(len(batch_ys[0]))
    # 迭代
    #sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

    prediction = sess.run(pred, feed_dict={x: batch_xs})
    print(prediction)
    print(len(prediction))
    print(len(prediction[0]))

    # 在特定的迭代回合进行数据的输出
    if k % 100 == 0:
        print("Iter " + str(k))
#         sk = 0
#         acck = 0
#         predictionk, y_k = sess.run(
#             [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: batch_xs, y: batch_ys})
#         for t in range(len(y_k)):
#             if y_k[t] != 33:
#                 sk = sk + 1
#                 if y_k[t] == predictionk[t]:
#                     acck = acck + 1
#
#         if sk != 0:
#             print("Iter " + str(k) + '-----------acc=' + str(acck / sk))

    k += 1

# save_path = saver.save(sess, "./ckpt_file/ace_bl.ckpt")
# print("Model saved in file: ", save_path)
# print(sess.run(W1))

# 载入测试集进行测试
length = len(ace_data_test)
test_accuracy = 0.0
p_s = 0  # 识别的个体总数
r_s = 0  # 测试集中存在个个体总数
pr_acc = 0  # 正确识别的个数

for i in range(length):
    test_len = len(ace_data_test[i])
    test_data = ace_data_test[i].reshape((-1, n_steps, n_input))  # 8
    test_label = ace_data_test_labels[i]
    # prediction识别出的结果，y_测试集中的正确结果
    prediction, y_ = sess.run(
        [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})

    for t in range(len(y_)):
        if prediction[t] != 33:
            p_s = p_s + 1

        if y_[t] != 33:
            r_s = r_s + 1
            if y_[t] == prediction[t]:
                pr_acc = pr_acc + 1

print('----------------------------------------------------')
print('共识别出：' + str(p_s))
print('识别正确：' + str(pr_acc))
print('触发词总数：' + str(r_s))
p = pr_acc / p_s
r = pr_acc / r_s
f = 2 * p * r / (p + r)
print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))

# print(tf.global_variables())
# for tens in tf.global_variables():
#     print(tens.name)
#     print(tens.eval())
#     print('-----------------------------------------')

# saver.restore(sess, "./ckpt_file/ace_bl.ckpt")
# print(sess.run(W1))
# # 载入测试集进行测试
# length = len(ace_data_test)
# test_accuracy = 0.0
# p_s = 0  # 识别的个体总数
# r_s = 0  # 测试集中存在个个体总数
# pr_acc = 0  # 正确识别的个数
#
# for i in range(length):
#     test_len = len(ace_data_test[i])
#     test_data = ace_data_test[i].reshape((-1, n_steps, n_input))  # 8
#     test_label = ace_data_test_labels[i]
#     # prediction识别出的结果，y_测试集中的正确结果
#     prediction, y_ = sess.run(
#         [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
#     for t in range(len(y_)):
#         if prediction[t] != 33:
#             p_s = p_s + 1
#
#         if y_[t] != 33:
#             r_s = r_s + 1
#             if y_[t] == prediction[t]:
#                 pr_acc = pr_acc + 1
#
# print('----------------------------------------------------')
# print('共识别出：' + str(p_s))
# print('识别正确：' + str(pr_acc))
# print('触发词总数：' + str(r_s))
# p = pr_acc / p_s
# r = pr_acc / r_s
# f = 2 * p * r / (p + r)
# print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))


sess.close()
