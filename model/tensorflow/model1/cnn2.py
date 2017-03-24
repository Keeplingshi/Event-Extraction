# coding:utf-8
"""
cnn

"""

import tensorflow as tf
import numpy as np
import pickle

data_f = open('./enACEdata/train_data2.data', 'rb')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = pickle.load(data_f)
data_f.close()

sess = tf.InteractiveSession()

def conv2d(a,b):
    return tf.nn.conv2d(a, b, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(a):
    return tf.nn.max_pool(a, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


sess.run(tf.initialize_all_variables())



# max_length=-1
# for i in range(len(X_train)):
#     max_length=max_length if max_length>len(X_train[i]) else len(X_train[i])
#
# for i in range(len(X_dev)):
#     max_length=max_length if max_length>len(X_dev[i]) else len(X_dev[i])
#
# for i in range(len(X_test)):
#     max_length=max_length if max_length>len(X_test[i]) else len(X_test[i])
#
# print(max_length)


