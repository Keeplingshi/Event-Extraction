# coding:utf-8
'''
Created on 2017年3月20日
事件识别  二分类
rnn
@author: chenbin
'''

import tensorflow as tf
import numpy as np
import pickle
import nltk
import itertools
import json
import sys
import time

data_f=open('./enACEdata/train_data2.data','rb')
X_train,Y_train,X_dev,Y_dev,X_test,Y_test=pickle.load(data_f)
data_f.close()

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# print(len(X_train))
# print(len(X_train[0]))
# print(len(X_train[0][0]))
# 
# print(X_train[0][0])
# 
# print(len(Y_train))
# print(len(Y_train[0]))
# print(len(Y_train[0][0]))




