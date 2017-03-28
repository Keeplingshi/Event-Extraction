"""
Created on 2017年3月20日
将34分类处理成两分类
@author: chenbin
"""

import numpy as np
import pickle
import sys

# homepath='D:/Code/pycharm/Event-Extraction/'
#
# ace_data_train_file = open(homepath+'ace_data_process/ace_eng_data2/ace_data_train.pkl', 'rb')
# X_train = pickle.load(ace_data_train_file)
#
# ace_data_train_labels_file = open(homepath+'ace_data_process/ace_eng_data2/ace_data_train_labels.pkl', 'rb')
# Y_train = pickle.load(ace_data_train_labels_file)
#
# ace_data_dev_file = open(homepath+'ace_data_process/ace_eng_data2/ace_data_dev.pkl', 'rb')
# X_dev = pickle.load(ace_data_dev_file)
#
# ace_data_dev_labels_file = open(homepath+'ace_data_process/ace_eng_data2/ace_data_dev_labels.pkl', 'rb')
# Y_dev = pickle.load(ace_data_dev_labels_file)
#
# ace_data_test_file = open(homepath+'ace_data_process/ace_eng_data2/ace_data_test.pkl', 'rb')
# X_test = pickle.load(ace_data_test_file)
#
# ace_data_test_labels_file = open(homepath+'ace_data_process/ace_eng_data2/ace_data_test_labels.pkl', 'rb')
# Y_test = pickle.load(ace_data_test_labels_file)
#
# ace_data_train_file.close()
# ace_data_train_labels_file.close()
# ace_data_dev_file.close()
# ace_data_dev_labels_file.close()
# ace_data_test_file.close()
# ace_data_test_labels_file.close()
#
# data=X_train,Y_train,X_dev,Y_dev,X_test,Y_test
# f=open('./enACEdata/train_data34.data','wb')
# pickle.dump(data,f)

data_f = open('./enACEdata/data2/train_data34.data', 'rb')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = pickle.load(data_f)

m_train=[]
for i in Y_train:
    n=[]
    for j in i:
        if j[33]==1.0:
            n.append([0.0,1.0])
        else:
            n.append([1.0,0.0])
    m_train.append(n)

m_test=[]
for i in Y_test:
    n=[]
    for j in i:
        if j[33]==1.0:
            n.append([0.0,1.0])
        else:
            n.append([1.0,0.0])
    m_test.append(n)

m_dev=[]
for i in Y_dev:
    n=[]
    for j in i:
        if j[33]==1.0:
            n.append([0.0,1.0])
        else:
            n.append([1.0,0.0])
    m_dev.append(n)

homepath='D:/Code/pycharm/Event-Extraction/'

data=X_train,m_train,X_dev,m_dev,X_test,m_test
f=open(homepath+'/model/tensorflow/enACEdata/data2/train_data2.data','wb')
pickle.dump(data,f)

