# coding:utf-8
'''
Created on 2017年3月19日
事件识别，识别出是否为触发词,二分类
词向量

结果 
正确识别的个体总数          识别出的个体总数               测试集中存在的个体总数                P                      R                   F
    172        331                269        0.6394052044609665 0.5196374622356495 0.5733333333333334
    173        320                269        0.6431226765799256   0.540625          0.5874363327674023
    160        283                269          0.5947955390334573   0.5653710247349824    0.5797101449275361

@author: chenbin
'''
from sklearn import cross_validation
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding,TimeDistributed,Flatten,Masking
from keras.layers import LSTM, SimpleRNN, GRU,Bidirectional
import pickle
import nltk
import itertools
import json
import sys
import time

batch_size=50
lstm_activation='tanh'
lstm_inner_activation='hard_sigmoid'
timeDistributed_dense_activation='softmax'
nb_epoch=10
maxlen = 180
dropout_W=0.
dropout_U=0.

def lstm():
    data_f=open('./chACEdata/train_data.data','rb')
    X_train,X_test,Y_train,Y_test=pickle.load(data_f)
#     X_train2=X_train
#     Y_train2=Y_train
    X_train2=[]
    Y_train2=[]
    for i,j in zip(X_train,Y_train):
        tmp=sum(j)
        if tmp>0:
            X_train2.append(i)
            Y_train2.append(j)
 
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    
    print('Pad sequences (samples x time)')

    X_train = sequence.pad_sequences(X_train2, maxlen=maxlen)
    Y_train1 = sequence.pad_sequences(Y_train2, maxlen=maxlen)
   
    Y_train = np.asarray([np_utils.to_categorical(j,2) for j in Y_train1])
      
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(180, 200)))
    model.add(LSTM(output_dim=128, activation=lstm_activation, inner_activation=lstm_inner_activation,dropout_W=dropout_W,dropout_U=dropout_U, return_sequences=True))    #,input_dim=200, input_length=180
    #model.add(LSTM(output_dim=128, activation=lstm_activation, inner_activation=lstm_inner_activation,dropout_W=dropout_W,dropout_U=dropout_U, return_sequences=True,go_backwards=True))    #,input_dim=200, input_length=180
    
    model.add(TimeDistributed(Dense(2,activation=timeDistributed_dense_activation))) 

    model.compile(loss='categorical_crossentropy',                                   
                  optimizer='rmsprop',                                               
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size,verbose=1, nb_epoch=nb_epoch)

    json_string  = model.to_json() 

    open('./chACEdata/my_model_architecture2.json','w').write(json_string) 

    model.save_weights('./chACEdata/my_model_weights2.h5') 



def lstm_test():

    data_f=open('./chACEdata/train_data.data','rb')

    X_train,X_test,Y_train,Y_test=pickle.load(data_f)
    model = model_from_json(open('./chACEdata/my_model_architecture2.json').read()) 
    model.load_weights('./chACEdata/my_model_weights2.h5')

    X_test1 = sequence.pad_sequences(X_test, maxlen=maxlen)
    Y_test1 = sequence.pad_sequences(Y_test, maxlen=maxlen)

    # for i,j in zip(Y_test1,Y_test):
    #     a1=np.array(j)
    #     a2=np.array(i[-1*len(j):])
    #     print a1-a2
    rs = model.predict_classes(X_test1)
    rs_all=0    #测试集中总个数
    rs_pre=0    #识别出来的个数
    rs_right=0  #识别正确的个数
    for i,j in zip(Y_test1,rs):
        for i1,j1 in zip(i,j):
            if i1==1:rs_all+=1
            if j1==1:rs_pre+=1
            if i1==j1 and i1==1:
                rs_right+=1
    P=rs_right/rs_all
    R=rs_right/rs_pre
    F=2*P*R/(P+R)
    
    log_f=open('./chACEdata/iden_event.log','a',encoding='utf8')
    log_f.write('\n')
    log_f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"\t参数记录与实验结果\n")
    log_f.write('batch_size='+str(batch_size)+"\n")
    log_f.write('lstm_activation='+str(lstm_activation)+"\n")
    log_f.write('lstm_inner_activation='+str(lstm_inner_activation)+"\n")
    log_f.write('timeDistributed_dense_activation='+str(timeDistributed_dense_activation)+"\n")
    log_f.write('nb_epoch='+str(nb_epoch)+"\n")
    log_f.write('maxlen='+str(maxlen)+"\n")
    log_f.write('dropout_W='+str(dropout_W)+"\n")
    log_f.write('dropout_U='+str(dropout_U)+"\n")
    log_f.write('实验结果（正确识别的个体总数  ，识别出的个体总数  ，测试集中存在的个体总数）：'+str(rs_right)+'\t'+str(rs_pre)+'\t'+str(rs_all)+'\t'+"\n")
    log_f.write('P,R,F：'+str(P)+'\t'+str(R)+'\t'+str(F)+"\n")
    log_f.write('---------------------------------------------------------------------------\n')
    print(rs_right,rs_pre,rs_all)
    print(P,R,F)
    print('测试结束')

if __name__ == '__main__':
#     pre_data()
    lstm()
    lstm_test()
    
#     for i in range(9):
#         for j in range(9):
#             dropout_W=i/10
#             dropout_U=j/10
#             lstm()
#             lstm_test()
#     
#     lstm_activation='sigmod'
#     for i in range(9):
#         for j in range(9):
#             dropout_W=i/10
#             dropout_U=j/10
#             lstm()
#             lstm_test()