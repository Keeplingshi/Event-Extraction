#coding=utf8
from sklearn import cross_validation
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding,TimeDistributed,Flatten
from keras.layers import LSTM, SimpleRNN, GRU
import pickle
import nltk
import itertools
import json
from nlp_hw3.chinese.others.type_index import EVENT_MAP
'''分类大法师'''


vocabulary_size=10000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
type_num=len(EVENT_MAP)

def pre_data():
    '''不用切词'''
    f=open('./others/event_class.data','rb')
    train_data=pickle.load(f)
    word_dict=dict()
    tokenized_sentences=[i[0] for i in train_data]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = word_freq.most_common(vocabulary_size-1)
    print("Found %d unique words tokens." % len(word_freq.items()))
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i+2) for i,w in enumerate(index_to_word)])
    word_index_f=open('./others/word_index2.json','w')
    json.dump(word_to_index,word_index_f)
    x=[]
    y=[]
    for item in train_data:
        x.append([word_to_index[word] if word in word_to_index else word_to_index[unknown_token] for word in item[0]])
        assert sum(item[1])>0
        y.append(item[1])
    X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(x,y,test_size=0.1, random_state=0)  
    print(X_train[5])
    print(Y_train[5])#输出来看是否对了，检查对了。
    data=X_train,X_test,Y_train,Y_test
    f=open('others/train_data_4class.data','wb')
    pickle.dump(data,f)

def lstm():
    maxlen = 180  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    data_f=open('./others/train_data_4class.data','rb')
    X_train,X_test,Y_train,Y_test=pickle.load(data_f)
    X_train2=[]
    Y_train2=[]
  
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    
    print('Pad sequences (samples x time)')

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    Y_train = sequence.pad_sequences(Y_train, maxlen=type_num)
    for i in Y_train:
        print(np.sum(i))
        assert np.sum(i)>0
    
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)

    model = Sequential()
    # model.add(Embedding(DICT_SIZE, EMBED_SIZE, input_length=MAX_SENTENCE_LEN))
    model.add(Embedding(vocabulary_size+3, 256, input_length=maxlen, mask_zero=True))
    model.add(LSTM(128, input_shape=(maxlen,256)))    
    # # model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax')))                           
    model.add(Dense(type_num,activation='softmax'))             

    model.compile(loss='categorical_crossentropy',                                   
                  optimizer='rmsprop',                                               
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=16,verbose=1, nb_epoch=10)

    json_string  = model.to_json() 

    open('my_class_model_architecture.json','w').write(json_string) 

    model.save_weights('my_class_model_weights.h5') 


def lstm_test():

    maxlen = 180  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    data_f=open('./others/train_data_4class.data','rb')

    X_train,X_test,Y_train,Y_test=pickle.load(data_f)
    model = model_from_json(open('my_class_model_architecture.json').read()) 
    model.load_weights('my_class_model_weights.h5')
    X_test1 = sequence.pad_sequences(X_test, maxlen=maxlen)
    Y_test1 = sequence.pad_sequences(Y_test, maxlen=type_num)

    # for i,j in zip(Y_test1,Y_test):
    #     a1=np.array(j)
    #     a2=np.array(i[-1*len(j):])
    #     print a1-a2
    rs = model.predict(X_test1)
    rs =np.array([[1 if i>0.5 else 0 for i in j] for j in rs]) 
    rs_all=0
    rs_pre=0
    rs_right=0
    line=0
    for i,j in zip(Y_test,rs):
        line+=1
        for i1,j1 in zip(i,j):
            if i1==1:rs_all+=1
            if j1==1:rs_pre+=1
            if i1==j1 and i1==1:
                print(line)
                rs_right+=1
    print(rs_all,rs_pre,rs_right)
    np.savetxt('rs22.txt',np.array(Y_test))
    np.savetxt('pre22.txt',rs)

if __name__ == '__main__':
    # pre_data()

    # lstm()
    lstm_test()