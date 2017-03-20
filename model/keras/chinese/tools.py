#coding=utf8

from lxml import etree  # @UnresolvedImport
import os
import json
import jieba 
from cb.chinese.type_index import EVENT_MAP


import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding,TimeDistributed,Flatten
from keras.layers import LSTM, SimpleRNN, GRU

vocabulary_size=10000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
type_num=len(EVENT_MAP)

def main():
    f_list=[os.path.join('train/tag',i) for i in os.listdir('train/tag')]
    type_list=set()
    for i in f_list:
        f=open(i)
        content=f.read().decode('utf8')
        doc=etree.fromstring(content)
        event=doc.xpath('//event')
        for j in event:
            _type=j.xpath("./@TYPE")[0]
            _subtype= j.xpath("./@SUBTYPE")[0]
            c=_type+'.'+_subtype
            type_list.add(c)
    json_data=dict()
    for index,i in enumerate(type_list):
        json_data[i]=index
    json_data['UNKOWN']=len(json_data)
    print(json_data)

def extract_text():
    '''提取test text到text3文件夹中'''
    f_list=os.listdir('./test/text')

    for i in f_list:
        filepath=os.path.join('./test/text',i)
        f2=open(filepath)
        content=f2.read()
        doc = etree.fromstring(content)
        content2=doc.xpath('//TURN//text()')
        _index=i.rfind('.')
        if not os.path.exists('text3'):
            os.mkdir('text3')
        newfname=os.path.join('./text3',i[:_index]+'.txt')
        print(filepath,'===>',newfname)
        f3=open(newfname,'w')
        f3.write(''.join(content2).encode('utf8'))


def test3_2_word_vec():
    '''text4的文件夹放句子的切词结果'''
    jieba.load_userdict('./others/train_dict.dict')
    f_list=[i for i in os.listdir("text3")]
    if not os.path.exists('text4'):
        os.mkdir('text4')
    for f in f_list:
        filepath=os.path.join('text3',f)
        content=open(filepath).read().decode('utf8')
        content=content.replace('\n\n','#')
        content=content.replace('\n','')
        content=content.replace(' ','')
        i_list=content.split('#')
        sen_list=[]
        for i in i_list:
            tt=i.split(u'。')
            sen_list.extend([ttt for ttt in tt if len(ttt)>3])
        sen_file_path=os.path.join('text4',f)
        rs_f=open(sen_file_path,'w')
        for t in sen_list:
            word_list=[i for i in jieba.cut(t)]
            rs_f.write(' '.join(word_list).encode('utf8')+'\n')

def sentence_type(filename):
    '''不用切词'''
    filepath=os.path.join('text4',filename)
    f=open(filepath)
    test_data=[]
    for i in f.readlines():
        test_data.append(i.decode('utf8').strip().split(' '))
    json_f=open('./others/word_index2.json','r')
    word_to_index=json.loads(json_f.read())
    X_test=[]
    for item in test_data:
        X_test.append([word_to_index[word] if word in word_to_index else word_to_index[unknown_token] for word in item])
    maxlen = 180  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    data_f=open('./others/train_data_4class.data','rb')

    model = model_from_json(open('my_class_model_architecture.json').read()) 
    model.load_weights('my_class_model_weights.h5')
    X_test1 = sequence.pad_sequences(X_test, maxlen=maxlen)
    
    # for i,j in zip(Y_test1,Y_test):
    #     a1=np.array(j)
    #     a2=np.array(i[-1*len(j):])
    #     print a1-a2
    rs = model.predict(X_test1)
    rs =np.array([[1 if i>0.5 else 0 for i in j] for j in rs]) 
    print(rs)

def sentence_trigger(filename):
    filepath=os.path.join('text4',filename)
    f=open(filepath)
    test_data=[]
    for i in f.readlines():
        test_data.append(i.decode('utf8').strip().split(' '))
    json_f=open('./others/word_index.json','r')
    word_to_index=json.loads(json_f.read())
    X_test=[]
    for item in test_data:
        X_test.append([word_to_index[word] if word in word_to_index else word_to_index[unknown_token] for word in item])
    print(X_test)
    maxlen = 180  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    data_f=open('./others/train_data.data','rb')

    model = model_from_json(open('my_model_architecture.json').read()) 
    model.load_weights('my_model_weights.h5')

    X_test1 = sequence.pad_sequences(X_test, maxlen=maxlen)
    rs = model.predict_classes(X_test1)
    print(rs)
    

if __name__ == '__main__':
    # test3_2_word_vec()
    # word_list2wordvec("CTS20001222.1300.0687.txt")
    sentence_trigger("CTS20001222.1300.0687.txt")
