# coding: utf-8
# @author: chenbin
# @date：2017-02-26
# 

from gensim.models import word2vec
import pickle
import sys
import os
import string
import time
import numpy as np
from model.tensorflow.data_process import process_english_data

homepath='D:/Code/pycharm/Event-Extraction/'

ace_type_dict = {
    'Be-Born': 0,
    'Die': 1,
    'Marry': 2,
    'Divorce': 3,
    'Injure': 4,
    'Transfer-Ownership': 5,
    'Transfer-Money': 6,
    'Transport': 7,
    'Start-Org': 8,
    'End-Org': 9,
    'Declare-Bankruptcy': 10,
    'Merge-Org': 11,
    'Attack': 12,
    'Demonstrate': 13,
    'Meet': 14,
    'Phone-Write': 15,
    'Start-Position': 16,
    'End-Position': 17,
    'Nominate': 18,
    'Elect': 19,
    'Arrest-Jail': 20,
    'Release-Parole': 21,
    'Charge-Indict': 22,
    'Trial-Hearing': 23,
    'Sue': 24,
    'Convict': 25,
    'Sentence': 26,
    'Fine': 27,
    'Execute': 28,
    'Extradite': 29,
    'Acquit': 30,
    'Pardon': 31,
    'Appeal': 32,
    'None-role': 33
}


"""
dict形式，word--vector
"""
def get_word2vec():
    word2vec_file=homepath+'/ace05/word2vec/wordvector'
    wordlist_file=homepath+'/ace05/word2vec/wordlist'

    wordvec={}
    word2vec_f=open(word2vec_file,'r')
    wordlist_f=open(wordlist_file,'r')
    word_len=19488
    for line in range(word_len):
        word=wordlist_f.readline().strip()
        vec=word2vec_f.readline().strip()
        temp=vec.split(',')
        temp = map(float, temp)
        vec_list = []
        for i in temp:
            vec_list.append(i)
        wordvec[word]=vec_list
    return wordvec


def word_process(word):

    return

def ace_data_pkl_process_english_word2vec(ace_train_path,word2vec_dict):
    print("-----------------------ace_data_pkl_process_english_word2vec start----------------------")

    ace_train_list = process_english_data.get_ace_event_list(ace_train_path)
    ace_train_list_len = len(ace_train_list)
    text_list = []
    trigger_list = []
    trigger_temp_list = []
    type_list = []
    type_temp_list = []

    for i in range(ace_train_list_len):
        ace_info = ace_train_list[i]
        if i < ace_train_list_len - 1:

            if ace_info.text == ace_train_list[i + 1].text:
                trigger_temp_list.append(ace_info.trigger)
                type_temp_list.append(ace_info.sub_type)
            else:
                text_list.append(ace_info.text)
                trigger_temp_list.append(ace_info.trigger)
                trigger_list.append(trigger_temp_list)
                trigger_temp_list = []
                type_temp_list.append(ace_info.sub_type)
                type_list.append(type_temp_list)
                type_temp_list = []
        else:
            trigger_temp_list.append(ace_info.trigger)
            trigger_list.append(trigger_temp_list)
            type_temp_list.append(ace_info.sub_type)
            type_list.append(type_temp_list)
            text_list.append(ace_info.text)

    text_list_len = len(text_list)

    ace_data = []
    ace_data_labels = []

    count=0
    for i in range(text_list_len):
        event_text = text_list[i]
        trigger_temp_list = trigger_list[i]
        type_temp_list = type_list[i]

        sentence_word2vec_arr = []  # 句子
        trigger_labels = []
        
        #event_text=event_text.translate(str.maketrans('','',string.punctuation))
        # event_text_ori=event_text
        # event_text=event_text.replace('(','').replace(')','').replace(': ',' : ').replace(', ',' , ').replace('\'s' ,' \'s').replace('\'d' ,' \'d').replace('. ',' . ')
        # event_text=event_text.replace('n\'t',' n\'t').replace('\'ll',' \'ll').replace('"',' ').replace(';',' ; ').replace('\' ',' \' ').replace('\'re',' \'re')
        # event_text=event_text.replace('$',' $ ')
        ace_text_list = event_text.split(' ')
        for word in ace_text_list:
            word=word.lower()

            if word!='':

                # 读取词向量，如果没有该单词，则None
                word_vector = word2vec_dict.get(word)

                if word_vector is not None:
                    # 将单词的词向量加入句子向量中
                    sentence_word2vec_arr.append(word_vector)
                    if word in trigger_temp_list:
                        val = ace_type_dict[type_temp_list[trigger_temp_list.index(word)]]
                        a = [0.0 for x in range(0, 34)]
                        a[val] = 1.0
                        trigger_labels.append(a)
                    else:
                        val = ace_type_dict['None-role']
                        a = [0.0 for x in range(0, 34)]
                        a[val] = 1.0
                        trigger_labels.append(a)
                else:
                    print(word)
                    print(event_text)
                    print(ace_text_list)
                    count += 1
                    print(count)


        if len(sentence_word2vec_arr)!=0:
            assert len(sentence_word2vec_arr) == len(trigger_labels), event_text
            ace_data.append(np.array(sentence_word2vec_arr))
            ace_data_labels.append(trigger_labels)

    return ace_data,ace_data_labels

    # ace_data_pkl = open(ace_data_pkl_path, 'wb')
    # pickle.dump(ace_data, ace_data_pkl)
    # ace_data_pkl.close()
    #
    # ace_data_labels_pkl = open(ace_label_pkl_path, 'wb')
    # pickle.dump(ace_data_labels, ace_data_labels_pkl)
    # ace_data_labels_pkl.close()
    #
    # print('---------------------------end--------------------------------')




if __name__ == "__main__":
    # print("-----------------------start----------------------")
    # lista=[[0],[1],[2]]
    # listb=[[3],[4]]
    # listc=lista.extend(listb)
    # print(lista)
    # print(listc)
    # sys.exit()

    wordvec=get_word2vec()

    ace_train_path = homepath+"/ace_en_experiment/train/"
    X_train,Y_train=ace_data_pkl_process_english_word2vec(ace_train_path, wordvec)

    ace_test_path = homepath+"/ace_en_experiment/test/"
    X_test,Y_test=ace_data_pkl_process_english_word2vec(ace_test_path, wordvec)

    ace_dev_path = homepath+"/ace_en_experiment/dev/"
    X_dev,Y_dev=ace_data_pkl_process_english_word2vec(ace_dev_path, wordvec)

    data=X_train,Y_train,X_test,Y_test,X_dev,Y_dev
    f=open(homepath+'/model/tensorflow/enACEdata/data2/train_data34.data','wb')
    pickle.dump(data,f)

    m_train = []
    for i in Y_train:
        n = []
        for j in i:
            if j[33] == 1.0:
                n.append([0.0, 1.0])
            else:
                n.append([1.0, 0.0])
        m_train.append(n)

    m_test = []
    for i in Y_test:
        n = []
        for j in i:
            if j[33] == 1.0:
                n.append([0.0, 1.0])
            else:
                n.append([1.0, 0.0])
        m_test.append(n)

    m_dev = []
    for i in Y_dev:
        n = []
        for j in i:
            if j[33] == 1.0:
                n.append([0.0, 1.0])
            else:
                n.append([1.0, 0.0])
        m_dev.append(n)

    data = X_train, m_train, X_dev, m_dev, X_test, m_test
    f = open(homepath + '/model/tensorflow/enACEdata/data2/train_data2.data', 'wb')
    pickle.dump(data, f)
    print("-----------------------end----------------------")

