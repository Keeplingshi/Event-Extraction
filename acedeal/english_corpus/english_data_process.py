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
from process_english_data import get_ace_event_list
import numpy as np

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

'''
将所有语料处理到一个文件内 
'''
def write_train_data():
    filelist=os.listdir('./source_data/')
     
    ofile = open('./traindata.txt', 'w',encoding= 'utf-8')
    for fr in filelist:
        for txt in open('./source_data/'+fr, 'r',encoding= 'utf-8'):
            #去除标点符号
            m=txt.translate(str.maketrans('','',string.punctuation))
            ofile.write(m)
             
        ofile.write('\n')
     
    ofile.close()


'''
word2vec进行训练
'''
def word2vec_train():
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
          
    sentences =word2vec.Text8Corpus("./traindata.txt")  # 加载语料  
    model =word2vec.Word2Vec(sentences, size=200,min_count=5,iter=15)  #训练skip-gram模型，默认window=5 
    # 保存模型，以便重用  
    model.save("./word2vec_data/news.model")  
    # 以一种c语言可以解析的形式存储词向量  
    model.save_word2vec_format("./word2vec_data/news.bin", binary=True)  
          
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))



def ace_data_pkl_process_english_word2vec(ace_train_path,word2vec_file,ace_data_pkl_path,ace_label_pkl_path):
    print("-----------------------start----------------------")
    
    model = word2vec.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

    ace_train_list = get_ace_event_list(ace_train_path)
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

    trigger_temp_list = []
    type_temp_list = []
    text_list_len = len(text_list)

    ace_data = []
    ace_data_labels = []

    for i in range(text_list_len):
        event_text = text_list[i]
        trigger_temp_list = trigger_list[i]
        type_temp_list = type_list[i]

        sentence_word2vec_arr = []  # 句子
        trigger_labels = []

        ace_text_list = event_text.split(' ')
        for word in ace_text_list:
            # 读取词向量，如果没有该单词，则None
            try:
                word_vector = model[word]
            except KeyError:
                word_vector = None

            #
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

        ace_data.append(np.array(sentence_word2vec_arr))
        ace_data_labels.append(trigger_labels)

    ace_data_pkl = open(ace_data_pkl_path, 'wb')
    pickle.dump(ace_data, ace_data_pkl)
    ace_data_pkl.close()

    ace_data_labels_pkl = open(ace_label_pkl_path, 'wb')
    pickle.dump(ace_data_labels, ace_data_labels_pkl)
    ace_data_labels_pkl.close()
    
    print('---------------------------end--------------------------------')
    

if __name__ == "__main__":
    print("-----------------------start----------------------")

    ace_train_path = "../../ace_en_experiment/test/"
    word2vec_file = "./word2vec_data/news.bin"
    ace_data_pkl_path = '../ace_data_process/ace_eng_data1/ace_data_test.pkl'
    ace_label_pkl_path = '../ace_data_process/ace_eng_data1/ace_data_test_labels.pkl'
   
    ace_data_pkl_process_english_word2vec(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path)
    
    ace_train_path = "../../ace_en_experiment/train/"
    word2vec_file = "./word2vec_data/news.bin"
    ace_data_pkl_path = '../ace_data_process/ace_eng_data1/ace_data_train.pkl'
    ace_label_pkl_path = '../ace_data_process/ace_eng_data1/ace_data_train_labels.pkl'
   
    ace_data_pkl_process_english_word2vec(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path)

    print("-----------------------end----------------------")

