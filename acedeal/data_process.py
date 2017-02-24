'''
Created on 2017年2月6日

@author: chenbin
'''
# 使用pickle模块将数据对象保存到文件

# import pickle
# coding:utf-8
'''
Created on 2017年2月23日
lstm
@author: chenbin
'''

from gensim.models import word2vec
# from acedeal.nlpir import *
# from acedeal.pre_process_ace import *
from nlpir import *
from pre_process_ace import *

import tensorflow as tf
import numpy as np
import pickle
import sys
import pprint
import random


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


def ace_data_pkl_process(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path):
    print("-----------------------start----------------------")

    model = word2vec.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

    # 获取ace事件基本信息，即text，type，ID，trigger等。
    ace_train_list = get_ace_event_list(ace_train_path)
    ace_train_list_len = len(ace_train_list)
    text_list = []
    trigger_list = []
    trigger_temp_list = []
    type_list = []
    type_temp_list = []

    for i in range(ace_train_list_len):
        # 获取一个ace事件的实体
        ace_info = ace_train_list[i]
        # 处理一个句子，多个事件的情况
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

#     print(len(text_list))
#     print(len(trigger_list))
#     print(len(type_list))

    trigger_temp_list = []
    type_temp_list = []
    text_list_len = len(text_list)

    ace_data = []
    ace_data_labels = []
    
    none_role_num=0

    for i in range(text_list_len):
        event_text = text_list[i]  # 某一句话，包含事件的一句话
        trigger_temp_list = trigger_list[i]  # 事件句中触发词，可能有多个
        type_temp_list = type_list[i]       # 事件句中触发词类型，数量与触发词数相等


        # 事件句分词结果
        ace_text_list = NLPIR_ParagraphProcess(event_text, 0).split(' ')
        for word in ace_text_list:

            try:
                word_vector = model[word]
            except KeyError:
                word_vector = None

            #如果存在词向量
            if word_vector is not None:
                
                none_role_num=none_role_num+1

                #sentence_word2vec_arr.append(word_vector)
                if word in trigger_temp_list:
                    val = ace_type_dict[type_temp_list[trigger_temp_list.index(word)]]
                    a = [0.0 for x in range(0, 34)]
                    a[val] = 1.0
                    ace_data_labels.append(a)
                    ace_data.append(word_vector)
                else:
                    val = ace_type_dict['None-role']
                    a = [0.0 for x in range(0, 34)]
                    a[val] = 1.0
                    ace_data_labels.append(a)
                    ace_data.append(word_vector)
                    
#                     # 每75个词取一个
#                     if none_role_num%75==0:
#                         # 获取一个随机数，如果小于一定数，则加入
#                         val = ace_type_dict['None-role']
#                         a = [0.0 for x in range(0, 34)]
#                         a[val] = 1.0
#                         ace_data_labels.append(a)
#                         ace_data.append(word_vector)
                    

                    


    ace_data=np.array(ace_data)
    ace_data_labels=np.array(ace_data_labels)

    ace_data_pkl = open(ace_data_pkl_path, 'wb')
    pickle.dump(ace_data, ace_data_pkl)
    ace_data_pkl.close()

    ace_data_labels_pkl = open(ace_label_pkl_path, 'wb')
    pickle.dump(ace_data_labels, ace_data_labels_pkl)
    ace_data_labels_pkl.close()

    print('---------------------------end--------------------------------')

    return ace_data_pkl_path, ace_label_pkl_path



if __name__ == "__main__":
    
    #训练集数据处理
    ace_train_path = "../ace_experiment/train/"
    word2vec_file = "./corpus_deal/ace_train_corpus2.bin"
    ace_data_pkl_path = './ace_data_process/ace_data6/ace_data_train.pkl'
    ace_label_pkl_path = './ace_data_process/ace_data6/ace_data_train_labels.pkl'
     
    ace_data_pkl_process(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path)
     
    #测试集数据处理
    ace_train_path = "../ace_experiment/test/"
    word2vec_file = "./corpus_deal/ace_train_corpus2.bin"
    ace_data_pkl_path = './ace_data_process/ace_data6/ace_data_test.pkl'
    ace_label_pkl_path = './ace_data_process/ace_data6/ace_data_test_labels.pkl'
    
    ace_data_pkl_process(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path)
    
    # 数据读取，训练集和测试集
    ace_data_train_file = open('./ace_data_process/ace_data6/ace_data_train.pkl', 'rb')
    ace_data_train = pickle.load(ace_data_train_file)
    
    ace_data_train_labels_file = open('./ace_data_process/ace_data6/ace_data_train_labels.pkl', 'rb')
    ace_data_train_labels = pickle.load(ace_data_train_labels_file)
    
    ace_data_test_file = open('./ace_data_process/ace_data6/ace_data_test.pkl', 'rb')
    ace_data_test = pickle.load(ace_data_test_file)
    
    ace_data_test_labels_file = open('./ace_data_process/ace_data6/ace_data_test_labels.pkl', 'rb')
    ace_data_test_labels = pickle.load(ace_data_test_labels_file)
    
    print(ace_data_train)
    print(len(ace_data_train))
    print(len(ace_data_train[0]))
    
    print(ace_data_train_labels)
    print(len(ace_data_train_labels))
    print(len(ace_data_train_labels[0]))
    
    
    print('-----------------------------------------')
    print(len(ace_data_train))
    print(len(ace_data_train_labels))
    print(len(ace_data_test))
    print(len(ace_data_test_labels))

