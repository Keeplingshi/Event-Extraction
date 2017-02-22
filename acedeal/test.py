'''
Created on 2017年2月6日

@author: chenbin
'''
# 使用pickle模块将数据对象保存到文件

# import pickle
# coding:utf-8
'''
Created on 2017年1月24日
lstm
@author: chenbin
'''

# from gensim.models import word2vec
# from acedeal.nlpir import *
# from acedeal.pre_process_ace import *
from nlpir import *
from pre_process_ace import *

import tensorflow as tf
import numpy as np
import pickle
import sys
import pprint

def ace_data_pkl_process(ace_train_path):
    print("-----------------------start----------------------")

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

    trigger_temp_list = []
    type_temp_list = []
    text_list_len = len(text_list)

    ace_data = []
    ace_data_labels = []
    
    vn_count=0
    other_count=0
    trigger_count=0

    for i in range(text_list_len):
        event_text = text_list[i]  # 某一句话，包含事件的一句话
        trigger_temp_list = trigger_list[i]  # 事件句中触发词，可能有多个
        type_temp_list = type_list[i]       # 事件句中触发词类型，数量与触发词数相等

        sentence_word2vec_arr = []  # 句子
        trigger_labels = []

        # 事件句分词结果
        ace_text_list = NLPIR_ParagraphProcess(event_text, 1).split(' ')
        for word_nominal in ace_text_list:
            # 读取词向量，如果没有该单词，则None
            word = word_nominal.split('/')
            if len(word) != 2:
                continue

            print(word[1])
            if get_partofspeech(word[1])==1:
                vn_count=vn_count+1
            else:
                other_count=other_count+1
            
            if word[0] in trigger_temp_list:
                trigger_count=trigger_count+1
    
    print(vn_count)
    print(other_count)
    print(trigger_count)


    print('---------------------------end--------------------------------')

    return 0



'''
获取词性，返回列表形式
'''
def get_partofspeech(part_of_speech):
    t = part_of_speech[0]
    if t in ['v']:
        return 1
    return 0
    
    # 所有词性类型
#     nominal = ['n', 'v', 't', 's', 'f', 'a', 'b', 'z', 'r', 'm',
#                'q', 'd', 'p', 'c', 'u', 'e', 'y', 'o', 'h', 'k', 'x', 'w']
#     for i in range(len(nominal)):
#         if t in nominal[i]:
#             return 0
#         else:
#             nominal_vector.append(0.0)
# 
#     return nominal_vector
    
    

if __name__ == "__main__":
    

#     ace_train_path = "../ace_experiment/test/"
#     word2vec_file = "./corpus_deal/ace_train_corpus2.bin"
#     ace_data_pkl_path = './corpus_deal/ace_data4/ace_data_test.pkl'
#     ace_label_pkl_path = './corpus_deal/ace_data4/ace_data_test_labels.pkl'
#   
#     ace_data_pkl_process(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path)
    
    ace_train_path = "../ace_experiment/train/"
    word2vec_file = "./corpus_deal/ace_train_corpus2.bin"
    ace_data_pkl_path = './corpus_deal/ace_data5/ace_data_test.pkl'
    ace_label_pkl_path = './corpus_deal/ace_data5/ace_data_test_labels.pkl'
  
    ace_data_pkl_process(ace_train_path)



#    print("-----------------------end----------------------")


