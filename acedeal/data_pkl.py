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

from gensim.models import word2vec
from acedeal.nlpir import *
from acedeal.pre_process_ace import *

import tensorflow as tf
import numpy as np
import pickle
import sys


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

if __name__ == "__main__":
    print("-----------------------start----------------------")
    

    model = word2vec.Word2Vec.load_word2vec_format(
        "./corpus_deal/ace_train_corpus.bin", binary=True)

    ace_train_path = "../ace_experiment/test/"
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

#     print(len(text_list))
#     print(len(trigger_list))
#     print(len(type_list))

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

        ace_text_list = NLPIR_ParagraphProcess(event_text, 0).split(' ')
        for word in ace_text_list:
            # 读取词向量，如果没有该单词，则None
            try:
                word_vector = model[word]
            except KeyError:
                word_vector = None

            #
            if word_vector is not None:
                sentence_word2vec_arr.append(word_vector)
                if word in trigger_temp_list:
                    val = ace_type_dict[
                        type_temp_list[trigger_temp_list.index(word)]]
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

    ace_data_pkl = open('./corpus_deal/ace_data/ace_data_test.pkl', 'wb')
    pickle.dump(ace_data, ace_data_pkl)
    ace_data_pkl.close()

    ace_data_labels_pkl = open(
        './corpus_deal/ace_data/ace_data_test_labels.pkl', 'wb')
    pickle.dump(ace_data_labels, ace_data_labels_pkl)
    ace_data_labels_pkl.close()

#     # 取出每一个事件实体
#     for ace_info in ace_train_list:
#
#         if ace_info.text not in text_list:
#             if trigger_temp_list:
#                 pass
#             else:
#                 trigger_temp_list.append(ace_info.trigger)
#             trigger_list.append(trigger_temp_list)
#             trigger_temp_list = []
#             text_list.append(ace_info.text)
#         else:
#             trigger_temp_list.append(ace_info.trigger)
#
#         print(text_list)
#         print(trigger_list)
#         print('-------------------------------')

#         # if
#         ace_text_list = NLPIR_ParagraphProcess(ace_info.text, 0).split(' ')
#         for word in ace_text_list:
#             # 读取词向量，如果没有该单词，则None
#             try:
#                 word_vector = model[word]
#             except KeyError:
#                 word_vector = None
#
#             #
#             if word_vector is not None:
#                 pass
#             else:
#                 print(word)

    # print(ace_text_list)
    # print(ace_info.toString())

    print("-----------------------end----------------------")


'''
创建pkl文件
'''
#
# data1 = {'a': [1, 2.0, 3, 4 + 6j],
#          'b': ('string', u'Unicode string'),
#          'c': None}
#
# selfref_list = [1, 2, 3]
# selfref_list.append(selfref_list)
#
# output = open('data.pkl', 'wb')
#
# # Pickle dictionary using protocol 0.
# pickle.dump(data1, output)
#
# # Pickle the list using the highest protocol available.
# pickle.dump(selfref_list, output, -1)
#
# output.close()


'''
读取pkl文件
'''
# 使用pickle模块从文件中重构python对象

# import pprint
# import pickle
#
# data_file = open('./corpus_deal/ace_data.pkl', 'rb')
#
# data1 = pickle.load(data_file)
# pprint.pprint(data1)
#
# labels_file = open('./corpus_deal/ace_data_labels.pkl', 'rb')
# data2 = pickle.load(labels_file)
# pprint.pprint(data2)
#
# data_file.close()
# labels_file.close()
