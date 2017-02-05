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


if __name__ == "__main__":
    print("-----------------------start----------------------")

    model = word2vec.Word2Vec.load_word2vec_format(
        "./corpus_deal/ace_train_corpus.bin", binary=True)

    ace_train_path = "../ace_experiment/train/"
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
                    trigger_labels.append()
                else:
                    trigger_labels

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
