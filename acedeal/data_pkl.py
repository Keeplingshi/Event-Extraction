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
# from acedeal.nlpir import *
# from acedeal.pre_process_ace import *
from nlpir import *
from pre_process_ace import *

import tensorflow as tf
import numpy as np
import pickle
import sys
import pprint


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

    for i in range(text_list_len):
        event_text = text_list[i]  # 某一句话，包含事件的一句话
        trigger_temp_list = trigger_list[i]  # 事件句中触发词，可能有多个
        type_temp_list = type_list[i]       # 事件句中触发词类型，数量与触发词数相等

        sentence_word2vec_arr = []  # 句子
        trigger_labels = []

        # 事件句分词结果
        ace_text_list = NLPIR_ParagraphProcess(event_text, 1).split(' ')
        word_distance = 0  # 单词的位置信息
        for word_nominal in ace_text_list:
            # 读取词向量，如果没有该单词，则None
            word = word_nominal.split('/')
            if len(word) != 2:
                continue

            try:
                word_vector = model[word[0]]
            except KeyError:
                word_vector = None

            #
            if word_vector is not None:
                # 将单词的词向量加入句子向量中
                # 单词特征向量，词向量+位置信息+词性
                # print(word_vector.tolist())
                characteristic_vector = []
                characteristic_vector.extend(word_vector.tolist())
                characteristic_vector.append(
                    word_distance / len(ace_text_list))
                nominal_vector = get_partofspeech(word[1])
                characteristic_vector.extend(nominal_vector)

                # print(characteristic_vector)

                sentence_word2vec_arr.append(characteristic_vector)
                if word[0] in trigger_temp_list:
                    val = ace_type_dict[
                        type_temp_list[trigger_temp_list.index(word[0])]]
                    a = [0.0 for x in range(0, 34)]
                    a[val] = 1.0
                    trigger_labels.append(a)
                else:
                    val = ace_type_dict['None-role']
                    a = [0.0 for x in range(0, 34)]
                    a[val] = 1.0
                    trigger_labels.append(a)

                word_distance = word_distance + 1

        ace_data.append(np.array(sentence_word2vec_arr))
        ace_data_labels.append(trigger_labels)

    ace_data_pkl = open(ace_data_pkl_path, 'wb')
    pickle.dump(ace_data, ace_data_pkl)
    ace_data_pkl.close()

    ace_data_labels_pkl = open(ace_label_pkl_path, 'wb')
    pickle.dump(ace_data_labels, ace_data_labels_pkl)
    ace_data_labels_pkl.close()

    print('---------------------------end--------------------------------')

    return ace_data_pkl_path, ace_label_pkl_path

'''
获取词性，返回列表形式
'''
def get_partofspeech(part_of_speech):
    t = part_of_speech[0]
    position = None
    # 所有词性类型
    nominal = ['n', 'v', 't', 's', 'f', 'a', 'b', 'z', 'r', 'm',
               'q', 'd', 'p', 'c', 'u', 'e', 'y', 'o', 'h', 'k', 'x', 'w']
    nominal_vector = []
    for i in range(len(nominal)):
        if t in nominal[i]:
            nominal_vector.append(1.0)
        else:
            nominal_vector.append(0.0)

    return nominal_vector

# '''
# 将数据分为训练集和测试集
# '''
# def get_train_test_data():
#     # 获取ace所有数据
#     ace_data_file = open('./corpus_deal/ace_data3/ace_data.pkl', 'rb')
#     ace_data = pickle.load(ace_data_file)
#     ace_data_labels_file = open('./corpus_deal/ace_data3/ace_data_labels.pkl', 'rb')
#     ace_data_labels = pickle.load(ace_data_labels_file)
#     
#     #根据标签分为测试集与训练集
#     #print(ace_data_train_labels)
#     labels=[]
#     for i in range(34):
#         labels.append(0)
#     
# #     for i in ace_data_train_labels:
# #         for m in i:
# #             labels[m.index(1.0)]=labels[m.index(1.0)]+1
#     
#     for i in ace_data_labels:
#         print(i)
#         for m in i:
#             labels[m.index(1.0)]=labels[m.index(1.0)]+1
#             
#             
#     new_ace_type_dict = {v:k for k,v in ace_type_dict.items()}
#     for i in new_ace_type_dict:
#         print(new_ace_type_dict[i]+'\t'+str(labels[i]))
#         
#         
#     ace_data_file.close()
#     ace_data_labels_file.close()
    
    
    
'''
返回数据信息
'''
def get_ace_data_info():
    # 数据读取，训练集和测试集
    ace_data_train_file = open('./corpus_deal/ace_data2/ace_data_train.pkl', 'rb')
    ace_data_train = pickle.load(ace_data_train_file)
    
    ace_data_train_labels_file = open(
        './corpus_deal/ace_data2/ace_data_train_labels.pkl', 'rb')
    ace_data_train_labels = pickle.load(ace_data_train_labels_file)
    
    ace_data_test_file = open('./corpus_deal/ace_data2/ace_data_test.pkl', 'rb')
    ace_data_test = pickle.load(ace_data_test_file)
    
    ace_data_test_labels_file = open(
        './corpus_deal/ace_data2/ace_data_test_labels.pkl', 'rb')
    ace_data_test_labels = pickle.load(ace_data_test_labels_file)
    
    
    #print(ace_data_train_labels)
    labels=[]
    for i in range(34):
        labels.append(0)
    
#     for i in ace_data_train_labels:
#         for m in i:
#             labels[m.index(1.0)]=labels[m.index(1.0)]+1
    
    for i in ace_data_test_labels:
        for m in i:
            labels[m.index(1.0)]=labels[m.index(1.0)]+1
            
            
    new_ace_type_dict = {v:k for k,v in ace_type_dict.items()}
    for i in new_ace_type_dict:
        print(new_ace_type_dict[i]+'\t'+str(labels[i]))
        #print(j+'\t'+str(labels[i]))
    # print(len(ace_data_train))
    # print(len(ace_data_train[0]))
    # print(len(ace_data_train[0][0]))
    #
    # sys.exit()
    
    
    ace_data_train_file.close()
    ace_data_train_labels_file.close()
    ace_data_test_file.close()
    ace_data_test_labels_file.close()
    print('---------------------------end--------------------------------')

if __name__ == "__main__":
    
    #get_train_test_data()

#     get_ace_data_info()

    #     p = "刚刚宣誓就任的 行政院长张俊雄也应邀参加成立典礼并且致词表示，我们的政治文化过去是以对抗、对立没有合 作的文化，以致于付出了很大的代价，但是现在环境不同了，应该以合作代替对抗"
    #     result = NLPIR_ParagraphProcess(p, 1).split(' ')
    #     for i in range(len(result)):
    #         print(result[i])
    #         t=result[i].split('/')
    #         print(get_partofspeech(t[1]))
    #     print(result)
    #     sys.exit()
    #
    #
    #     test_list=['北', '韩', '代表团', '成员', '则', '是', '向', '记者', '表示', '，', '会谈', '的', '', '。', '气氛', '严肃', '，', '但', '是', '十分', '具有', '建设性', '']
    #     regular =['',' ',',','.','!','?',':',';','"','',' ','，','。','！','？','：','；','“','”']
    # #     regular_eng=['',' ',',','.','!','?',':',';','"']
    # #     regular_ch=['',' ','，','。','！','？','：','；','“','”']
    #     for i in range(len(test_list)):
    #         if test_list[i] in regular:
    #             print(test_list[i])
    #
    #     sys.exit()

#     ace_train_path = "../ace05/data/Chinese/"
#     word2vec_file = "./corpus_deal/ace_train_corpus.bin"
#     ace_data_pkl_path = './corpus_deal/ace_data3/ace_data.pkl'
#     ace_label_pkl_path = './corpus_deal/ace_data3/ace_data_labels.pkl'
#  
#     ace_data_pkl_process(ace_train_path, word2vec_file, ace_data_pkl_path, ace_label_pkl_path)


#     data_file = open(ace_data_pkl_path, 'rb')
#
#     data1 = pickle.load(data_file)
#     pprint.pprint(data1)
#
#     labels_file = open(ace_label_pkl_path, 'rb')
#     data2 = pickle.load(labels_file)
#     pprint.pprint(data2)
#
#     data_file.close()
#     labels_file.close()

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

#    print("-----------------------end----------------------")


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
