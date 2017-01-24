#coding:utf-8
'''
Created on 2017年1月24日
lstm
@author: chenbin
'''

from gensim.models import word2vec
from acedeal.nlpir import *
from acedeal.pre_process_ace import *


if __name__ == "__main__":
    print("-----------------------start----------------------")
    
    model =word2vec.Word2Vec.load_word2vec_format("./corpus_deal/ace_train_corpus.bin",binary=True)
    
    ace_train_path="../ace_experiment/train/"
    ace_train_list=get_ace_event_list(ace_train_path)
    for ace_info in ace_train_list:
        ace_text_list=NLPIR_ParagraphProcess(ace_info.text,0).split(' ')
        for word in ace_text_list:
            #读取词向量，如果没有该单词，则None
            try:  
                word_vector = model[word] 
            except KeyError:  
                word_vector = None  
            
            # 
            if word_vector is not None:
                pass
            else:
                print(word)
            
            
        #print(ace_text_list)
        #print(ace_info.toString())


    print("-----------------------end----------------------")
