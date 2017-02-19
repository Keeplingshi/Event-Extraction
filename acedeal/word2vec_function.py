#coding:utf-8
'''
Created on 2017年1月24日
word2vec处理
@author: chenbin
'''
#import word2vec
from gensim.models import word2vec  
import logging 

if __name__ == "__main__":
    print("-----------------------start----------------------")
#     logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)  
#     sentences =word2vec.Text8Corpus("./corpus_deal/ace_train_corpus.txt")  # 加载语料  
#     model =word2vec.Word2Vec(sentences, size=200,min_count=1)  #训练skip-gram模型，默认window=5 
#     print(model)
    # 计算两个词的相似度/相关程度  
#     try:  
#         y1 = model.similarity("今天", "今日")  
#     except KeyError:  
#         y1 = 0  
#     print("相似度为：", y1  )
#     print("-----\n")
       
    # 保存模型，以便重用  
#    model.save("./corpus_deal/ace_train_corpus.model")  
    # 对应的加载方式  
    #model_2 =word2vec.Word2Vec.load("./corpus_deal/text8.model")  
       
    # 以一种c语言可以解析的形式存储词向量  
#    model.save_word2vec_format("./corpus_deal/ace_train_corpus.bin", binary=True)  
#     # 对应的加载方式  
#     model_3 =word2vec.Word2Vec.load_word2vec_format("./corpus_deal/result0.bin",binary=True)
#     print(model_3["战火连连"])
#     try:  
#         y1 = model.similarity("战火", "战火连连")  
#     except KeyError:  
#         y1 = 0  
#     print("相似度为：", y1  )
#     print("-----\n")
#     model = gensim.models.Word2Vec.load_word2vec_format('./corpus_deal/vectorsSougou.bin', binary=True,encoding="ISO-8859-1")  # C binary format
#     print(model["中国".encode('UTF8')])
#     model = word2vec.load('./corpus_deal/vectorsSougou.bin',encoding="ISO-8859-1")
#     print(model['中国'])

    model =word2vec.Word2Vec.load_word2vec_format("./corpus_deal/ace_train_corpus.bin",binary=True)
    #print(model["被处"])
    try:  
        y1 = model.similarity("中国", "中华")  
    except KeyError:  
        y1 = 0  
    print("相似度为：", y1  )
    print("-----\n")

    print("-----------------------end------------------------")