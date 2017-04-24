'''
Created on 2017年3月20日

@author: chenbin
'''
import os
from lxml import etree  # @UnresolvedImport
import jieba
import pickle
import re
import sys
from xml_parse import xml_parse_base
from xml.dom import minidom
from sklearn import cross_validation
from gensim.models import word2vec
import random
import numpy as np
from model.keras.test2.type_index import EVENT_MAP

homepath='D:/Code/pydev/EventExtract/'

def content2wordvec(text_content,start_end_type_list):
    jieba.load_userdict(homepath+'/ace_ch_experiment/trigger.dict')
    s_e_sorted=sorted(start_end_type_list,key=lambda x:x[0])
    label=[]
    tmp_i=0

    word_list=[t for t in jieba.cut(text_content)]
    for t in word_list:
        tmp_j=tmp_i+len(t)-1
        flag=0
        for start_end_type in s_e_sorted:
            if tmp_i==start_end_type[0] and tmp_j==start_end_type[1]:
                flag=1
        
        if flag==1:
            label.append(start_end_type[2])
        else:
            label.append(0)

        tmp_i=tmp_j+1


    assert len(word_list)==len(label),'word_list与label长度不相等'

    return (word_list,label)


def get_text_from_sgm(sgm_file):
    if '/bn/' in sgm_file:
        foldorname="bn"
    elif '/nw/' in sgm_file:
        foldorname="nw"
    else:
        foldorname="wl"

    text=""
    doc = minidom.parse(sgm_file)
    root = doc.documentElement

    if foldorname=="bn":
        turn_nodes = xml_parse_base.get_xmlnode(None,root, 'TURN')
        for turn_node in turn_nodes:
            text+=xml_parse_base.get_nodevalue(None,turn_node,0).replace("\n", "")

    elif foldorname=="nw":
        text_node = xml_parse_base.get_xmlnode(None,root, 'TEXT')[0]
        text+=xml_parse_base.get_nodevalue(None,text_node,0).replace("\n", "")

    else:
        post_node=xml_parse_base.get_xmlnode(None,root, 'POST')[0]
        text+=xml_parse_base.get_nodevalue(None,post_node,4).replace("\n", "")

    return text

'''
读入文件名称，获取词向量
'''
def read_answer(filename_prefix):

    corpus_path=homepath+'/ace_ch_experiment/corpus/'

    # 获取apf文件位置
    tag_filename = filename_prefix+'.apf.xml'
    tag_filepath=corpus_path+tag_filename
    tag_f=open(tag_filepath, 'rb')
    tag_content=tag_f.read()

    text_filename= filename_prefix+".sgm"
    text_filepath=corpus_path+text_filename
    text_content=get_text_from_sgm(text_filepath)

    try:
        doc=etree.fromstring(tag_content)
        trigger_list=[]
        sen_list=[]
        event_list=[]

        start_end_type_list = []

        for i in doc.xpath("//event"):
            assert len(i.xpath(".//anchor"))>0,'len(i.xpath(".//anchor"))>0报错'
            cur_ele = i.xpath(".//anchor")

            event_type = i.xpath("./@TYPE")[0]+'.'+i.xpath("./@SUBTYPE")[0]
            event_num = EVENT_MAP[event_type]

            ldc_scope_ele=i.xpath(".//ldc_scope")
            for ldc_scope in ldc_scope_ele:
                sentence_str=ldc_scope.xpath("./charseq/text()")[0].replace('\n','')
                sen_list.append(sentence_str)

            for anchor in cur_ele:
                trigger_str = anchor.xpath("./charseq/text()")[0].replace('\n','')
                trigger_list.append(trigger_str)
                event_list.append(event_num)

        assert len(trigger_list)==len(sen_list),'触发词数目与句子数目不相等'
        assert len(trigger_list)==len(event_list),'触发词数目与事件类型数目不相等'

        trilen=len(trigger_list)
        for i in range(trilen):
            # 触发词在事件句中的位置
            tri_position=sen_list[i].index(trigger_list[i])
            # 事件句在文章中的位置
            sen_position=text_content.index(sen_list[i])

            assert tri_position!=-1,'（'+trigger_list[i]+'）不在（'+sen_list[i]+'）中'
            assert sen_position!=-1,'（'+sen_list[i]+'）不在（'+text_content+'）中'

            #触发词在文章中的位置
            tri_start=tri_position+sen_position
            tri_end=tri_start+len(trigger_list[i])-1

            start_end_type_list.append((int(tri_start),int(tri_end),event_list[i]))

        return content2wordvec(text_content,start_end_type_list)
    except Exception as e:
        print(e)
        print(filename_prefix,'droped')
        return []

def prepare_data():
    train_data=[]
#     f=lambda x:x[:x.rfind('.')]
#     f_list=[f(i) for i in os.listdir('./text2')]
    doclist=homepath+'/ace_ch_experiment/doclist/ACE_Chinese_all';
    f_list=[i.replace('\n','') for i in open(doclist,'r')]

    for i in f_list:
        tmp=read_answer(i)
        if len(tmp)>1:
            train_data.append(tmp)
    train_data=[i for i in train_data if len(i[0])>0]
    rs_f=open('./chACEdata/class_pre_data2_1.txt','w', encoding='utf8')
    for item in train_data:
        word=item[0]
        strtemp=' '.join([i for i in word])
        rs_f.write(strtemp)
        rs_f.write('\n')
        rs_f.write(str(item[1]))
        rs_f.write('\n')
    new_train_data=[]
    for item in train_data:
        sentence=item[0]
        label=item[1]
        tmp_i=0
        for index,i in enumerate(sentence):
            if i==u'。':
                new_train_data.append((sentence[tmp_i:index],label[tmp_i:index]))
                tmp_i=index+1

    rs_f=open('./chACEdata/class_pre_data2_2.txt','w', encoding='utf8')
    for item in new_train_data:
        word=item[0]
        rs_f.write(' '.join([i for i in word]))
        rs_f.write('\n')
        rs_f.write(str(item[1]))
        rs_f.write('\n')
    rs_f=open('./chACEdata/class_pre_data2.data','wb')
    pickle.dump(new_train_data,rs_f)

def pre_word2vec_data():
    '''不用切词，值'''
    f=open('./chACEdata/class_pre_data2.data','rb')
    train_data=pickle.load(f)
#
    word2vec_file=homepath+'/acedeal/corpus_deal/ace_train_corpus.bin'
    model = word2vec.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

    x=[]
    y=[]
    for item in train_data:
        sen_vec=[]
        for word in item[0]:
            try:
                word_vector = model[word]
            except KeyError:
                word_vector = np.array([0.0 for i in range(200)])

            #如果存在词向量
            sen_vec.append(word_vector)

        x.append(sen_vec)
        #x.append([word_to_index[word] if word in word_to_index else word_to_index[unknown_token] for word in item[0]])
        y.append(item[1])
    X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(x,y,test_size=0.1, random_state=0)

    data=X_train,X_test,Y_train,Y_test
    f=open('./chACEdata/class_train_data.data','wb')
    pickle.dump(data,f)


# 规范句子长度
def padding_mask(x, y,max_len):

    y_form=[]
    for a in y:
        y_temp=[]
        for b in a:
            label=[0.0 for i in range(34)]
            label[b]=1.0
            y_temp.append(label)
        y_form.append(y_temp)


    X_train=[]
    Y_train=[]
    x_zero_list=[0.0 for i in range(200)]
    y_zero_list=[0.0 for i in range(34)]
    y_zero_list[0]=1.0
    for i, (x, y_form) in enumerate(zip(x, y_form)):
        if max_len>len(x):
            for j in range(max_len-len(x)):
                x.append(x_zero_list)
                y_form.append(y_zero_list)
        else:
            x=x[:max_len]
            y_form=y_form[:max_len]
        X_train.append(x)
        Y_train.append(y_form)
    return X_train,Y_train


def form_data():

    data_f = open('./chACEdata/class_train_data.data', 'rb')
    X_train,X_test,Y_train,Y_test=pickle.load(data_f)
    data_f.close()

    max_len=60
    X_train,Y_train=padding_mask(X_train,Y_train,max_len)
    X_test,Y_test=padding_mask(X_test,Y_test,max_len)

    data=X_train,Y_train,X_test,Y_test
    f=open('./chACEdata/class_train_form_data.data','wb')
    pickle.dump(data,f)

    print(np.array(X_train).shape)
    print(np.array(Y_train).shape)
    print(np.array(X_test).shape)
    print(np.array(Y_test).shape)


if __name__ == '__main__':
    print('--------------------------main start-----------------------------')
    #read_answer('/bn/adj/CTV20001228.1330.1196')
    prepare_data()

    pre_word2vec_data()

    form_data()

    # f=open('./chACEdata/class_train_data.data','rb')
    # X_train,X_test,Y_train,Y_test=pickle.load(f)
    #
    #
    # print(np.array(X_train).shape)
    # print(len(X_test[0]))
    # print(len(X_train[1]))
    # print(len(X_train[0][0]))
    # print(len(X_train[1][1]))
    # print(Y_test)
    #
    # l=[34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]
    # print(len(l))

#


#     m=read_answer('/wl/adj/CTV20001030.1330.0326.0844')
#     for i in range(len(m[1])):
#         if m[1][i]==1:
#             print(m[0][i])
# 
#     print(m[0])
#     print(m[1])
#     path=homepath+'/ace_ch_experiment/corpus/'+'/wl/adj/LIUYIFENG_20050126.0844.sgm'
#     text=get_text_from_sgm(path)
#     print(text)

    print('--------------------------main end-----------------------------')
