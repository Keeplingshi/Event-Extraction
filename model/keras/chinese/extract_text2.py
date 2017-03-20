#coding=utf8

from lxml import etree  # @UnresolvedImport
import os

import nltk
import itertools
import re
import json
import jieba
import pickle
from cb.chinese.type_index import EVENT_MAP

vocabulary_size = 10000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
type_num=34


'''为文本分类提取数据'''
def content2wordvec(text_content,start_end_type_list):
    jieba.load_userdict('./others/train_dict.dict')
    s_e_sorted=sorted(start_end_type_list,key=lambda x:x[0])
    tmp_set=dict()
    for i,j,k in s_e_sorted:
        if (i,j) not in tmp_set:
            tmp_set[(i,j)]=[]
        tmp_set[(i,j)].append(k)
    s_e_sorted=sorted(tmp_set.iteritems(),key=lambda x:x[0])
    new_text=''
    tmp_i=0
    train_data=[]
    for t in s_e_sorted:
        i,j=t[0]
        label_list=t[1]
        sent1=text_content[tmp_i:i]
        label=[0 for tt in range(type_num)]
        label[type_num-1]=1
        tmp_i=j+1
        train_data.append((sent1,label))

        sent2=text_content[i:j+1]
        label=[0 for tt in range(type_num)]
        for i in label_list:
            label[i]=1
        train_data.append((sent2,label))
    rs_train_data=[]
    def clean(x):
        x=x.replace('\n','')
        x=x.replace(' ','')
        return x
    for i in train_data:
        x=clean(i[0])
        y=i[1]
        x_list=[i for i in jieba.cut(x)]
        if len(x_list)>5:
            if y[type_num-1]==0:
                #待分类的句子
                rs_train_data.append((x_list,y))
            else:
                #不知道分类的句子，但是需要根据句号进行拆分
                tmp_ii=0
                for index,i in enumerate(x_list):
                    if i==u'。':
                        if len(x_list[tmp_ii:index])>5:
                            rs_train_data.append((x_list[tmp_ii:index],y))
                        tmp_ii=index+1
                if len(x_list[tmp_ii:-1])>5:
                    rs_train_data.append((x_list[tmp_ii:-1],y))
    return rs_train_data


def read_answer(filename_prefix):
    
    tag_filename = filename_prefix+'.apf.xml'
    tag_filepath=os.path.join('./train/tag',tag_filename)
    tag_f=open(tag_filepath)
    tag_content=tag_f.read().decode('utf8')

    text_filename= filename_prefix+".txt"
    text_filepath=os.path.join('./text2',text_filename)
    text_f=open(text_filepath)
    text_content=text_f.read().decode('utf8')

    text2_filename= filename_prefix+".sgm"
    text2_filepath=os.path.join('./train/text',text2_filename)
    text2_f=open(text2_filepath)
    text2_content=text2_f.read().decode('utf8')

    doc=etree.fromstring(text2_content)

    sentence=doc.xpath('//TEXT//text()')

    begin_sen=''.join(sentence)
    begin_sen=begin_sen.replace('\n','')
    begin_sen=begin_sen[:3]
    begin_index=text_content.find(begin_sen)

    try:
        assert begin_index!=-1
        begin_index-=1
        new_text=text_content[begin_index:]
        doc=etree.fromstring(tag_content)
        trigger=[]
        
        start_end_type_list = []
        for i in doc.xpath("//event"):
            assert len(i.xpath(".//ldc_scope"))>0
            cur_ele = i.xpath(".//ldc_scope")
            event_type = i.xpath("./@TYPE")[0]+'.'+i.xpath("./@SUBTYPE")[0]
            event_num = EVENT_MAP[event_type]
            for ldc_scope in cur_ele:
                start = ldc_scope.xpath("./charseq/@START")[0]
                end = ldc_scope.xpath("./charseq/@END")[0]
                start = int(start)-begin_index
                end = int(end)- begin_index
                real_str = ldc_scope.xpath("./charseq/text()")[0]
                my_str = new_text[start:end+1]
                assert real_str == my_str
                start_end_type_list.append((int(start),int(end),event_num))
        return content2wordvec(new_text,start_end_type_list)
    except Exception as e:
        print(e)
        print(filename_prefix,'droped')
        return []

    
def prepare_data():
    train_data=[]
    f=lambda x:x[:x.rfind('.')]
    f_list=[f(i) for i in os.listdir('./text2')]
    for i in f_list:
        tmp=read_answer(i)
        if len(tmp)>0:
            train_data.extend(tmp)
    train_data=[i for i in train_data if len(i[0])>0]
    sentence3_f=open('sentence3.txt','w')
    for i in train_data:
        print>>sentence3_f ,' '.join(i[0]).encode('utf8')
        print>>sentence3_f ,i[1]
    rs_f=open('./others/event_class.data','wb')
    pickle.dump(train_data,rs_f)


if __name__ == '__main__':
    prepare_data()
    # read_answer("XIN20001231.0200.0020")
