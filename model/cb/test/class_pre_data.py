# coding:utf-8
'''
Created on 2017年3月16日
事件分类  数据处理
二分类，one-hot向量
a) word转为词序号
从训练语料统计获得单词列表，并按照词频从大到小排序，序号从0开始，然后将句子中单词全部转为序号
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
import nltk
import itertools
import json
from cb.test.type_index import EVENT_MAP

homepath='D:/Code/pydev/EventExtract/'
type_num=34


'''为文本分类提取数据'''
def content2wordvec(text_content,start_end_type_list):
    jieba.load_userdict(homepath+'/ace_ch_experiment/trigger.dict')
    s_e_sorted=sorted(start_end_type_list,key=lambda x:x[0])
    word_list=[t for t in jieba.cut(text_content)]
    
    tmp_i=0
    
    train_data=[]
    sen_list=[]
    label_list=[0 for tt in range(type_num)]
    for t in word_list:
        if t==u'。':
            if 1 not in label_list:
                label_list[type_num-1]=1
            train_data.append((sen_list,label_list))
            sen_list=[]
            label_list=[0 for tt in range(type_num)]
        else:
            sen_list.append(t)
        
        tmp_j=tmp_i+len(t)-1
        for start_end_type in s_e_sorted:
            if tmp_i==start_end_type[0] and tmp_j==start_end_type[1]:
                label_list[start_end_type[2]]=1
            
        tmp_i=tmp_j+1
    
    return train_data


def get_text_from_sgm(sgm_file):
    foldorname=""
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
            
            event_type = i.xpath("./@TYPE")[0]+'.'+i.xpath("./@SUBTYPE")[0]
            event_num = EVENT_MAP[event_type]
            
            ldc_scope_ele=i.xpath(".//ldc_scope")
            for ldc_scope in ldc_scope_ele:
                sentence_str=ldc_scope.xpath("./charseq/text()")[0].replace('\n','')
                sen_list.append(sentence_str)
            
            cur_ele = i.xpath(".//anchor")
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
    doclist=homepath+'/ace_ch_experiment/doclist/ACE_Chinese_all';
    f_list=[i.replace('\n','') for i in open(doclist,'r')]
    for i in f_list:
        tmp=read_answer(i)
        if len(tmp)>0:
            train_data.extend(tmp)
    train_data=[i for i in train_data if len(i[0])>0]
    sentence3_f=open('./chACEdata/sentence3.txt','w', encoding='utf8')
    for i in train_data:
        sentence3_f.write(' '.join(i[0]))
        sentence3_f.write('\n')
        sentence3_f.write(str(i[1]))
        sentence3_f.write('\n')
    rs_f=open('./chACEdata/event_class.data','wb')
    pickle.dump(train_data,rs_f)

if __name__ == '__main__':
    
    prepare_data()
    
#     str='澳门特区行政长官办公室、各司长办公室、行政会以及政府总部、辅助部门将从30号起从原来的 临时办公处迁出，正式在政府总部办公。 澳门特区政府总部位于前澳督府及政务司大楼，由三座建筑物和花园组成，面积约1万平方米，经 过近7个月的修缮，建筑物外观基本保留原有风格，对内部年久失修的天花板和下水道依照消防和 现代技术要求进行重新改造，而具有殖民色彩的照片等物品已送往澳门博物馆收藏。修缮改造后 的政府总部整体给人以明亮、简洁、大方的感觉，其中改动最大的是原立法会主席办公室现在改 为待客厅，并在厅内鲜花装饰立体莲花图案，成为最具澳门特区色彩部分。另外还新开辟了记者 室，安装了传真机等设备，为记者提供方便。据了解特区政府原来临时所使用的宋裕生广场办公 用房将尽快退租交还业主。中央台驻澳门记者报道。'
#     start_end_type_list=[(4,7,1),(49, 50, 28)]
#     content2wordvec(str, start_end_type_list)
#     
#     print('----------------------------------------------')
    
#     start_end_type_list=[(0,3),(49, 50)]
#     content2wordvec2(str, start_end_type_list)
    #read_answer("/wl/adj/DAVYZW_20041223.1020")
