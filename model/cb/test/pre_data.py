# coding:utf-8
'''
Created on 2017年3月15日
数据处理
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

homepath='D:/Code/pydev/EventExtract/'

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
# 
#     # txt文件是将sgm文件移除尖括号<>所得
#     text_filename= filename_prefix+".txt"
#     text_filepath=corpus_path+text_filename
#     text_f=open(text_filepath, 'rb')
#     text_content=text_f.read().decode('utf8')
#     text_content=text_content.replace('\n','')
# 
    text_filename= filename_prefix+".sgm"
    text_filepath=corpus_path+text_filename
    text_content=get_text_from_sgm(text_filepath)
#     text2_f=open(text2_filepath, 'rb')
#     text2_content=text2_f.read().decode('utf8')
#     
#     print(text2_content)
# 
#     doc=etree.fromstring(text2_content)
#     
# 
#     sentence=doc.xpath('//TEXT//text()')
#     
#     begin_sen=''.join(sentence)
#     
#     begin_sen=begin_sen.replace('\n','')
#     begin_sen=begin_sen[:3]
#     begin_index=text_content.find(begin_sen)
# 
#     try:
#         assert begin_index!=-1,'begin_index==-1'
#         begin_index-=1
#         new_text=text_content[begin_index:]
#         doc=etree.fromstring(tag_content)
#         trigger=[]
#         
#         start_end_type_list = []
#         for i in doc.xpath("//event"):
#             assert len(i.xpath(".//anchor"))>0,'len(i.xpath(".//anchor"))>0报错'
#             cur_ele = i.xpath(".//anchor")
#             event_type = i.xpath("./@TYPE")[0]+'.'+i.xpath("./@SUBTYPE")[0]
#            
#             for anchor in cur_ele:
#                 start = anchor.xpath("./charseq/@START")[0]
#                 end = anchor.xpath("./charseq/@END")[0]
#                 start = int(start)-begin_index
#                 end = int(end)- begin_index
#                 real_str = anchor.xpath("./charseq/text()")[0].replace('\n','')
#                 my_str = new_text[start:end+1].replace('\r','')
#                 assert real_str == my_str,real_str+'\t'+my_str+'报错'
#                 start_end_type_list.append((int(start),int(end)))
#         
#         print(new_text)
#         return []
#     except Exception as e:
#         print(e)
#         print(filename_prefix,'droped')
#         return []


if __name__ == '__main__':
    print('--------------------------main start-----------------------------')
    
    read_answer('/wl/adj/LIUYIFENG_20050126.0844')
    
#     path=homepath+'/ace_ch_experiment/corpus/'+'/wl/adj/LIUYIFENG_20050126.0844.sgm'
#     text=get_text_from_sgm(path)
#     print(text)

    print('--------------------------main end-----------------------------')
