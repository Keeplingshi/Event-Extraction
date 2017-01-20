#coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

from acedeal.xml_parse import xml_parse_base
import os
import re
from xml.dom import minidom

'''
ACE  event实体
'''
class ACE_info:
    # go gain the event mention from ACE dataset
    def __init__(self):
        self.id = None
        self.text = None
        self.trigger = None
        self.sub_type = None  # sub-type of this event
        
    def toString(self):
        return 'id:'+str(self.id)+'\t text:'+str(self.text)+'\t trigger:'+str(self.trigger)+'\t sub_type:'+str(self.sub_type)


'''
抽取单个apf.xml中的事件实体ACE_info
apf_file ：单个.apf.xml文件路径，如：apf_file = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.apf.xml"
'''
def extract_ace_info(apf_file):
    
    # 存储事件实体的list
    R=[]
    
    doc = minidom.parse(apf_file)
    root = doc.documentElement
    
    event_nodes = xml_parse_base.get_xmlnode(None,root, 'event')
    for node in event_nodes:
        R_element = ACE_info()
        # 获取事件id
        R_element.id=xml_parse_base.get_attrvalue(None, node, 'ID')
        # 获取事件子类型
        R_element.sub_type = xml_parse_base.get_attrvalue(None,node, 'SUBTYPE')
        #获取事件mention
        mention_nodes = xml_parse_base.get_xmlnode(None,node, 'event_mention')
        for mention_node in mention_nodes:
            # 获取事件所在语句
            mention_ldc_scope=xml_parse_base.get_xmlnode(None,mention_node, 'ldc_scope')
            mention_ldc_scope_charseq=xml_parse_base.get_xmlnode(None,mention_ldc_scope[0], 'charseq')
            R_element.text=xml_parse_base.get_nodevalue(None,mention_ldc_scope_charseq[0],0).replace("\n", "")
            
            # 获取事件触发词
            mention_anchor=xml_parse_base.get_xmlnode(None,mention_node, 'anchor')
            mention_anchor_charseq=xml_parse_base.get_xmlnode(None,mention_anchor[0], 'charseq')
            R_element.trigger=xml_parse_base.get_nodevalue(None,mention_anchor_charseq[0],0).replace("\n", "")
            
        R.append(R_element)
        
    return R


'''
抽取整个ace语料中的所有事件
ace_file_path ： ACE语料路径，如：ace_file_path = "../ace05/data/Chinese/"
'''
def get_ace_event_list(ace_file_path):
    ace_list=[]
    
    for filename in os.listdir(ace_file_path):
        # adj文件夹所在地
        adj_file_path=os.path.join(ace_file_path,filename,'adj')
        for apf_file in os.listdir(adj_file_path):
            # 获取.apf.xml的文件
            if ".apf.xml" in apf_file:
                # apf文件
                apf_file_path=os.path.join(adj_file_path,apf_file)
                ace_info_list=extract_ace_info(apf_file_path)
                ace_list.extend(ace_info_list)
                
    return ace_list


'''
保存ace事件到txt
ace_list：list of ACE_info
save_path：保存路径，如：save_path = "./ch.txt"
'''
def save_ace_event_list(ace_list,save_path):
    f_out = open(save_path, 'w',encoding="utf-8")
    for ace_info in ace_list:
        f_out.write(ace_info.toString())
        f_out.write('\n')
        
    f_out.close()


'''
抽取ACE语料所有文章内容，到txt文件中
ace_file_path： ACE语料路径，如：ace_file_path = "../ace05/data/Chinese/"
save_path：保存路径，如：save_path = "./ace_corpus.txt"
'''
def extract_corpus(ace_file_path,save_path):
    
    f_out = open(save_path, 'w',encoding="utf-8")
    
    for filename in os.listdir(ace_file_path):
        # adj文件夹所在地
        adj_file_path=os.path.join(ace_file_path,filename,'adj')
        for sgm_file in os.listdir(adj_file_path):
            # 获取.sgm 的文件
            if ".sgm" in sgm_file:
                text=""
                sgm_file_path=os.path.join(adj_file_path,sgm_file)
                doc = minidom.parse(sgm_file_path)
                root = doc.documentElement
                # text_node = xml_parse_base.get_xmlnode(None,root, 'TEXT')[0]
                # text_node = xml_parse_base.get_xmlnode(None,root, 'TEXT')[0]
                if filename=="bn":
                    turn_nodes = xml_parse_base.get_xmlnode(None,root, 'TURN')
                    for turn_node in turn_nodes:
                        #print(xml_parse_base.get_nodevalue(None,turn_node,0).replace("\n", ""))
                        text+=xml_parse_base.get_nodevalue(None,turn_node,0).replace("\n", "")
                        
                elif filename=="nw":
                    text_node = xml_parse_base.get_xmlnode(None,root, 'TEXT')[0]
                    text+=xml_parse_base.get_nodevalue(None,text_node,0).replace("\n", "")
                    
                else:
                    post_node=xml_parse_base.get_xmlnode(None,root, 'POST')[0]
                    text+=xml_parse_base.get_nodevalue(None,post_node,4).replace("\n", "")
                    print(text)
                    
                f_out.write(text)
                f_out.write('\n')
                
                
    f_out.close()

