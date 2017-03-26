'''
Created on 2017年2月26日
处理英文ACE语料 
@author: chenbin
'''
from model.tensorflow.data_process import xml_parse
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
    
    event_nodes = xml_parse.xml_parse_base.get_xmlnode(None,root, 'event')
    for node in event_nodes:
        R_element = ACE_info()
        # 获取事件id
        R_element.id=xml_parse.xml_parse_base.get_attrvalue(None, node, 'ID')
        # 获取事件子类型
        R_element.sub_type = xml_parse.xml_parse_base.get_attrvalue(None,node, 'SUBTYPE')
        #获取事件mention
        mention_nodes = xml_parse.xml_parse_base.get_xmlnode(None,node, 'event_mention')
        for mention_node in mention_nodes:
            # 获取事件所在语句
            mention_ldc_scope=xml_parse.xml_parse_base.get_xmlnode(None,mention_node, 'ldc_scope')
            mention_ldc_scope_charseq=xml_parse.xml_parse_base.get_xmlnode(None,mention_ldc_scope[0], 'charseq')
            R_element.text=xml_parse.xml_parse_base.get_nodevalue(None,mention_ldc_scope_charseq[0],0).replace("\n", " ")
            
            # 获取事件触发词
            mention_anchor=xml_parse.xml_parse_base.get_xmlnode(None,mention_node, 'anchor')
            mention_anchor_charseq=xml_parse.xml_parse_base.get_xmlnode(None,mention_anchor[0], 'charseq')
            R_element.trigger=xml_parse.xml_parse_base.get_nodevalue(None,mention_anchor_charseq[0],0).replace("\n", " ")
            
        R.append(R_element)
        
    return R

'''
获取ACE事件
'''
def get_ace_event_list(ace_file_path):
    ace_list = []

    for filename in os.listdir(ace_file_path):
        # adj文件夹所在地
        adj_file_path = os.path.join(ace_file_path, filename, 'timex2norm')
        for apf_file in os.listdir(adj_file_path):
            # 获取.apf.xml的文件
            if ".apf.xml" == apf_file[-8:]:
                # apf文件
                apf_file_path = os.path.join(adj_file_path, apf_file)
                ace_info_list = extract_ace_info(apf_file_path)
                ace_list.extend(ace_info_list)

    return ace_list



if __name__ == "__main__":
    print("-----------------------start----------------------")

    ace_train_path = '../../ace_en_experiment/test/'
    ace_train_list=get_ace_event_list(ace_train_path)
    for ace_info in ace_train_list:
        if ace_info.trigger =='winning':
            print(ace_info.text)

    print(len(ace_train_list))
#     num = 0
#     for filename in os.listdir(ace_file_path):
#         # adj文件夹所在地
#         adj_file_path = os.path.join(ace_file_path, filename, 'timex2norm')
#         for apf_file in os.listdir(adj_file_path):
#             # 获取.apf.xml的文件
#             if ".apf.xml" == apf_file[-8:]:
#                 num = num + 1
# #             if ".apf.xml" in apf_file:
# #                 # apf文件
# #                 apf_file_path = os.path.join(adj_file_path, apf_file)
# #                 num = num + 1
# #                 ace_info_list = extract_ace_info(apf_file_path)
# #                 ace_list.extend(ace_info_list)
#
#     print(num)

    print("-----------------------end----------------------")
