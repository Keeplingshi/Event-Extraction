#coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

from acedeal.xml_parse import xml_parse_base
from xml.dom import minidom

class ACE_info:
    # go gain the event mention from ACE dataset
    def __init__(self):
        self.id = None
        self.text = None
        self.trigger = None
        self.sub_type = None  # sub-type of this event
        
    def toString(self):
        print('id:'+str(self.id)+'\t text:'+str(self.text)+'\t trigger:'+str(self.trigger)+'\t sub_type:'+str(self.sub_type))


def extract_info(apf_name,sgm_name):
    
    R=[]
    
    doc = minidom.parse(apf_name)
    root = doc.documentElement
    
    event_nodes = xml_parse_base.get_xmlnode(None,root, 'event')
    for node in event_nodes:
        R_element = ACE_info()
        R_element.id=xml_parse_base.get_attrvalue(None, node, 'ID')
        R_element.sub_type = xml_parse_base.get_attrvalue(None,node, 'SUBTYPE')
        mention_nodes = xml_parse_base.get_xmlnode(None,node, 'event_mention')
        for mention_node in mention_nodes:
            mention_ldc_scope=xml_parse_base.get_xmlnode(None,mention_node, 'ldc_scope')
            mention_ldc_scope_charseq=xml_parse_base.get_xmlnode(None,mention_ldc_scope[0], 'charseq')
            R_element.text=xml_parse_base.get_nodevalue(None,mention_ldc_scope_charseq[0],0).replace("\n", "")
            
            mention_anchor=xml_parse_base.get_xmlnode(None,mention_node, 'anchor')
            mention_anchor_charseq=xml_parse_base.get_xmlnode(None,mention_anchor[0], 'charseq')
            R_element.trigger=xml_parse_base.get_nodevalue(None,mention_anchor_charseq[0],0)
            
        R.append(R_element)
        
    return R
#         R = []
#         doc = minidom.parse(file_name)
#         root = doc.documentElement
# 
#         relation_nodes = self.get_xmlnode(root, 'relation')
#         for node in relation_nodes:
#             relation_type = self.get_attrvalue(node, 'TYPE')
#             relation_sub_type = self.get_attrvalue(node, 'SUBTYPE')
#             mention_nodes = self.get_xmlnode(node, 'relation_mention')
#             for mention_node in mention_nodes:
#                 R_element = ACE_info()
#                 R_element.type = relation_type
#                 R_element.sub_type = relation_sub_type
# 
#                 # gain the attribute info of mention
#                 mention_extent = self.get_xmlnode(mention_node, 'charseq')
#                 R_element.mention_pos[0] = int(self.get_attrvalue(mention_extent[0], 'START'))
#                 R_element.mention_pos[1] = int(self.get_attrvalue(mention_extent[0], 'END'))
#                 R_element.mention = self.get_nodevalue(mention_extent[0])
# 
#                 argument_nodes = self.get_xmlnode(mention_node, 'relation_mention_argument')
#                 for argument_node in argument_nodes:
#                     if self.get_attrvalue(argument_node, 'ROLE') == 'Arg-1':
#                         # gain the attribute info of arg1
#                         R_element.arg1_pos[0] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'START'))
#                         R_element.arg1_pos[1] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'END'))
#                         #relation_arg1 = get_nodevalue(get_xmlnode(argument_node, 'charseq')[0]).encode('utf-8', 'ignore')
#                         #print(R_element.arg1_pos[0],R_element.arg1_pos[1])
#                     elif self.get_attrvalue(argument_node, 'ROLE') == 'Arg-2':
#                         # gain the attribute info of age2
#                         R_element.arg2_pos[0] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'START'))
#                         R_element.arg2_pos[1] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'END'))
#                         #relation_arg2 = get_nodevalue(get_xmlnode(argument_node, 'charseq')[0]).encode('utf-8', 'ignore')
#                         #print(R_element.arg2_pos[0],R_element.arg2_pos[1])
#                 R_element.combine()
#                 R.append(R_element)
#         return R
