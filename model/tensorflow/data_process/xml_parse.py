#coding:utf-8
"""
Created on 2017年1月19日

@author: chenbin
"""

class xml_parse_base:

    def get_attrvalue(self, node, attrname):
        return node.getAttribute(attrname) if node else ''

    def get_nodevalue(self, node, index = 0):
        return node.childNodes[index].nodeValue if node else ''

    def get_xmlnode(self, node, name):
        return node.getElementsByTagName(name) if node else''


