#coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

from acedeal.pre_process_ace import get_ace_event_list, save_ace_event_list


if __name__ == "__main__":
    
    save_path = "./ch.txt"
    ace_file_path = "../ace05/data/Chinese/"
#     ace_file_path = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.apf.xml"
#     sgm_file_path = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.sgm"
    
    ace_list=get_ace_event_list(ace_file_path)
    
    save_ace_event_list(ace_list,save_path)
    
    