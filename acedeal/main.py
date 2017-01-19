#coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

from acedeal.pre_process_ace import extract_info


if __name__ == "__main__":
    
    save_path = "./ch.txt"
    apf_file_path = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.apf.xml"
    sgm_file_path = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.sgm"
    
    ace_info_list=extract_info(apf_file_path,sgm_file_path)
    
    for ace_info in ace_info_list:
        ace_info.toString()
    
#     language = "ch"  # you can change en to "ch" if the input in a chinese data set
# 
#     # First-Part: extraction ACE mention from raw dataset
#     # to gain all the mention from files , you should change the file name by yourself
#     other_save_path = "./ch.txt"
#     # mention_save_path = "out/ace2005/en_wl.txt"
#     directory = "../../ace05/data/Chinese/bn/adj"
# 
#     ace_parse = ACE_parse(other_save_path, directory, language)
#     ace_parse.get_other()
#     ace_parse.get_info()