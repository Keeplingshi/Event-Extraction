#coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

import os.path
import string

import jpype

from acedeal.nlpir import *
from acedeal.pre_process_ace import *


if __name__ == "__main__":
    print("-----------------------start----------------------")
#     save_path = "./trigger.txt"
#     ace_file_path = "../ace05/data/Chinese/"
#     ace_file_path = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.apf.xml"
#     sgm_file_path = "../ace05/data/Chinese/nw/adj/XIN20001002.0200.0004.sgm"
    
    # 抽取所有事件
    # ace_list=get_ace_event_list(ace_file_path)
    
    # extract_corpus(ace_file_path,save_path)
    
    # 将ACE事件保存到文件
    # save_path = "./ch.txt"
    # save_ace_event_list(ace_list,save_path)
    
    
#     startJVM(getDefaultJVMPath(), "-ea")
#     java.lang.System.out.println("Hello World")
#     shutdownJVM()
    
    # 获取lib文件夹，即jar包所在路径
#     jarpath = os.path.join(os.path.abspath('..'),'lib\\nlpir_ppl.jar')
#     print(jarpath)
#     jpype.startJVM(jpype.getDefaultJVMPath(),"-ea", "-Djava.class.path=" + jarpath) 
# #     String content="8月19日晚，315国道德令哈段(480KM+700M)发生一起交通事故";
# #     String result=NlpirMethod.NLPIR_ParagraphProcess(content,1);
# #     System.out.println(result);
# 
#     jprint = jpype.java.lang.System.out.println
#     #n = jpype.JPackage('com').nlpir.OSInfo
#     n=jpype.JClass("com.nlpir.OSInfo")
#     print(n.getModulePath("8月19日晚，315国道德令哈段(480KM+700M)发生一起交通事故"))
#     #result=com.nlpir.NlpirMethod.NLPIR_ParagraphProcess("8月19日晚，315国道德令哈段(480KM+700M)发生一起交通事故",1);
#     #jprint(result) 
#     jpype.shutdownJVM()
#     JDClass = JClass("jpype.JpypeDemo")  
#     jd = JDClass()  
#     #jd = JPackage("jpype").JpypeDemo() #两种创建jd的方法  
#       
#     jprint(jd.sayHello("waw"))  
#     jprint(jd.calc(2,4))
    
#     p = "今天天气不错。"
#     print(NLPIR_ParagraphProcess(p,1))

#     save_path = "./corpus_deal/ace_corpus.txt"
#     out_path="./corpus_deal/result0.txt"
#     NLPIR_FileProcess(save_path,out_path,0)
    
#     for ace_info in ace_list:
#         f_out.write(ace_info.toString())
#         f_out.write('\n')
         

    #NLPIR_AddUserWord("今天天气  n")
#     p = "今天天气不错。"
#     print(NLPIR_ParagraphProcess(p,1))


    print("-----------------------end------------------------")

    