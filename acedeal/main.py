#coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

from acedeal.pre_process_ace import *
from jpype import *
import os.path


if __name__ == "__main__":
    
    save_path = "./ace_corpus.txt"
    ace_file_path = "../ace05/data/Chinese/"
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
    jarpath = os.path.join(os.path.abspath('..'),'lib')
    print(jarpath)
    startJVM(getDefaultJVMPath(),"-ea", "-Djava.class.path=%s" % (jarpath + 'jpype.jar'))  
#     String content="8月19日晚，315国道德令哈段(480KM+700M)发生一起交通事故";
#     String result=NlpirMethod.NLPIR_ParagraphProcess(content,1);
#     System.out.println(result);

    jprint = java.lang.System.out.println
    NlpirMethod=JClass("jpype.JpypeDemo")
    result=NlpirMethod.NLPIR_ParagraphProcess("8月19日晚，315国道德令哈段(480KM+700M)发生一起交通事故",1);
    jprint(result) 
    shutdownJVM()
#     JDClass = JClass("jpype.JpypeDemo")  
#     jd = JDClass()  
#     #jd = JPackage("jpype").JpypeDemo() #两种创建jd的方法  
#       
#     jprint(jd.sayHello("waw"))  
#     jprint(jd.calc(2,4))
    