# coding:utf-8
'''
Created on 2017年1月19日

@author: chenbin
'''

import os.path
import string
import tensorflow as tf
#import jpype
import sys
import pickle
import pprint

# from acedeal.nlpir import *
# from acedeal.pre_process_ace import *
# 
# # 神经网络的参数
# n_input = 200  # 输入层的n
# n_steps = 1  # 28长度
# n_hidden = 128  # 隐含层的特征数
# n_classes = 34  # 输出的数量，因为是分类问题，这里一共有34个

if __name__ == "__main__":
    print("-----------------------start----------------------")
    
    
    labels_file = open('./corpus_deal/ace_data/ace_data_train_labels.pkl', 'rb')
    data2 = pickle.load(labels_file)
    k=0
    t=0
    for list in data2:
        flag=False
        t=t+1
        for label_list in list:
            if label_list.index(1.0)!=34:
                flag=True
                k=k+1
                break
        
        if flag==False:
            print(list)
    
    print(k)
    print(t)
    #pprint.pprint(data2)
    
    sys.exit()
    
#     sess = tf.InteractiveSession()
#     x = tf.placeholder("float", [None, n_steps, n_input])
#     y = tf.placeholder("float", [None, n_classes])
# 
#     W1 = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]), name="W1")
#     b1 = tf.Variable(tf.random_normal([n_classes]), name="b1")

#     saver = tf.train.Saver()
#     saver.restore(sess, "./ckpt_file/ace_bl.ckpt")
#
#     print(sess.run(W1))

#     print(sess.run(tf.is_variable_initialized(W1)))
#     print(tf.report_uninitialized_variables([W1]))
# 
#     sess.close()

#     ace_train_path="../ace_experiment/train/"
#     ace_train_list=get_ace_event_list(ace_train_path)
#     for ace_info in ace_train_list:
#         print(ace_info.toString())


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
