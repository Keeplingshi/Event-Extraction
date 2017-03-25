# coding:utf-8

from lxml import etree
import jieba
import pickle
import re
from xml.dom import minidom
from model.tensorflow.data_process import xml_parse
from model.tensorflow.data_process import type_index
import sys


homepath='D:/Code/pycharm/Event-Extraction/'
acepath=homepath+'ace05/data/English/'
# trainpath=homepath+'/ace_en_experiment/train/'
# testpath=homepath+'/ace_en_experiment/test/'
# devpath=homepath+'/ace_en_experiment/dev/'

def get_text_from_sgm(sgm_file):
    foldorname = ""
    if '/bn/' in sgm_file:
        foldorname = "bn"
    elif '/nw/' in sgm_file:
        foldorname = "nw"
    else:
        foldorname = "wl"

    text = ""
    doc = minidom.parse(sgm_file)
    root = doc.documentElement

    if foldorname == "bn":
        turn_nodes = xml_parse.xml_parse_base.get_xmlnode(None, root, 'TURN')
        for turn_node in turn_nodes:
            text += xml_parse.xml_parse_base.get_nodevalue(None, turn_node, 0).replace("\n", "")

    elif foldorname == "nw":
        text_node = xml_parse.xml_parse_base.get_xmlnode(None, root, 'TEXT')[0]
        text += xml_parse.xml_parse_base.get_nodevalue(None, text_node, 0).replace("\n", "")

    else:
        post_node = xml_parse.xml_parse_base.get_xmlnode(None, root, 'POST')[0]
        text += xml_parse.xml_parse_base.get_nodevalue(None, post_node, 4).replace("\n", "")

    return text


"""
替换字符串中对应位置
"""
def str_replace_by_position(str,replacestr,start,end):
    tmp1=str[:start]
    #tmp2=str[start:end]
    tmp3=str[end:]
    tmp=tmp1+replacestr+tmp3
    return tmp


"""
读取apf文件内容
"""
def read_documnet(file_name,wordlist,phrase_posi_dict):
    # 获取apf文件位置
    apf_filename = file_name+'.apf.xml'
    apf_f=open(apf_filename, 'rb')
    apf_content=apf_f.read()

    #sgm文件内容
    sgm_filename = file_name + ".sgm"
    sgm_content = open(sgm_filename, 'rb').read().decode('utf-8')
    # 除去尖括号等标签，只留下内容
    otherstr = re.findall("<[^>]+>|</[^>]+>", sgm_content)
    for rmstr in otherstr:
        sgm_content = sgm_content.replace(rmstr, '')

    try:
        doc = etree.fromstring(apf_content)
        start_end_type_list = []

        for i in doc.xpath("//event"):
            assert len(i.xpath(".//anchor")) > 0, 'len(i.xpath(".//anchor"))>0报错'
            cur_ele = i.xpath(".//anchor")

            event_type = i.xpath("./@TYPE")[0] + '.' + i.xpath("./@SUBTYPE")[0]
            event_num = type_index.EVENT_MAP[event_type]

            for anchor in cur_ele:
                start = int(anchor.xpath("./charseq/@START")[0])
                end = int(anchor.xpath("./charseq/@END")[0])+1
                trigger_str=anchor.xpath("./charseq/text()")[0]
                if trigger_str!='Q&A':
                    assert trigger_str == sgm_content[start:end], "触发词不匹配"

                #触发词为短语的情况
                if '\n' in trigger_str:
                    trigger_str=trigger_str.replace('\n',' ')

                if ' ' in trigger_str:
                    trigger_tmp1=trigger_str.replace(' ','_')   #下划线，短语，有词向量
                    trigger_tmp2 = trigger_str.replace(' ', '-')#中间线，短语，有词向量
                    if trigger_tmp1 in wordlist:
                        sgm_content = str_replace_by_position(sgm_content, trigger_tmp1, start, end)
                    elif trigger_tmp2 in wordlist:
                        sgm_content = str_replace_by_position(sgm_content, trigger_tmp2, start, end)
                    else:
                        #如果二者都没有，选取其中某个词来做
                        trigger_tmp3=trigger_str.replace('\n', ' ')
                        if trigger_tmp3 in phrase_posi_dict.keys():
                            new_trigger=trigger_tmp3.split(' ')[int(phrase_posi_dict[trigger_tmp3])-1]
                            new_trigger_start=trigger_tmp3.find(new_trigger)
                            start+=new_trigger_start
                            end=start+len(new_trigger)
                            #print(trigger_tmp3,new_trigger,start,end)
                        else:
                            continue

                #print(trigger_str,start,end,event_num)
                start_end_type_list.append((start,end,event_num))
        #print(start_end_type_list)
        return sgm_content,start_end_type_list

    except Exception as e:
        print(e)
        print(file_name, 'droped')

    return None

def content2vec(sgm_content,start_end_type_list,wordlist):
    for (start,end,event_num) in start_end_type_list:
        word=sgm_content[start:end].replace('\n',' ').lower()
        if word in wordlist:
            pass
        else:
            pass


def str_process(ss):
    strlist=ss.split(' ')
    k=len(strlist)
    s=strlist[0]
    for i in range(1,k-1):
        s=s+' '+strlist[i]
    return s,strlist[k-1]

if __name__ == '__main__':
    # ss='1 2 3 4'
    # a,b=str_process(ss)
    # print(a)
    # print(b)

    # d = {'name':1,'age':2,'sex':3}
    # if 'namea' in d.keys():
    #     print('111')
    # else:
    #     print('222')
    # # print(d['name'])
    # sys.exit()

    filename1=acepath+'/nw/timex2norm/AFP_ENG_20030323.0020'
    filename2=acepath+'/nw/timex2norm/AFP_ENG_20030509.0345'
    filename3 = acepath + '/bc/timex2norm/CNN_IP_20030330.1600.05-2'
    filename4 = acepath + 'bn/timex2norm/CNN_ENG_20030424_113549.11'
    filename5 = acepath + 'un/timex2norm/alt.corel_20041228.0503'
    filename6 = acepath + '/wl/timex2norm/AGGRESSIVEVOICEDAILY_20041226.1712'
    filename7 = acepath + '/cts/timex2norm/fsh_29191'
    filename8=acepath+'un/timex2norm/alt.obituaries_20041121.1339'


    wordlist_file = homepath + '/ace05/word2vec/wordlist'
    wordlist = [i.replace('\n', '') for i in open(wordlist_file, 'r')]
    phrase_posi_file=homepath+'/ace05/word2vec/phrase_posi.txt'
    phrase_posi_dict={}
    for i in open(phrase_posi_file, 'r'):
        a,b=str_process(i.replace('\n', ''))
        phrase_posi_dict[a]=b

    # sgm_content,start_end_type_list=read_documnet(filename8,wordlist,phrase_posi_dict)
    # if content2vec(sgm_content,start_end_type_list,wordlist)==1:
    #     print('111111111111111111111111')
    # sys.exit()

    doclist=homepath+'/ace05/new_filelist_ACE_full.txt';
    f_list=[i.replace('\n','') for i in open(doclist,'r')]
    k=0
    for i in f_list:
        path=acepath+i
        sgm_content,start_end_type_list=read_documnet(path,wordlist,phrase_posi_dict)
        content2vec(sgm_content,start_end_type_list,wordlist)
        k+=1
    print(k)
    #read_documnet(filename1)

    # read_documnet(filename2)

