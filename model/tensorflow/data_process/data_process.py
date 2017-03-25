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
def read_documnet(file_name,wordlist):
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
        trigger_list = []
        sen_list = []
        event_list = []

        start_end_type_list = []

        for i in doc.xpath("//event"):
            assert len(i.xpath(".//anchor")) > 0, 'len(i.xpath(".//anchor"))>0报错'
            cur_ele = i.xpath(".//anchor")

            event_type = i.xpath("./@TYPE")[0] + '.' + i.xpath("./@SUBTYPE")[0]
            event_num = type_index.EVENT_MAP[event_type]


            for anchor in cur_ele:
                start = int(anchor.xpath("./charseq/@START")[0])
                end = int(anchor.xpath("./charseq/@END")[0])+1
                # print(start,end)
                # print(sgm_content1[start:end])
                trigger_str=anchor.xpath("./charseq/text()")[0]
                #print(trigger_str,sgm_content1[start:end])
                #print(anchor.xpath("./charseq/text()")[0])
                if trigger_str!='Q&A':
                    assert trigger_str == sgm_content[start:end], "触发词不匹配"

                if ' ' in trigger_str:
                    trigger_tmp1=trigger_str.replace(' ','_')
                    trigger_tmp2 = trigger_str.replace(' ', '-')
                    #sgm_content=sgm_content.replace(sgm_content[start:end],trigger_str)
                    #print(trigger_str)
                    # if 'assault_w'==trigger_tmp:
                    #     print(file_name)
                    if trigger_tmp1 in wordlist:
                        sgm_content = str_replace_by_position(sgm_content, trigger_tmp1, start, end)
                        #print(trigger_tmp1)
                    elif trigger_tmp2 in wordlist:
                        sgm_content = str_replace_by_position(sgm_content, trigger_tmp2, start, end)
                    else:
                        pass

        # print(sgm_content)

                # trigger_str = anchor.xpath("./charseq/text()")[0].replace('\n', '')
                # trigger_list.append(trigger_str)
                # event_list.append(event_num)

        # assert len(trigger_list) == len(sen_list), '触发词数目与句子数目不相等'
        # assert len(trigger_list) == len(event_list), '触发词数目与事件类型数目不相等'

        # trilen = len(trigger_list)
        # for i in range(trilen):
        #     # 触发词在事件句中的位置
        #     tri_position = sen_list[i].index(trigger_list[i])
        #     # 事件句在文章中的位置
        #     sen_position = text_content.index(sen_list[i])
        #
        #     assert tri_position != -1, '（' + trigger_list[i] + '）不在（' + sen_list[i] + '）中'
        #     assert sen_position != -1, '（' + sen_list[i] + '）不在（' + text_content + '）中'
        #
        #     # 触发词在文章中的位置
        #     tri_start = tri_position + sen_position
        #     tri_end = tri_start + len(trigger_list[i]) - 1
        #
        #     start_end_type_list.append((int(tri_start), int(tri_end), event_list[i]))

    except Exception as e:
        print(e)
        print(file_name, 'droped')

    return


if __name__ == '__main__':

    # str='123 456'
    # print(str_replace_by_position(str,'_',3,4))
    # sys.exit()

    filename1=acepath+'/nw/timex2norm\AFP_ENG_20030323.0020'
    filename2=acepath+'/nw/timex2norm/AFP_ENG_20030509.0345'
    filename3 = acepath + '/bc/timex2norm/CNN_IP_20030330.1600.05-2'
    filename4 = acepath + 'bn/timex2norm/CNN_ENG_20030424_113549.11'
    filename5 = acepath + 'un/timex2norm/alt.corel_20041228.0503'
    filename6 = acepath + '/wl/timex2norm/AGGRESSIVEVOICEDAILY_20041226.1712'
    filename7 = acepath + '/cts/timex2norm/fsh_29191'
    filename8=acepath+'bc/timex2norm/CNN_CF_20030304.1900.04'
    # read_documnet(filename8)
    # sys.exit()

    wordlist_file = homepath + '/ace05/word2vec/wordlist'
    wordlist = [i.replace('\n', '') for i in open(wordlist_file, 'r')]
    # for i in wordlist:
    #     if 'took_on' in i:
    #         print(i)
    #     #print(i)
    # # if 'took on' in wordlist:
    # #     print('took on')
    # sys.exit()


    doclist=homepath+'/ace05/new_filelist_ACE_full.txt';
    f_list=[i.replace('\n','') for i in open(doclist,'r')]
    k=0
    for i in f_list:
        path=acepath+i
        read_documnet(path,wordlist)
        k=k+1
    print(k)
    #read_documnet(filename1)

    # read_documnet(filename2)

