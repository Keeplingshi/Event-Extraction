# coding:utf-8

from lxml import etree
import jieba
import pickle
import re
from xml.dom import minidom
from model.tensorflow.data_process import xml_parse
from model.tensorflow.data_process import type_index
import sys
import string



homepath='D:/Code/pycharm/Event-Extraction/'
acepath=homepath+'ace05/data/English/'
punctuation = """!"#$%&'()*+,./:;<=>?@[\]^`{|}~"""
endpunc="""!"#$%&'()*+,./:;<=>?@[\]^`{|}~"""
# trainpath=homepath+'/ace_en_experiment/train/'
# testpath=homepath+'/ace_en_experiment/test/'
# devpath=homepath+'/ace_en_experiment/dev/'

def get_word2vec():
    word2vec_file=homepath+'/ace05/word2vec/wordvector'
    wordlist_file=homepath+'/ace05/word2vec/wordlist'

    wordvec={}
    word2vec_f=open(word2vec_file,'r')
    wordlist_f=open(wordlist_file,'r')
    word_len=19488
    for line in range(word_len):
        word=wordlist_f.readline().strip()
        vec=word2vec_f.readline().strip()
        temp=vec.split(',')
        temp = map(float, temp)
        vec_list = []
        for i in temp:
            vec_list.append(i)
        wordvec[word]=vec_list
    return wordvec

def get_text_from_sgm(sgm_file):
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
        start_end_type_list = {}

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
                start_end_type_list[(start,end)]=event_num
                #start_end_type_list.append(((start,end),event_num))
        #print(start_end_type_list)
        return sgm_content,start_end_type_list

    except Exception as e:
        print(e)
        print(file_name, 'droped')

    return None


"""
内容转化为词向量
"""
def content2list(sgm_content, start_end_type_list):
    word_list=[]
    type_list=[]
    sgm_content=sgm_content.replace('\n',' ')
    wi=0
    word=''
    start=0
    for w in sgm_content:
        if w!=' ':
            word=word+w

        if w==' ':
            end=wi
            word_list.append(word)
            #解决触发词最后一个字符为标点符号的情况
            w_end=-1
            for character in word:
                if character in punctuation:
                    w_end=word.index(character)
                    break
            if w_end==-1:
                if (start,end) in start_end_type_list.keys():
                    type_list.append(start_end_type_list[(start,end)])
                else:
                    type_list.append(34)
            else:
                end=end-len(word)+w_end
                if (start,end) in start_end_type_list.keys():
                    type_list.append(start_end_type_list[(start,end)])
                else:
                    type_list.append(34)

            word=''
            start=wi+1

        wi+=1
    assert len(type_list)==len(word_list),'content2list单词数目与实践类型数目不匹配'
    return word_list,type_list


"""

"""
def list2vec(word_list,type_list,vec_dict):
    assert len(type_list)==len(word_list),'list2vec单词数目与实践类型数目不匹配'
    length=len(word_list)
    document_list=[]    #存储整个文档的向量
    document_label_list=[]
    sen_list=[]     #存储句子向量
    label_list=[]

    for i in range(length):
        word=word_list[i].lower()       #取单词小写
        w_end=-1
        for character in word:
            if character in punctuation:
                w_end=word.index(character)
                break

        if w_end==-1:
            #说明没有标点符号，则直接查找词向量
            if word in vec_dict.keys():
                if vec_dict.get(word) is not None:
                    sen_list.append(vec_dict.get(word))
                    a = [0.0 for x in range(0, 34)]
                    a[type_list[i]-1] = 1.0
                    label_list.append(a)
        else:
            #如果有标点符号，判断标点符号是否为结束符，如果是，则断句处理。否则，特殊处理
            if word in vec_dict.keys():
                if vec_dict.get(word) is not None:
                    sen_list.append(vec_dict.get(word))
                    a = [0.0 for x in range(0, 34)]
                    a[type_list[i]-1] = 1.0
                    label_list.append(a)
            else:
                wordtmp=word[w_end:]
                flag=False
                for tmp in wordtmp:
                    if tmp.isalpha():
                        flag=True
                        break
                if ~flag:
                    wordtmp=word[:w_end]
                    if wordtmp in vec_dict.keys():
                        sen_list.append(vec_dict.get(wordtmp))
                        a = [0.0 for x in range(0, 34)]
                        a[type_list[i]-1] = 1.0
                        label_list.append(a)
                else:
                    #处理doesn't,wasn't,isn't等
                    if 'n\'t' in word:
                        if word[-3:]=='n\'t':
                            if word[:-3] in vec_dict.keys():
                                sen_list.append(vec_dict.get(word[:-3]))
                                a = [0.0 for x in range(0, 34)]
                                a[type_list[i]-1] = 1.0
                                label_list.append(a)
                                a = [0.0 for x in range(0, 34)]
                                a[33] = 1.0
                                sen_list.append(vec_dict.get('n\'t'))
                                label_list.append(a)
                    if '\'s' in word:
                        if word[-2:]=='\'s':
                            if word[:-2] in vec_dict.keys():
                                sen_list.append(vec_dict.get(word[:-2]))
                                a = [0.0 for x in range(0, 34)]
                                a[type_list[i]-1] = 1.0
                                label_list.append(a)
                                sen_list.append(vec_dict.get('\'s'))
                                a = [0.0 for x in range(0, 34)]
                                a[33] = 1.0
                                label_list.append(a)
                    if '\'re' in word:
                        if word[-3:]=='\'re':
                            if word[:-3] in vec_dict.keys():
                                sen_list.append(vec_dict.get(word[:-3]))
                                a = [0.0 for x in range(0, 34)]
                                a[type_list[i]-1] = 1.0
                                label_list.append(a)
                                sen_list.append(vec_dict.get('\'re'))
                                a = [0.0 for x in range(0, 34)]
                                a[33] = 1.0
                                label_list.append(a)
                    if '\'ve' in word:
                        if word[-3:]=='\'ve':
                            if word[:-3] in vec_dict.keys():
                                sen_list.append(vec_dict.get(word[:-3]))
                                a = [0.0 for x in range(0, 34)]
                                a[type_list[i]-1] = 1.0
                                label_list.append(a)
                                sen_list.append(vec_dict.get('\'ve'))
                                a = [0.0 for x in range(0, 34)]
                                a[33] = 1.0
                                label_list.append(a)
                    if '\'ll' in word:
                        if word[-3:]=='\'ll':
                            if word[:-3] in vec_dict.keys():
                                sen_list.append(vec_dict.get(word[:-3]))
                                a = [0.0 for x in range(0, 34)]
                                a[type_list[i]-1] = 1.0
                                label_list.append(a)
                                sen_list.append(vec_dict.get('\'ll'))
                                a = [0.0 for x in range(0, 34)]
                                a[33] = 1.0
                                label_list.append(a)
                    if '\'d' in word:
                        if word[-2:]=='\'d':
                            if word[:-2] in vec_dict.keys():
                                sen_list.append(vec_dict.get(word[:-2]))
                                a = [0.0 for x in range(0, 34)]
                                a[type_list[i]-1] = 1.0
                                label_list.append(a)
                                sen_list.append(vec_dict.get('\'d'))
                                a = [0.0 for x in range(0, 34)]
                                a[33] = 1.0
                                label_list.append(a)

            #断句操作
            if '.' in word or '!' in word or '?' in word:
                #print(len(sen_list))
                assert len(sen_list)==len(label_list),'句子，标注不相等'
                if len(sen_list)>=5:
                    document_list.append(sen_list)
                    document_label_list.append(label_list)
                sen_list=[]
                label_list=[]

    return document_list,document_label_list



"""
获取短语取哪个词作为触发词
"""
def get_phrase_posi(ss):
    strlist=ss.split(' ')
    k=len(strlist)
    s=strlist[0]
    for i in range(1,k-1):
        s=s+' '+strlist[i]
    return s,strlist[k-1]


def pre_data():
    wordlist_file = homepath + '/ace05/word2vec/wordlist'
    wordlist = [i.replace('\n', '') for i in open(wordlist_file, 'r')]
    phrase_posi_file=homepath+'/ace05/word2vec/phrase_posi.txt'
    phrase_posi_dict={}
    for i in open(phrase_posi_file, 'r'):
        a,b=get_phrase_posi(i.replace('\n', ''))
        phrase_posi_dict[a]=b

    vec_dict=get_word2vec()

    doclist_train=homepath+'/ace05/new_filelist_ACE_training.txt'
    doclist_train_f=[i.replace('\n','') for i in open(doclist_train,'r')]
    doclist_test=homepath+'/ace05/new_filelist_ACE_test.txt';
    doclist_test_f=[i.replace('\n','') for i in open(doclist_test,'r')]
    doclist_dev=homepath+'/ace05/new_filelist_ACE_dev.txt';
    doclist_dev_f=[i.replace('\n','') for i in open(doclist_dev,'r')]

    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    X_dev=[]
    Y_dev=[]


    for i in doclist_train_f:
        path=acepath+i
        sgm_content,start_end_type_list=read_documnet(path,wordlist,phrase_posi_dict)
        word_list,type_list=content2list(sgm_content, start_end_type_list)
        document_list,document_label_list=list2vec(word_list,type_list,vec_dict)
        X_train.extend(document_list)
        Y_train.extend(document_label_list)

    for i in doclist_test_f:
        path=acepath+i
        sgm_content,start_end_type_list=read_documnet(path,wordlist,phrase_posi_dict)
        word_list,type_list=content2list(sgm_content, start_end_type_list)
        document_list,document_label_list=list2vec(word_list,type_list,vec_dict)
        X_test.extend(document_list)
        Y_test.extend(document_label_list)

    for i in doclist_dev_f:
        path=acepath+i
        sgm_content,start_end_type_list=read_documnet(path,wordlist,phrase_posi_dict)
        word_list,type_list=content2list(sgm_content, start_end_type_list)
        document_list,document_label_list=list2vec(word_list,type_list,vec_dict)
        X_dev.extend(document_list)
        Y_dev.extend(document_label_list)

    data=X_train,Y_train,X_test,Y_test,X_dev,Y_dev
    f=open(homepath+'/model/tensorflow/enACEdata/data2/train_data34.data','wb')
    pickle.dump(data,f)

    m_train = []
    for i in Y_train:
        n = []
        for j in i:
            if j[33] == 1.0:
                n.append([0.0, 1.0])
            else:
                n.append([1.0, 0.0])
        m_train.append(n)

    m_test = []
    for i in Y_test:
        n = []
        for j in i:
            if j[33] == 1.0:
                n.append([0.0, 1.0])
            else:
                n.append([1.0, 0.0])
        m_test.append(n)

    m_dev = []
    for i in Y_dev:
        n = []
        for j in i:
            if j[33] == 1.0:
                n.append([0.0, 1.0])
            else:
                n.append([1.0, 0.0])
        m_dev.append(n)

    data = X_train, m_train, X_dev, m_dev, X_test, m_test
    f = open(homepath + '/model/tensorflow/enACEdata/data2/train_data2.data', 'wb')
    pickle.dump(data, f)


if __name__ == '__main__':
    pre_data()
    # filename1=acepath+'/nw/timex2norm/AFP_ENG_20030323.0020'
    # filename2=acepath+'/nw/timex2norm/AFP_ENG_20030509.0345'
    # filename3 = acepath + '/bc/timex2norm/CNN_IP_20030330.1600.05-2'
    # filename4 = acepath + 'bn/timex2norm/CNN_ENG_20030424_113549.11'
    # filename5 = acepath + 'un/timex2norm/alt.corel_20041228.0503'
    # filename6 = acepath + '/wl/timex2norm/AGGRESSIVEVOICEDAILY_20041226.1712'
    # filename7 = acepath + '/cts/timex2norm/fsh_29191'
    # filename8=acepath+'nw/timex2norm/APW_ENG_20030304.0555'
    #
    #
    # wordlist_file = homepath + '/ace05/word2vec/wordlist'
    # wordlist = [i.replace('\n', '') for i in open(wordlist_file, 'r')]
    # phrase_posi_file=homepath+'/ace05/word2vec/phrase_posi.txt'
    # phrase_posi_dict={}
    # for i in open(phrase_posi_file, 'r'):
    #     a,b=get_phrase_posi(i.replace('\n', ''))
    #     phrase_posi_dict[a]=b
    #
    # vec_dict=get_word2vec()
    #
    # sgm_content,start_end_type_list=read_documnet(filename8,wordlist,phrase_posi_dict)
    # word_list,type_list=content2list(sgm_content, start_end_type_list)
    # list2vec(word_list,type_list,vec_dict)

    #
    # doclist=homepath+'/ace05/new_filelist_ACE_full.txt'
    # f_list=[i.replace('\n','') for i in open(doclist,'r')]
    # # doclist2=homepath+'/ace05/new_filelist_ACE_training.txt';
    # # f_list2=[i.replace('\n','') for i in open(doclist2,'r')]
    #
    # vec_dict=get_word2vec()
    # k=0
    # for i in f_list:
    #     path=acepath+i
    #     sgm_content,start_end_type_list=read_documnet(path,wordlist,phrase_posi_dict)
    #     word_list,type_list=content2list(sgm_content, start_end_type_list)
    #     list2vec(word_list,type_list,vec_dict)
    #     k+=1
    # print(k)
    #read_documnet(filename1)

    # read_documnet(filename2)

