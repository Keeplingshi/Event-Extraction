"""
max_len=60
单触发词，34类
"""

import xml.etree.ElementTree as ET
import pickle, re, sys, os
import numpy as np
import nltk
from gensim.models import word2vec
import time
import random

homepath='D:/Code/pycharm/Event-Extraction/'
acepath=homepath+'ace05/data/English/'

doclist_train=homepath+'/ace05/split1.0/new_filelist_ACE_training.txt'
doclist_test=homepath+'/ace05/split1.0/new_filelist_ACE_test.txt'
doclist_dev=homepath+'/ace05/split1.0/new_filelist_ACE_dev.txt'

posi_embed_path=homepath+"/model/tensorflow2/data/posi_embed.bin"

data_save_path=homepath+"/model/tensorflow2/data/trigger_data/3/trigger_train_data.data"
form_data_save_path=homepath+"/model/tensorflow2/data/trigger_data/3/trigger_train_data_form.data"
#添加位置信息
form_addposi_data_save_path=homepath+"/model/tensorflow2/data/trigger_data/3/trigger_train_addposi_data_form.data"
#在位置信息添加完之后，添加词性信息
form_posi_postag_data_save_path=homepath+"/model/tensorflow2/data/trigger_data/1/trigger_train_posi_postag_data_form.data"

class_size=34
max_len=60
sen_min_len=5


def get_dot_word():
    wordlist_file=homepath+'/ace05/word2vec/wordlist'

    wordlist_f=open(wordlist_file,'r')
    word_dot_list=dict()
    for line in wordlist_f:
        word=line.strip()
        if "." in word:
            if "."!=word and "..."!=word:
                temp=word
                word_dot_list[temp.replace("."," <dot> ")]=word
    return word_dot_list

word_dot_list=get_dot_word()

def number_form(s):
    num_list = re.findall("\d+\s,\s\d+", s)
    for re_num in num_list:
        s = s.replace(re_num, re_num.replace(" ", ""))

    if s in word_dot_list.keys():
        s=word_dot_list.get(s)
    return s


def read_file(xml_path, text_path, event_type):
    apf_tree = ET.parse(xml_path)
    root = apf_tree.getroot()

    event_start = {}
    event_end = {}

    event_ident = {}
    event_map = {}
    event = dict()

    for events in root.iter("event"):
        ev_type = events.attrib["TYPE"] + "_" + events.attrib["SUBTYPE"]
        if ev_type not in event_type:
            event_type.append(ev_type)
        for mention in events.iter("event_mention"):
            ev_id = mention.attrib["ID"]
            anchor = mention.find("anchor")
            for charseq in anchor:
                start = int(charseq.attrib["START"])
                end = int(charseq.attrib["END"]) + 1
                text = re.sub(r"\n", r"", charseq.text)
                event_tupple = (ev_type, start, end, text)
                if event_tupple in event_ident:
                    sys.stderr.write("dulicapte event {}\n".format(ev_id))
                    event_map[ev_id] = event_ident[event_tupple]
                    continue
                event_ident[event_tupple] = ev_id
                event[ev_id] = [ev_id, ev_type, start, end, text]
                event_start[start] = ev_id
                event_end[end] = ev_id

    doc = open(text_path).read()
    doc = re.sub(r"<[^>]+>", r"", doc)
    doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)
    offset = 0
    size = len(doc)
    try:
        header, _, finish = doc.split(r"\n\n\n\n")
        current = len(header) + 4
        end = len(doc) - len(finish)
    except:
        end = len(doc)
        current = 0
    regions = []
    tokens = []
    anchors = []
    for i in range(size):
        if i in event_start:
            inc = 0
            new = clean_str(doc[current:i])
            regions.append(new)
            tokens += new.split()
            anchors += [0 for _ in range(len(new.split()))]
            inc = 0
            current = i
            ent = event_start[i]
            event[ent][2] += offset + inc
        if i in event_end:
            ent = event_end[i]
            event[ent][3] += offset
            new = clean_str(doc[event[ent][2] : event[ent][3]])
            regions.append(new)
            tokens += [new]
            anchors += [event_type.index(event[ent][1])]
            offset += inc
            current = event[ent][3]
    new = clean_str(doc[current : end])
    regions.append(new)
    tokens += new.split()
    anchors += [0 for _ in range(len(new.split()))]
    doc = "".join(regions)
    if len(tokens) == 0:
        print(doc)
        print(text_path)
    for e in  event.values():
        if "\n" in doc[int(e[2]) : int(e[3])]:
            l = []
            l.append(doc[0 : int(e[2])])
            l.append(doc[int(e[2]) : int(e[3])].replace("\n", " "))
            l.append(doc[int(e[3]) :])
            doc = "".join(l)

    return tokens, anchors


def encode_corpus(flag):
    if flag=='train':
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_train,'r')]
        return doclist_train_f
    if flag=='test':
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_test,'r')]
        return doclist_train_f
    if flag=='dev':
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_dev,'r')]
        return doclist_train_f



def read_corpus(event_type,flag):
    count = 0
    file_list = encode_corpus(flag)
    tokens, anchors = [], []
    for file_path in file_list:
        tok, anc = read_file(file_path + ".apf.xml", file_path + ".sgm", event_type)
        count += 1
        tokens.append(tok)
        anchors.append(anc)
    print(event_type)
    return tokens, anchors


def clean_str(string, TREC=False):
    string = re.sub(r"\n\n", "<dot2>", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`<>]", " ", string)
    string = re.sub(r"\'m", r" 'm", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " <dot> ", string)
    string = re.sub(r"\,", r" , ", string)
    string = re.sub(r"!", " <dot2> ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " <dot2> ", string)
    string = re.sub(r"\s{2,}", " ", string)

    # print(string)
    string=number_form(string)
    return string.strip() if TREC else string.strip().lower()


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


def list2vec(tokens,anchors,phrase_posi_dict):
    vec_dict=get_word2vec()

    X=[]
    Y=[]
    W=[]

    sen_list=[]     #存储句子向量
    label_list=[]
    sen_word_list=[]

    length=len(tokens)
    assert len(tokens)==len(anchors), '句子数目不相等'
    for i in range(length):
        token=tokens[i]
        anchor=anchors[i]
        assert len(token)==len(anchor), '句子数目不相等'
        for j in range(len(token)):
            #如果是句号，结束符
            if "<dot>" in token[j] or "<dot2>" in token[j]:
                if len(sen_list)>=sen_min_len:
                    X.append(sen_list)
                    Y.append(label_list)
                    W.append(sen_word_list)
                sen_list=[]
                label_list=[]
                sen_word_list=[]
                continue

            if vec_dict.get(token[j]) is not None:
                sen_list.append(vec_dict.get(token[j]))
                sen_word_list.append(token[j])
                a = [0.0 for x in range(0, class_size)]
                a[anchor[j]] = 1.0
                label_list.append(a)
            else:
                trigger_tmp3=token[j].replace('\n', ' ')
                if ' ' in token[j]:
                    if trigger_tmp3 in phrase_posi_dict.keys():
                        new_trigger=trigger_tmp3.split(' ')[int(phrase_posi_dict[trigger_tmp3])-1]
                        sen_list.append(vec_dict.get(new_trigger))
                        sen_word_list.append(new_trigger)
                        a = [0.0 for x in range(0, class_size)]
                        a[anchor[j]] = 1.0
                        label_list.append(a)
                else:
                    sen_list.append(np.random.uniform(-0.25, 0.25, 300).tolist())
                    sen_word_list.append(token[j])
                    a = [0.0 for x in range(0, class_size)]
                    a[anchor[j]] = 1.0
                    label_list.append(a)

    return X,Y,W


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

    phrase_posi_file=homepath+'/ace05/word2vec/phrase_posi.txt'
    phrase_posi_dict={}
    for i in open(phrase_posi_file, 'r'):
        a,b=get_phrase_posi(i.replace('\n', ''))
        phrase_posi_dict[a]=b

    event_type = [None]
    train_tokens, train_anchors=read_corpus(event_type,'train')
    test_tokens, test_anchors=read_corpus(event_type,'test')
    dev_tokens, dev_anchors=read_corpus(event_type,'dev')

    X_train,Y_train,W_train=list2vec(train_tokens,train_anchors,phrase_posi_dict)
    X_test,Y_test,W_test=list2vec(test_tokens,test_anchors,phrase_posi_dict)
    X_dev,Y_dev,W_dev=list2vec(dev_tokens,dev_anchors,phrase_posi_dict)

    data=X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev
    f=open(data_save_path,'wb')
    pickle.dump(data,f)


# 规范句子长度
def padding_mask(x, y,w,max_len):
    X_train=[]
    Y_train=[]
    W_train=[]
    x_zero_list=[0.0 for i in range(300)]
    y_zero_list=[0.0 for i in range(class_size)]
    y_zero_list[0]=1.0
    unknown='#'
    for i, (x, y,w) in enumerate(zip(x, y,w)):
        if max_len>len(x):
            for j in range(max_len-len(x)):
                x.append(x_zero_list)
                y.append(y_zero_list)
                w.append(unknown)
        else:
            x=x[:max_len]
            y=y[:max_len]
            w=w[:max_len]
        X_train.append(x)
        Y_train.append(y)
        W_train.append(w)
    return X_train,Y_train,W_train


def form_data():

    data_f = open(data_save_path, 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()

    X_train,Y_train,W_train=padding_mask(X_train,Y_train,W_train,max_len)
    X_test,Y_test,W_test=padding_mask(X_test,Y_test,W_test,max_len)
    X_dev,Y_dev,W_dev=padding_mask(X_dev,Y_dev,W_dev,max_len)

    data=X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev
    f=open(form_data_save_path,'wb')
    pickle.dump(data,f)

    print(np.array(X_train).shape)
    print(np.array(Y_train).shape)
    print(np.array(W_train).shape)
    print(np.array(X_test).shape)
    print(np.array(Y_test).shape)
    print(np.array(W_test).shape)
    print(np.array(X_dev).shape)
    print(np.array(Y_dev).shape)
    print(np.array(W_dev).shape)


def add_posi():

    data_f = open(form_data_save_path, 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()

    posi_model =word2vec.Word2Vec.load_word2vec_format(posi_embed_path,binary=True)

    zero_posi=[0.0 for i in range(5)]

    X_train_addposi=[]
    word=[]
    sen_add_posi=[]
    for sentence in X_train:
        for i in range(len(sentence)):
            word.extend(sentence[i])
            if word[0]==0:
                word.extend(zero_posi)
            else:
                word.extend(posi_model[str(i)])
            sen_add_posi.append(word)
            word=[]
        X_train_addposi.append(sen_add_posi)
        sen_add_posi=[]

    X_test_addposi=[]
    word=[]
    sen_add_posi=[]
    for sentence in X_test:
        for i in range(len(sentence)):
            word.extend(sentence[i])
            if word[0]==0:
                word.extend(zero_posi)
            else:
                word.extend(posi_model[str(i)])
            sen_add_posi.append(word)
            word=[]
        X_test_addposi.append(sen_add_posi)
        sen_add_posi=[]


    X_dev_addposi=[]
    word=[]
    sen_add_posi=[]
    for sentence in X_dev:
        for i in range(len(sentence)):
            word.extend(sentence[i])
            if word[0]==0:
                word.extend(zero_posi)
            else:
                word.extend(posi_model[str(i)])
            sen_add_posi.append(word)
            word=[]
        X_dev_addposi.append(sen_add_posi)
        sen_add_posi=[]

    data=X_train_addposi,Y_train,W_train,X_test_addposi,Y_test,W_test,X_dev_addposi,Y_dev,W_dev
    f=open(form_addposi_data_save_path,'wb')
    pickle.dump(data,f)

    print(np.array(X_train_addposi).shape)
    print(np.array(Y_train).shape)
    print(np.array(W_train).shape)
    print(np.array(X_test_addposi).shape)
    print(np.array(Y_test).shape)
    print(np.array(W_test).shape)
    print(np.array(X_dev_addposi).shape)
    print(np.array(Y_dev).shape)
    print(np.array(W_dev).shape)


"""
添加词性标注
"""
def get_pos_tag(word_list,vec_list):
    tag_list=['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB','unknown']

    word_tag_list=nltk.pos_tag(word_list)

    new_vec_list=[]
    for i in range(len(word_list)):
        word_tag=word_tag_list[i]
        tag_array=[0.0 for x in range(len(tag_list))]
        try:
            tag_index=tag_list.index(word_tag[1])
        except:
            tag_index=35

        tag_array[tag_index]=1.0
        word_vec=vec_list[i]
        word_vec.extend(tag_array)
        new_vec_list.append(word_vec)

    return new_vec_list



def add_pos_tag():
    data_f = open(form_addposi_data_save_path, 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()

    new_X_train=[]
    new_X_test=[]
    new_X_dev=[]
    for word_list,vec_list in zip(W_train,X_train):
        new_vec_list=get_pos_tag(word_list,vec_list)
        new_X_train.append(new_vec_list)

    for word_list,vec_list in zip(W_test,X_test):
        new_vec_list=get_pos_tag(word_list,vec_list)
        new_X_test.append(new_vec_list)

    for word_list,vec_list in zip(W_dev,X_dev):
        new_vec_list=get_pos_tag(word_list,vec_list)
        new_X_dev.append(new_vec_list)


    data=new_X_train,Y_train,W_train,new_X_test,Y_test,W_test,new_X_dev,Y_dev,W_dev
    f=open(form_posi_postag_data_save_path,'wb')
    pickle.dump(data,f)

    print(np.array(new_X_train).shape)
    print(np.array(Y_train).shape)
    print(np.array(W_train).shape)
    print(np.array(new_X_test).shape)
    print(np.array(Y_test).shape)
    print(np.array(W_test).shape)
    print(np.array(new_X_dev).shape)
    print(np.array(Y_dev).shape)
    print(np.array(W_dev).shape)


if __name__ == "__main__":

    # pre_data()
    #
    # form_data()


    data_f = open(form_data_save_path, 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()

    target = np.argmax(Y_test, 2)

    classify_r = 0  # 测试集中存在个个体总数

    for i in range(len(target)):
        for j in range(max_len):
            if target[i][j]!=0:
                classify_r+=1

    print(classify_r)




    # add_posi()

    # add_pos_tag()
    #
    # # word_list=["word","unknow_word","run","at",",","#"]
    # # word_tag_list=nltk.pos_tag(word_list)
    # # print(word_tag_list)

