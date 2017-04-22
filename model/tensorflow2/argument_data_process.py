import xml.etree.ElementTree as ET
import pickle, re, sys, os
import numpy as np
import nltk
from gensim.models import word2vec
import time

homepath='D:/Code/pycharm/Event-Extraction/'
acepath=homepath+'ace05/data/English/'

doclist_train=homepath+'/ace05/split1.0/new_filelist_ACE_training.txt'
doclist_test=homepath+'/ace05/split1.0/new_filelist_ACE_test.txt'
doclist_dev=homepath+'/ace05/split1.0/new_filelist_ACE_dev.txt'

stop_words_path=homepath+'/ace05/word2vec/stop_words'

arg_type_num=36

def get_stop_words():
    stop_words_f=open(stop_words_path,'r')
    for line in stop_words_f.readlines():
        print(line.strip())
    return 0


# get_stop_words()
# sys.exit()

def number_form(s):
    num_list = re.findall("\d+\s<dot>\s\d+", s)
    for re_num in num_list:
        s = s.replace(re_num, re_num.replace(" <dot> ", "."))
    num_list = re.findall("\d+\s,\s\d+", s)
    for re_num in num_list:
        s = s.replace(re_num, re_num.replace(" ", ""))
    return s


def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'m", r" 'm", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " <dot> ", string)
    string = re.sub(r"\,", r" , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

"""
获取语料列表
"""
def encode_corpus(flag):
    if flag=='train':
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_train,'r')]
        return doclist_train_f
    if flag=='test':
        doclist_test_f=[acepath+i.replace('\n','') for i in open(doclist_test,'r')]
        return doclist_test_f
    if flag=='dev':
        doclist_dev_f=[acepath+i.replace('\n','') for i in open(doclist_dev,'r')]
        return doclist_dev_f


def read_file(xml_path, text_path, argument_type):
    apf_tree = ET.parse(xml_path)
    root = apf_tree.getroot()

    # argment process
    argument_start = {}
    argument_end = {}

    argument_ident={}
    event_argument=dict()

    for events in root.iter("event"):

        for mention in events.iter("event_mention"):

            mention_arguments = mention.findall("event_mention_argument")
            for mention_argument in mention_arguments:
                ev_id = mention_argument.attrib["REFID"]
                # 时间要素类型
                arg_type=mention_argument.attrib["ROLE"]
                if arg_type not in argument_type:
                    argument_type.append(arg_type)

                for extend in mention_argument:

                    for charseq in extend:
                        start = int(charseq.attrib["START"])
                        end = int(charseq.attrib["END"]) + 1
                        text = re.sub(r"\n", r"", charseq.text)
                        argument_tupple = (arg_type, start, end, text)
                        if argument_tupple in argument_ident:
                            #sys.stderr.write("dulicapte event {}\n".format(ev_id))
                            # argument_map[ev_id] = argument_ident[argument_tupple]
                            continue
                        argument_ident[argument_tupple] = ev_id
                        event_argument[ev_id] = [ev_id, arg_type, start, end, text]
                        argument_start[start] = ev_id
                        argument_end[end] = ev_id

    # print(len(argument_ident))
    # print(len(event_argument))
    # print(argument_ident)
    # print(event_argument)
    # print('--------------------------------------------------------')

    doc = open(text_path).read()
    doc = re.sub(r"<[^>]+>", r"", doc)
    doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)
    offset = 0
    size = len(doc)
    try:
        header, _, finish = doc.split(r"\n\n\n\n")
        current = len(header) + 4
        end = len(doc) - len(finish)
    except :
        end = len(doc)
        current = 0
    regions = []
    tokens = []
    arguments = []
    for i in range(size):
        if i in argument_start:
            new = clean_str(doc[current:i])
            regions.append(new)
            tokens += new.split()
            arguments += [0 for _ in range(len(new.split()))]
            inc = 0
            current = i
            ent = argument_start[i]
            event_argument[ent][2] += offset + inc
        if i in argument_end:
            ent = argument_end[i]
            event_argument[ent][3] += offset
            new = clean_str(doc[event_argument[ent][2] : event_argument[ent][3]])
            regions.append(new)
            tokens += [new]
            arguments += [argument_type.index(event_argument[ent][1])]
            offset += inc
            current = event_argument[ent][3]
    new = clean_str(doc[current : end])
    regions.append(new)
    tokens += new.split()
    arguments += [0 for _ in range(len(new.split()))]
    doc = "".join(regions)
    if len(tokens) == 0:
        print(doc)
        print(text_path)
    for e in  event_argument.values():
        if "\n" in doc[int(e[2]) : int(e[3])]:
            l = []
            l.append(doc[0 : int(e[2])])
            l.append(doc[int(e[2]) : int(e[3])].replace("\n", " "))
            l.append(doc[int(e[3]) :])
            doc = "".join(l)

    return tokens, arguments


"""
从语料读取信息
"""
def read_corpus(argument_type,flag):
    file_list = encode_corpus(flag)
    tokens, arguments = [], []
    for file_path in file_list:
        tok, arg = read_file(file_path + ".apf.xml", file_path + ".sgm", argument_type)
        assert len(tok) == len(arg), file_path+"\t长度不一"
        tokens.append(tok)
        arguments.append(arg)
    return tokens, arguments


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


def list2vec(tokens, arguments):
    vec_dict=get_word2vec()

    X=[]
    Y=[]
    W=[]

    sen_list=[]     #存储句子向量
    label_list=[]
    sen_word_list=[]

    length=len(tokens)
    assert len(tokens)==len(arguments), '句子数目不相等'
    for i in range(length):
        token=tokens[i]
        anchor=arguments[i]
        assert len(token)==len(anchor), '句子数目不相等'
        for j in range(len(token)):
            #如果是句号，结束符
            token[j]=number_form(token[j])
            if "<dot>" in token[j]:
                if "u <dot> s <dot>" in token[j]:
                    token[j] = token[j].replace("u <dot> s <dot>", "u.s.")
                else:
                    if len(sen_list)>=5:
                        X.append(sen_list)
                        Y.append(label_list)
                        W.append(sen_word_list)
                    sen_list=[]
                    label_list=[]
                    sen_word_list=[]

            if vec_dict.get(token[j]) is not None:
                sen_list.append(vec_dict.get(token[j]))
                sen_word_list.append(token[j])
                a = [0.0 for x in range(0, arg_type_num)]
                a[anchor[j]] = 1.0
                label_list.append(a)
            else:
                arg_tokens=token[j].replace('\n', ' ')
                if ' ' in arg_tokens:
                    # arg_tokens = number_form(arg_tokens)
                    arg_words=arg_tokens.split(' ')
                    if len(arg_words)<=5:
                        for k in range(len(arg_words)):
                            if vec_dict.get(arg_words[k]) is not None:
                                sen_list.append(vec_dict.get(arg_words[k]))
                                sen_word_list.append(arg_words[k])
                                a = [0.0 for x in range(0, arg_type_num)]
                                a[anchor[j]] = 1.0
                                label_list.append(a)
                    else:
                        for k in range(len(arg_words)):
                            if vec_dict.get(arg_words[k]) is not None:
                                sen_list.append(vec_dict.get(arg_words[k]))
                                sen_word_list.append(arg_words[k])
                                a = [0.0 for x in range(0, arg_type_num)]
                                a[0] = 1.0
                                label_list.append(a)

    return X,Y,W


def pre_data():

    argument_type = [None]
    train_tokens, train_arguments=read_corpus(argument_type,'train')
    test_tokens, test_arguments=read_corpus(argument_type,'test')
    dev_tokens, dev_arguments=read_corpus(argument_type,'dev')

    print(argument_type)
    print(len(argument_type))
    print('-------------------------------')

    X_train,Y_train,W_train=list2vec(train_tokens,train_arguments)
    X_test,Y_test,W_test=list2vec(test_tokens,test_arguments)
    X_dev,Y_dev,W_dev=list2vec(dev_tokens,dev_arguments)

    data=X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev
    f=open(homepath+'/model/tensorflow2/data/8/argument_train_data34.data','wb')
    pickle.dump(data,f)



def get_arg_tuple_list(Y_train):
    arg_list=[]
    for i in range(len(Y_train)):
        for j in range(len(Y_train[i])):
            arg_type=np.argmax(Y_train[i][j])
            if arg_type!=0:
                word_num=0
                if j+1<len(Y_train[i]):
                    arg_type_next=np.argmax(Y_train[i][j+1])
                    if arg_type_next==arg_type:
                        for jj in range(j+1,len(Y_train[i])):
                            word_num+=1
                            if np.argmax(Y_train[i][jj])==0:
                                break
                flag=True
                for (a,b,c,d) in arg_list:
                    if a==i and d==arg_type:
                        if j+word_num<=b+c:
                            flag=False
                if flag:
                    arg_tuple=(i,j,word_num,arg_type)
                    arg_list.append(arg_tuple)

    return arg_list
    # for arg_tuple in arg_list:
    #     print(arg_tuple)

def number_form2(s):
    num_list = re.findall("\d+\s<dot>\s\d+", s)
    for re_num in num_list:
        s = s.replace(re_num, re_num.replace(" <dot> ", "."))
    num_list = re.findall("\d+\s,\s\d+", s)
    for re_num in num_list:
        s = s.replace(re_num, re_num.replace(" ", ""))
    return s


if __name__ == "__main__":
    pre_data()

    # s="1 <dot> 0 million dollars"
    # s = "aaa <dot> 0 million dollars"
    # print(number_form2(s))

    # argument_type=[None]
    # file_path=acepath+"nw/timex2norm/AFP_ENG_20030430.0075"
    # tok, arg = read_file(file_path + ".apf.xml", file_path + ".sgm", argument_type)
    # print(tok)
    #
    # for i in range(len(arg)):
    #     if arg[i]!=0:
    #         print(tok[i])


    # data_f = open('./data/8/argument_train_data34.data', 'rb')
    # X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    # data_f.close()
    #
    # arg_list=get_arg_tuple_list(Y_test)
    # for (i,j,word_num,arg_type) in arg_list:
    #     a=[]
    #     if word_num==0:
    #         a.append(W_test[i][j])
    #     for m in range(word_num):
    #         a.append(W_test[i][j+m])
    #     print(a)

