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

data_save_path=homepath+"/model/tensorflow2/data/argument_data/5/argument_train_data.data"
form_data_save_path=homepath+"/model/tensorflow2/data/argument_data/5/argument_train_data_form.data"

def get_dot_word():
    wordlist_file=homepath+'/ace05/word2vec/wordlist'

    wordlist_f=open(wordlist_file,'r')
    word_dot_list=dict()
    for line in wordlist_f:
        word=line.strip()

        if "." in word:
            if "."!=word and "..."!=word:
                temp=word
                word_dot_list[temp.replace("."," <dot> ").strip()]=word
                print(word)
    return word_dot_list

word_dot_list=get_dot_word()
# for i in word_dot_list:
#     print(i)

sys.exit()

def number_form(s):
    num_list = re.findall("\d+\s,\s\d+", s)
    for re_num in num_list:
        s = s.replace(re_num, re_num.replace(" ", ""))

    if s.strip() in word_dot_list.keys():
        s=word_dot_list.get(s.strip())

    return s

def clean_str(string, TREC=False):
    string = re.sub(r"\n\n", " <dot2> ", string)
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

    return_str=string.strip() if TREC else string.strip().lower()
    return_str=number_form(return_str).strip()
    return return_str


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

        ev_type = "Trigger_"+events.attrib["TYPE"] + "_" + events.attrib["SUBTYPE"]
        if ev_type not in argument_type:
            argument_type.append(ev_type)
        for mention in events.iter("event_mention"):
            ev_id = mention.attrib["ID"]
            anchor = mention.find("anchor")
            for charseq in anchor:
                start = int(charseq.attrib["START"])
                end = int(charseq.attrib["END"]) + 1
                text = re.sub(r"\n", r"", charseq.text)
                event_tupple = (ev_type, start, end, text)
                if event_tupple in argument_ident:
                    # sys.stderr.write("dulicapte event {}\n".format(ev_id))
                    # event_map[ev_id] = event_ident[event_tupple]
                    continue
                argument_ident[event_tupple] = ev_id
                event_argument[ev_id] = [ev_id, ev_type, start, end, text]
                argument_start[start] = ev_id
                argument_end[end] = ev_id



        for mention in events.iter("event_mention"):

            mention_arguments = mention.findall("event_mention_argument")
            for mention_argument in mention_arguments:
                ev_id = mention_argument.attrib["REFID"]
                # 事件要素类型
                arg_type="Argument_"+mention_argument.attrib["ROLE"]
                if arg_type not in argument_type:
                    argument_type.append(arg_type)

                for extend in mention_argument:

                    for charseq in extend:
                        start = int(charseq.attrib["START"])
                        end = int(charseq.attrib["END"]) + 1
                        text = re.sub(r"\n", r"", charseq.text)
                        argument_tupple = (arg_type, start, end, text)
                        if argument_tupple in argument_ident:
                            sys.stderr.write("dulicapte event {}\n".format(ev_id))
                            # argument_map[ev_id] = argument_ident[argument_tupple]
                            continue
                        argument_ident[argument_tupple] = ev_id
                        event_argument[ev_id] = [ev_id, arg_type, start, end, text]
                        argument_start[start] = ev_id
                        argument_end[end] = ev_id


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
            print(doc[current:i])
            print(new)
            print("======================================")
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
            if "<dot>" in token[j] or "<dot2>" in token[j]:
                if len(sen_list)>=5:
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
                a = [0.0 for x in range(0, arg_type_num)]
                a[anchor[j]] = 1.0
                label_list.append(a)
            else:
                arg_tokens=token[j].replace('\n', ' ')
                if ' ' in arg_tokens:
                    arg_words=arg_tokens.split(' ')

                    for k in range(len(arg_words)):
                        if vec_dict.get(arg_words[k]) is not None:

                            sen_list.append(vec_dict.get(arg_words[k]))
                            sen_word_list.append(arg_words[k])
                            a = [0.0 for x in range(0, arg_type_num)]
                            if k == 0:
                                a[anchor[j]] = 1.0
                            else:
                                a[anchor[j]+36] = 1.0
                            label_list.append(a)


                    # if len(arg_words)<=5:
                    #     for k in range(len(arg_words)):
                    #         if vec_dict.get(arg_words[k]) is not None:
                    #             sen_list.append(vec_dict.get(arg_words[k]))
                    #             sen_word_list.append(arg_words[k])
                    #             a = [0.0 for x in range(0, arg_type_num)]
                    #             if arg_words[k] not in stop_word_list:
                    #                 a[anchor[j]] = 1.0
                    #             else:
                    #                 a[0] = 1.0
                    #             label_list.append(a)
                    # else:
                    #     for k in range(len(arg_words)):
                    #         if vec_dict.get(arg_words[k]) is not None:
                    #             sen_list.append(vec_dict.get(arg_words[k]))
                    #             sen_word_list.append(arg_words[k])
                    #             a = [0.0 for x in range(0, arg_type_num)]
                    #             a[0] = 1.0
                    #             label_list.append(a)

                else:
                    sen_list.append(np.random.uniform(-0.25, 0.25, 300).tolist())
                    sen_word_list.append(token[j])
                    a = [0.0 for x in range(0, arg_type_num)]
                    a[anchor[j]] = 1.0
                    # if token[j] not in stop_word_list:
                    #     a[anchor[j]] = 1.0
                    # else:
                    #     a[0] = 1.0
                    label_list.append(a)

    return X,Y,W



if __name__ == "__main__":
    trigger_argument_type=[None]
    file_name="nw/timex2norm/APW_ENG_20030311.0775"
    xml_path=acepath+file_name+".apf.xml"
    text_path=acepath+file_name+".sgm"
    tokens, arguments=read_file(xml_path, text_path, trigger_argument_type)
    print(tokens)
    print(arguments)
    print(trigger_argument_type)

    for i in range(len(tokens)):
        if arguments[i]!=0:
            print(tokens[i]+"\t"+trigger_argument_type[arguments[i]])


    a="Turkish party leader Recep Tayyip Erdogan named prime minister, may push to allow in U.S. troops"
    print(clean_str(a))
    a=" U.S. "
    print(clean_str(a))



