import xml.etree.ElementTree as ET
import pickle, re, sys, os
import numpy as np
import nltk


homepath='D:/Code/pycharm/Event-Extraction/'
acepath=homepath+'ace05/data/English/'

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
    check= 0
    for i in range(size):
        if i in event_start:
            inc = 0
            new = clean_str(doc[current:i])
            regions.append(new)
            tokens += new.split()
            check = 1
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
        doclist_train=homepath+'/ace05/split1.0/new_filelist_ACE_training.txt'
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_train,'r')]
        return doclist_train_f
    if flag=='test':
        doclist_train=homepath+'/ace05/split1.0/new_filelist_ACE_test.txt'
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_train,'r')]
        return doclist_train_f
    if flag=='dev':
        doclist_train=homepath+'/ace05/split1.0/new_filelist_ACE_dev.txt'
        doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_train,'r')]
        return doclist_train_f

def get_apf_info(test_tokens,test_anchors,event_type):
    trigger=[]
    length=len(test_tokens)
    for i in range(length):
        token=test_tokens[i]
        anchor=test_anchors[i]
        if anchor!=0:
            trigger.append(token)
        # token_len=len(token)
        # for j in range(token_len):
        #     if anchor[j]!=0:
        #         trigger.append(token[j]+":"+event_type[anchor[j]])
    return trigger


def read_corpus(event_type,flag):
    count = 0
    file_list = encode_corpus(flag)
    tokens, anchors = [], []
    for file_path in file_list:
        tok, anc = read_file(file_path + ".apf.xml", file_path + ".sgm", event_type)
        # print(file_path)
        # print(get_apf_info(tok,anc,event_type))
        count += 1
        tokens.append(tok)
        anchors.append(anc)
    #print(count, len(event_type))
    print(event_type)
    return tokens, anchors

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """

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
    # print(string)
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
            if "<dot>" in token[j]:
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
                a = [0.0 for x in range(0, 34)]
                a[anchor[j]] = 1.0
                label_list.append(a)
            else:
                trigger_tmp3=token[j].replace('\n', ' ')
                if ' ' in token[j]:
                    if trigger_tmp3 in phrase_posi_dict.keys():
                        new_trigger=trigger_tmp3.split(' ')[int(phrase_posi_dict[trigger_tmp3])-1]
                        sen_list.append(vec_dict.get(new_trigger))
                        sen_word_list.append(new_trigger)
                        a = [0.0 for x in range(0, 34)]
                        a[anchor[j]] = 1.0
                        label_list.append(a)

                # sen_list.append(np.random.uniform(-0.25, 0.25, 300).tolist())
                # sen_word_list.append(token[j])
                # a = [0.0 for x in range(0, 34)]
                # a[0] = 1.0
                # label_list.append(a)

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
    f=open(homepath+'/model/tensorflow2/data/2/train_data34.data','wb')
    pickle.dump(data,f)

# 规范句子长度
def padding_mask(x, y,w,max_len):
    X_train=[]
    Y_train=[]
    W_train=[]
    x_zero_list=[0.0 for i in range(300)]
    y_zero_list=[0.0 for i in range(34)]
    y_zero_list[0]=1.0
    unknown='unknow_word'
    for i, (x, y) in enumerate(zip(x, y)):
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

#[None, 'Movement_Transport', 'Personnel_Elect', 'Personnel_Start-Position', 'Personnel_Nominate', 'Conflict_Attack', 'Personnel_End-Position', 'Contact_Meet', 'Life_Marry', 'Contact_Phone-Write', 'Transaction_Transfer-Money', 'Justice_Sue', 'Conflict_Demonstrate', 'Business_End-Org', 'Life_Injure', 'Life_Die', 'Justice_Arrest-Jail', 'Transaction_Transfer-Ownership', 'Business_Start-Org', 'Justice_Execute', 'Justice_Trial-Hearing', 'Justice_Sentence', 'Life_Be-Born', 'Justice_Charge-Indict', 'Justice_Convict', 'Business_Declare-Bankruptcy', 'Justice_Release-Parole', 'Justice_Fine', 'Justice_Pardon', 'Justice_Appeal', 'Business_Merge-Org', 'Justice_Extradite', 'Life_Divorce', 'Justice_Acquit']
#sen_list.append(np.random.uniform(-0.25, 0.25, 300).tolist())

'''
规范句子长度
补零的词汇用np.random.uniform(-0.25, 0.25, 300).tolist()代替
'''
def padding_mask_random(x, y,w,max_len):
    X_train=[]
    Y_train=[]
    W_train=[]
    L_train=[]

    x_zero_list=np.random.uniform(-0.25, 0.25, 300).tolist()
    y_zero_list=[0.0 for i in range(34)]
    y_zero_list[0]=1.0
    unknown='unknow_word'
    for i, (x, y) in enumerate(zip(x, y)):
        sen_len=len(x)
        if max_len>len(x):
            for j in range(max_len-len(x)):
                x.append(x_zero_list)
                y.append(y_zero_list)
                w.append(unknown)
        else:
            sen_len=max_len
            x=x[:max_len]
            y=y[:max_len]
            w=w[:max_len]
        L_train.append(sen_len)
        X_train.append(x)
        Y_train.append(y)
        W_train.append(w)
    return X_train,Y_train,W_train,L_train


# 规范句子长度
def padding_mask_full(x, y,w,max_len):
    X_train=[]
    Y_train=[]
    W_train=[]
    x_zero_list=[0.0 for i in range(300)]
    y_zero_list=[0.0 for i in range(34)]
    y_zero_list[0]=1.0
    unknown='unknow_word'
    for i, (x, y) in enumerate(zip(x, y)):
        x_len=len(x)
        iter_num=int(x_len/max_len)
        if iter_num==0:
            for j in range(max_len-len(x)):
                x.append(x_zero_list)
                y.append(y_zero_list)
                w.append(unknown)
            X_train.append(x)
            Y_train.append(y)
            W_train.append(w)
        else:
            for j in range(iter_num):
                if (j+2)*max_len>x_len:
                    a=x[(j+1)*max_len:x_len]
                    b=y[(j+1)*max_len:x_len]
                    c=w[(j+1)*max_len:x_len]
                    for k in range((j+2)*max_len-x_len):
                        a.append(x_zero_list)
                        b.append(y_zero_list)
                        c.append(unknown)
                    X_train.append(a)
                    Y_train.append(b)
                    W_train.append(c)
                else:
                    X_train.append(x[j*max_len:(j+1)*max_len])
                    Y_train.append(y[j*max_len:(j+1)*max_len])
                    W_train.append(w[j*max_len:(j+1)*max_len])
                    # print(len(w[j*max_len:(j+1)*max_len]))


    return X_train,Y_train,W_train

def pos_tag_add():
    data_f = open('./data/2/train_data_form34.data', 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()

    pos_tag=[]
    word_list=[]
    for sentence in W_test:
        for word in sentence:
            # tag=nltk.pos_tag(word)[0][1]
            word_list.append(word)
            # if tag not in pos_tag:
            #     pos_tag.append(tag)
    # print(word_list)
    # print(nltk.pos_tag(word_list))
    # print(pos_tag)


def get_posi():
    data_f = open('./data/2/train_data_form34.data', 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()
    posi=[]
    for sentence in X_train:
        for i in range(len(sentence)):
            word=sentence[i]
            if word[0]==0:
                posi.append(i)
                break

    for sentence in X_test:
        for i in range(len(sentence)):
            word = sentence[i]
            if word[0] == 0:
                posi.append(i)
                break

    for sentence in X_dev:
        for i in range(len(sentence)):
            word = sentence[i]
            if word[0] == 0:
                posi.append(i)
                break

    posi_f = open('./data/2/posi.txt', 'w')
    for i in posi:
        for j in range(i):
            posi_f.write(str(j)+' ')
        posi_f.write('\n')

if __name__ == "__main__":

    get_posi()
    # pos_tag_add()

    # Rfiltered =nltk.pos_tag('today')
    #
    # print(Rfiltered[0][1])
    # pass
    # data_f = open('./data/4/train_data34.data', 'rb')
    # X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    # data_f.close()
    #
    # for i in X_train:
    #     if len(i)>60:
    #         print(len(i))
    #
    # num=0
    # for i in range(len(Y_test)):
    #     for j in range(len(Y_test[i])):
    #         if Y_test[i][j][0]!=1.0:
    #             num+=1
    #
    # print(num)
    #
    # max_len=60
    # # X_train,Y_train,W_train=padding_mask_full(X_train,Y_train,W_train,max_len)
    # X_test,Y_test,W_test=padding_mask_full(X_test,Y_test,W_test,max_len)
    # # X_dev,Y_dev,W_dev=padding_mask_full(X_dev,Y_dev,W_dev,max_len)
    #
    # # data=X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev
    # # f=open(homepath+'/model/tensorflow2/data/4/train_data_len60_form34.data','wb')
    # # pickle.dump(data,f)
    # #
    # # print(np.array(X_train).shape)
    # # print(np.array(Y_train).shape)
    # # print(np.array(W_train).shape)
    # print(np.array(X_test).shape)
    # print(np.array(Y_test).shape)
    # print(np.array(W_test).shape)
    # # print(np.array(X_dev).shape)
    # # print(np.array(Y_dev).shape)
    # # print(np.array(W_dev).shape)
    #
    # num=0
    # for i in range(len(Y_test)):
    #     for j in range(len(Y_test[i])):
    #         if Y_test[i][j][0]!=1.0:
    #             num+=1
    #
    # print(num)

    # pre_data()
    # data_f = open('./data/3/train_data34.data', 'rb')
    # X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    # data_f.close()
    #
    # max_len=60
    # X_train,Y_train,W_train,L_train=padding_mask_random(X_train,Y_train,W_train,max_len)
    # X_test,Y_test,W_test,L_test=padding_mask_random(X_test,Y_test,W_test,max_len)
    # X_dev,Y_dev,W_dev,L_dev=padding_mask_random(X_dev,Y_dev,W_dev,max_len)
    #
    # data=X_train,Y_train,W_train,L_train,X_test,Y_test,W_test,L_test,X_dev,Y_dev,W_dev,L_dev
    # f=open(homepath+'/model/tensorflow2/data/3/train_data_form34.data','wb')
    # pickle.dump(data,f)
    #
    # print(np.array(X_train).shape)
    # print(np.array(Y_train).shape)
    # print(np.array(W_train).shape)
    # print(np.array(X_test).shape)
    # print(np.array(Y_test).shape)
    # print(np.array(W_test).shape)
    # print(np.array(X_dev).shape)
    # print(np.array(Y_dev).shape)
    # print(np.array(W_dev).shape)
    # print(L_train)
    # print(L_test)
    # print(L_dev)






    # print(X_train[0][0])
    # print(X_train[0][1])
    # print(Y_train[0][0])
    # print(Y_train[0][1])
    # print(W_train[0][0])
    # print(W_train[0][1])

    # print(np.array(X_train).shape)
    # print(np.array(X_test).shape)
    #
    # print(W_train[0])

