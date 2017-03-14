# coding:utf-8
import os
from lxml import etree  # @UnresolvedImport
#from lxml import etree
import jieba
import pickle
import re
import sys

homepath='D:/Code/pydev/EventExtract/'


'''
内容转为词向量，这里使用结巴分词
'''
def content2wordvec(text_content,start_end_type_list):
    jieba.load_userdict(homepath+'/ace_ch_experiment/trigger.dict')
    s_e_sorted=sorted(start_end_type_list,key=lambda x:x[0])
    word_list=[]
    label=[]
    tmp_i=0
    def clean(x):
        x=x.replace('\n','')
        x=x.replace(' ','')
        return x
    for i,j in s_e_sorted:
        sentence=clean(text_content[tmp_i:i])
        tmp=[t for t in jieba.cut(sentence)]
        word_list.extend(tmp)
        tmp_i=j+1
        label.extend([0 for t in range(len(tmp))])
        word_list.append(clean(text_content[i:j+1]))
        label.append(1)
    assert len(word_list)==len(label)
    return (word_list,label)


'''
读入文件名称，获取词向量
'''
def read_answer(filename_prefix):
    
    corpus_path=homepath+'/ace_ch_experiment/corpus/'
    
    # 获取apf文件位置
    tag_filename = filename_prefix+'.apf.xml'
    tag_filepath=corpus_path+tag_filename
    tag_f=open(tag_filepath, 'rb')
    tag_content=tag_f.read()

    # txt文件是将sgm文件移除尖括号<>所得
    text_filename= filename_prefix+".txt"
    text_filepath=corpus_path+text_filename
    text_f=open(text_filepath, 'rb')
    text_content=text_f.read().decode('utf8')
    text_content=text_content.replace('\n','')

    text2_filename= filename_prefix+".sgm"
    text2_filepath=corpus_path+text2_filename
    text2_f=open(text2_filepath, 'rb')
    text2_content=text2_f.read().decode('utf8')

    doc=etree.fromstring(text2_content)

    sentence=doc.xpath('//TEXT//text()')
    
    
    begin_sen=''.join(sentence)
    begin_sen=begin_sen.replace('\n','')
    begin_sen=begin_sen[:3]
    begin_index=text_content.find(begin_sen)

    try:
        assert begin_index!=-1,'begin_index==-1'
        begin_index-=1
        new_text=text_content[begin_index:]
        doc=etree.fromstring(tag_content)
        trigger=[]
        
        start_end_type_list = []
        for i in doc.xpath("//event"):
            assert len(i.xpath(".//anchor"))>0,'len(i.xpath(".//anchor"))>0报错'
            cur_ele = i.xpath(".//anchor")
            event_type = i.xpath("./@TYPE")[0]+'.'+i.xpath("./@SUBTYPE")[0]
           
            for anchor in cur_ele:
                start = anchor.xpath("./charseq/@START")[0]
                end = anchor.xpath("./charseq/@END")[0]
                start = int(start)-begin_index
                end = int(end)- begin_index
                real_str = anchor.xpath("./charseq/text()")[0].replace('\n','')
                my_str = new_text[start:end+1].replace('\r','')
                assert real_str == my_str,real_str+'\t'+my_str+'报错'
                start_end_type_list.append((int(start),int(end)))
        
        print(new_text)
        print(start_end_type_list)
        return content2wordvec(new_text,start_end_type_list)
    except Exception as e:
        print(e)
        print(filename_prefix,'droped')
        return []

    
def prepare_data():
    train_data=[]
#     f=lambda x:x[:x.rfind('.')]
#     f_list=[f(i) for i in os.listdir('./text2')]
    doclist=homepath+'/ace_ch_experiment/doclist/ACE_Chinese_test0';
    f_list=[i.replace('\n','') for i in open(doclist,'r')]
    for i in f_list:
        tmp=read_answer(i)
        if len(tmp)>1:
            train_data.append(tmp)
    train_data=[i for i in train_data if len(i[0])>0]
    rs_f=open('./chACEdata/sentence1.txt','w', encoding='utf8')
    for item in train_data:
        word=item[0]
        strtemp=' '.join([i for i in word])
        rs_f.write(strtemp)
        rs_f.write(str(item[1]))
    new_train_data=[]
    for item in train_data:
        sentence=item[0]
        label=item[1]
        tmp_i=0
        for index,i in enumerate(sentence):
            if i==u'。':
                new_train_data.append((sentence[tmp_i:index],label[tmp_i:index]))
                tmp_i=index+1

    rs_f=open('./chACEdata/sentence2.txt','w', encoding='utf8')
    for item in new_train_data:
        word=item[0]
        rs_f.write(' '.join([i for i in word]))
        rs_f.write(str(item[1]))
    rs_f=open('./chACEdata/trigger_iden.data','wb')
    pickle.dump(new_train_data,rs_f)


'''
将所有ACE内容抽取出来
'''
def writeToTxt():
    doclist=homepath+'/ace_ch_experiment/doclist/ACE_Chinese_all'
    f_list=[i.replace('\n','') for i in open(doclist,'r')]
    for filename in f_list:
        filepath= homepath+'/ace_ch_experiment/corpus/'+filename+".sgm"
        content=open(filepath, 'rb').read().decode('utf-8')
        #除去尖括号等标签，只留下内容
        otherstr=re.findall("<[^>]+>|</[^>]+>", content)
        for rmstr in otherstr:
            content=content.replace(rmstr,'')
        
        newfile=open(homepath+'/ace_ch_experiment/corpus/'+filename+".txt",'w', encoding='utf8')
        newfile.write(content)
        newfile.close()
    
    return 0


if __name__ == '__main__':
    #prepare_data()
    read_answer('/wl/adj/LIUYIFENG_20050126.0844')

    #writeToTxt()
    
#     doclist='D:/Code/pydev/nlp_hw/ace_ch_experiment/doclist/';
#     f_list=[i.replace('\n','') for i in open(doclist+'ACE_Chinese_test0','r')]
#     print(f_list)
    
    