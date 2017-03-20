#coding=utf8

from lxml import etree  # @UnresolvedImport
import os



def extract_text():
    '''提取train text到text2文件夹中'''
    f_list=os.listdir('./train/text')

    for i in f_list:
        filepath=os.path.join('./train/text',i)
        f2=open(filepath)
        content=f2.read()
        doc = etree.fromstring(content)
        content2=doc.xpath('//text()')
        _index=i.rfind('.')
        if not os.path.exists('text2'):
            os.mkdir('text2')
        newfname=os.path.join('./text2',i[:_index]+'.txt')
        print(filepath,'===>',newfname)
        f3=open(newfname,'w')
        f3.write(''.join(content2).encode('utf8'))


def read_answer(filename_prefix):
    text_filename= filename_prefix+".txt"
    tag_filename = filename_prefix+'.apf.xml'
    tag_filepath=os.path.join('./train/tag',tag_filename)
    tag_f=open(tag_filepath)
    tag_content=tag_f.read()

    text_filepath=os.path.join('./text2',text_filename)
    text_f=open(text_filepath)
    text_content=text_f.read().decode('utf8')

    doc=etree.fromstring(tag_content)
    try:
        trigger=[]
        for i in doc.xpath("//event"):
            assert len(i.xpath(".//anchor"))>0
            cur_ele=i.xpath(".//anchor")
            event_type=i.xpath("./@TYPE")[0]+'.'+i.xpath("./@SUBTYPE")[0]
            for anchor in cur_ele:
                start = anchor.xpath("./charseq/@START")[0]
                end = anchor.xpath("./charseq/@END")[0]
                real_str=''.join(anchor.xpath("./charseq/text()"))
                mystr=text_content[int(start):int(end)+1]
                assert real_str==mystr
                trigger.append(real_str)
        return trigger
    except AssertionError as e:
        print(e)
        print(filename_prefix , 'droped')
        return []

def pre_my_dict():
    '''得到那些trigger的词汇，保证trigger的词汇都被切分出来'''
    train_data=[]
    f=lambda x:x[:x.rfind('.')]
    f_list=[f(i) for i in os.listdir('text2')]
    for i in f_list:
        tmp=read_answer(i)
        if len(tmp)>0:
            train_data.extend(tmp)
    def clean(x):
        x=x.replace('\n','')
        x=x.replace(' ','')
        return x
    train_data=[clean(i) for i in train_data]
    train_data=set(train_data)
    rs_f=open('./others/train_dict.dict','w')
    for i in train_data:
        print>>rs_f,i.encode('utf8')

if __name__ == '__main__':
   # extract_text()
   pre_my_dict()
   # read_answer("DAVYZW_20050201.1538")
