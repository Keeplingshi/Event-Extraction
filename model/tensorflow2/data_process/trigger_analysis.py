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

doclist_full=homepath+'/ace05/split1.0/new_filelist_ACE_full.txt'


class ACE_info:
    # go gain the event mention from ACE dataset
    def __init__(self):
        self.id = None
        self.text = None
        self.trigger = None
        self.sub_type = None  # sub-type of this event

    def toString(self):
        return 'id:' + str(self.id)  + '\t trigger:' + str(self.trigger) + '\t sub_type:' + str(self.sub_type)+ '\t text:' + str(self.text)
        # return 'id:' + str(self.id) + '\t text:' + str(self.text) + '\t trigger:' + str(
        #     self.trigger) + '\t sub_type:' + str(self.sub_type)


def read_file(xml_path,event_list, event_type):
    apf_tree = ET.parse(xml_path)
    root = apf_tree.getroot()

    # event_list = []

    for events in root.iter("event"):
        ev_type = events.attrib["TYPE"] + "_" + events.attrib["SUBTYPE"]
        if ev_type not in event_type:
            event_type.append(ev_type)
        for mention in events.iter("event_mention"):
            anchor = mention.find("anchor")
            for charseq in anchor:
                text = re.sub(r"\n", r"", charseq.text)
                event_element = ACE_info()
                event_element.sub_type=ev_type
                event_element.text=mention.find("ldc_scope").find("charseq").text.replace("\n"," ")
                event_element.trigger=text
                event_list.append(event_element)
                print(event_element.toString())

    return event_list


def encode_corpus():
    doclist_train_f=[acepath+i.replace('\n','') for i in open(doclist_full,'r')]
    return doclist_train_f


def read_corpus(event_type):
    count = 0
    file_list = encode_corpus()
    event_list=[]
    for file_path in file_list:
        read_file(file_path + ".apf.xml",event_list, event_type)
        count += 1
    print(event_type)


if __name__ == "__main__":

    trigger_type=[None]
    # file_name="nw/timex2norm/APW_ENG_20030311.0775"
    # # file_name="bc/timex2norm/CNN_CF_20030303.1900.00"
    # # file_name="nw/timex2norm/APW_ENG_20030304.0555"
    # xml_path=acepath+file_name+".apf.xml"
    # read_file(xml_path,trigger_argument_type)

    read_corpus(trigger_type)



