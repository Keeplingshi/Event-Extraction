import os
import shutil

homepath='D:/Code/pycharm/Event-Extraction/'
acepath=homepath+'ace05/data/English/'
experiment_path=homepath+"ace_en_experiment/"

doclist_train=homepath+'/ace05/split1.0/new_filelist_ACE_full.txt'

def encode_corpus():
    doclist_f=[acepath+i.replace('\n','') for i in open(doclist_train,'r')]
    for file_path in doclist_f:
        apf_file=file_path + ".apf.xml"
        sgm_file=file_path + ".sgm"
        file_dir=apf_file.rpartition("/")[0]
        apf_file_name=apf_file.rpartition("/")[2]
        sgm_file_name=sgm_file.rpartition("/")[2]

        file_dir=file_dir.replace(acepath,experiment_path)

        # 判断目录是否存在
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        shutil.copy(apf_file, file_dir+"/"+apf_file_name)
        shutil.copy(sgm_file, file_dir+"/" + sgm_file_name)


if __name__ == "__main__":
    encode_corpus()
