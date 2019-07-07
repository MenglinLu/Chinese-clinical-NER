# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:28:30 2019

@author: eileenlu
"""

import os
import codecs
import json

###将训练集数据转化为BIO格式
class Data_Format(object):
    def __init__(self,origin_path,after_path):
        self.origin_path=origin_path
        self.after_path=after_path
              
    def formatt(self):
        with codecs.open(self.after_path,'w',encoding='utf-8') as ff:
            with codecs.open(self.origin_path,'r',encoding='utf-8-sig') as f:
                for line in f.readlines():
                    if(line.strip()!=''):
                        line_dict=json.loads(line)
                        origin_text=line_dict['originalText']
                        entities=line_dict['entities']
                        label_text=['O' for _ in range(len(origin_text))]
                        for entity in entities:
                            label_type=entity['label_type']
                            start_pos=entity['start_pos']
                            end_pos=entity['end_pos']
                            label_text[start_pos]='B_'+label_type
                            for j in range(start_pos+1,end_pos):
                                label_text[j]='I_'+label_type
                        for ii in range(len(origin_text)):
                            ff.write(origin_text[ii] + ' ' + str(label_text[ii]) + '\n')
                            if(origin_text[ii] in '!?。;'):
                                ff.write('\n')
        ff.close()

if __name__=='__main__':
    dir_path=os.path.dirname(os.path.realpath(__file__))
    origin_path=os.path.join(dir_path,'data/dev.txt')
    after_transfer_path=os.path.join(dir_path,'data/dev_after.txt')
    DF=Data_Format(origin_path,after_transfer_path)
    DF.formatt()