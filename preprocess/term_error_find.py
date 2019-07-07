# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:41:44 2019

@author: Jolin
"""
import pandas as pd
import os
import codecs
import json

class term_error_find:
    def __init__(self,termerror_path,before_revise_path):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.term_error_path=termerror_path
        self.term_error=pd.read_csv(self.term_error_path,header=0,encoding='gbk')
        self.doc_path=before_revise_path
        self.errorterm_list=list(self.term_error['term'])
        self.errorlabel_list=list(self.term_error['label'])
        
    def find_pos(self):
#        f=open(os.path.join(self.dir_path,'data/term_error_pos.txt'),'w',encoding='utf-8')
        lineno=0
        with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                if(line.strip()!=''):   
                    lineno=lineno+1
                    row_dict=json.loads(line)
                    originalText=row_dict['originalText']
                    entities=row_dict['entities']
                    for entity in entities:
                        label_type=entity['label_type']
                        start=entity['start_pos']
                        end=entity['end_pos']
                        text=originalText[start:end]
                        if(text==''):
                            print('errorline:'+str(lineno)+'***errorterm:'+text+'***errorlabel:'+label_type+'***'+str([start,end])+'\n')
                        if(text in self.errorterm_list):
                            pos=self.errorterm_list.index(text)
                            if(self.errorlabel_list[pos]==label_type):
                                print('errorline:'+str(lineno)+'***errorterm:'+text+'***errorlabel:'+label_type+'***'+str([start,end])+'\n')
#        f.close()
        

        