# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:48:46 2019

@author: Jolin
"""
import os
import codecs
import json

class index_error_find:
    def __init__(self,doc_path,indexerror_output_path):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.doc_path=doc_path
        self.index_error_path=indexerror_output_path
        self.chs_punctuations = '＃＄＆＇＊＋，－／：；＝＠＼＾＿｀｜ ～､　、〜〟〰〾〿–—‛„‟…‧﹏﹑﹔·！？｡。'
        self.eng_punctuations = '!"#$&\'*+,-./:;=\?@\\\^_`|~'
        self.double_match_left=['"','(','（','“','[','【']
        self.double_match_right=['"',')','）','”',']','】']
    
    ###分析是否存在index标注的错误（空格以及首尾的标点符号）
    def index_error(self):
        f_index=open(self.index_error_path,'w',encoding='utf-8')
        with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
            line_no=0
            for line in f.readlines():
                line_no=line_no+1
                if(line.strip()!=''):   
                    row_dict=json.loads(line)
                    originalText=row_dict['originalText']
                    entities=row_dict['entities']
                    for entity in entities:
                        start=entity['start_pos']
                        end=entity['end_pos']
                        a=originalText[start:end].strip(self.chs_punctuations).strip(self.eng_punctuations).strip()
                        b=originalText[start:end]
                        if(a!=b):
                            f_index.write('errorline:'+str(line_no)+':'+str(start)+'-'+str(end)+'***'+'errorentity:'+'***'+b+'trueentity:'+'***'+a+'\n')
                        for i_i in range(len(self.double_match_left)):
                            c=b.count(self.double_match_left[i_i])
                            d=b.count(self.double_match_right[i_i])
                            if(c!=d):
                                f_index.write('errorline:'+str(line_no)+':'+str(start)+'-'+str(end)+'***'+'errorentity:'+'***'+b+'\n')                    
        f_index.close()
        f.close()
