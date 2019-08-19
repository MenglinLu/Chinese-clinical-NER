# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:18:10 2019

@author: Jolin
"""
###数据处理第一步
###数据转换（将originalText中的中文标点符号全部转换成英文，大写英文全部转换成小写英文）
import os
import codecs
import json
import re

class data_consistent:
    
    def __init__(self,original_doc_path,output_doc_path,charunification=False):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.doc_path=original_doc_path
        self.output_path=output_doc_path
        self.rep_char=re.compile(r'[^\u4e00-\u9fa5^A-Z^a-z^0-9^。^!^%^?]')
        self.table={ord(f):ord(t) for f,t in zip(u'，！？【】（）％＃＠＆１２３４５６７８９０：', u',!?[]()%#@&1234567890:')}
        self.rep_num=re.compile(r'[0-9]')
        self.charunification=charunification
    ###标点符号全部转换成英文
    def punctuation_translate(self,sent:str) -> str:
        sent_1=sent.translate(self.table)
        return sent_1
    
    ###全角转半角
    def strQ2B(self,sent:str) -> str:
        ss = []
        for s in sent:
            rstring = ""
            for uchar in s:
                
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif inside_code==12290:
                    inside_code=12290
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        res=''.join(ss)
        return res
    
    def lower(self,sent:str) -> str:
        sent_2=sent.lower()
        return sent_2
    
    def main_consistent(self):
        f_1=open(self.output_path,'w',encoding='utf-8')
        with codecs.open(self.doc_path, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                if(line.strip()!=''):  
                    row_dict_1=dict()
                    row_dict=json.loads(line)
                    originalText=row_dict['originalText']
                    entities=row_dict['entities']
                    originalText_1=self.punctuation_translate(originalText)
                    originalText_2=self.strQ2B(originalText_1)
                    originalText_3=self.lower(originalText_2)
                    originalText_3=originalText_3.replace(' ','叒')
                    if(self.charunification):
                        originalText_4=re.sub(self.rep_char,'Q',originalText_3)
                        originalText_4=re.sub(self.rep_num,'0',originalText_4)
                    else:
                        originalText_4=originalText_3
                    row_dict_1['originalText']=originalText_4
                    row_dict_1['entities']=entities
                    line_1=json.dumps(row_dict_1,ensure_ascii=False)
                    f_1.write(line_1+'\n')
        f_1.close()
        f.close()
        
if __name__=='__main__':
    DC=data_consistent(r'data/test_sample.txt',r'data/test_sample3.txt',charunification=False)
    DC.main_consistent()

        