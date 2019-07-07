# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:08:23 2019

@author: eileenlu
"""

import os
import codecs
import json
import pandas as pd
from collections import Counter

###Data analysis
class Data_Analysis:
    
    def __init__(self,doc_path):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.doc_path=doc_path
        self.class_list=['疾病和诊断','影像检查','实验室检验','手术','药物','解剖部位']
         
    ###统计每个doc的文本长度        
    def doc_length(self):
        doc_num=0
        res=[]
        with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                if(line.strip()!=''):
                    doc_num=doc_num+1
                    row_dict=json.loads(line)
                    originalText=row_dict['originalText']
                    length=len(originalText)
                    res.append([doc_num,length])
        res_df=pd.DataFrame(res)
        res_df.columns=['doc_id','length']
        res_df.to_csv(self.dir_path+'/res/doc_length.csv',index=False,encoding='utf-8-sig')
    
    ###判断是否有overlap
    def has_overlap(self):
        doc_num=0
        overlap_num=0
        with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                if(line.strip()!=''):
                    doc_num=doc_num+1
                    row_dict=json.loads(line)
                    entities=row_dict['entities']
                    for entity in entities:
                        is_overlap=entity['overlap']
                        if(is_overlap==1):
                            print('the %d th doc. has overlap' % doc_num)
                            overlap_num=overlap_num+1
        if(overlap_num==0):
            print('No overlap...')
       
#    ###统计各预定义类别出现的次数（未去重）
#    def label_frequency(self):        
#        label_list=[]
#        with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
#            for line in f.readlines():
#                if(line.strip()!=''):   
#                    row_dict=json.loads(line)
#                    entities=row_dict['entities']
#                    for entity in entities:
#                        label_type=entity['label_type']
#                        label_list.append(label_type)
#        res=dict(Counter(label_list))
#        res_df=pd.DataFrame.from_dict(res,orient='index',columns=['frequency'])
#        res_df=res_df.reset_index()
#        res_df.columns=['label_type','frequency']
#        res_df.to_csv(self.dir_path+'/res/label_frequency.csv',encoding='utf-8-sig',index=False)
    
        ###统计不同预定义类别下各词的词频
    def term_frequency(self):
        for label_class in self.class_list:
            label_term=[]
            with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
                for line in f.readlines():
                    if(line.strip()!=''):   
                        row_dict=json.loads(line)
                        originalText=row_dict['originalText']
                        entities=row_dict['entities']
                        for entity in entities:
                            label_type=entity['label_type']
                            start=entity['start_pos']
                            end=entity['end_pos']
                            if(label_type==label_class):
                                label_term.append(originalText[start:end]) 
            term_counter=dict(Counter(label_term))
            res_df=pd.DataFrame.from_dict(term_counter,orient='index',columns=['frequency'])
            res_df=res_df.reset_index()
            res_df.columns=['term','frequency']
            res_df=res_df.sort_values(by='frequency', ascending=False)
            res_df.to_csv(os.path.join(self.dir_path,'res/term_frequency/'+str(label_class)+'_term_frequency.csv'),encoding='utf-8-sig',index=False)
    
    ###分析去重之后各预定义类别的词项个数               
    def type_count(self):
        type_frequency=[]
        for label_class in self.class_list:
            label_term=[]
            with codecs.open(self.doc_path,'r',encoding='utf-8-sig') as f:
                for line in f.readlines():
                    if(line.strip()!=''):   
                        row_dict=json.loads(line)
                        originalText=row_dict['originalText']
                        entities=row_dict['entities']
                        for entity in entities:
                            label_type=entity['label_type']
                            start=entity['start_pos']
                            end=entity['end_pos']
                            if(label_type==label_class):
                                label_term.append(originalText[start:end]) 
            label_term_dropdup=set(label_term)
            type_frequency.append([label_class,len(label_term_dropdup)])
        nodup=pd.DataFrame.from_dict(dict(type_frequency),orient='index',columns=['frequency'])
        nodup=nodup.reset_index()
        nodup.columns=['label_type','frequency']
        nodup.to_csv(self.dir_path+'/res/label_frequency_nodup.csv',encoding='utf-8-sig',index=False)
      
      


###数据分析步骤（data_clean)
###----------------------------------------------
#1. 统计是否存在训练数据集中的overlap--DA.has_overlap() 若存在overlap则修改，若输出 'No overlap...'则说明训练数据集标注中无overlap
#2. 分析各预定义类别下的term及其词频--DA.term_frequency() 分析各预定义类别下是否存在标注错误的term，针对性进行修改，难以处理的term分别列一个表深入分析。都修改完之后进行4
#3. 分析去重之后各预定义类别的词项个数--DA.type_count()
###上述完成了data clean，后续分析overlap的情况


            
                        
        
                    
            
                            