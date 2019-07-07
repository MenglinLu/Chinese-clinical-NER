# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:11:47 2019

@author: eileenlu
"""
import pandas as pd
import os 
###分析是否存在类别间完全的overlap，即某个term同时出现在两个预定义类别中，一般可辅助判断标注错误
class type_whole_overlap:
    def __init__(self):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.term_frequency_csv_list=os.listdir(os.path.join(self.dir_path,'res/term_frequency'))
    
    def type_overlap(self):
        term_list_csv=[]
        for csv in self.term_frequency_csv_list:
            term_list_i=set(list(pd.read_csv(open(os.path.join(self.dir_path+'/res/term_frequency',csv),encoding='utf-8-sig'),header=0)['term']))
            term_list_csv.append(term_list_i)
        overlap_list=[]
        for i in range(len(term_list_csv)):
            for j in range(i+1,len(term_list_csv)):
                i_termlist=term_list_csv[i]
                j_termlist=term_list_csv[j]
                overlap_term=list(i_termlist & j_termlist)
                for iii in overlap_term:
                    overlap_list.append([iii,self.term_frequency_csv_list[i],self.term_frequency_csv_list[j]])
        nodup=pd.DataFrame(overlap_list)
        nodup=nodup.reset_index()
        if(len(nodup)>0):
            nodup.columns=['index','term','label_type_1','label_type_2']
        nodup.to_csv(self.dir_path+'/res/whole_overlap.csv',encoding='utf-8-sig',index=False)


###分析类别内部存在的overlap
class overlap_intype:
    def __init__(self):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.term_frequency_csv_list=os.listdir(os.path.join(self.dir_path,'res/term_frequency'))
          
    def overlap(self):
        for csv in self.term_frequency_csv_list:
            res=[]
            term_list=set(list(pd.read_csv(open(os.path.join(self.dir_path+'/res/term_frequency',csv),encoding='utf-8-sig'),header=0)['term']))
            for term in term_list:
                res_term=[]
                for term_i in term_list:
                    if(term in term_i and term!=term_i):
                        res_term.append(term_i)
                if(len(res_term)>0):
                    res.append([term,res_term])
            res_df=pd.DataFrame(res)
            res_df.columns=['term','overlap_term']
            res_df.to_csv(os.path.join(self.dir_path+'/res/overlap_intype',csv).replace('term_frequency.csv','overlap.csv'),index=False,encoding='utf-8-sig')


###分析类别之间存在的overlap
class overlap_betweentype:
    def __init__(self):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.term_frequency_csv_list=os.listdir(os.path.join(self.dir_path,'res/term_frequency'))

    def type_overlap(self,file_path1,file_path2):
        term_list1=set(list(pd.read_csv(open(file_path1,encoding='utf-8-sig'),header=0)['term']))
        term_list2=set(list(pd.read_csv(open(file_path2,encoding='utf-8-sig'),header=0)['term']))
        res1=[]
        res2=[]
        for term in term_list1:
            res_term=[]
            for term_i in term_list2:
                if(term in term_i):
                    res_term.append(term_i)
            if(len(res_term)>0):
                res1.append([term,res_term])
        res_df1=pd.DataFrame(res1)
        if(len(res_df1)>0):
            res_df1.columns=['term','overlap_term']
        
        for term in term_list2:
            res_term=[]
            for term_i in term_list1:
                if(term in term_i):
                    res_term.append(term_i)
            if(len(res_term)>0):
                res2.append([term,res_term])
        res_df2=pd.DataFrame(res2)
        if(len(res_df2)>0):
            res_df2.columns=['term','overlap_term']
        
        csv1=file_path1.split('//')[-1].split('_')[0]
        csv2=file_path2.split('//')[-1].split('_')[0]
        writer = pd.ExcelWriter(os.path.join(self.dir_path,'res/overlap_betweentype/'+csv1+'+'+csv2+'.xlsx'))
        res_df1.to_excel(writer,sheet_name=csv1+'&'+csv2,index=False,encoding='utf-8-sig')
        res_df2.to_excel(writer,sheet_name=csv2+'&'+csv1,index=False,encoding='utf-8-sig')
        writer.save()
        
    def main_overlap_between(self): 
        for csv_i in range(len(self.term_frequency_csv_list)):
            file_path_i=os.path.join(self.dir_path,'res/term_frequency')+'//'+self.term_frequency_csv_list[csv_i]
            for csv_j in range(csv_i+1,len(self.term_frequency_csv_list)):
                file_path_j=os.path.join(self.dir_path,'res/term_frequency')+'//'+self.term_frequency_csv_list[csv_j]
                self.type_overlap(file_path_i,file_path_j)


###分析overlap的步骤-----
#1. 分析是否存在类别间完全的overlap，即一个词是否同时属于两个预定义类别，辅助判断标注错误，对原数据集进行修改
#2. 在data clean得到预处理后数据集的基础上，分析类别内部的部分overlap和类别间的部分overlap，得到分析结果


