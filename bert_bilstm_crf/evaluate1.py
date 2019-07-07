# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:55:32 2019

@author: eileenlu
"""

import os
import pandas as pd

class evaluate1:
    
    def __init__(self,true_file_path,predict_file_path):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.label_list=['疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术','@']
        self.true_file=true_file_path
        self.predict_file=predict_file_path

    def evaluate(self,category):
        true_file=open(self.true_file,'r',encoding='utf-8')
        predict_file=open(self.predict_file,'r',encoding='utf-8')
        category_true=[]
        category_predict=[]
        for (line_true, line_predict) in zip(true_file.readlines(), predict_file.readlines()):
    #        print(line_true)
            if(line_true.strip()!=''):
                line_no_true=line_true.split('@@')[0]
                entity_true=line_true.strip().split('@@')[1].split(';;')[:-1]
                for i in entity_true:
                    if(category in i):
                        category_true.append(str(line_no_true)+'@@'+i)
            if(line_predict.strip()!=''):
                line_no_predict=line_true.split('@@')[0]
                entity_predict=line_predict.strip().split('@@')[1].split(';;')[:-1]
                for j in entity_predict:
                    if(category in j):
                        category_predict.append(str(line_no_predict)+'@@'+j)
    
        ###求准确率
        len_predict=len(category_predict)
        len_predicttrue=0
        error_predict=[]
        for iii in category_predict:
            if(iii in category_true):
                len_predicttrue=len_predicttrue+1
            if(iii not in category_true):
                error_predict.append(iii)
        if(len_predict>0):
            precision=len_predicttrue*1.0/len_predict
        else:
            precision=-1
        ###求召回率
        len_true=len(category_true)
        not_predict=[]
        not_predict_1=[]
        predict_true=set(category_predict) & set(category_true)
        for i_i in category_predict:
            if(i_i not in predict_true):
                not_predict_1.append(i_i)
        for i_j in category_true:
            if(i_j not in predict_true):
                not_predict.append(i_j)
        
        if(len_true>0):
            recall=len(predict_true)/len_true
        else:
            recall=-1
        ###求F1
        if(precision+recall>0):
            f1=2*precision*recall/(precision+recall)
        else:
            f1=-1
        
        return [precision, recall, f1, error_predict, not_predict]
    
    def evaluate_main(self):
        rrres=[]
        for typee in self.label_list:
            c_res=self.evaluate(typee)
            p=c_res[0]
            r=c_res[1]
            f1=c_res[2]
            error_predict=c_res[3]
            not_predict=c_res[4]
            rrres.append([typee,p,r,f1,error_predict,not_predict])
        r_df=pd.DataFrame(rrres,columns=['category','precision','recall','F1','error_predict','not_predict'])
        return r_df
    
    