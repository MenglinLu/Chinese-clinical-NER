# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:01:37 2019

@author: eileenlu
"""

import pandas as pd
import os
import codecs

def gen_ourdict():
    dir_path=os.path.dirname(os.path.realpath(__file__))
    type_list=['疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
    termterml=[]
    with codecs.open(os.path.join(dir_path,'data/our_dict1.txt'),'w',encoding='utf-8') as f:
        for typee in type_list:
            df_i=pd.read_csv(open(os.path.join(os.path.dirname(dir_path),'analysis/res/term_frequency/'+typee+'_term_frequency.csv'),encoding='utf-8-sig'),header=0)
            term_list=list(df_i['term'])
            for term in term_list:
                termterml.append(term)
                f.write(term+'@@'+str(100000 + (len(term) - 1) * 10000)+'@@'+typee+'\n')
        with open(os.path.join(dir_path,'data/jieba_dict.txt'),'r',encoding='utf-8') as ff:
            for line in ff.readlines():
                wo=line.split(' ')[0]
                fre=line.split(' ')[1]
                tag=line.split(' ')[2]
                if(wo not in termterml):
                    f.write(wo+'@@'+fre+'@@'+tag)
    f.close()

###生成自定义词典
#gen_ourdict()

    
    