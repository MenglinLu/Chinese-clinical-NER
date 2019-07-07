# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:50:59 2019

@author: Jolin
"""

import os
import json
import matplotlib.pyplot as plt

def pos_distri(doc_path,pattern='relative'):
#pattern_list=['absolute pos','relative pos']
#pattern='absolute pos'
    print('Two pattern: relative and absolute\n')
    print('Now pattern: '+pattern+'\n')
    class_list=['疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
    class_pos_list=[[],[],[],[],[],[]]
    dir_path=os.path.dirname(os.path.abspath(__file__))
    f=open(doc_path,'r',encoding='utf-8')
    for line in f.readlines():
       if(line.strip()!=''):   
           row_dict=json.loads(line)
           originalText=row_dict['originalText']
           len_text=len(originalText)
           entities=row_dict['entities']
           for entity in entities:
               label_type=entity['label_type']
               start=entity['start_pos']
               end=entity['end_pos']
               if(pattern=='absolute'):
                   median=(start+end-1)/2
               if(pattern=='relative'):
                   median=((start+end-1)/2)/len_text*100
               class_pos_list[class_list.index(label_type)].append(median)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig=plt.figure(figsize=(30,12))
    for i in range(len(class_list)):
        a=class_pos_list[i]
        hueHist = plt.subplot(2,3,i+1)
        num_bins = 300
        hueHist.hist(a, num_bins, facecolor='red')
        hueHist.set_title(class_list[i],fontsize=12,color='black') 
    plt.show()
    fig.savefig(os.path.join(dir_path,'res/pos_distribution_'+pattern+'.png'))


           