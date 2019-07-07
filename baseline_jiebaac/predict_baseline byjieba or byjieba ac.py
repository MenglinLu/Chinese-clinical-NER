# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:36:07 2019

@author: Jolin
"""

from gen_dict import gen_ourdict
from predict_byjieba import predict_byjieba
from predict_jiebaac import predict_by_jieba_ac
import os
import sys
sys.path.append("..")
from metrics.evaluate1 import evaluate1
import pandas as pd
from metrics.confusion_matrix1 import exact_confusion_matrix, soft_confusion_matrix

dir_path=os.path.dirname(os.path.realpath(__file__))
do_predict_byjieba=1
do_predict_byjiebaac=1

###生成基于训练集的自定义词典
gen_ourdict()
###基于jieba预测
if(do_predict_byjieba==1):
    doc_path=os.path.join(dir_path,'data/subtask1_training_afterrevise_charUnification.txt')
    true_path=os.path.join(dir_path,'res/true_label_byjieba.txt')
    predict_path=os.path.join(dir_path,'res/predict_label_byjieba.txt')
    PJB=predict_byjieba(doc_path,true_path,predict_path)
    PJB.main()
    res=evaluate1(true_path,predict_path)
    res=res.evaluate_main()
    r_df=pd.DataFrame(res,columns=['category','precision','recall','F1','error_predict','not_predict'])
    r_df.to_csv(os.path.join(dir_path,'res/performance_report_byjieba.csv'),index=False,encoding='utf-8-sig')
    cm_exact=exact_confusion_matrix(os.path.join(dir_path,'res/true_label_byjieba.txt'),os.path.join(dir_path,'res/predict_label_byjieba.txt'))
    cm_exact.to_csv(os.path.join(dir_path,'res/confusion_matrix_jieba_exact.csv'),encoding='utf-8-sig')
    cm_soft=soft_confusion_matrix(os.path.join(dir_path,'res/true_label_byjieba.txt'),os.path.join(dir_path,'res/predict_label_byjieba.txt'))
    cm_soft.to_csv(os.path.join(dir_path,'res/confusion_matrix_jieba_soft.csv'),encoding='utf-8-sig')

###基于jiebaac预测    
if(do_predict_byjiebaac==1):
    doc_path=os.path.join(dir_path,'data/subtask1_training_afterrevise_charUnification.txt')
    true_path=os.path.join(dir_path,'res/true_label_byjiebaac.txt')
    predict_path=os.path.join(dir_path,'res/predict_label_byjiebaac.txt')
    PJBAC=predict_by_jieba_ac(doc_path,true_path,predict_path)
    PJBAC.main()
    
    res=evaluate1(true_path,predict_path)
    res=res.evaluate_main()
    r_df=pd.DataFrame(res,columns=['category','precision','recall','F1','error_predict','not_predict'])
    r_df.to_csv(os.path.join(dir_path,'res/performance_report_byjiebaac.csv'),index=False,encoding='utf-8-sig')
    cm_exact=exact_confusion_matrix(os.path.join(dir_path,'res/true_label_byjiebaac.txt'),os.path.join(dir_path,'res/predict_label_byjiebaac.txt'))
    cm_exact.to_csv(os.path.join(dir_path,'res/confusion_matrix_jiebaac_exact.csv'),encoding='utf-8-sig')
    cm_soft=soft_confusion_matrix(os.path.join(dir_path,'res/true_label_byjiebaac.txt'),os.path.join(dir_path,'res/predict_label_byjiebaac.txt'))
    cm_soft.to_csv(os.path.join(dir_path,'res/confusion_matrix_jiebaac_soft.csv'),encoding='utf-8-sig')

    
    



