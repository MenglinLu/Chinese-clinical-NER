# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:09:30 2019

@author: Jolin
"""

###1.数据一致化
###2.index_error发现与修正
###3.term_error发现与修正
import os
from data_consistent import data_consistent
from index_error_find import index_error_find
from index_error_revise import error_revise
from term_error_find import term_error_find

do_consistent=1
do_indexerror_find=0
do_indexerror_revise=0
do_termerror_find=0
dir_path=os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    if(do_consistent==1):
        doc_path=os.path.join(dir_path,'data_origin/subtask1_training_afterrevise.txt',charunification=False)
        output_path=os.path.join(dir_path,'data_after/subtask1_training_afterrevise_charunification.txt')
        DC=data_consistent(doc_path,output_path)
        DC.main_consistent()
    
    if(do_indexerror_find==1):
        doc_path=os.path.join(dir_path,'data_before/subtask1_training_afterrevise.txt')
        indexerror_output_path=os.path.join(dir_path,'middle/index_error.txt')
        index_error=index_error_find(doc_path,indexerror_output_path)
        index_error.index_error()
        
    if(do_indexerror_revise==1):
        index_error_path=os.path.join(dir_path,'middle/index_error_revise.txt')
        befor_revise_path=os.path.join(dir_path,'data_before/subtask1_training_afterconsistent.txt')
        after_revise_path=os.path.join(dir_path,'data_after/subtask1_training_afterrevise.txt')
        ER=error_revise(index_error_path,befor_revise_path,after_revise_path)
        ER.revise()
        
    if(do_termerror_find==1):
        termerror_path=os.path.join(dir_path,'middle/term_error.csv')
        before_path=os.path.join(dir_path,'data_after/subtask1_training_afterrevise.txt')
        TER=term_error_find(termerror_path,before_path)
        TER.find_pos()
        
    
        
    
        
    

    
