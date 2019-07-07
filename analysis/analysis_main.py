# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:11:09 2019

@author: Jolin
"""

###term_frequency的统计分析
###overlap的情况分析

from data_analysis import Data_Analysis
from overlap import overlap_betweentype, overlap_intype, type_whole_overlap
import os
from pos_distribution import pos_distri

do_analysis=1
do_overlap_betweentype=1
do_overlap_intype=1
do_whole_overlap=1
do_posdistri_analysis=1
dir_path=os.path.dirname(os.path.realpath(__file__))

if __name__=='__main__':
    
    doc_path=os.path.join(dir_path,'data/subtask1_training_afterrevise.txt')
    if(do_analysis==1):
        DA=Data_Analysis(doc_path)
        DA.has_overlap()
        DA.term_frequency()
        DA.type_count()
        
    if(do_overlap_betweentype==1):
        OBT=overlap_betweentype()
        OBT.main_overlap_between()
        
    if(do_overlap_intype==1):
        OL=overlap_intype()
        OL.overlap()
        
    if(do_whole_overlap==1):
        TWO=type_whole_overlap()
        TWO.type_overlap()
        
    if(do_posdistri_analysis):
        pos_distri(doc_path,pattern='relative')
        
