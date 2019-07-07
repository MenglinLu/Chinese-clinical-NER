# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:00:54 2019

@author: Jolin
"""
import os

class error_revise:
    def __init__(self,indexerror_path,before_revise_path,after_revise_path):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.index_error_path=indexerror_path
        self.before_revise_path=before_revise_path
        self.after_revise_path=after_revise_path
        
    def revise(self):
        index_error=open(self.index_error_path,'r',encoding='utf-8')
        error_lineno_list=[]
        error_index_list=[]
        true_index_list=[]
        for error_line in index_error.readlines():
    #        break
            lineno=int(error_line.split(':')[1])
            errorindex=[int(iii) for iii in list(error_line.split(':')[2].split('-'))]
            trueindex=[int(jjj) for jjj in list(error_line.split(':')[3].split('***')[0].split('-'))]
            error_lineno_list.append(lineno)
            error_index_list.append(errorindex)
            true_index_list.append(trueindex)
        f=open(self.before_revise_path,'r',encoding='utf-8')
        ff=open(self.after_revise_path,'w',encoding='utf-8')
        i=0
        for line in f.readlines():
            i=i+1
            line=line.strip()
            if(i in error_lineno_list):
                id1 = [j for j,x in enumerate(error_lineno_list) if x==i]
                for no in id1:
                    line=line.replace('"start_pos": '+str(error_index_list[no][0]),'"start_pos": '+str(true_index_list[no][0]))
                    line=line.replace('"end_pos": '+str(error_index_list[no][1]),'"end_pos": '+str(true_index_list[no][1]))
            ff.write(line+'\n')
                
        f.close()
        ff.close()

