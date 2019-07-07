# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:15:52 2019

@author: Jolin
"""

import os
import json
import jieba
import jieba.posseg
from ac_automaton import ACA
import pandas as pd
import numpy as np

class predict_by_jieba_ac:
    def __init__(self,doc_path,true_path,predict_path):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.doc_path=doc_path
        self.true_path=true_path
        self.predict_path=predict_path
        self.label_list=['疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术','@']
        self._jieba=jieba.Tokenizer(dictionary=None)
        self._jieba.set_dictionary(os.path.join(self.dir_path,'data/our_dict1.txt'))
        self._jieba.initialize()
        self._jieba_posseg=jieba.posseg.POSTokenizer(tokenizer=self._jieba)
        self.aca: ACA = ACA()
        type_list=['疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
        self.term_list=[]
        self.term_label_dict=dict()
        for typee in type_list:
            file_i=pd.read_csv(open(os.path.join(os.path.dirname(self.dir_path),'analysis/res/term_frequency/'+typee+'_term_frequency.csv'),encoding='utf-8-sig'),header=0)
            self.term_list.extend(file_i['term'])
            for term_i in file_i['term']:
                self.term_label_dict[term_i]=typee
        self.aca.add_words(self.term_list)
        
    def ac_match(self,sentence):
        res=[]
        for last_idx, term in self.aca.get_hits_with_index(sentence):
            res.append([last_idx+1-len(term),last_idx+1,term,self.term_label_dict[term]])
        res_final=[]
        for i in res:
            flag=0
            start_pos_i=i[0]
            end_pos_i=i[1]
            for j in res:
                if(i!=j):
                    start_pos_j=j[0]
                    end_pos_j=j[1]
                    if(start_pos_i>=start_pos_j and end_pos_i<=end_pos_j):
                        flag=1
            if(flag==0):
                _tmp=dict()
                _tmp['source_word']=i[2]
                _tmp['category']=i[3]
                _tmp['start_pos']=i[0]
                _tmp['end_pos']=i[1]
                if(_tmp['category'] in self.label_list):
                    res_final.append(_tmp)
        return res_final
                  
    def get_nes(self,sentence, _jieba_posseg):
        
        ne_pairs = _jieba_posseg.lcut(sentence, HMM=False)
        pos_start=0
        nes = list()
        for word, flag in ne_pairs:
            now_pairs=dict()
            now_pairs['source_word']=word
            now_pairs['category']=flag
            now_pairs['start_pos']=pos_start
            now_pairs['end_pos']=now_pairs['start_pos']+len(word)
            pos_start=now_pairs['end_pos']
            if(flag in self.label_list):
                nes.append(now_pairs)
        global aa
        aa=nes
        res_jieba=nes
#        f123=open(r'overlap_between_jieba_ac.txt','w',encoding='utf-8')
        res_ac=self.ac_match(sentence)
        for i_jieba in res_jieba:
            start_pos_jieba=i_jieba['start_pos']
            end_pos_jieba=i_jieba['end_pos']
            jiebajieba_list=list(np.arange(start_pos_jieba,end_pos_jieba))
            for i_ac in res_ac:
                start_pos_ac=i_ac['start_pos']
                end_pos_ac=i_ac['end_pos']
                acac_list=list(np.arange(start_pos_ac,end_pos_ac))
                list_intersection=list(set(acac_list) & set(jiebajieba_list))
                
                ###输出含有overlap的部分
                if(len(list_intersection)>0 and jiebajieba_list != acac_list):
                    print(sentence+'***'+str([i_jieba,i_ac])+'\n')
#                    b123=sentence+json.dumps(i_jieba)+json.dumps(i_ac)
                    
                if(sorted(list_intersection)==sorted(jiebajieba_list) and jiebajieba_list != acac_list):
                    if(i_ac not in nes):
                        nes=[i_ac if x==i_jieba else x for x in nes]
                    if(i_ac in nes and i_jieba in nes):
                        nes.remove(i_jieba)
                    break
#        f123.close()
        return nes
    
    def predict(self):
        f1_path=self.true_path
        f2_path=self.predict_path
        f1=open(f1_path,'w',encoding='utf-8')
        f2=open(f2_path,'w',encoding='utf-8')
        i=0
        with open(self.doc_path,'r',encoding='utf-8-sig') as f:
            for line in f.readlines():
                i=i+1
                res_line=str(i)+'@@'
                line_dict=json.loads(line)
                originaltext=line_dict['originalText']
                entities=line_dict['entities']
                for entity in entities:
                    res_line=res_line+originaltext[entity['start_pos']:entity['end_pos']]+'@'+str(entity['start_pos'])+'@'+str(entity['end_pos'])+'@'+entity['label_type']+';;'
                res_line=res_line+'\n'
                f1.write(res_line)
                     
                nes=self.get_nes(originaltext,self._jieba_posseg)
                res_line2=str(i)+'@@'
                for nes_i in nes:
                    res_line2=res_line2+nes_i['source_word']+'@'+str(nes_i['start_pos'])+'@'+str(nes_i['end_pos'])+'@'+nes_i['category']+';;'
                res_line2=res_line2+'\n'
                f2.write(res_line2)
        
        f1.close()
        f2.close()        

###基于jieba的NER
    def main(self):
        self.predict()