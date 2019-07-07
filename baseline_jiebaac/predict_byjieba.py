# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:16:56 2019

@author: eileenlu
"""

import os
import json
import jieba
import jieba.posseg

class predict_byjieba:
    def __init__(self,doc_path,true_path,predict_path):
        self.dir_path=os.path.dirname(os.path.realpath(__file__))
        self.doc_path=doc_path
        self.label_list=['疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术',' ']
        self._jieba=jieba.Tokenizer(dictionary=None)
        self._jieba.set_dictionary(os.path.join(self.dir_path,'data/our_dict1.txt'))
        self._jieba.initialize()
        self._jieba_posseg=jieba.posseg.POSTokenizer(tokenizer=self._jieba)
        self.true_path=true_path
        self.predict_path=predict_path

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