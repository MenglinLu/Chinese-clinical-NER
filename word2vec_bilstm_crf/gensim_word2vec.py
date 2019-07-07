# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:42:14 2019

@author: Jolin
"""

from gensim.models import word2vec
import os
import json

cur=os.path.dirname(os.path.abspath(__file__))
doc_path=os.path.join(cur,'data/subtask1_training_afterrevise.txt')
txt_path=os.path.join(cur,'data/text.txt')

f1=open(txt_path,'w',encoding='utf-8')
with open(doc_path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        json_dict=json.loads(line)
        line_text=json_dict['originalText']
        l=' '.join(list(line_text.strip()))
        f1.write(l+'\n')
f1.close()

###word2vec
sentences = word2vec.LineSentence(txt_path)
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=4,size=200) 

#model.save(r'F:/CCKS_Clinic/word2vec/v2/word2vec.model')

#model=gensim.models.Word2Vec.load(r'F:/CCKS_Clinic/word2vec/v2/word2vec.model')
model.wv.save_word2vec_format(os.path.join(cur,'data/word2vec.bin'),binary=False)
###data_format
#训练集

#测试集


        
    