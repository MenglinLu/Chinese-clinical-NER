# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:12:12 2019

@author: eileenlu
"""

import os
from utils import get_x_data, get_y_data
from bert_ner import Bert_ner

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

cur=os.path.dirname(os.path.abspath(__file__))
#cur=r'F:\bbbert'
class_dict={'O':0, 'B_疾病和诊断':1, 'I_疾病和诊断':2, 'B_解剖部位':3, 'I_解剖部位':4, 'B_实验室检验':5, 'I_实验室检验':6,'B_影像检查':7,'I_影像检查':8,'B_手术':9, 'I_手术':10,'B_药物':11,'I_药物':12}
max_seq_len=500

XX,CC,L=get_x_data(os.path.join(cur,'data/train.txt'))
XX_dev,CC_dev,L_dev=get_x_data(os.path.join(cur,'data/dev.txt'))
LL=get_y_data(L, class_dict, max_seq_len)
LL_dev=get_y_data(L_dev, class_dict, max_seq_len)
config_path = os.path.join(cur, 'chinese_L-12_H-768_A-12/bert_config.json')
checkpoint_path =os.path.join(cur,  'chinese_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(cur, 'chinese_L-12_H-768_A-12/vocab.txt')
model=Bert_ner(config_path,checkpoint_path,dict_path).model_build_bert_dense()
model.fit(x=[XX,CC],y=LL,batch_size=32,epochs=3,validation_data=([XX_dev,CC_dev],LL_dev))
model.save_weights(os.path.join(cur,'bert_dense.weights'))