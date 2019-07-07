# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:12:34 2019

@author: eileenlu
"""
import os 
from bert_ner import Bert_ner
from utils import get_x_data, get_y_data, token_dict, get_y_orig, get_entity_index
from evaluate1 import evaluate1

cur=os.path.dirname(os.path.abspath(__file__))
class_dict={'O':0, 'B_疾病和诊断':1, 'I_疾病和诊断':2, 'B_解剖部位':3, 'I_解剖部位':4, 'B_实验室检验':5, 'I_实验室检验':6,'B_影像检查':7,'I_影像检查':8,'B_手术':9, 'I_手术':10,'B_药物':11,'I_药物':12}
max_seq_len=500

config_path = os.path.join(cur, 'chinese_L-12_H-768_A-12/bert_config.json')
checkpoint_path =os.path.join(cur,  'chinese_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(cur, 'chinese_L-12_H-768_A-12/vocab.txt')
model=Bert_ner(config_path,checkpoint_path,dict_path).model_build_bert_crf()
model.load_weights(os.path.join(cur,'model/bert_crf_model.weights'))

XX_test,CC_test,L_test=get_x_data(os.path.join(cur,'data/test.txt'))
LL_test=get_y_data(L_test, class_dict, max_seq_len)
pred1=model.predict([XX_test,CC_test])
pred_list1, true_list1=get_y_orig(pred1,LL_test)
path_true=os.path.join(cur, r'res/true.txt')
path_pre=os.path.join(cur, r'res/pre.txt')
char2index_dict={key:value for key,value in enumerate(token_dict)}
XX_char_test=[[char2index_dict[j] for j in i] for i in XX_test]
get_entity_index(XX_char_test, pred_list1, path_pre)
get_entity_index(XX_char_test, true_list1, path_true)
E=evaluate1()
res=E.evaluate_main(path_true,path_pre)
res.to_csv(os.path.join(cur,r'res/bert_crf_performance.csv'))
