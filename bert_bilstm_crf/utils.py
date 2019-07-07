# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:12:55 2019

@author: eileenlu
"""

import os
from keras_bert import Tokenizer
import codecs
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

cur=os.path.dirname(os.path.abspath(__file__))
token_dict = {}
max_seq_len=500


with codecs.open(os.path.join(cur,r'chinese_L-12_H-768_A-12/vocab.txt'), 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
    
    
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
    
tokenizer = OurTokenizer(token_dict)

def get_x_data(data_path):
    f=open(data_path,'r',encoding='utf-8')
    data=[]
    text=''
    label=[]
    for line in f.readlines():
        lineline=line.strip()
        if(len(lineline)>0):
            t=lineline.split()[0]
            c=lineline.split()[1]
            text=text+t
            label.append(c)
        if(len(lineline)==0):
            data.append([text,label])
            text=''
            label=[]
    
    X=[]
    C=[]
    L=[]
    for d in data:
        dd=d[0]
        ll=['O']
        ll.extend(d[1])
        ll.append('O')
#        tokens = tokenizer.tokenize(dd)
        x,c=tokenizer.encode(first=dd)
        L.append(ll)
        X.append(x)
        C.append(c)  
    XX=pad_sequences(X, maxlen=max_seq_len, dtype='int32',
                                    padding='post', truncating='post', value=0)
    CC=pad_sequences(C, maxlen=max_seq_len, dtype='int32',
                                    padding='post', truncating='post', value=0)    
    return XX,CC,L
    
def get_y_data(tag_data, label2index, max_length):
    index_data = []
    for l in tag_data:
        index_data.append([label2index[s] for s in l])
    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)
    index_array = to_categorical(index_array, num_classes=13) # (20863, 574, 7)

    # return np.expand_dims(index_array, -1)
    return index_array

def get_y_orig(y_pred, y_true):
    label = ['O', 'B_疾病和诊断', 'I_疾病和诊断', 'B_解剖部位', 'I_解剖部位', 'B_实验室检验', 'I_实验室检验','B_影像检查','I_影像检查','B_手术', 'I_手术','B_药物','I_药物']
    index2label = dict()
    idx = 0
    for c in label:
        index2label[idx] = c
        idx += 1
    n_sample = y_pred.shape[0]
    pred_list = []
    true_list = []
    for i in range(n_sample):
        pred_label = [index2label[idx] for idx in np.argmax(y_pred[i], axis=1)]
        pred_list.append(pred_label)
        true_label = [index2label[idx] for idx in np.argmax(y_true[i], axis=1)]
        true_list.append(true_label)
    return pred_list, true_list

def get_entity_index(X_data, y_data, file_path):
    """
    :param X_data: 以character_level text列表为元素的列表
    :param y_data: 以entity列表为元素的列表
    :return: [{'entity': [phrase or word], ....}, ...]
    """
    n_example = len(X_data)
    entity_list = []
    entity_name = ''
#    
    for i in range(n_example):
        d = defaultdict(list)
        s_index=0
        for c, l in zip(X_data[i], y_data[i]):
            s_index=s_index+1
            if l[0] == 'B':
                d[l[2:]].append(str(s_index))
                ad0=d[l[2:]]
                if(len(ad0)>0):
                    d[l[2:]][-1] += ','+str(s_index)
                entity_name += ','+str(s_index)
                
            elif (l[0] == 'I') & (len(entity_name) > 0):
                ad1=d[l[2:]]
                if(len(ad1)>0):
                    d[l[2:]][-1] += ','+str(s_index)
            elif l == 'O':
                entity_name = ''
        entity_list.append(d)
    
    line_no=0
    f=open(file_path,'w',encoding='utf-8')
    for j in entity_list:
        rr=''
        for jj in j.keys():
            value_list=j[jj]
            val_index=[]
            for val in value_list:
                start_pos=int((val.strip(',').split(',')[0]))-1
                end_pos=int(val.strip(',').split(',')[-1])
                text_i=''.join(X_data[line_no][start_pos:end_pos])
                val_after=text_i+'@'+str(start_pos)+'@'+str(end_pos)+'@'+jj
                val_index.append(val_after)
                rr=rr+val_after+';;'
            j[jj]=val_index
        f.write(str(line_no+1)+'@@'+rr+'\n')
        line_no=line_no+1
    f.close()
    return entity_list