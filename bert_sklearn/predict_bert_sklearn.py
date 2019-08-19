# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:20:20 2019

@author: eileenlu
"""


# score model on dev data

import os
import pandas as pd
from bert_sklearn import load_model
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,7"

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_CoNLL2003_format(filename, idx=3):
    """Read file in CoNLL-2003 shared task format"""
    # read file
    lines =  open(filename,encoding='utf-8').read().strip()
    
    # find sentence-like boundaries
    lines = lines.split("\n\n")  
    
     # split on newlines
    lines = [line.split("\n") for line in lines]
    
    # get tokens
    tokens = [[l.split()[0] for l in line] for line in lines]
    
    # get labels/tags
    labels = [[l.split()[idx] for l in line] for line in lines]
    
    #convert to df
    data= {'tokens': tokens, 'labels': labels}
    df=pd.DataFrame(data=data)
    
    return df

cur=os.path.dirname(os.path.abspath(__file__))
DATADIR=os.path.join(cur,'data')
def get_data(file_path):

    data = read_CoNLL2003_format(file_path, 1)
    print("Test data: %d sentences, %d tokens"%(len(data),len(flatten(data.tokens))))
    return data

test = get_data(DATADIR+'/task1 test2.txt')
X_test, y_test = test['tokens'], test['labels']

cur=os.path.dirname(os.path.abspath(__file__))
model=load_model(os.path.join(cur,r'checkpoint/bert_sklearn2.h5'))

# get predictions on test data
# calculate the probability of each class
#y_probs = model.predict_proba(X_test)

# print report on classifier stats
#print(classification_report(flatten(y_test), flatten(y_preds)))

from collections import defaultdict
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
    return 0

def get_submit(X_data, y_data, original_path, file_path):
    """
    :param X_data: 以character_level text列表为元素的列表
    :param y_data: 以entity列表为元素的列表
    :return: [{'entity': [phrase or word], ....}, ...]
    """
    char_test_line=[]
    for line in open(original_path,encoding='utf-8').readlines():
        if(line.strip()!=''):
            textt=json.loads(line)
            char_test_line.append(textt['originalText'])
            
    
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
        res_i_json=dict()
        res_i_json['originalText']=char_test_line[line_no]
        entity_res=[]
        for jj in j.keys():
            value_list=j[jj]
            for val in value_list:
                entity_i=dict()
                start_pos=int((val.strip(',').split(',')[0]))-1
                end_pos=int(val.strip(',').split(',')[-1])
                entity_i['start_pos']=start_pos
                entity_i['end_pos']=end_pos
                entity_i['label_type']=jj
                entity_res.append(entity_i)
        res_i_json['entities']=entity_res
        res_i=json.dumps(res_i_json,ensure_ascii=False)
        f.write(res_i+'\n')
        line_no=line_no+1
    f.close()
    return 0

y_preds = model.predict(X_test)
y_preds1=[[str(j) for j in i] for i in y_preds]
x_data=[list(i) for i in X_test]
y_test1=[[str(j) for j in i] for i in y_test]

predict_path=os.path.join(cur,r'res/bert_sklearn_pred_test.txt')
true_path=os.path.join(cur,r'res/bert_sklearn_true_test.txt')
#get_entity_index(x_data, y_preds1, predict_path)
#get_entity_index(x_data, y_test1, true_path)
get_submit(x_data, y_preds1, 'data/task1 test.json', 'res/result.json')

#E=evaluate1(true_path,predict_path)
#res=E.evaluate_main()
#res.to_csv('res/bert_sklearn_performance.csv',encoding='utf-8-sig')
