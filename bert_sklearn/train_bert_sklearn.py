# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:31:26 2019

@author: eileenlu
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from bert_sklearn import BertTokenClassifier, load_model

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

def get_data(file_path):

    data = read_CoNLL2003_format(file_path, 1)
    print("Test data: %d sentences, %d tokens"%(len(data),len(flatten(data.tokens))))
    return data

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    cur=os.path.dirname(os.path.abspath(__file__))
    train_path=os.path.join(cur,'data/train.txt')
    dev_path=os.path.join(cur,'data/dev.txt')
    test_path=os.path.join(cur,'data/test.txt')
    train, dev, test = get_data(train_path),get_data(dev_path),get_data(test_path)
    
    X_train, y_train = train['tokens'], train['labels']
    X_dev, y_dev = dev['tokens'], dev['labels']
    X_test, y_test = test['tokens'], test['labels']
    
    label_list = np.unique(flatten(y_train))
    label_list = list(label_list)
    
    model = BertTokenClassifier(bert_model='bert-base-chinese',
                                epochs=20,
                                learning_rate=2e-5,
                                train_batch_size=16,
                                eval_batch_size=16,
                                ignore_label=['O'])
    
    print("Bert wordpiece tokenizer max token length in train: %d tokens"% model.get_max_token_len(X_train))
    print("Bert wordpiece tokenizer max token length in dev: %d tokens"% model.get_max_token_len(X_dev))
    print("Bert wordpiece tokenizer max token length in test: %d tokens"% model.get_max_token_len(X_test))
    
    
    model.max_seq_length = 512
    print(model)
    
    # finetune model on train data
    model.fit(X_train, y_train)
    model.save(os.path.join(cur,r'checkpoint/bert_sklearn07311.h5'))
    
    f1_dev = model.score(X_dev, y_dev)
    print("Dev f1: %0.02f"%(f1_dev))
    
    # score model on test data
    f1_test = model.score(X_test, y_test)
    print("Test f1: %0.02f"%(f1_test))
