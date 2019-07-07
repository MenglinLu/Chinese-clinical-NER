# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:50:24 2019

@author: eileenlu
"""

import os

cur=os.path.dirname(os.path.abspath(__file__))
f=open(os.path.join(cur,'data/test.txt'),'r',encoding='utf-8')
len_list=[]

l=0
for line in f.readlines():
    if(len(line.strip().split())>0):
        i=line.strip().split()[0]
        l=l+1
        if(i in '!;。?'):
            len_list.append(l)
            l=0
            
ll=max(len_list)