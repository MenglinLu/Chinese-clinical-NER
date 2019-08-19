# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:19:02 2019

@author: eileenlu
"""
from collections import defaultdict

        entity_list = []
        entity_name = ''
#    

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
                isend=0
                
            elif (l[0] == 'I') & (len(entity_name) > 0) & (isend != 1):
                ad1=d[l[2:]]
                if(len(ad1)>0):
                    d[l[2:]][-1] += ','+str(s_index)
                    
            elif (l[0] == 'E') & (len(entity_name) > 0)  & (isend != 1):
                ad1=d[l[2:]]
                if(len(ad1)>0):
                    d[l[2:]][-1] += ','+str(s_index)
                isend=1
  
            elif (l[0]=='S'):
                d[l[2:]].append(str(s_index))
                ad0=d[l[2:]]
                if(len(ad0)>0):
                    d[l[2:]][-1] += ','+str(s_index)
                isend=1  

            elif l == 'O':
                entity_name = ''
                isend=0
        entity_list.append(d)