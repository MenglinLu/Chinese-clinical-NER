# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:28:07 2019

@author: eileenlu
"""

from sklearn.metrics import confusion_matrix
import pandas as pd

##exact match
def exact_confusion_matrix(true_path,predict_path):
    class_dict={'疾病和诊断':1,'解剖部位':2,'影像检查':3,'实验室检验':4,'药物':5,'手术':6}
    true_file=open(true_path,'r',encoding='utf-8')
    predict_file=open(predict_path,'r',encoding='utf-8')
    y_true_l=[]
    y_pre_l=[]
    for line_predict,line_true in zip(predict_file.readlines(),true_file.readlines()):
        line_no_predict=line_predict.split('@@')[0]
        tag_predict=line_predict.split('@@')[1].split(';;')[:-1]
        line_no_true=line_true.split('@@')[0]
        tag_true=line_true.split('@@')[1].split(';;')[:-1]
        if(line_no_predict!=line_no_true):
            print('maybe error...')
        else:
            line_true_dict={} 
            index=[1]
            for i_true in tag_true:
                term_i=line_no_true+'@@'+'@'.join(i_true.split('@')[:-1])
                start=term_i.split('@')[-2]
                end=term_i.split('@')[-1]
                keykey=line_no_true+'-'+start+'-'+end
                index.append(int(start))
                index.append(int(end))
                y_true_i=i_true.split('@')[-1]
                line_true_dict[keykey]=class_dict[y_true_i]
            index.sort()
            for j in range(len(index)-1):
                key_j=line_no_true+'-'+str(index[j])+'-'+str(index[j+1])
                if(key_j not in line_true_dict.keys()):
                    line_true_dict[key_j]=0
            
            index_p=[1]
            line_pre_dict={} 
            for i_pre in tag_predict:
                term_i=line_no_predict+'@@'+'@'.join(i_pre.split('@')[:-1])
                start=term_i.split('@')[-2]
                end=term_i.split('@')[-1]
                keykey=line_no_predict+'-'+start+'-'+end
                y_pre_i=i_pre.split('@')[-1]
                line_pre_dict[keykey]=class_dict[y_pre_i]
                
            index_p.sort()
            for j in range(len(index_p)-1):
                key_j=line_no_predict+'-'+str(index_p[j])+'-'+str(index_p[j+1])
                if(key_j not in line_pre_dict.keys()):
                    line_pre_dict[key_j]=0
            
            line_prediction_dict={}
            for k in line_true_dict.keys():
                start_k=int(k.split('-')[1])
                end_k=int(k.split('-')[2])
                flag=0
                for k_p in line_pre_dict.keys():
    #                if(k_p==k):
    #                    line_prediction_dict[k]=line_pre_dict[k_p]
    #                    break
    #                else:
                    start_k_p=int(k_p.split('-')[1])
                    end_k_p=int(k_p.split('-')[2])
                    if(start_k_p>=start_k and end_k_p<=end_k):
                        flag=1
                        line_prediction_dict[k]=line_pre_dict[k_p]
                        break
                if(flag == 0):
                    line_prediction_dict[k]=0
            
            
            for k in line_true_dict.keys():
                now=k+'@'+str(line_true_dict[k])
                y_true_l.append(now)
                
            for k in line_prediction_dict.keys():
                now=k+'@'+str(line_prediction_dict[k])
                y_pre_l.append(now)
            
            y_true_l.sort()
            y_pre_l.sort()
            
            y_true=[(i.split('@')[0]) for i in y_true_l]
            y_pre=[i.split('@')[0] for i in y_pre_l]
            
            if(y_true != y_pre):
                print('not match...')
            else:
                y_true1=[int(i.split('@')[1]) for i in y_true_l]
                y_pre1=[int(i.split('@')[1]) for i in y_pre_l]

    bbb=pd.DataFrame(confusion_matrix(y_true1, y_pre1,labels=[0,1,2,3,4,5,6]))
    bbb.columns=['O','疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
    bbb.index=['O','疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
    return bbb

##soft match
def soft_confusion_matrix(true_path,predict_path):
    class_dict={'疾病和诊断':1,'解剖部位':2,'影像检查':3,'实验室检验':4,'药物':5,'手术':6}
    true_file=open(true_path,'r',encoding='utf-8')
    predict_file=open(predict_path,'r',encoding='utf-8')
    y_true_l=[]
    y_pre_l=[]
    for line_predict,line_true in zip(predict_file.readlines(),true_file.readlines()):
        line_no_predict=line_predict.split('@@')[0]
        tag_predict=line_predict.split('@@')[1].split(';;')[:-1]
        line_no_true=line_true.split('@@')[0]
        tag_true=line_true.split('@@')[1].split(';;')[:-1]
        if(line_no_predict!=line_no_true):
            print('maybe error...')
        else:
            line_true_dict={} 
            index=[1]
            for i_true in tag_true:
                term_i=line_no_true+'@@'+'@'.join(i_true.split('@')[:-1])
                start=term_i.split('@')[-2]
                end=term_i.split('@')[-1]
                keykey=line_no_true+'-'+start+'-'+end
                index.append(int(start))
                index.append(int(end))
                y_true_i=i_true.split('@')[-1]
                line_true_dict[keykey]=class_dict[y_true_i]
                
            line_pre_dict={} 
            for i_pre in tag_predict:
                term_i=line_no_predict+'@@'+'@'.join(i_pre.split('@')[:-1])
                start=term_i.split('@')[-2]
                end=term_i.split('@')[-1]
                index.append(int(start))
                index.append(int(end))
                keykey=line_no_predict+'-'+start+'-'+end
                y_pre_i=i_pre.split('@')[-1]
                line_pre_dict[keykey]=class_dict[y_pre_i]
            
            index=list(set(index))
            index.sort()        
            
            line_true_dict1={}
            for j in range(len(index)-1):
                key_j=line_no_true+'-'+str(index[j])+'-'+str(index[j+1])
                start_j=int(index[j])
                end_j=int(index[j+1])
                if(key_j in line_true_dict.keys()):
                    line_true_dict1[key_j]=line_true_dict[key_j]
                else:
                    flag=0
                    for k in line_true_dict.keys():
                        start=int(k.split('-')[1])
                        end=int(k.split('-')[2])
                        if(start<=start_j and end>=end_j):
                            line_true_dict1[key_j]=line_true_dict[k]
                            flag=1
                            break
                    if(flag==0):
                        line_true_dict1[key_j]=0
                        
            line_pre_dict1={}
            for j_pre in range(len(index)-1):
                key_j_pre=line_no_true+'-'+str(index[j_pre])+'-'+str(index[j_pre+1])
                start_j_pre=int(index[j_pre])
                end_j_pre=int(index[j_pre+1])
                if(key_j_pre in line_pre_dict.keys()):
                    line_pre_dict1[key_j_pre]=line_pre_dict[key_j_pre]
                else:
                    flag=0
                    for k in line_pre_dict.keys():
                        start=int(k.split('-')[1])
                        end=int(k.split('-')[2])
                        if(start<=start_j_pre and end>=end_j_pre):
                            line_pre_dict1[key_j_pre]=line_pre_dict[k]
                            flag=1
                            break
                    if(flag==0):
                        line_pre_dict1[key_j_pre]=0            
     
            for k in line_true_dict1.keys():
                now=k+'@'+str(line_true_dict1[k])
                y_true_l.append(now)
                
            for k in line_pre_dict1.keys():
                now=k+'@'+str(line_pre_dict1[k])
                y_pre_l.append(now)
            
            y_true_l.sort()
            y_pre_l.sort()
            
            y_true=[(i.split('@')[0]) for i in y_true_l]
            y_pre=[i.split('@')[0] for i in y_pre_l]
            
            if(y_true != y_pre):
                print('not match...')
            else:
                y_true1=[int(i.split('@')[1]) for i in y_true_l]
                y_pre1=[int(i.split('@')[1]) for i in y_pre_l]
            
    bbb=pd.DataFrame(confusion_matrix(y_true1, y_pre1,labels=[0,1,2,3,4,5,6]))
    bbb.columns=['O','疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
    bbb.index=['O','疾病和诊断','解剖部位','影像检查','实验室检验','药物','手术']
    return bbb

#for i,j in zip(y_true_l,y_pre_l):
#    if(int(i.split('@')[1])==2 and int(j.split('@')[1])==1):
#        print(i,j)
#dir_path=os.path.dirname(os.path.abspath(__file__))
#predcit_path_byjieba=r'E:\CCKS2019_Clinic\baseline_jiebaac\res/predict_label_byjieba.txt'
#true_path=r'E:\CCKS2019_Clinic\baseline_jiebaac\res/true_label_byjieba.txt'