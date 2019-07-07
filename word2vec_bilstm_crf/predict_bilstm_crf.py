# coding=utf-8
import numpy as np
from bilstm_crf import BiLSTM_CRF
from collections import defaultdict
import preprocess as p
from evaluate import evaluate1
import os
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

def get_X_orig(X_data, index2char):
    """
    :param X_data: index_array
    :param index2char: dict
    :return: 以character_level text列表为元素的列表
    """
    X_orig = []
    for n in range(X_data.shape[0]):
        orig = [index2char[i] if i > 0 else 'None' for i in X_data[n]]
        X_orig.append(orig)
    return X_orig

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

def get_entity(X_data, y_data):
    """
    :param X_data: 以character_level text列表为元素的列表
    :param y_data: 以entity列表为元素的列表
    :return: [{'entity': [phrase or word], ....}, ...]
    """
    n_example = len(X_data)
    entity_list = []
    entity_name = ''
    for i in range(n_example):
        d = defaultdict(list)
        for c, l in zip(X_data[i], y_data[i]):
            if l[0] == 'B':
                d[l[2:]].append('')
                ad0=d[l[2:]]
                if(len(ad0)>0):
                    d[l[2:]][-1] += c
                entity_name += c
            elif (l[0] == 'I') & (len(entity_name) > 0):
                ad1=d[l[2:]]
                if(len(ad1)>0):
                    d[l[2:]][-1] += c
            elif l == 'O':
                entity_name = ''
        entity_list.append(d)

    return entity_list

def micro_evaluation(pred_entity, true_entity):
    n_example = len(pred_entity)
    t_pos, true, pred = [], [], []
    for n in range(n_example):
        et_p = pred_entity[n]
        et_t = true_entity[n]
        print('the prediction is', et_p.items(), '\n',
              'the true is', et_t.items())
        t_pos.extend([len(set(et_p[k]) & set(et_t[k]))
                      for k in (et_p.keys() & et_t.keys())])
        pred.extend([len(v) for v in et_p.values()])
        true.extend([len(v) for v in et_t.values()])

    precision = sum(t_pos) / sum(pred) + 1e-8
    recall = sum(t_pos) / sum(true) + 1e-8
    f1 = 2 / (1 / precision + 1 / recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)

if __name__ == '__main__':

    char_embedding_mat = np.load('data/char_embedding_matrix.npy')

    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    true_path='res/true.txt'
    predict_path='res/predict.txt'
    
    ner_model = BiLSTM_CRF(n_input=500, n_vocab=char_embedding_mat.shape[0],
                           n_embed=200, embedding_mat=char_embedding_mat,
                           keep_prob=0.5, n_lstm=150, keep_prob_lstm=0.5,
                           n_entity=13, optimizer='adam', batch_size=512, epochs=100)

    model_file = 'checkpoints/bilstm_crf_weights_best.hdf5'
    ner_model.model.load_weights(model_file)

    y_pred = ner_model.model.predict(X_test[:, :])

    char2vec, n_char, n_embed, char2index = p.get_char2object()

    index2char = {i: w for w, i in char2index.items()}

    X_list = get_X_orig(X_test[:, :], index2char) # list

    pred_list, true_list = get_y_orig(y_pred, y_test[:, :]) # list

    get_entity_index(X_list, pred_list, predict_path)
    get_entity_index(X_list, true_list, true_path)
    
    E=evaluate1(true_path,predict_path)
    res=E.evaluate_main()
    res.to_csv('res/word2vec_bilstm_crf_performance.csv',encoding='utf-8-sig')
    
#    pred_entity, true_entity = get_entity(X_list, pred_list), get_entity(X_list, true_list)
#    precision1, recall1, f11 = micro_evaluation(pred_entity, true_entity)
#
#    print('微平均：'+ str([precision1, recall1, f11]))
     #print('宏平均：'+ str([precision2, recall2, f12]))    
    # Just test 'get_entity' function:
#     X_data = [['火','箭','队','的','主','场','在','休','斯','顿',',',
#                '当','家','球','星','为','哈','登','和','保','罗'],
#               ['北', '京', '故', '宫', '主','场','在','休','斯','顿',',',
#                '当','家','球','星','为','哈','登','和','保','罗']]
#     y_data = [['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-LOC',
#                   'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER',
#                   'I-PER', 'O', 'B-PER', 'I-PER'],['B-LOC', 'I-LOC', 'B-ORG',
#                     'I-ORG', 'O', 'O', 'O', 'B-LOC',
#                     'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER',
#                     'I-PER', 'O', 'B-PER', 'I-PER']]
#     entity_list = get_entity(X_true, pred_list)
    # print(entity_list)
    

