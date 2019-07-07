# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:01:44 2019

@author: eileenlu
"""

from keras.layers import *
from keras_bert import load_trained_model_from_checkpoint
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from crf_layer import CRF
from keras.losses import categorical_crossentropy

class Bert_ner(): 
    def __init__(self,config_path,checkpoint_path,dict_path):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path
        
    
    def model_bert_bilstm_crf(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
            
        x_input1=Input(shape=(None,))
        x_input2=Input(shape=(None,))
        x=bert_model([x_input1,x_input2])
        bilstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.35, recurrent_dropout=0.35), name='BiLSTM')(x)
        hidden = TimeDistributed(Dense(32, activation=None), name='hidden_layer')(bilstm)
        crf = CRF(units=13, learn_mode='join',
                  test_mode='viterbi', sparse_target=False)
        output = crf(hidden)
        model = Model(inputs=[x_input1,x_input2], outputs=output)
        adam=Adam(lr=2e-4)
        model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        return model
    
    def model_build_bert_crf(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
            
        x_input1=Input(shape=(None,))
        x_input2=Input(shape=(None,))
        x=bert_model([x_input1,x_input2])
#        bilstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='BiLSTM')(x)
#        hidden = TimeDistributed(Dense(32, activation=None), name='hidden_layer')(bilstm)
        crf = CRF(units=13, learn_mode='join',
                  test_mode='viterbi', sparse_target=False)
        output = crf(x)
        model = Model(inputs=[x_input1,x_input2], outputs=output)
        adam=Adam(lr=2e-4)
        model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        return model
    
    def model_build_bert_dense(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
            
        x_input1=Input(shape=(None,))
        x_input2=Input(shape=(None,))
        x=bert_model([x_input1,x_input2])
#        bilstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='BiLSTM')(x)
#        hidden = TimeDistributed(Dense(32, activation=None), name='hidden_layer')(bilstm)
        output = Dense(units=13, activation='softmax')(x)
        model = Model(inputs=[x_input1,x_input2], outputs=output)
        adam=Adam(lr=2e-4)
        model.compile(optimizer=adam, loss=categorical_crossentropy, metrics='acc')
        model.summary()
        return model
    
    





