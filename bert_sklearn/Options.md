
# model parameters

```python3
model = BertClassifier(   
                bert_model='bert-base-uncased',
                label_list=None, 
                num_mlp_hiddens=500,
                num_mlp_layers=0,
                epochs=3,
                max_seq_length=128,
                train_batch_size=32,
                eval_batch_size=8,
                learning_rate=2e-5,
                warmup_proportion=0.1,
                gradient_accumulation_steps=1,
                fp16=False,
                loss_scale=0,
                local_rank=-1,
                use_cuda=True,
                random_state=42,
                validation_fraction=0.1,
                logfile='bert_sklearn.log')
```

### bert model options

`bert_model`: one of Google AI's pretrained BERT models.  See [pytorch-pretrained-BERT doc](https://github.com/huggingface/pytorch-pretrained-BERT#Doc) for more info.
 * `'bert-base-uncased'` (default)
 * `'bert-large-uncased'`
 * `'bert-base-cased'`
 * `'bert-large-cased'`
 * `'bert-base-multilingual-uncased'`
 * `'bert-base-multilingual-cased'`
 * `'bert-base-chinese'`

### final classifier/regressor options
`label_list`: list of classifier labels. If `None`, then the labels will be inferred from training data in model.fit(). Default: `None`

`num_mlp_layers`: the number of mlp layers in final mlp classifier/regressor. If set to 0, then defaults 
    to the linear classifier/regresor in the original Google paper and code. Default: 0

`num_mlp_hiddens`: the number of hidden neurons in each layer of the mlp. Default: 500

`ignore_label`: labels to be excluded when calculating f1 for BertTokenClassifier. Default: None


### training/finetuning options

`learning_rate`: inital learning rate of Bert Optimizer. Default: 2e-5

`epochs`: number of (finetune) training epochs. Default: 3        

`warmup_proportion`: proportion of training to perform learning rate warmup . Default: 0.1

`max_seq_length`: maximum length of input text sequence. Default: 128

`train_batch_size`: batch size for training. Default: 32

`validation_fraction`: fraction of training set to use for validation during finetuning. Deafult: 0.1

`eval_batch_size`: batch_size for validation. Default: 8

`gradient_accumulation_steps`: number of update steps to accumulate before performing a backward/update pass. Default: 1      
`fp16`: whether to use 16-bit float precision instead of 32-bit. [Nvidia apex](https://github.com/NVIDIA/apex) must be installed to enable this option. Default: False

`loss_scale`: loss scaling to improve fp16 numeric stability. Only used when fp16 set to True. See [pytorch-pretrained-BERT fpint16 section](https://github.com/huggingface/pytorch-pretrained-BERT#Training-large-models-introduction,-tools-and-examples) for more info.  Default: False

### Other

`use_cuda`: use GPU(s) if available. Default: True

`random_state`: random seed to initialize numpy and torch random number generators. Default: 42

`logname`: path name for logfile. Default: 'bert_sklearn.log'

`local_rank`: local_rank for distributed training on gpus. `-1` indicates to train locally. Currently this parameter is only exposed in case someone may want to do this. Distrbuted training/inference has not been tested in this release. Default: -1

