# scikit-learn wrapper for BERT

A scikit-learn model for text and token sequence classification/regression based on the [huggingface pytorch](https://github.com/huggingface/pytorch-pretrained-BERT) port of [Google's BERT](https://github.com/google-research/bert)(Bidirectional Encoder Representations from Transformers) model.

* Added an MSE loss for regression tasks
* Added configurable MLP as final classifier/regressor
* Added Token Sequence Classifier for NER, PoS, and chunking tasks

## basic operation

**`model.fit(X,y)`** where

* **`X`**: list, pandas dataframe, or numpy array of text, text pairs, or token lists

* **`y`** : list, pandas dataframe, or numpy array of labels/targets

```python3
from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model

# define model
model = BertClassifier()         # text/text pair classification
# model = BertRegressor()        # text/text pair regression
# model = BertTokenClassifier()  # token sequence classification
 
# fit model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# make probabilty predictions
y_pred = model.predict_proba(X_test)

# score model on test data
model.score(X_test, y_test)

# save model to disk
savefile='/data/mymodel.bin'
model.save(savefile)

# load model from disk
new_model = load_model(savefile)

# do stuff with new model
new_model.score(X_test, y_test)
```
See [demo](https://github.com/charles9n/bert-sklearn/blob/master/demo.ipynb) notebook.

## model options

```python3
# try different options...
model.bert_model = 'bert-large-uncased'
model.num_mlp_layers = 3
model.max_seq_length = 196
model.epochs = 4
model.learning_rate = 4e-5
model.gradient_accumulation_steps = 4

# fit model
model.fit(X_train, y_train)

# do stuff...
model.score(X_test, y_test)
```
See [options](https://github.com/charles9n/bert-sklearn/blob/master/Options.md)


## hyperparameter tuning

```python3
from sklearn.model_selection import GridSearchCV

params = {'epochs':[3, 4], 'learning_rate':[2e-5, 3e-5, 5e-5]}

# wrap classifier in GridSearchCV
clf = GridSearchCV(BertClassifier(validation_fraction=0), 
                    params,
                    scoring='accuracy',
                    verbose=True)

# fit gridsearch 
clf.fit(X_train ,y_train)
```
See [demo_tuning_hyperparameters](https://github.com/charles9n/bert-sklearn/blob/master/demo_tuning_hyperparams.ipynb) notebook.

## GLUE datasets
The train and dev data sets from the [GLUE(Generalized Language Understanding Evaluation) ](https://github.com/nyu-mll/GLUE-baselines) benchmarks were used with `bert-base-uncased` model and compared against the reported results in the Google paper and [GLUE leaderboard](https://gluebenchmark.com/leaderboard).

|    | MNLI(m/mm)| QQP   | QNLI | SST-2| CoLA | STS-B | MRPC | RTE |
| - | - | - | - | - |- | - | - | - |
|BERT base(leaderboard) |84.6/83.4  | 89.2 | 90.1 | 93.5 | 52.1 | 87.1  | 84.8 | 66.4 | 
| bert-sklearn  |83.7/83.9| 90.2 |88.6 |92.32 |58.1| 89.7 |86.8 | 64.6 |

Individual task demos can be found [here](https://github.com/charles9n/bert-sklearn/tree/master/glue_examples).

## CoNLL-2003 Named Entity Recognition(NER)

NER results for [**`CoNLL-2003`**](https://www.clips.uantwerpen.be/conll2003/ner/)  shared task

|    | dev f1 | test f1   |
| - | - | - |
| BERT paper| 96.4 | 92.4|
| bert-sklearn | 96.04 | 91.97|

Span level stats on test:
```bash
processed 46666 tokens with 5648 phrases; found: 5740 phrases; correct: 5173.
accuracy:  98.15%; precision:  90.12%; recall:  91.59%; FB1:  90.85
              LOC: precision:  92.24%; recall:  92.69%; FB1:  92.46  1676
             MISC: precision:  78.07%; recall:  81.62%; FB1:  79.81  734
              ORG: precision:  87.64%; recall:  90.07%; FB1:  88.84  1707
              PER: precision:  96.00%; recall:  96.35%; FB1:  96.17  1623
```
See [ner_english](https://github.com/charles9n/bert-sklearn/blob/master/other_examples/ner_english.ipynb) for a demo using `'bert-base-cased'` model.

## Other examples

* See [IMDb](https://github.com/charles9n/bert-sklearn-tmp/blob/master/other_examples/IMDb.ipynb) for a text classification demo on the Internet Movie Database review sentiment task.

* See [chunking_english](https://github.com/charles9n/bert-sklearn/blob/master/other_examples/chunker_english.ipynb) for a demo on syntactic chunking using the [**`CoNLL-2000`**](https://www.clips.uantwerpen.be/conll2003/ner/) chunking task data.

* See [ner_chinese](https://github.com/charles9n/bert-sklearn/blob/master/other_examples/ner_chinese.ipynb) for a demo using `'bert-base-chinese'` for Chinese NER.

## installation

requires python >= 3.5 and pytorch >= 0.4.1

```bash
# setup bert-sklearn locally
git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip install .
```

## tests

Run tests with pytest:
```bash
python -m pytest -sv tests/
```
## references

* [Google's original tf code](https://github.com/google-research/bert)  and [paper](https://arxiv.org/abs/1810.04805)

* [huggingface pytorch port](https://github.com/huggingface/pytorch-pretrained-BERT)

