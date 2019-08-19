"""sklearn interface to finetuning BERT.

Overall flow:
-------------

    # define model
    model = BertClassifier()       # text/text pair classification
    model = BertRegressor()        # text/text pair regression
    model = BertTokenClassifier()  # token sequence classification

    # fit model to training data
    model.fit(X_train, y_train)

    # score model on holdout data
    model.score(X_dev, y_dev)

    # predict model on new inputs
    model.predict(X_test)


Model inputs X, y:
------------------

    For text pair tasks:
        X = [X1, X2]
            Model inputs are triples : (text_a, text_b, label/target)
            X1 is 1D list-like of text_a strings
            X2 is 1D list-like of text_b strings

    For single text tasks:
        X = 1D list-like of text strings

    For text classification tasks:
        y = 1D list-like of string labels

    For text regression  tasks:
        y = 1D list-like of floats

    For token classification tasks:
        X = 2D list-like of token strings
        y = 2D list-like of token tags
"""

import logging
from collections import Counter

import statistics as stats
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

from .config import model2config
from .data import get_test_dl
from .data.utils import flatten
from .model import get_model
from .model import get_tokenizer
from .model import get_basic_tokenizer
from .utils import prepare_model_and_device
from .utils import get_logger
from .utils import set_random_seed
from .utils import to_numpy
from .utils import unpack_data
from .finetune import finetune
from .finetune import eval_model


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

SUPPORTED_MODELS = ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased',
                    'bert-large-cased', 'bert-base-multilingual-uncased',
                    'bert-base-multilingual-cased', 'bert-base-chinese')


class BaseBertEstimator(BaseEstimator):
    """
    Base Class for Bert Classifiers and Regressors.

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS, i.e 'bert-base-uncased', 'bert-large-uncased'...
    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp
    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code
    restore_file : string
        file to restore model state from previous savepoint
    epochs : int
        number of finetune training epochs
    max_seq_length : int
        maximum length of input text sequence (text_a + text_b)
    train_batch_size : int
        batch size for training
    eval_batch_size : int
        batch_size for validation
    label_list :list of strings
        list of classifier labels. For regressors this is None.
    learning_rate :float
        inital learning rate of Bert Optimizer
    warmup_proportion : float
        proportion of training to perform learning rate warmup
    gradient_accumulation_steps : int
        number of update steps to accumulate before performing a backward/update pass
    fp16 : bool
        whether to use 16-bit float precision instead of 32-bit
    loss_scale : float
        loss scaling to improve fp16 numeric stability. Only used when
        fp16 set to True
    local_rank : int
        local_rank for distributed training on gpus
    use_cuda : bool
        use GPU(s) if available
    random_state : intt
        seed to initialize numpy and torch random number generators
    validation_fraction : float
        fraction of training set to use for validation
    logname : string
        path name for logfile
    ignore_label : list of strings
        Labels to be ignored when calculating f1 for token classifiers
    """
    def __init__(self, label_list=None, bert_model='bert-base-uncased',
                 num_mlp_hiddens=500, num_mlp_layers=0, restore_file=None,
                 epochs=3, max_seq_length=128, train_batch_size=32,
                 eval_batch_size=8, learning_rate=2e-5, warmup_proportion=0.1,
                 gradient_accumulation_steps=1, fp16=False, loss_scale=0,
                 local_rank=-1, use_cuda=True, random_state=42,
                 validation_fraction=0.1, logfile='bert_sklearn.log',
                 ignore_label=None):

        self.id2label, self.label2id = {}, {}
        self.input_text_pairs = None

        self.label_list = label_list
        self.bert_model = bert_model
        self.num_mlp_hiddens = num_mlp_hiddens
        self.num_mlp_layers = num_mlp_layers
        self.restore_file = restore_file
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.local_rank = local_rank
        self.use_cuda = use_cuda
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.logfile = logfile
        self.ignore_label = ignore_label
        self.majority_label = None

        # a BertPlusMLP model and BertTokenizer will be loaded during fit()
        self.model = None
        self.tokenizer = None

        # if given a restore_file, then finish loading a previously finetuned
        # model. Normally a user wouldn't do this directly. This is called from
        # load_model() to finish constructing the object
        if restore_file is not None:
            self.restore_finetuned_model(restore_file)

        # the sklearn convention is to only call validate_params in fit, but I
        # feel like it should be in init as well
        self._validate_hyperparameters()

        # not good to use 'isinstance' :-( but the init code for these
        # classes are identical. So keep the ugly hack for now...
        if isinstance(self, BertClassifier):
            print("Building sklearn text classifier...")
            self.model_type = "text_classifier"
        elif isinstance(self, BertTokenClassifier):
            print("Building sklearn token classifier...")
            self.model_type = "token_classifier"
        elif isinstance(self, BertRegressor):
            print("Building sklearn text regressor...")
            self.model_type = "text_regressor"
            self.num_labels = 1

        self.logger = get_logger(logfile)
        self.logger.info("Loading model:\n" + str(self))

    def load_tokenizer(self):
        """
        Load Basic and BERT Wordpiece Tokenizers
        """
        # use lower case for uncased models
        self.do_lower_case = True if 'uncased' in self.bert_model else False

        # get basic tokenizer
        self.basic_tokenizer = get_basic_tokenizer(self.do_lower_case)

        # get bert wordpiece tokenizer
        self.tokenizer = get_tokenizer(self.bert_model, self.do_lower_case)

        return self.tokenizer

    def load_bert(self, state_dict=None):
        """
        Load a BertPlusMLP model from a pretrained checkpoint.

        This will be an pretrained BERT ready to be finetuned.
        """

        # load a vanilla bert model ready to finetune:
        # pretrained bert LM + untrained classifier/regressor

        self.model = get_model(bert_model=self.bert_model,
                               num_labels=self.num_labels,
                               model_type=self.model_type,
                               num_mlp_layers=self.num_mlp_layers,
                               num_mlp_hiddens=self.num_mlp_hiddens,
                               state_dict=state_dict,
                               local_rank=self.local_rank)

    def _validate_hyperparameters(self):
        """
        Check hyperpameters are within allowed values.
        """
        if self.bert_model not in SUPPORTED_MODELS:
            raise ValueError("The bert model '%s' is not supported. Supported "
                             "models are %s." % (self.bert_model, SUPPORTED_MODELS))

        if (not isinstance(self.num_mlp_hiddens, int) or self.num_mlp_hiddens < 1):
            raise ValueError("num_mlp_hiddens must be an integer >= 1, got %s"%
                             self.num_mlp_hiddens)

        if (not isinstance(self.num_mlp_layers, int) or self.num_mlp_layers < 0):
            raise ValueError("num_mlp_layers must be an integer >= 0, got %s"%
                             self.num_mlp_layers)

        if (not isinstance(self.epochs, int) or self.epochs < 1):
            raise ValueError("epochs must be an integer >= 1, got %s" %self.epochs)

        if (not isinstance(self.max_seq_length, int) or self.max_seq_length < 2 or \
                           self.max_seq_length > 512):
            raise ValueError("max_seq_length must be an integer >=2 and <= 512, "
                             "got %s" %self.max_seq_length)

        if (not isinstance(self.train_batch_size, int) or self.train_batch_size < 1):
            raise ValueError("train_batch_size must be an integer >= 1, got %s" %
                             self.train_batch_size)

        if (not isinstance(self.eval_batch_size, int) or self.eval_batch_size < 1):
            raise ValueError("eval_batch_size must be an integer >= 1, got %s" %
                             self.eval_batch_size)

        if self.learning_rate < 0 or self.learning_rate >= 1:
            raise ValueError("learning_rate must be >= 0 and < 1, "
                             "got %s" % self.learning_rate)

        if self.warmup_proportion < 0 or self.warmup_proportion >= 1:
            raise ValueError("warmup_proportion must be >= 0 and < 1, "
                             "got %s" % self.warmup_proportion)

        if (not isinstance(self.gradient_accumulation_steps, int) or \
                self.gradient_accumulation_steps > self.train_batch_size or \
                self.gradient_accumulation_steps < 1):
            raise ValueError("gradient_accumulation_steps must be an integer"
                             " >= 1 and <= train_batch_size, got %s" %
                             self.gradient_accumulation_steps)

        if not isinstance(self.fp16, bool):
            raise ValueError("fp16 must be either True or False, got %s." %
                             self.fp16)

        if not isinstance(self.use_cuda, bool):
            raise ValueError("use_cuda must be either True or False, got %s." %
                             self.fp16)

        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)

    def fit(self, X, y, load_at_start=True):
        """
        Finetune pretrained Bert model.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, text pair, or token list of data features

        y : 1D or 2D list-like of strings or floats
            Labels/targets for text or token data

        load_at_start : bool
            load model from saved checkpoint file at the start of the fit

        """
        # validate params
        self._validate_hyperparameters()

        # set random seed for reproducability
        set_random_seed(self.random_state, self.use_cuda)

        # unpack data
        texts_a, texts_b, labels = unpack_data(X, y)
        self.input_text_pairs = not texts_b is None

        if is_classifier(self):

            # if the label_list not specified, then infer it from training data
            if self.label_list is None:
                if self.model_type == "text_classifier":
                    self.label_list = np.unique(labels)
                elif self.model_type == "token_classifier":
                    self.label_list = np.unique(flatten(labels))

            # build label2id and id2label maps
            self.num_labels = len(self.label_list)
            for (i, label) in enumerate(self.label_list):
                self.label2id[label] = i
                self.id2label[i] = label

            # calculate majority label for token_classifier
            if self.model_type == "token_classifier":
                c = Counter(flatten(y))
                self.majority_label = c.most_common()[0][0]
                self.majority_id = self.label2id[self.majority_label]

        # load model and tokenizer from checkpoint
        if load_at_start:
            self.load_tokenizer()
            self.load_bert()

        # to fix BatchLayer1D prob in rare case last batch is a singlton w MLP
        drop_last_batch = False if self.num_mlp_layers == 0 else True

        # create a finetune config object
        config = model2config(self)
        config.drop_last_batch = drop_last_batch
        config.train_sampler = 'random'

        # check lengths match
        assert len(texts_a) == len(labels)
        if texts_b is not None:
            assert len(texts_a) == len(texts_b)

        # finetune model!
        self.model = finetune(self.model, texts_a, texts_b, labels, config)

        return self

    def setup_eval(self, texts_a, texts_b, labels):
        """
        Get dataloader and device for eval.
        """
        config = model2config(self)
        _, device = prepare_model_and_device(self.model, config)
        config.device = device

        dataloader = get_test_dl(texts_a, texts_b, labels, config)
        self.model.eval()
        return dataloader, config

    def score(self, X, y, verbose=True):
        """
        Score model on test/eval data.
        
        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, text pair, or token list of data features

        y : 1D or 2D list-like of strings or floats
            Labels/targets for text or token data

        Returns
        ----------
        accy: float
            classification accuracy, or pearson for regression     
        """
        texts_a, texts_b, labels = unpack_data(X, y)

        dataloader, config = self.setup_eval(texts_a, texts_b, labels)

        res = eval_model(self.model, dataloader, config, "Testing")
        loss, accy = res['loss'], res['accy']

        if verbose:
            msg = "\nLoss: %0.04f, Accuracy: %0.02f%%"%(loss, accy)
            if 'f1' in res:
                msg += ", f1: %0.02f"%(100 * res['f1'])
            print(msg)

        return accy

    def save(self, filename):
        """
        Save model state to disk.
        """
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        state = {
            'params': self.get_params(),
            'class_name' : type(self).__name__,
            'model_type' : self.model_type,
            'num_labels' : self.num_labels,
            'id2label'   : self.id2label,
            'label2id'   : self.label2id,
            'do_lower_case': self.do_lower_case,
            'state_dict' : model_to_save.state_dict(),
            'input_text_pairs' : self.input_text_pairs
        }
        torch.save(state, filename)

    def restore_finetuned_model(self, restore_file):
        """
        Restore a previously finetuned model from a restore_file

        This is called from the BertClassifier or BertRegressor. The saved model
        is a finetuned BertPlusMLP
        """
        print("Loading model from %s..."%(restore_file))
        state = torch.load(restore_file,map_location='cpu')

        self.model_type = state['model_type']
        self.num_labels = state['num_labels']
        self.do_lower_case = state['do_lower_case']
        self.input_text_pairs = state['input_text_pairs']
        self.id2label = state['id2label']
        self.label2id = state['label2id']

        params = state['params']
        self.set_params(**params)

        # get tokenizers
        self.load_tokenizer()

        # load bert with finetuned weights
        self.load_bert(state_dict=state['state_dict'])


class BertClassifier(BaseBertEstimator, ClassifierMixin):
    """
    A text classifier built on top of a pretrained Bert model.
    """

    def predict_proba(self, X):
        """
        Make class probability predictions.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text or text pairs
            
        Returns
        ----------
        probs: numpy 2D array of floats
            probability estimates for each class
        """

        texts_a, texts_b = unpack_data(X)

        dataloader, config = self.setup_eval(texts_a, texts_b, labels=None)
        device = config.device

        probs = []
        batch_iter = tqdm(dataloader, desc="Predicting", leave=False)
        for batch in batch_iter:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = self.model(*batch)
                prob = F.softmax(logits, dim=-1)
            prob = prob.detach().cpu().numpy()
            probs.append(prob)
        return np.vstack(tuple(probs))

    def predict(self, X):
        """
        Predict most probable class.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, or text pairs

        Returns
        ----------
        y_pred: numpy array of strings
            predicted class estimates
        """
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        y_pred = np.array([self.id2label[y] for y in y_pred])
        return y_pred


class BertRegressor(BaseBertEstimator, RegressorMixin):
    """
    A text regressor built on top of a pretrained Bert model.
    """
    def predict(self, X):
        """
        Predict method for regression.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, or text pairs
            
        Returns
        ----------
        y_pred: 1D numpy array of float
            predicted regressor float value
        """

        texts_a, texts_b = unpack_data(X)

        dataloader, config = self.setup_eval(texts_a, texts_b, labels=None)
        device = config.device

        ypred_list = []
        batch_iter = tqdm(dataloader, desc="Predicting", leave=False)
        for batch in batch_iter:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                y_pred = self.model(*batch)
            ypred_list.append(y_pred.detach().cpu().numpy())
        y_pred = np.vstack(tuple(ypred_list)).reshape(-1,)
        return y_pred


class BertTokenClassifier(BaseBertEstimator, ClassifierMixin):
    """
    A token sequence classifier built on top of a pretrained Bert model.
    """
    def get_max_token_len(self, token_list):
        """
        Get max length of bert tokens for a token list

        Parameters
        ----------
        token_list: list of list of token strings
        """
        if self.tokenizer is None:
            self.load_tokenizer()

        bert_token_lengths = [len(flatten([self.tokenizer.tokenize(token) for token in tokens]))
                              for tokens in token_list]

        return np.max(bert_token_lengths)

    def predict_proba(self, X):
        """
        Make class probability predictions.

        Parameters
        ----------
        X : 2D list of list of token strings
        
        Returns
        ----------
        y_probs: 3D numpy array of floats
            probability estimates for each tag in for each token in each 
            input token list in X
        """
        y_probs = []

        texts_a, texts_b = to_numpy(X), None
        dataloader, config = self.setup_eval(texts_a, texts_b, labels=None)
        device = config.device

        batch_iter = tqdm(dataloader, desc="Predicting", leave=False)

        for batch in batch_iter:
            # get the token_starts mask from batch
            x1, x2, x3, token_starts_mask = batch

            # put BERT input features onto device
            batch = (x1, x2, x3)
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = self.model(*batch)

            # valid tokens are where mask is '1'
            logits = logits[token_starts_mask == 1]

            # calculate the original token list lengths from token_starts mask
            lengths = torch.sum(token_starts_mask, 1)

            # softmax over logits
            y_prob = F.softmax(logits, dim=-1)

            # to numpy
            y_prob = y_prob.detach().cpu().numpy()

            # re-assemble the tokens based on the lengths
            start = 0
            for length in  lengths:
                y_probs.append(y_prob[start:start + length])
                start += length

        # predict majority label for any tokens that have been truncated by max_seq_len
        for i, (x, y_prob) in enumerate(zip(X, y_probs)):
            if len(x) > len(y_prob):

                # create rows for all the truncated tokens with prob=1
                # for majority_label/majority_id
                y_prob_xtra = np.zeros_like(y_prob[-1])
                y_prob_xtra[self.majority_id] = 1.0

                length = len(x) - len(y_prob)
                y_prob_xtra = np.tile(y_prob_xtra, (length, 1))

                y_probs[i] = np.vstack((y_prob, y_prob_xtra))
        return y_probs

    def predict(self, X):
        """
        Make most probable class prediction on input data.

        Parameters
        ----------
        X : 2D list of list of token strings
        
        Returns
        ----------
        y_preds: 2D numpy array of string
            predicted tag for each token in each input token list 
        """
        y_preds = []

        texts_a, texts_b = to_numpy(X), None
        dataloader, config = self.setup_eval(texts_a, texts_b, labels=None)
        device = config.device

        batch_iter = tqdm(dataloader, desc="Predicting", leave=False)

        for batch in batch_iter:
            # peel off the token_starts mask from batch
            x1, x2, x3, token_starts_mask = batch

            # put BERT input features onto device
            batch = (x1, x2, x3)
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = self.model(*batch)

            # valid tokens are where mask is '1'
            logits = logits[token_starts_mask == 1]

            # calculate the original token list lengths from token_starts mask
            lengths = torch.sum(token_starts_mask, 1)

            # get predicts
            _, y_pred = torch.max(logits, 1)
            y_pred = y_pred.detach().cpu().numpy()

            # convert to class_ids to class labels
            y_pred = [self.id2label[y_i] for y_i in y_pred]

            # re-assemble the tokens into their original input form from the lengths
            start = 0
            for length in  lengths:
                y_preds.append(y_pred[start:start + length])
                start += length

        # predict majority label for any tokens that have been truncated by max_seq_len
        for x, y in zip(X, y_preds):
            if len(x) > len(y):
                y.extend([self.majority_label] * (len(x) - len(y)))
        return y_preds

    def score(self, X, y):
        """
        Score model on test/eval data.

        Parameters
        ----------
        X : 2D list of list of token strings 
        y : 2D list of list of token tags/labels

        Returns
        ----------
        f1: float
            f1 wrt to the ignore_labels i.e 'O' for NER
        """ 

        y_preds = self.predict(X)
        label_list = self.label_list

        if self.ignore_label is not None:
            label_list = list(set(label_list) - set(self.ignore_label))

        f1 = 100 * f1_score(flatten(y), flatten(y_preds), average='micro', labels=label_list)
        return f1


    def tokens_proba(self, tokens, prob=None, verbose=True):
        """
        Print tag probabilities for each token.
        """
        if prob is None:
            prob = self.predict_proba([tokens])
            prob = np.array(prob)[0]
        if verbose:
            cols = list(self.id2label.values())
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            df = pd.DataFrame(prob, columns=cols)
            df.insert(0, "token", tokens)
            print(df)
        return prob

    def tag_text_proba(self, text, verbose=True):
        """
        Tokenize text and print tag probabilities for each token.
        """
        tokens = self.basic_tokenizer.tokenize(text)
        return self.tokens_proba(tokens, verbose=verbose)

    def tag_text(self, text, verbose=True):
        """
        Tokenize text and print most probable token tags.
        """
        tokens = self.basic_tokenizer.tokenize(text)
        tags = self.predict(np.array([tokens]))[0]
        if verbose:
            data = {"token": tokens, "predicted tags": tags}
            df = pd.DataFrame(data=data)
            print(df)
        return tags


def load_model(filename):
    """
    Load BertClassifier, BertRegressor, or BertTokenClassifier from a disk file.
    
        Parameters
        ----------
        filename : string
            filename of saved model file

        Returns
        ----------
        model : BertClassifier, BertRegressor, or BertTokenClassifier model
    """
    state = torch.load(filename,map_location='cpu')
    class_name = state['class_name']

    classes = {
        'BertClassifier': BertClassifier,
        'BertRegressor' : BertRegressor,
        'BertTokenClassifier' : BertTokenClassifier}

    # call the constructor to load the model
    model_ctor = classes[class_name]
    model = model_ctor(restore_file=filename)
    return model
