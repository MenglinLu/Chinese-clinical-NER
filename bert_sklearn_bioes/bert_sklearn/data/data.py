"""Torch Datasets and Dataloaders for Text and Token tasks"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from .utils import convert_text_to_features, convert_tokens_to_features


class TextFeaturesDataset(Dataset):
    """
    A pytorch dataset for Bert text features.

    Parameters
    ----------

    X1 : list of strings
        text_a for input data
    X2 : list of strings
        text_b for input data text pairs
    y : list of string or list of floats
        labels/targets for data
    model_type : string
        specifies 'text_classifier' or 'text_regressor' model
    label2id : dict map of string to int
        label map for classifer labels
    max_seq_length : int
        maximum length of input text sequence (text_a + text_b)
    tokenizer : BertTokenizer)
        word tokenizer followed by WordPiece Tokenizer
    """
    def __init__(self,
                 X1, X2, y,
                 model_type,
                 label2id,
                 max_seq_length,
                 tokenizer):

        self.X1 = X1
        self.X2 = X2
        self.y = y

        self.len = len(self.X1)
        self.model_type = model_type
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        if self.X2 is not None:
            text_a = str(self.X1[index])
            text_b = str(self.X2[index])
        else:
            text_a = str(self.X1[index])
            text_b = None

        feature = convert_text_to_features(text_a, text_b,
                                           self.max_seq_length,
                                           self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)

        if self.y is not None:

            label = self.y[index]

            if self.model_type == 'text_classifier':
                label_id = self.label2id[label]
                target = torch.tensor(label_id, dtype=torch.long)
            elif self.model_type == 'text_regressor':
                target = torch.tensor(label, dtype=torch.float32)
            return input_ids, segment_ids, input_mask, target
        else:
            return input_ids, segment_ids, input_mask

    def __len__(self):
        return self.len


class TokenFeaturesDataset(Dataset):
    """
    A pytorch dataset for Bert token features.

    Parameters
    ----------

    X : list of list of strings
        input token lists

    y : list of list of strings
        input token tags

    model_type : string
        specifies 'classifier' or 'regressor' model
    label2id : dict map of string to int
        label map for classifer labels
    max_seq_length : int
        maximum length of input tokens
    tokenizer : BertTokenizer
        word tokenizer followed by WordPiece Tokenizer
    """
    def __init__(self,
                 X, y,
                 label2id,
                 max_seq_length,
                 tokenizer):

        self.X = X
        self.y = y

        self.len = len(self.X)
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):


        tokens = self.X[index]

        feature = convert_tokens_to_features(tokens,
                                             self.max_seq_length,
                                             self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        token_starts = feature.token_starts

        if self.y is not None:
            labels = self.y[index]

            # convert to label ids
            labels = [self.label2id[label] for label in labels]
            
            # create token labels for all tokens. Set the non-start tokens to 
            # have label ids = "-1". We will flag them to ignored in the loss
            # function
            token_labels = [-1] * self.max_seq_length
            for idx, label in zip(token_starts, labels):
                token_labels[idx] = label

            token_labels = torch.tensor(token_labels, dtype=torch.long)
            return input_ids, segment_ids, input_mask, token_labels

        else:
            token_starts_mask = [0] *  self.max_seq_length
            for idx in token_starts:
                token_starts_mask[idx] = 1
            token_starts_mask = torch.tensor(token_starts_mask, dtype=torch.long)
            return input_ids, segment_ids, input_mask, token_starts_mask

    def __len__(self):
        return self.len


def get_dataset(X1, X2, y, config):
    """
    Get daatset.

    Parameters
    ----------
    X1 : list of strings
        text_a for input data pairs for text classification/regression
        OR
        list of list of strings
        for sequence/token tasks

    X2 : list of strings
        text_b for input data text pairs for text classification/regression

    y : list of string or list of floats
        labels/targets for data

    config : FinetuneConfig
        Parameters for finetuning BER
    """
    # text/text pair  classification and regression tasks
    if (config.model_type == 'text_classifier') or (config.model_type == 'text_regressor'):

        text_a, text_b, labels = X1, X2, y
        dataset = TextFeaturesDataset(text_a, text_b, labels,
                                      config.model_type,
                                      config.label2id,
                                      config.max_seq_length,
                                      config.tokenizer)
    # token sequence  tasks
    elif config.model_type == 'token_classifier':
        tokens, labels = X1, y
        dataset = TokenFeaturesDataset(tokens,
                                       labels,
                                       config.label2id,
                                       config.max_seq_length,
                                       config.tokenizer)
    return dataset


def get_test_dl(X1, X2, y, config):
    """
    Get test dataloaders.

    Parameters
    ----------
    X1 : list of strings
        text_a for input data pairs for text classification/regression
        OR
        list of list of strings
        for sequence/token tasks

    X2 : list of strings
        text_b for input data text pairs for text classification/regression

    y : list of string or list of floats
        labels/targets for data

    config : FinetuneConfig
        Parameters for finetuning BERT
    """

    dataset = get_dataset(X1, X2, y, config)
    test_dl = DataLoader(dataset, batch_size=config.eval_batch_size, num_workers=5,
                         drop_last=config.drop_last_batch, shuffle=False)

    return test_dl
    

def get_train_val_dl(X1, X2, y, config):
    """
    Get train and validation dataloaders.

    Parameters
    ----------
    X1 : list of strings
        text_a for input data pairs for text classification/regression
        OR
        list of list of strings
        for sequence/token tasks

    X2 : list of strings
        text_b for input data text pairs for text classification/regression

    y : list of string or list of floats
        labels/targets for data

    config : FinetuneConfig
        Parameters for finetuning BERT
    """

    dataset = get_dataset(X1, X2, y, config)

    # get train and val datasets
    val_len = int(len(dataset) * config.val_frac)
    if val_len > 0:
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
        val_dl = DataLoader(val_ds, batch_size=config.eval_batch_size,
                            num_workers=5, shuffle=False)
    else:
        val_dl = None
        train_ds = dataset

    if config.local_rank == -1:
        train_sampler = RandomSampler(train_ds) if config.train_sampler == 'random' else None
    else:
        train_sampler = DistributedSampler(train_ds)

    train_dl = DataLoader(train_ds, sampler=train_sampler,
                          batch_size=config.train_batch_size, num_workers=5,
                          drop_last=config.drop_last_batch, shuffle=False)
    return train_dl, val_dl
