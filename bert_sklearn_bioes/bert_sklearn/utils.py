import logging
import random

import numpy as np
import torch

from .model.pytorch_pretrained import BertAdam, warmup_linear


def set_random_seed(seed=42, use_cuda=True):
    """Seed all random number generators to enable repeatable runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def to_numpy(X):
    """
    Convert input to numpy ndarray
    """
    if hasattr(X, 'iloc'):              # pandas
        return X.values
    elif isinstance(X, list):           # list
        return np.array(X)
    elif isinstance(X, np.ndarray):     # ndarray
        return X
    else:
        raise ValueError("Unable to handle input type %s"%str(type(X)))


def unpack_text_pairs(X):
    """
    Unpack text pairs
    """
    if X.ndim == 1:
        texts_a = X
        texts_b = None
    else:
        texts_a = X[:, 0]
        texts_b = X[:, 1]

    return texts_a, texts_b


def unpack_data(X, y=None):
    """
    Prepare data
    """
    X = to_numpy(X)
    texts_a, texts_b = unpack_text_pairs(X)

    if y is not None:
        y = to_numpy(y)
        labels = y
        return texts_a, texts_b, labels
    else:
        return texts_a, texts_b


def get_logger(logname, no_stdout=True):
    logger = logging.getLogger()
    handler = logging.StreamHandler(open(logname, "a"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if no_stdout:
        logger.removeHandler(logger.handlers[0])

    return logger



def get_device(local_rank, use_cuda):
    """
    Get torch device and number of gpus.

    Parameters
    ----------
    local_rank : int
        local_rank for distributed training on gpus
    use_cuda : bool
        use cuda if available
    """
    if local_rank == -1 or not use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will
        # take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    return device, n_gpu


def prepare_model_and_device(model, config):
    """
    Prepare model for training and get torch device

    Parameters
    ----------
    model : BertPlusMLP
        BERT model plud mlp head

    len_train_data : int
        length of training data

    config : FinetuneConfig
        Parameters for finetuning BERT
    """
    device, n_gpu = get_device(config.local_rank, config.use_cuda)

    if config.fp16:
        model.half()

    model.to(device)

    if config.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from \
            https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, device



def get_optimizer(params, len_train_data, config):
    """
    Get and prepare Bert Adam optimizer.

    Parameters
    ----------
    params :
        model parameters to be optimized
    len_train_data : int
        length of training data
    config : FinetuneConfig
        Parameters for finetuning BERT

    Returns
    -------
    optimizer : FusedAdam or BertAdam
        Optimizer for training model
    num_opt_steps : int
        number of optimization training steps
    """

    num_opt_steps = len_train_data / config.train_batch_size
    num_opt_steps = int(num_opt_steps / config.gradient_accumulation_steps) * config.epochs

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]

    if config.local_rank != -1:
        num_opt_steps = num_opt_steps // torch.distributed.get_world_size()

    if config.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/\
                                nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(grouped_params,
                              lr=config.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)

        if config.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=config.loss_scale)
    else:
        optimizer = BertAdam(grouped_params,
                             lr=config.learning_rate,
                             warmup=config.warmup_proportion,
                             t_total=num_opt_steps)

    return optimizer, num_opt_steps


def update_learning_rate(optimizer, global_step, num_opt_steps, config):
    """Update learning rate for optimizer for special warm up BERT uses

    if args.fp16 is False, BertAdam is used that handles this automatically
    """
    lr, warmup = config.learning_rate, config.warmup_proportion
    if config.fp16:
        lr_this_step = lr * warmup_linear(global_step/num_opt_steps, warmup)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step


class OnlinePearson():
    """
    Online pearson stats calculator

    Calculates online pearson coefficient via running covariance
    ,variance, and mean  estimates.

    Ref: https://stats.stackexchange.com/questions/23481/\
    are-there-algorithms-for-computing-running-linear-or-logistic-regression-param
    """
    def __init__(self):
        self.num_points = 0.
        self.mean_X = self.mean_Y = 0.
        self.var_X = self.var_Y = self.cov_XY = 0.
        self.pearson = 0.

    def add(self, x, y):
        """Add data point to online calculation"""
        self.num_points += 1
        n = self.num_points
        delta_x = x - self.mean_X
        delta_y = y - self.mean_Y
        self.var_X += (((n - 1)/n) * delta_x * delta_x - self.var_X)/n
        self.var_Y += (((n - 1)/n) * delta_y * delta_y - self.var_Y)/n

        self.cov_XY += (((n - 1)/n) * delta_x * delta_y - self.cov_XY)/n
        self.mean_X += delta_x/n
        self.mean_Y += delta_y/n

        if self.var_X * self.var_Y != 0:
            self.pearson = self.cov_XY/ np.sqrt((self.var_X * self.var_Y))


class OnlineF1():
    """
    Online F1 for NER and Token tasks
    """
    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label
        self.num_correct_predicts = 0.
        self.num_predicts = 0.
        self.num_actuals = 0.
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.

    def add(self, y_true, y_pred):
        """Add data point to online calc"""
        correct_predicts = y_true[y_pred == y_true]

        ignore = self.ignore_label

        # filter labels i.e 'O' labels for NER
        if ignore is not None:
            self.num_correct_predicts += len([y for y in correct_predicts if y not in ignore])

            # total number of named entities found
            self.num_predicts += len([y for y in y_pred if y not in ignore])

            # actual number of named entities in data
            self.num_actuals += len([y for y in y_true if y not in ignore])

        else:
            self.num_correct_predicts += len(correct_predicts)
            self.num_predicts += len(y_pred)
            self.num_actuals += len(y_true)

        # calculate stats
        if self.num_predicts == 0:
            self.num_predicts = 1.

        if self.num_actuals == 0:
            self.num_actuals = 1.

        self.precision = self.num_correct_predicts / self.num_predicts
        self.recall = self.num_correct_predicts / self.num_actuals

        if (self.precision + self.recall) == 0:
            self.f1 = 0.
        else:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
