"""
Module for finetuning BERT.

Overall flow:
-------------

    # Input data to BertPlusMLP consists of text pairs and labels:
    X1, X2, y = texts_a, texts_b, labels

    # get a BertTokenizer
    tokenizer = model.utils.get_tokenizer('bert-base-uncased',do_lower_case)

    # get a BertPlusMLP model
    model = model.utils.get_model('bert-base-uncased',...)

    # set tokenizer and training parameters in config
    config = FinetuneConfig(tokenizer=tokenizer, epochs=3,...)

    # finetune model
    model = finetune(model, X1, X2, y, config)

"""
import sys

import numpy as np
from tqdm import tqdm
import torch

from .data import get_train_val_dl
from .utils import prepare_model_and_device
from .utils import get_optimizer
from .utils import update_learning_rate
from .utils import OnlinePearson
from .utils import OnlineF1


def finetune(model, X1, X2, y, config):
    """
    Finetune pretrained Bert model.

    A training wrapper based on: https://github.com/huggingface/\
    pytorch-pretrained-BERT/blob/master/examples/run_classifier.py

    Parameters
    ----------
    Bert model inputs are triples of: (text_a,text_b,label).
    For single text tasks text_b = None

    model : BertPlusMLP
        pretrained Bert model with a MLP classifier/regressor head

    X1 : list of strings
        First of a pair of input text data, texts_a

    X2 : list of strings
        Second(optional) of a pair of input text data, texts_b

    y : list of string/floats
        labels/targets for input text data

    config : FinetuneConfig
        Parameters for finetuning BERT

    Returns
    --------
    model : BertPlusMLP
        finetuned BERT model plus mlp head

    """

    def log(msg, logger=config.logger, console=True):
        if logger:
            logger.info(msg)
        if console:
            print(msg)
            sys.stdout.flush()

    grad_accum_steps = config.gradient_accumulation_steps

    # change batch_size if we do gradient accumulation
    config.train_batch_size = int(config.train_batch_size / grad_accum_steps)

    # build dataloaders from input texts and labels
    train_dl, val_dl = get_train_val_dl(X1, X2, y, config)
    log("train data size: %d, validation data size: %d"%
        (len(train_dl.dataset), len(val_dl.dataset) if val_dl else 0))

    # prepare model i.e multiple gpus and fpint16
    model, device = prepare_model_and_device(model, config)
    config.device = device

    # get and prepare BertAdam optimizer
    params = list(model.named_parameters())
    optimizer, num_opt_steps = get_optimizer(params, len(train_dl.dataset), config)
    log("Number of train optimization steps is : %d"%(num_opt_steps), console=False)

    #=========================================================
    #                 main training loop
    #=========================================================
    global_step = 0

    for epoch in range(int(config.epochs)):

        model.train()
        losses = []
        batch_iter = tqdm(train_dl, desc="Training", leave=True)

        for step, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            loss, _ = model(*batch)

            loss = loss.mean()

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

            if config.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # step the optimizer every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                update_learning_rate(optimizer, global_step, num_opt_steps, config)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            losses.append(loss.item() * grad_accum_steps)
            batch_iter.set_postfix(loss=np.mean(losses))

        if val_dl is not None:
            res = eval_model(model, val_dl, config)
            test_loss, test_accy = res['loss'], res['accy']
            msg = "Epoch %d, Train loss: %0.04f, Val loss: %0.04f, Val accy: %0.02f%%"
            msg = msg%(epoch+1, np.mean(losses), test_loss, test_accy)
            if 'f1' in res:
                msg += ", f1: %0.02f"%(100 * res['f1'])
            log(msg)
        else:
            msg = "Epoch %d, Train loss : %0.04f"%(epoch+1, np.mean(losses))
            log(msg, console=False)

    return model


def eval_model(model, dataloader, config, desc="Validation"):
    """
    Evaluate model on validation data.

    Parameters
    ----------
    model : BertPlusMLP
        Bert model plus mlp head
    dataloader : Dataloader
        validation dataloader
    device : torch.device
        device to run validation on
    model_type : string
         'text_classifier' | 'text_regressor' | 'token_classifier'

    Returns
    -------
    loss : float
        Loss calculated on eval data
    accy : float
        Classification accuracy for classifiers.
        Pearson coorelation for regressors.
    """
    device = config.device
    model_type = config.model_type
    ignore_label = config.ignore_label_id

    regression_stats = OnlinePearson()
    stats = OnlineF1(ignore_label=ignore_label)

    model.to(device)
    model.eval()
    loss = accy = 0.
    total_evals = 0
    res = {}
    batch_iter = tqdm(dataloader, desc=desc, leave=False)

    for eval_steps, batch in enumerate(batch_iter):
        batch = tuple(t.to(device) for t in batch)
        _, _, _, y = batch
        with torch.no_grad():
            tmp_eval_loss, output = model(*batch)
        loss += tmp_eval_loss.mean().item()

        if model_type == "text_classifier":
            _, y_pred = torch.max(output, 1)
            accy += torch.sum(y_pred == y)

        elif model_type == "text_regressor":
            y_pred = output

            # add to online stats calculator
            for xi, yi in zip(y.detach().cpu().numpy(),
                              y_pred.detach().cpu().numpy()):
                regression_stats.add(xi, yi)

        elif model_type == "token_classifier":

            output = output.view(-1, output.shape[-1])
            y_true = y.view(-1)
            valid_tokens = y_true != -1

            _, y_pred = torch.max(output, 1)

            accy += torch.sum(y_pred[valid_tokens] == y_true[valid_tokens])
            total_evals += torch.sum(valid_tokens).item()

            y_true = y_true[valid_tokens].detach().cpu().numpy()
            y_pred = y_pred[valid_tokens].detach().cpu().numpy()

            # add to online stats calculator
            stats.add(y_true=y_true, y_pred=y_pred)

    loss = loss/(eval_steps+1)

    if model_type == "text_classifier":
        accy = 100 * (accy.item() / len(dataloader.dataset))
    elif model_type == "text_regressor":
        accy = 100 * regression_stats.pearson
    elif model_type == "token_classifier":
        accy = 100 * (accy.item() / total_evals)
        res['f1'] = stats.f1
        res['precision'] = stats.precision
        res['recall'] = stats.recall

    res['loss'] = loss
    res['accy'] = accy
    return res
