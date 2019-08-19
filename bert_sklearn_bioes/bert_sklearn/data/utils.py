import numpy as np


class TextFeatures:
    """
    Input features for the BERT model for text and text pair tasks.
    """
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class TokenFeatures(TextFeatures):
    """
    Input features for the BERT model for sequence/token tasks.
    """
    def __init__(self, input_ids, input_mask, segment_ids, token_starts):
        TextFeatures.__init__(self,input_ids, input_mask, segment_ids)
        self.token_starts = token_starts


def flatten(l):
    return [item for sublist in l for item in sublist]


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def pad_and_get_ids(tokens, segment_ids, max_seq_length, tokenizer):
    """Convert tokens and segment_ids to BERT input TextFeatures.

    convert tokens to tokenids. Build input mask and pad to max_seq_length.
    """
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return TextFeatures(input_ids, input_mask, segment_ids)


# The convention in BERT is:
# (a) For sequence pairs:
#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
# (b) For single sequences:
#  tokens:   [CLS] the dog is hairy . [SEP]
#  type_ids: 0   0   0   0  0     0 0
#
# Where "type_ids" are used to indicate whether this is the first
# sequence or the second sequence. The embedding vectors for `type=0` and
# `type=1` were learned during pre-training and are added to the wordpiece
# embedding vector (and position vector). This is not *strictly* necessary
# since the [SEP] token unambigiously separates the sequences, but it makes
# it easier for the model to learn the concept of sequences.
#
# For classification tasks, the first vector (corresponding to [CLS]) is
# used as as the "sentence vector". Note that this only makes sense because
# the entire model is fine-tuned.


def convert_text_to_features(text_a, text_b, max_seq_length, tokenizer):
    """
    Convert text pairs to BERT input features.

    Adapted from 'convert_examples_to_features' in
    pytorch-pretrained-BERT/examples/run_classifier.py

    """
    tokens_a = tokenizer.tokenize(text_a)

    tokens_b = None
    if text_b is not None:
        tokens_b = tokenizer.tokenize(text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    features = pad_and_get_ids(tokens, segment_ids, max_seq_length, tokenizer)
    return features


def convert_tokens_to_features(input_tokens, max_seq_length, tokenizer):
    """
    Convert token sequence to BERT input features.

    Input tokens will need to be tokenized by BERT wordpiece tokenizer. This will
    require us to keep track of the token starts within the subtoken list.
    As in the Google paper, we will only track the loss associated with the
    first token.
    """

    # do Wordpiece tokenization on input tokens
    tokens = [tokenizer.tokenize(tok) for tok in input_tokens]
    lengths = [len(sub_toks) for sub_toks in tokens]

    # calculate original token starts in the list of sub-tokens from
    # wordpiece subtoken lengths
    token_starts = [0] + np.cumsum(lengths).tolist()[:-1]

    # flatten the list of lists to get a single list of Wordpiece tokens
    tokens = flatten(tokens)

    # truncate tokens and token starts if needed
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
        token_starts = [t for t in token_starts if t < (max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # increment token starts due to [CLS]
    token_starts = [(t + 1) for t in token_starts]

    # set segment ids
    segment_ids = [0] * len(tokens)
    
    feature = pad_and_get_ids(tokens, segment_ids, max_seq_length, tokenizer)

    return TokenFeatures(feature.input_ids, feature.input_mask,
                         feature.segment_ids, token_starts)
