# Functions are adapted from Huggingface's transformers library:
# https://github.com/huggingface/transformers

""" Helper functions. """
import random
import logging
from collections import namedtuple

import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset


def retokenize(examples, tokenization_function):

    for i, example in tqdm.tqdm(enumerate(examples), desc='retokenizing examples...'):
        if type(example).__name__ == 'ClassificationExample':
            assert example.tokens_a
            if example.tokens_b is not None:
                assert example.tokens_b
            assert example.label

            new_tokens_a = []
            for token_a in example.tokens_a:
                new_tokens_a.extend(tokenization_function(token_a))
            example = example._replace(tokens_a=new_tokens_a if new_tokens_a else [''])

            if example.tokens_b is not None:
                new_tokens_b = []
                for token_b in example.tokens_b:
                    new_tokens_b.extend(tokenization_function(token_b))
                example = example._replace(tokens_b=new_tokens_b if new_tokens_b else [''])

        elif type(example).__name__ == 'SequenceLabellingExample':
            tokens = example.token_sequence
            labels = example.label_sequence
            assert tokens
            assert len(tokens) == len(labels)
            new_tokens, new_labels = [], []
            for token, label in zip(tokens, labels):
                retokenized_token = tokenization_function(token)
                if retokenized_token != [token]:
                    if label != 'O':
                        label_pos = label[:2]
                        label_type = label.split('-')[-1]
                        if label_pos == 'B-':
                            new_label = [label] + (len(retokenized_token) - 1) * ['I-' + label_type]
                        elif label_pos == 'I-':
                            new_label = [label] * len(retokenized_token)
                    else:
                        new_label = [label] * len(retokenized_token)
                    new_tokens.extend(retokenized_token)
                    new_labels.extend(new_label)
                else:
                    new_tokens.append(token)
                    new_labels.append(label)
            if new_tokens:
                example = example._replace(token_sequence=new_tokens)
                example = example._replace(label_sequence=new_labels)
            else:
                example = example._replace(token_sequence=[''])
                example = example._replace(label_sequence=['O'])
        examples[i] = example


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


def convert_examples_to_features__classification(
        args, tokenizer, examples, labels,
        pad_token_id, pad_token_label_id, max_seq_length):
    """Converts classification examples into pytorch tensors."""

    InputFeatures = namedtuple(
        'ClassificationFeatures',
        ['input_ids', 'input_mask', 'segment_ids', 'label_id'])

    label_map = {label : i for i, label in enumerate(labels)}

    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = example.tokens_a

        tokens_b = None
        if example.tokens_b:
            tokens_b = example.tokens_b
            seq_len = len(tokens_a) + len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
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
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        # Tokenization of inputs
        if args.embedding == 'bert-base-uncased':
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokenizer.as_padded_tensor([tokens], maxlen=max_seq_length)[0]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(tokens)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_mask)
        if args.embedding == 'bert-base-uncased':
            input_ids += [pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 3:
            logging.info("*** Example ***")
            logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
        )
    return features


def convert_examples_to_features__tagging(
        args, tokenizer, examples, labels,
        pad_token_id, pad_token_label_id, max_seq_length):
    """Converts tagging examples into pytorch tensors."""

    InputFeatures = namedtuple(
        'SequenceLabelligFeatures',
        ['input_ids', 'input_mask', 'segment_ids', 'label_ids'])

    label_map = {label: i for i, label in enumerate(labels)}

    data_iterator = tqdm.tqdm(enumerate(examples), total=len(examples))

    features = []
    for i, example in data_iterator:
        tokens = example.token_sequence
        labels = example.label_sequence

        #label_ids = [label_map[label] for label in labels]
        label_ids = []
        for token, label in zip(tokens, labels):
            if token.startswith('##'):
                label_ids.append(pad_token_label_id)
            else:
                label_ids.append(label_map[label])

        # Account for [CLS] and [SEP] with "- 2"
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # Handle [SEP]
        tokens += ["[SEP]"]
        label_ids += [pad_token_label_id]
        segment_ids = [0] * len(tokens)  # Only one sentence, so all segment ids are 0

        # Handle [CLS]
        tokens = ["[CLS]"] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [0] + segment_ids

        # Tokenization of inputs
        if args.embedding == 'bert-base-uncased':
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokenizer.as_padded_tensor([tokens], maxlen=max_seq_length)[0]

        # The mask has 1 for real tokens and 0 for padding tokens.
        # Only real tokens are attended to.
        input_mask = [1] * len(tokens)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_mask)
        if args.embedding == 'bert-base-uncased':
            input_ids += [pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if i < 3:
            logging.info("*** Example ***")
            logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logging.info("labels: %s", " ".join([str(x) for x in labels]))
            logging.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids)
        )
    return features


def build_features(
        args, split, tokenizer, examples, labels,
        pad_token_id, pad_token_label_id, max_seq_length):

    logging.info("Building features from data...")
    if args.task == 'sequence_labelling':
        func = convert_examples_to_features__tagging
    else:
        func = convert_examples_to_features__classification
    features = func(
        args, tokenizer, examples, labels,
        pad_token_id, pad_token_label_id, max_seq_length
    )

    # Convert to Tensors and build dataset
    if args.embedding == 'bert-base-uncased':
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    else:
        all_input_ids = torch.tensor([f.input_ids.tolist() for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.task == 'sequence_labelling':
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
