""" Tools for loading datasets as Classification/SequenceLabelling Examples. """
import os
import logging
from collections import namedtuple

from tqdm import tqdm
from transformers import BasicTokenizer

from utils.data import retokenize

DATA_PATH = 'data/'
ClassificationExample = namedtuple(
    'ClassificationExample', ['id', 'tokens_a', 'tokens_b', 'label'])
SequenceLabellingExample = namedtuple(
    'SequenceLabellingExample', ['id', 'token_sequence', 'label_sequence'])


def load_classification_dataset(step, do_lower_case):
    """ Loads classification exampels from a dataset. """
    assert step in ['train', 'test']
    basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    path = os.path.join(DATA_PATH, 'classification', f'{step}.txt')
    examples = []
    with open(path, 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        for i, line in tqdm(enumerate(lines), desc=f'reading `{os.path.basename(path)}`...'):
            # example: __label__negative I don't like tomatoes.
            splitline = line.strip().split()
            label = splitline[0].split('__label__')[-1]
            tokens = ' '.join(splitline[1:])
            examples.append(
                ClassificationExample(
                    id=i,
                    tokens_a=basic_tokenizer.tokenize(tokens),
                    tokens_b=None,
                    label=label,
                )
            )
    logging.info('Number of `%s` examples: %d', step, len(examples))
    return examples

def load_sequence_labelling_dataset(step, do_lower_case):
    """ Loads sequence labelling examples from a dataset. """
    assert step in ['train', 'test']
    path = os.path.join(DATA_PATH, 'sequence_labelling', f'{step}.txt')
    i = 0
    examples = []
    with open(path, 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        token_sequence = []
        label_sequence = []
        for line in tqdm(lines, desc=f'reading `{os.path.basename(path)}`...'):
            # example:
            #          My O
            #          name O
            #          is O
            #          Hicham B-PER
            #          . O
            splitline = line.strip().split()
            if splitline:
                token, label = splitline
                token_sequence.append(token)
                label_sequence.append(label)
            else:
                examples.append(
                    SequenceLabellingExample(
                        id=i,
                        token_sequence=token_sequence,
                        label_sequence=label_sequence,
                    )
                )
                i += 1
                token_sequence = []
                label_sequence = []

    # Don't forget to add the last example
    if token_sequence:
        examples.append(
            SequenceLabellingExample(
                id=i,
                token_sequence=token_sequence,
                label_sequence=label_sequence,
            )
        )

    retokenize(
        examples,
        tokenization_function=BasicTokenizer(do_lower_case=do_lower_case).tokenize)
    logging.info('Number of `%s` examples: %d', step, len(examples))
    return examples
