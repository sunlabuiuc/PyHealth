# Adapted from Huggingface's transformers library:
# https://github.com/allenai/allennlp/

""" Main script. """
import os
import logging
import argparse
import datetime
from collections import Counter

import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForSequenceClassification
)
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel

from data import load_classification_dataset, load_sequence_labelling_dataset

from utils.misc import set_seed
from utils.data import retokenize, build_features
from utils.training import train, evaluate

from download import MODEL_TO_URL
AVAILABLE_MODELS = list(MODEL_TO_URL.keys()) + ['bert-base-uncased']

def parse_args():
    """ Parse command line arguments and initialize experiment. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=['classification', 'sequence_labelling'],
        help="The evaluation task."
    )
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS,
        help="The model to use."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to apply lowercasing during tokenization."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size to use for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--validation_ratio",
        default=0.5, type=float, help="Proportion of training set to use as a validation set.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_ratio",
        default=0.1, type=int, help="Linear warmup over warmup_ratio*total_steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Do training & validation."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Do prediction on the test set."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()
    args.start_time = datetime.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')
    args.output_dir = os.path.join(
        'results',
        args.task,
        args.embedding,
        f'{args.start_time}__seed-{args.seed}')

    # --------------------------------- INIT ---------------------------------

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO)

    # Check for GPUs
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1  # This script doesn't support multi-gpu
        args.device = torch.device("cuda")
        logging.info("Using GPU (`%s`)", torch.cuda.get_device_name(0))
    else:
        args.device = torch.device("cpu")
        logging.info("Using CPU")

    # Set random seed for reproducibility
    set_seed(seed_value=args.seed)

    return args

def main(args):
    """ Main function. """

    # --------------------------------- DATA ---------------------------------

    # Tokenizer
    logging.disable(logging.INFO)
    try:
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join('pretrained-models', args.embedding),
            do_lower_case=args.do_lower_case)
    except OSError:
        # For CharacterBert models use BertTokenizer.basic_tokenizer for tokenization
        # and CharacterIndexer for indexing 
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join('pretrained-models', 'bert-base-uncased'),
            do_lower_case=args.do_lower_case)
        tokenizer = tokenizer.basic_tokenizer
        characters_indexer = CharacterIndexer()
    logging.disable(logging.NOTSET)

    tokenization_function = tokenizer.tokenize

    # Pre-processsing: apply basic tokenization (both) then split into wordpieces (BERT only)
    data = {}
    for split in ['train', 'test']:
        if args.task == 'classification':
            func = load_classification_dataset
        elif args.task == 'sequence_labelling':
            func = load_sequence_labelling_dataset
        else:
            raise NotImplementedError

        data[split] = func(step=split, do_lower_case=args.do_lower_case)
        retokenize(data[split], tokenization_function)

    logging.info('Splitting training data into train / validation sets...')
    data['validation'] = data['train'][:int(args.validation_ratio * len(data['train']))]
    data['train'] = data['train'][int(args.validation_ratio * len(data['train'])):]
    logging.info('New number of training sequences: %d', len(data['train']))
    logging.info('New number of validation sequences: %d', len(data['validation']))

    # Count target labels or classes
    if args.task == 'classification':
        counter_all = Counter(
            [example.label for example in data['train'] + data['validation'] + data['test']])
        counter = Counter(
            [example.label for example in data['train']])

        # Maximum sequence length is either 512 or maximum token sequence length + 3
        max_seq_length = min(
            512,
            3 + max(
                map(len, [
                    e.tokens_a if e.tokens_b is None else e.tokens_a + e.tokens_b
                    for e in data['train'] + data['validation'] + data['test']
                ])
            )
        )
    elif args.task == 'sequence_labelling':
        counter_all = Counter(
            [label
             for example in data['train'] + data['validation'] + data['test']
             for label in example.label_sequence])
        counter = Counter(
            [label
             for example in data['train']
             for label in example.label_sequence])

        # Maximum sequence length is either 512 or maximum token sequence length + 5
        max_seq_length = min(
            512,
            5 + max(
                map(len, [
                    e.token_sequence
                    for e in data['train'] + data['validation'] + data['test']
                ])
            )
        )
    else:
        raise NotImplementedError
    labels = sorted(counter_all.keys())
    num_labels = len(labels)

    logging.info("Goal: predict the following labels")
    for i, label in enumerate(labels):
        logging.info("* %s: %s (count: %s)", label, i, counter[label])

    # Input features: list[token indices] (BERT) or list[list[character indices]] (CharacterBERT)
    pad_token_id = None
    if 'character' not in args.embedding:
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    pad_token_label_id = None
    if args.task == 'sequence_labelling':
        pad_token_label_id = CrossEntropyLoss().ignore_index

    dataset = {}
    logging.info("Maximum sequence lenght: %s", max_seq_length)
    for split in data:
        dataset[split] = build_features(
            args,
            split=split,
            tokenizer=tokenizer \
                if 'character' not in args.embedding \
                else characters_indexer,
            examples=data[split],
            labels=labels,
            pad_token_id=pad_token_id,
            pad_token_label_id=pad_token_label_id,
            max_seq_length=max_seq_length)

    del data  # Not used anymore

    # --------------------------------- MODEL ---------------------------------

    # Initialize model
    if args.task == 'classification':
        model = BertForSequenceClassification
    elif args.task == 'sequence_labelling':
        model = BertForTokenClassification
    else:
        raise NotImplementedError

    logging.info('Loading `%s` model...', args.embedding)
    logging.disable(logging.INFO)
    config = BertConfig.from_pretrained(
        os.path.join('pretrained-models', args.embedding),
        num_labels=num_labels)
    if 'character' not in args.embedding:
        model = model.from_pretrained(
            os.path.join('pretrained-models', args.embedding),
            config=config)
    else:
        model = model(config=config)
        model.bert = CharacterBertModel.from_pretrained(
            os.path.join('pretrained-models', args.embedding),
            config=config)
    logging.disable(logging.NOTSET)

    model.to(args.device)
    logging.info('Model:\n%s', model)

    # ------------------------------ TRAIN / EVAL ------------------------------

    # Log args
    logging.info('Using the following arguments for training:')
    for k, v in vars(args).items():
        logging.info("* %s: %s", k, v)

    # Training
    if args.do_train:
        global_step, train_loss, best_val_metric, best_val_epoch = train(
            args=args,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            pad_token_label_id=pad_token_label_id
        )
        logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
        logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)

    # Evaluation on test data
    if args.do_predict:

        # Load best model
        if args.task == 'classification':
            model = BertForSequenceClassification
        elif args.task == 'sequence_labelling':
            model = BertForTokenClassification
        else:
            raise NotImplementedError

        logging.disable(logging.INFO)
        if 'character' not in args.embedding:
            model = model.from_pretrained(args.output_dir)
        else:
            state_dict = torch.load(
                os.path.join(args.output_dir, 'pytorch_model.bin'), map_location='cpu')
            model = model(config=config)
            model.bert = CharacterBertModel(config=config)
            model.load_state_dict(state_dict, strict=True)
        logging.disable(logging.NOTSET)
        model.to(args.device)

        # Compute predictions and metrics
        results, _ = evaluate(
            args=args,
            eval_dataset=dataset["test"],
            model=model, labels=labels,
            pad_token_label_id=pad_token_label_id
        )

        # Save metrics
        with open(os.path.join(args.output_dir, 'performance_on_test_set.txt'), 'w') as f:
            f.write(f'best validation score: {best_val_metric}\n')
            f.write(f'best validation epoch: {best_val_epoch}\n')
            f.write('--- Performance on test set ---\n')
            for k, v in results.items():
                f.write(f'{k}: {v}\n')

if __name__ == "__main__":
    main(parse_args())
