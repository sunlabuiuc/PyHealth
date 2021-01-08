# Functions are adapted from Huggingface's transformers library:
# https://github.com/huggingface/transformers

""" Defines training and evaluation functions. """
import os
import logging
import datetime

import tqdm
import numpy as np
import sklearn.metrics as sklearn_metrics
import metrics.sequence_labelling as seqeval_metrics

import torch
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.misc import set_seed


def train(args, dataset, model, tokenizer, labels, pad_token_label_id):
    """ Trains the given model on the given dataset. """

    train_dataset = dataset['train']
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)

    n_train_steps__single_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    n_train_steps = n_train_steps__single_epoch * args.num_train_epochs
    args.logging_steps = n_train_steps__single_epoch

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio*n_train_steps),
        num_training_steps=n_train_steps
    )

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info(
        "  Total train batch size (w. accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", n_train_steps)
    logging.info("  Using linear warmup (ratio=%s)", args.warmup_ratio)
    logging.info("  Using weight decay (value=%s)", args.weight_decay)
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    best_metric, best_epoch = -1.0, -1  # Init best -1 so that 0 > best

    model.zero_grad()
    train_iterator = tqdm.trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    set_seed(seed_value=args.seed)  # Added here for reproductibility
    for num_epoch in train_iterator:
        epoch_iterator = tqdm.tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    # Log metrics
                    # -- Only evaluate when single GPU otherwise metrics may not average well
                    results, _ = evaluate(
                        args=args,
                        eval_dataset=dataset["validation"],
                        model=model, labels=labels,
                        pad_token_label_id=pad_token_label_id
                    )

                    logging_loss = tr_loss
                    metric = results['f1']

                    if metric > best_metric:
                        best_metric = metric
                        best_epoch = num_epoch

                        # Save model checkpoint
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        model.save_pretrained(args.output_dir)
                        if 'character' not in args.embedding:
                            tokenizer.save_pretrained(args.output_dir)
                        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                        logging.info("Saving model checkpoint to %s", args.output_dir)

                        #torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
                        #torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
                        #logging.info("Saving optimizer and scheduler states to %s", args.output_dir)

    return global_step, tr_loss / global_step, best_metric, best_epoch


def evaluate(args, eval_dataset, model, labels, pad_token_label_id):
    """ Evaluates the given model on the given dataset. """

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)

    # Evaluate!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    label_map = {i: label for i, label in enumerate(labels)}
    if args.task == 'classification':
        preds_list = np.argmax(preds, axis=1)
        results = {
            "loss": eval_loss,
            "precision": sklearn_metrics.precision_score(out_label_ids, preds_list, average='micro'),
            "recall": sklearn_metrics.recall_score(out_label_ids, preds_list, average='micro'),
            "f1": sklearn_metrics.f1_score(out_label_ids, preds_list, average='micro'),
            "accuracy": sklearn_metrics.accuracy_score(out_label_ids, preds_list),
        }
    else:
        preds = np.argmax(preds, axis=2)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": seqeval_metrics.precision_score(out_label_list, preds_list),
            "recall": seqeval_metrics.recall_score(out_label_list, preds_list),
            "f1": seqeval_metrics.f1_score(out_label_list, preds_list),
        }

    logging.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logging.info("  %s = %s", key, str(results[key]))

    return results, preds_list
