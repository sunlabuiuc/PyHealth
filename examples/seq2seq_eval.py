import argparse
from pyhealth.evaluator import Evaluator
from model_wrapped import Seq2SeqModel
from data_utils import build_vocab, load_pairs_jsonl, PyHealthSeq2SeqDataset
import torch
import re
import sqlite3
from rdflib import Graph

# Custom metric functions
def remove_literals(tokens):
    return [t for t in tokens if not re.fullmatch(r'".*?"', t) and not re.fullmatch(r'\d+(\.\d+)?', t)]

def logic_form_accuracy(preds, labels):
    return {"logic_acc": sum(p == l for p, l in zip(preds, labels)) / len(preds)}

def structural_match_accuracy(preds, labels):
    return {"structure_acc": sum(remove_literals(p) == remove_literals(l) for p, l in zip(preds, labels)) / len(preds)}

# Main evaluation script
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

# Load data
vocab = build_vocab(args.data_path)
pairs = load_pairs_jsonl(args.data_path)
dataset = PyHealthSeq2SeqDataset(pairs, vocab)

# Load model
model_config = {"emb": 256, "hid": 512}
model = Seq2SeqModel(dataset, model_config)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

# Initialize evaluator
evaluator = Evaluator(
    model=model,
    dataset=dataset,
    metrics=[logic_form_accuracy, structural_match_accuracy]
)

# Run evaluation
evaluator.evaluate()
