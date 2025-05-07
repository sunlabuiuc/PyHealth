import argparse
from pyhealth.trainer import Trainer
from model_wrapped import Seq2SeqModel
from data_utils import build_vocab, load_pairs_jsonl, PyHealthSeq2SeqDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./results")
args = parser.parse_args()

# Load data and vocab
vocab = build_vocab(args.data_path)
pairs = load_pairs_jsonl(args.data_path)

# Initialize dataset
dataset = PyHealthSeq2SeqDataset(pairs, vocab)

# Model configuration
model_config = {
    "emb": 256,
    "hid": 512
}

# Initialize model
model = Seq2SeqModel(dataset, model_config)

# Initialize trainer
trainer = Trainer(
    model=model,
    dataset=dataset,
    metrics=["bleu"],
    output_dir=args.output_dir
)

# Start training
trainer.train()
