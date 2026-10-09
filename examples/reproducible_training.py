"""Reproducible training: one seed, identical predictions.

Trains the same model twice with the same seed and once with another, on an
in-memory and a disk-backed dataset, and compares the predictions. With
``torch.manual_seed`` set before the loaders and the model are created, the
shuffle order, the initial weights and therefore the predictions are
bit-identical; a different seed gives different ones.

Runs in under a minute on CPU with synthetic data; no download.

Usage:
    python examples/reproducible_training.py
"""

import hashlib

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.trainer import Trainer

samples = [
    {
        "patient_id": f"p{i}",
        "codes": [f"c{(i + j) % 9}" for j in range(1 + i % 4)],
        "label": int(i % 3 == 0),
    }
    for i in range(60)
]


def run(seed: int, in_memory: bool) -> str:
    torch.manual_seed(seed)
    dataset = create_sample_dataset(
        samples, {"codes": "sequence"}, {"label": "binary"}, in_memory=in_memory
    )
    train_loader = get_dataloader(dataset, batch_size=8, shuffle=True)
    trainer = Trainer(model=RNN(dataset=dataset), enable_logging=False)
    trainer.train(train_loader, get_dataloader(dataset, 16), epochs=3, monitor="roc_auc")
    _, y_prob, _ = trainer.inference(get_dataloader(dataset, 16))
    return hashlib.sha256(y_prob.tobytes()).hexdigest()[:12]


def main():
    for in_memory in (True, False):
        kind = "in-memory" if in_memory else "disk-backed"
        a, b, c = run(0, in_memory), run(0, in_memory), run(1, in_memory)
        print(f"{kind:12s} seed 0: {a}  seed 0 again: {b}  seed 1: {c}")
        print(f"{'':12s} same seed identical: {a == b}, different seed differs: {a != c}")


if __name__ == "__main__":
    main()
