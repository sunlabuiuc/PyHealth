"""Weight positive examples for a rare binary outcome.

With 10% positives, the plain loss lets a model score well by leaning towards
"negative". ``set_pos_weight("balanced", train)`` weights each positive by the
negative/positive ratio of the training split. The script trains both and
reports held-out scores side by side; how much weighting helps depends on the
data. It also pushes predicted probabilities up, so check calibration before
reading them as risks.

Runs in under a minute on CPU with synthetic data; no download.

Usage:
    python examples/imbalanced_outcome_pos_weight.py
"""

import random

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.trainer import Trainer


def make_samples(n=400, seed=0):
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        label = int(rng.random() < 0.10)  # rare outcome
        codes = rng.sample(["A", "B", "C", "D", "E", "F"], 3)
        if label and rng.random() < 0.8:
            codes.append("RISK")  # a code that marks most positives
        samples.append({"patient_id": f"p{i}", "codes": codes, "label": label})
    return samples


def train_and_evaluate(dataset, train, test, pos_weight):
    torch.manual_seed(0)
    model = RNN(dataset=dataset)
    if pos_weight is not None:
        model.set_pos_weight(pos_weight, train)
    trainer = Trainer(
        model=model, metrics=["pr_auc", "recall", "f1"], enable_logging=False
    )
    trainer.train(get_dataloader(train, batch_size=32, shuffle=True), epochs=5)
    scores = trainer.evaluate(get_dataloader(test, batch_size=64))
    weight = None if model.pos_weight is None else round(float(model.pos_weight), 2)
    return weight, {k: round(v, 3) for k, v in scores.items()}


def main():
    dataset = create_sample_dataset(
        samples=make_samples(),
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="imbalanced_demo",
    )
    train, _, test = split_by_patient(dataset, [0.7, 0.1, 0.2], seed=0)
    for pos_weight in (None, "balanced"):
        weight, scores = train_and_evaluate(dataset, train, test, pos_weight)
        print(f"pos_weight={weight}: {scores}")


if __name__ == "__main__":
    main()
