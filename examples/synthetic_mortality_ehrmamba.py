"""Small reproducible EHRMamba hyperparameter ablation on synthetic PyHealth samples.

No credentials or clinical data required. This tests engineering behavior, not
clinical utility. Run from repository root:
    python -m examples.synthetic_mortality_ehrmamba --epochs 5 --output ablation.csv

Ablations vary embedding width, SSM depth, state size, and dropout. All
configurations share the same patient-level split and synthetic label rule.
CSV records per-configuration held-out metrics, parameter counts, and time.

EHRMamba Hyperparameter Ablation Study
=====================================

Dataset:
    Synthetic patient-level binary classification using
    PyHealth's SampleDataset infrastructure.

Experimental setup:
    - 64 training patients
    - 16 validation patients
    - 16 test patients
    - 5 training epochs
    - Random seed: 17
    - Model selection: minimum validation loss

Configurations:
    baseline:      embedding_dim=16, layers=1, state_dim=4
    wider:         embedding_dim=32, layers=1, state_dim=4
    deeper:        embedding_dim=16, layers=2, state_dim=4
    larger_state:  embedding_dim=16, layers=1, state_dim=8
    dropout:       embedding_dim=16, layers=1, state_dim=4,
                   dropout=0.2

Results (seed 17):
    Configuration  Accuracy    F1
    baseline       0.6875      0.6154
    wider          0.8125      0.7692
    deeper         0.6875      0.6667
    larger_state   0.7500      0.7500
    dropout        0.6875      0.6154

Findings:
    Increasing embedding dimension produced the highest
    test accuracy and F1 in this experiment.

    Increasing state dimension also improved F1 relative
    to the baseline.

    Increasing depth or adding dropout did not improve
    test accuracy under the tested settings.

Limitations:
    Results are based on a small synthetic dataset and
    one random seed. They are not clinical benchmarks
    and do not establish statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EHRMamba


def build_samples(n=96, seed=17):
    rng = random.Random(seed)
    records = []
    for i in range(n):
        risk = i % 2 == 0
        conditions = rng.choices(["DX_A", "DX_B", "DX_C", "DX_D"], k=rng.randint(1, 5))
        drugs = rng.choices(["RX_A", "RX_B", "RX_C"], k=rng.randint(1, 4))
        if risk:
            conditions.insert(rng.randrange(len(conditions) + 1), "DX_RISK")
            drugs.insert(rng.randrange(len(drugs) + 1), "RX_RISK")
        records.append(
            {
                "patient_id": f"synthetic_{i:04d}",
                "conditions": conditions,
                "drugs": drugs,
                "label": int(risk),
            }
        )
    rng.shuffle(records)
    return records


def evaluate(model, loader):
    model.eval()
    labels, probs, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            result = model(**batch)
            labels.extend(result["y_true"].reshape(-1).cpu().tolist())
            probs.extend(result["y_prob"].reshape(-1).cpu().tolist())
            losses.append((result["loss"].item(), len(batch["label"])))
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs)
    predictions = (probs >= 0.5).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "loss": sum(v * n for v, n in losses) / sum(n for _, n in losses),
        "accuracy": float((predictions == labels).mean()),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", default="ehrmamba_ablation.csv")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    dataset = create_sample_dataset(
        samples=build_samples(seed=args.seed),
        input_schema={"conditions": "sequence", "drugs": "sequence"},
        output_schema={"label": "binary"},
        dataset_name=f"ehrmamba_ablation_synthetic_{args.seed}",
    )
    # Exactly one sample per synthetic patient; fixed disjoint partitions.
    from torch.utils.data import Subset

    train = get_dataloader(
        Subset(dataset, list(range(64))), batch_size=16, shuffle=True
    )
    val = get_dataloader(
        Subset(dataset, list(range(64, 80))), batch_size=16, shuffle=False
    )
    test = get_dataloader(
        Subset(dataset, list(range(80, 96))), batch_size=16, shuffle=False
    )
    configs = [
        ("baseline", 16, 1, 4, 0.0),
        ("wider", 32, 1, 4, 0.0),
        ("deeper", 16, 2, 4, 0.0),
        ("larger_state", 16, 1, 8, 0.0),
        ("dropout", 16, 1, 4, 0.2),
    ]
    rows = []
    for name, dim, layers, state, dropout in configs:
        torch.manual_seed(args.seed)
        model = EHRMamba(
            dataset=dataset,
            embedding_dim=dim,
            num_layers=layers,
            state_dim=state,
            dropout=dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        start = time.perf_counter()
        best_val = float("inf")
        best_weights = None
        for epoch in range(args.epochs):
            model.train()
            for batch in train:
                optimizer.zero_grad()
                loss = model(**batch)["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss for {name}")
                loss.backward()
                optimizer.step()
            metrics = evaluate(model, val)
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                best_weights = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
        model.load_state_dict(best_weights)
        result = evaluate(model, test)
        row = {
            "config": name,
            "embedding_dim": dim,
            "num_layers": layers,
            "state_dim": state,
            "dropout": dropout,
            "epochs": args.epochs,
            "seed": args.seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_val_loss": round(best_val, 5),
            "test_loss": round(result["loss"], 5),
            "test_accuracy": round(result["accuracy"], 4),
            "test_f1": round(result["f1"], 4),
            "seconds": round(time.perf_counter() - start, 2),
        }
        rows.append(row)
        print(row, flush=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.output}")
    print(
        "Interpretation: synthetic benchmark only; compare test metrics, parameter count, and runtime."
    )


if __name__ == "__main__":
    main()
