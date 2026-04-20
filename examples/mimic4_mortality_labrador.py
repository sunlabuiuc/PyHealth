"""Ablation example: LabradorModel for mortality prediction.

This script intentionally uses synthetic data so it can be run quickly while
showing the expected PR example format for model contributions.

Paper: Bellamy et al., "Labrador: Exploring the Limits of Masked Language
Modeling for Laboratory Data" (ML4H 2024)
https://arxiv.org/abs/2312.11502
"""

from __future__ import annotations

import random
from typing import Dict, List

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LabradorModel
from pyhealth.trainer import Trainer


def build_synthetic_samples(n: int = 40, seed: int = 7) -> List[Dict]:
    """Builds small synthetic patient-visit samples for a fast example run."""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        seq_len = rng.choice([4, 5, 6])
        lab_codes = [rng.randint(1, 60) for _ in range(seq_len)]
        lab_values = [round(rng.uniform(0.0, 1.0), 3) for _ in range(seq_len)]
        label = int(sum(lab_values) / seq_len > 0.55)
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "lab_codes": lab_codes,
                "lab_values": lab_values,
                "label": label,
            }
        )
    return samples


def run_ablation() -> None:
    """Runs a tiny ablation over hidden_dim / num_layers / dropout."""
    samples = build_synthetic_samples(n=40)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_dataset = create_sample_dataset(
        samples=train_samples,
        input_schema={"lab_codes": "sequence", "lab_values": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="mimic4_labrador_train_synth",
    )
    val_dataset = create_sample_dataset(
        samples=val_samples,
        input_schema={"lab_codes": "sequence", "lab_values": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="mimic4_labrador_val_synth",
    )

    train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=8, shuffle=False)

    configs = [
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.1},
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.1},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.3},
    ]

    print("=== Labrador tiny ablation (synthetic) ===")
    for cfg in configs:
        model = LabradorModel(dataset=train_dataset, vocab_size=64, **cfg)
        trainer = Trainer(model=model, metrics=["accuracy", "roc_auc"])
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=2,
            monitor="roc_auc",
        )
        score = trainer.evaluate(val_loader)
        print(f"config={cfg} -> {score}")


if __name__ == "__main__":
    run_ablation()
