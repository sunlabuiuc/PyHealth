"""RNN with nested-code pooling strategies.

Demonstrates the ``code_pooling`` parameter of :class:`pyhealth.models.RNN`
on a synthetic EHR dataset where each sample contains a list of visits and
each visit contains a list of ICD-like diagnosis codes (nested_sequence).

The four available strategies are compared side-by-side:

- ``"sum"``       — original baseline (sum over codes in a visit).
- ``"mean"``      — mask-aware mean; removes scale bias from visit length.
- ``"max"``       — element-wise max across codes in a visit.
- ``"attention"`` — learnable per-code importance score (recommended for
                    tasks where the primary diagnosis dominates, e.g.,
                    in-hospital mortality prediction).

Usage::

    python examples/rnn_code_pooling_demo.py
"""

from __future__ import annotations

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN

# ---------------------------------------------------------------------------
# Synthetic EHR data: each patient has one "visit record" where
# conditions is a list-of-lists (nested_sequence):
#   outer list  → visits
#   inner lists → ICD codes recorded during that visit
# ---------------------------------------------------------------------------
SAMPLES = [
    {
        "patient_id": "p0",
        "visit_id": "v0",
        "conditions": [
            ["I10", "E11.9", "K92.1", "Z87.39"],   # visit 1: 4 codes
            ["I10", "J18.9"],                        # visit 2: 2 codes
            ["E11.9"],                               # visit 3: 1 code
        ],
        "label": 1,
    },
    {
        "patient_id": "p1",
        "visit_id": "v1",
        "conditions": [
            ["A41.9", "N17.9"],
            ["A41.9", "J96.0", "D65"],
        ],
        "label": 0,
    },
    {
        "patient_id": "p2",
        "visit_id": "v2",
        "conditions": [
            ["I50.9"],
            ["I50.9", "N18.3", "E87.6"],
            ["I50.9", "N18.4"],
        ],
        "label": 1,
    },
    {
        "patient_id": "p3",
        "visit_id": "v3",
        "conditions": [
            ["C34.10", "J98.2"],
            ["C34.10"],
        ],
        "label": 0,
    },
]


def build_dataset():
    return create_sample_dataset(
        samples=SAMPLES,
        input_schema={"conditions": "nested_sequence"},
        output_schema={"label": "binary"},
        dataset_name="rnn_pooling_demo",
    )


def run_one_epoch(model: RNN, loader) -> dict[str, float]:
    """Single forward pass; returns loss and predicted probabilities."""
    batch = next(iter(loader))
    with torch.no_grad():
        ret = model(**batch)
    probs = ret["y_prob"].squeeze(-1).tolist()
    labels = ret["y_true"].view(-1).tolist()
    return {
        "loss": float(ret["loss"]),
        "probs": [round(float(p), 3) for p in probs],
        "labels": [int(l) for l in labels],
    }


def main():
    dataset = build_dataset()
    loader = get_dataloader(dataset, batch_size=len(SAMPLES), shuffle=False)

    pooling_modes = ["sum", "mean", "max", "attention"]

    print("=" * 60)
    print("RNN nested-code pooling comparison")
    print(f"  Dataset: {len(SAMPLES)} synthetic EHR patients")
    print(f"  Input:   nested_sequence (visits × ICD codes)")
    print(f"  Task:    binary classification (mortality proxy)")
    print("=" * 60)

    for mode in pooling_modes:
        torch.manual_seed(42)
        model = RNN(
            dataset=dataset,
            embedding_dim=32,
            hidden_dim=32,
            code_pooling=mode,
        )
        result = run_one_epoch(model, loader)

        n_pooling_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "code_pooling_layers" in name
        )

        print(f"\n[{mode:>9}]  loss={result['loss']:.4f}  "
              f"pooling_params={n_pooling_params}")
        print(f"           probs={result['probs']}")
        print(f"           true ={result['labels']}")

    print("\n" + "=" * 60)
    print("Notes:")
    print("  - 'sum' and 'mean' have 0 extra trainable params.")
    print("  - 'attention' adds embedding_dim params per nested feature,")
    print("    learning which codes matter most for the task.")
    print("  - Losses differ because each mode initializes to a different")
    print("    effective representation (same random seed, different pooling).")
    print("=" * 60)


if __name__ == "__main__":
    main()
