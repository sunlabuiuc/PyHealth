"""
EBCL: training loop + validation metrics (synthetic data).

Demonstrates :class:`~pyhealth.trainer.Trainer` with :class:`~pyhealth.models.EBCL`
when ``supervised_weight > 0`` so the linear probe is trained jointly with the
contrastive loss. Validation reports **roc_auc** (and other binary metrics) via
``pyhealth.metrics``.

Requires **no** MIMIC files. For contrastive-only pretraining (probe not trained),
use ``supervised_weight=0`` and either omit ``val_dataloader`` or monitor **loss**
only (see ``Trainer.evaluate`` when classification metrics are not meaningful).

Paper: https://arxiv.org/abs/2312.10308

Run::

    cd /path/to/PyHealth
    python examples/ebcl_eval_probe_demo.py
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCL
from pyhealth.trainer import Trainer


def _samples(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = ["A", "B", "C", "D", "E", "F", "G"]
    out = []
    for i in range(n):
        out.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "conditions_pre": [pool[rng.randrange(7)], pool[rng.randrange(7)]],
                "conditions_post": [pool[rng.randrange(7)], pool[rng.randrange(7)]],
                "label": i % 2,
            }
        )
    return out


def main() -> None:
    # One dataset so SequenceProcessor vocab is shared; split for train/val.
    n_total = 40
    n_train = 28
    all_samples = _samples(n_total, seed=42)
    full_ds = create_sample_dataset(
        samples=all_samples,
        input_schema={
            "conditions_pre": "sequence",
            "conditions_post": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="ebcl_probe_demo",
    )
    train_ds = full_ds.subset(range(n_train))
    val_ds = full_ds.subset(range(n_train, n_total))

    model = EBCL(
        train_ds,
        embedding_dim=48,
        hidden_dim=48,
        projection_dim=24,
        temperature=0.15,
        supervised_weight=0.3,
        dropout=0.2,
    )

    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)

    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "f1"],
        enable_logging=False,
    )

    print("Training EBCL (contrastive + probe). Val metrics: roc_auc, pr_auc, f1\n")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=8,
        optimizer_params={"lr": 3e-3},
        monitor="roc_auc",
        monitor_criterion="max",
        patience=4,
        load_best_model_at_last=False,
    )
    print("\nDone. Best checkpoint not saved (enable_logging=False).")


if __name__ == "__main__":
    main()
