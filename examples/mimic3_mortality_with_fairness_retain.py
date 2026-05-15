"""Ablation: how do RETAIN hyperparameters affect per-cohort fairness?

Demonstrates :class:`pyhealth.tasks.MortalityPredictionWithFairnessMIMIC3`
and :func:`pyhealth.tasks.fairness_utils.audit_predictions` end-to-end on
MIMIC-III. Trains three RETAIN configurations and produces a side-by-side
fairness comparison across 7 demographic groupings.

Paper: Hoche et al., *FAMEWS: A Fairness Auditing Tool for Medical
Early-Warning Systems*, CHIL 2024. https://proceedings.mlr.press/v248/hoche24a.html

Usage:
    python examples/mimic3_mortality_with_fairness_retain.py \\
        --root /srv/local/data/physionet.org/files/mimiciii/1.4 \\
        --epochs 5

Notes:
    - Uses an existing PyHealth dataset (MIMIC-III) and an existing PyHealth
      model (RETAIN); this script's novelty is the task + audit pipeline.
    - On real MIMIC-III (~40k patients) the ablation surfaces genuine
      cohort-level disparities that shift with hyperparameters. On the
      demo dataset (100 patients) sample sizes are too small for reliable
      Bonferroni-corrected significance.

Contributor: Rahul Joshi (rahulpj2@illinois.edu)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RETAIN
from pyhealth.tasks import MortalityPredictionWithFairnessMIMIC3
from pyhealth.tasks.fairness_utils import audit_predictions
from pyhealth.trainer import Trainer


CONFIGS: List[Dict] = [
    {"name": "RETAIN_default",     "hidden_dim": 128, "dropout": 0.5},
    {"name": "RETAIN_larger",      "hidden_dim": 256, "dropout": 0.5},
    {"name": "RETAIN_low_dropout", "hidden_dim": 128, "dropout": 0.1},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Path to MIMIC-III CSVs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out-dir", default="./outputs/mimic3_fairness_ablation")
    p.add_argument("--n-bootstrap", type=int, default=100)
    return p.parse_args()


def train_and_predict(samples, train, test, cfg, epochs):
    model = RETAIN(
        dataset=samples,
        feature_keys=["conditions", "procedures", "drugs"],
        label_key="mortality",
        mode="binary",
        embedding_dim=cfg["hidden_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=16, shuffle=True),
        epochs=epochs,
    )
    inf = trainer.inference(get_dataloader(test, batch_size=16, shuffle=False))
    y_true = np.array(inf[0]).astype(int).ravel()
    y_prob = np.array(inf[1]).astype(float).ravel()
    return y_true, y_prob


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MIMIC-III via PyHealth...")
    ds = MIMIC3Dataset(
        root=args.root,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    )

    print("Applying MortalityPredictionWithFairnessMIMIC3 task...")
    task = MortalityPredictionWithFairnessMIMIC3()
    samples = ds.set_task(task)
    print(f"  {len(samples)} samples after task")

    train, _val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=42)
    test_list = [test[i] for i in range(len(test))]

    summary_rows: List[Dict] = []
    for cfg in CONFIGS:
        print(f"\n=== config: {cfg['name']} ===")
        y_true, y_prob = train_and_predict(samples, train, test, cfg, args.epochs)

        audit = audit_predictions(
            test_list, list(y_prob), list(y_true),
            n_bootstrap=args.n_bootstrap,
        )
        audit.to_csv(out_dir / f"audit_{cfg['name']}.csv", index=False)

        sig = audit[audit["significantly_worse"]]
        print(f"  significantly-worse cohorts (Bonferroni): {len(sig)}")
        for _, r in sig.sort_values("delta", ascending=False).head(5).iterrows():
            print(
                f"    {r['category']:<18s} {r['metric']:<10s} "
                f"cohort={r['median_cohort']:.3f} rest={r['median_rest']:.3f} "
                f"Δ={r['delta']:.3f} p={r['pvalue']:.1e}"
            )

        for _, r in audit.sort_values("delta", ascending=False).head(3).iterrows():
            summary_rows.append({"config": cfg["name"], **r.to_dict()})

    pd.DataFrame(summary_rows).to_csv(out_dir / "ablation_summary.csv", index=False)
    print(f"\nWrote per-config audits and ablation_summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
