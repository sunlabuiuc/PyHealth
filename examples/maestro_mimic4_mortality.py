"""Train MAESTRO on in-hospital mortality from the MIMIC-IV demo dataset.

Runs on the open MIMIC-IV demo (100 patients), so it needs no PhysioNet
credentialing. Download the demo first and point ``--root`` at its folder,
which must contain ``hosp/`` and ``icu/``::

    https://physionet.org/content/mimic-iv-demo/2.2/

Usage::

    python examples/maestro_mimic4_mortality.py --root path/to/mimic-iv-demo/2.2

The demo is far too small for meaningful metrics; the script shows the full
pipeline (task, split, training, evaluation) on real MIMIC-IV data.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MAESTRO
from pyhealth.tasks import MAESTROMortalityMIMIC4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="MIMIC-IV demo folder (2.2)")
    parser.add_argument("--cache-dir", default=None, help="Folder for processed data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MIMIC4EHRDataset(
        root=args.root,
        tables=["chartevents", "labevents"],
        cache_dir=args.cache_dir,
        num_workers=1,
    )
    task = MAESTROMortalityMIMIC4()
    train, val, test = dataset.set_task(
        task, split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=args.seed)
    )
    print(f"stays train/val/test: {len(train)}/{len(val)}/{len(test)}")

    torch.manual_seed(args.seed)
    model = MAESTRO(
        dataset=train,
        d_model=32,
        n_heads=4,
        num_layers_per_modal=1,
        num_layers=1,
        num_experts=2,
        dropout=0.1,
        modality_dropout=0.2,
        input_length=24,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: last batch loss {loss.item():.4f}")

    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in get_dataloader(test, batch_size=args.batch_size):
            output = model(**batch)
            y_true.append(output["y_true"].cpu().numpy())
            y_prob.append(output["y_prob"].cpu().numpy())
    y_true = np.concatenate(y_true).ravel()
    y_prob = np.concatenate(y_prob).ravel()
    print(f"test stays: {len(y_true)}, deaths: {int(y_true.sum())}")
    if len(np.unique(y_true)) > 1:
        scores = binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"])
        print(f"test metrics (demo-sized, not meaningful): {scores}")
    else:
        print("test set has one class; no AUC to report")


if __name__ == "__main__":
    main()
