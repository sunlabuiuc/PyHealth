# Author: Anish Gupta
# NetID: anishg8
# Paper Title: N/A (original task contribution, not a paper reproduction)
# Paper Link: N/A
# Description: End-to-end example running SepsisPredictionMIMIC4 on real
#     MIMIC-IV data: load the dataset, build the task's sample dataset,
#     split by patient, and train/evaluate a small RNN.
"""End-to-end example: sepsis prediction on MIMIC-IV with PyHealth.

Requires access to MIMIC-IV (PhysioNet credentialing:
https://physionet.org/content/mimiciv/), including its ICU module
(``icu/chartevents.csv.gz``, ``icu/d_items.csv.gz``) for vitals.

Run:

    python examples/sepsis_prediction_mimic4_demo.py --root /path/to/mimic-iv/2.2

See ``pyhealth.tasks.SepsisPredictionMIMIC4`` for the label definition
(qSOFA-based Sepsis-3 approximation) and its documented limitations.
"""

import argparse

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import SepsisPredictionMIMIC4
from pyhealth.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Root of the MIMIC-IV dataset (directory containing hosp/ and icu/)",
    )
    args = parser.parse_args()

    dataset = MIMIC4EHRDataset(
        root=args.root,
        tables=["admissions", "prescriptions", "labevents", "chartevents"],
    )
    dataset.stats()

    sample_dataset = dataset.set_task(SepsisPredictionMIMIC4())
    n_positive = sum(int(s["sepsis"]) for s in sample_dataset)
    print(
        f"Sepsis samples: {len(sample_dataset)} total, {n_positive} positive "
        f"({n_positive / max(len(sample_dataset), 1):.1%})"
    )

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    model = RNN(dataset=sample_dataset, hidden_dim=64)
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=5,
        monitor="pr_auc",
    )
    # Sepsis is a rare-outcome task: report pr_auc/roc_auc, not just
    # accuracy, which would be misleadingly high for a model that mostly
    # predicts the majority class.
    metrics = trainer.evaluate(test_dataloader)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    # BaseDataset spawns Dask worker processes; keep the main-module guard.
    main()
