"""PhysioNet 2012 in-hospital mortality: task-variation ablation.

Authors: Sadegh Nassiri, Kaushal Seelam (Georgia Tech, CSE 6250 project).

Shows what the options of
:class:`~pyhealth.tasks.PhysioNet2012InHospitalMortality` change, using PyHealth's
:class:`~pyhealth.models.LogisticRegression` (a linear model, so the differences
come from the task, not model capacity). The
task follows the input layout of TransEHR (Xu et al., 2023,
https://proceedings.mlr.press/v225/xu23a.html).

For every variant the script

1. loads the existing :class:`~pyhealth.datasets.PhysioNet2012Dataset` once,
2. fits the normalization on the training patients only
   (:func:`~pyhealth.tasks.physionet2012_normalization`),
3. builds samples with the variant's task options,
4. trains ``pyhealth.models.LogisticRegression`` with the ``Trainer`` (model
   selected on validation AUROC),
5. reports test AUROC / AUPRC.

Variants: all variables, vitals only, labs only, a 24 h window, outlier
clipping at 10 scale units, and ICU type 1 only (a cohort restriction).
All variants share one stratified patient split.

Usage::

    # no data needed: a small random cohort in a temp dir (about a minute)
    python examples/physionet2012_mortality_logreg.py --synthetic

    # real data (downloaded to --root on first use; open access, about 50 MB)
    python examples/physionet2012_mortality_logreg.py --root data/physionet2012 \
        --download

The synthetic mode only checks that the pipeline runs; its scores mean nothing.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split

from pyhealth.datasets import PhysioNet2012Dataset, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.tasks import (
    PhysioNet2012InHospitalMortality,
    physionet2012_normalization,
)
from pyhealth.tasks.physionet2012_inhospital_mortality import (
    LAB_VARIABLES,
    MORTALITY_VARIABLES,
)
from pyhealth.trainer import Trainer

VARIANTS = {
    "all": {},
    "vitals_only": {"feature_groups": ["vitals"]},
    "labs_only": {"feature_groups": ["labs"]},
    "24h": {"observation_hours": 24},
    "clip10": {"clip": 10.0},
    "icu_type_1": {"icu_types": [1]},
}


# ---------------------------------------------------------------- synthetic data
def write_synthetic_root(root: Path, n_stays: int = 160, seed: int = 0) -> None:
    """Write ``n_stays`` random records in the raw PhysioNet 2012 layout (set-a).

    Every variable is observed several times per stay, and the chance of dying
    grows with a hidden severity that also raises the measured values, so there
    is a weak signal to learn. Intended only for a smoke test.
    """
    rng = np.random.default_rng(seed)
    folder = root / "set-a"
    folder.mkdir(parents=True)
    outcomes = ["RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"]
    for i in range(n_stays):
        record_id = 100000 + i
        severity = rng.normal()
        died = int(rng.random() < 1 / (1 + np.exp(-(severity - 0.8))))
        lines = [
            "Time,Parameter,Value",
            f"00:00,RecordID,{record_id}",
            f"00:00,Age,{int(rng.integers(20, 90))}",
            f"00:00,Gender,{int(rng.integers(0, 2))}",
            f"00:00,Height,{float(rng.normal(170, 10)):.0f}",
            f"00:00,ICUType,{int(rng.integers(1, 5))}",
            "00:00,Weight,80",
        ]
        for variable in MORTALITY_VARIABLES:
            n_obs = int(rng.integers(3, 9)) if variable in LAB_VARIABLES else 20
            for minute in sorted(rng.integers(0, 48 * 60, n_obs)):
                value = 50 + 10 * severity + rng.normal(scale=5)
                lines.append(f"{minute // 60:02d}:{minute % 60:02d},{variable},"
                             f"{value:.2f}")
        (folder / f"{record_id}.txt").write_text("\n".join(lines) + "\n")
        outcomes.append(f"{record_id},10,3,5,-1,{died}")
    (root / "Outcomes-a.txt").write_text("\n".join(outcomes) + "\n")


# ------------------------------------------------------------------------ data
def split_patients(dataset, seed: int) -> dict[str, list[str]]:
    """Stratified 64/16/20 patient split."""
    df = (
        dataset.global_event_df.filter(pl.col("event_type") == "patients")
        .select("patient_id", "patients/in_hospital_death")
        .collect()
        .sort("patient_id")
    )
    ids = df["patient_id"].to_list()
    y = df["patients/in_hospital_death"].cast(pl.Float64).cast(pl.Int64).to_list()
    rest, test, y_rest, _ = train_test_split(
        ids, y, test_size=0.2, stratify=y, random_state=seed
    )
    train, val = train_test_split(
        rest, test_size=0.2, stratify=y_rest, random_state=seed
    )
    return {"train": train, "val": val, "test": test}


class GridMortality(PhysioNet2012InHospitalMortality):
    """The task without the integer-typed lab-event stream.

    PyHealth's linear model embeds every input as floats, so the long-typed
    ``event_types`` / ``event_times`` stream is left out of the schema. Everything
    else (options, cohort, label, normalization) is the task's own.
    """

    input_schema: ClassVar[dict] = {
        k: v
        for k, v in PhysioNet2012InHospitalMortality.input_schema.items()
        if k in ("measurements", "observed", "static")
    }


def subset(samples, patient_ids):
    """The samples of the given patients."""
    index = [i for p in patient_ids for i in samples.patient_to_index.get(p, [])]
    return samples.subset(index)


def run_variant(dataset, split, name, options, args) -> dict:
    """Train PyHealth's LogisticRegression on one task variant; score the test part."""
    torch.manual_seed(args.seed)
    center, scale = physionet2012_normalization(dataset, patient_ids=split["train"])
    task = GridMortality(center=center, scale=scale, **options)
    samples = dataset.set_task(task)
    parts = {k: subset(samples, ids) for k, ids in split.items()}
    loaders = {
        k: get_dataloader(v, batch_size=args.batch_size, shuffle=(k == "train"))
        for k, v in parts.items()
    }
    model = LogisticRegression(samples, embedding_dim=args.embedding_dim)
    trainer = Trainer(
        model, metrics=["roc_auc", "pr_auc"], enable_logging=False,
        output_path=args.output_dir,
    )
    trainer.train(
        train_dataloader=loaders["train"], val_dataloader=loaders["val"],
        epochs=args.epochs, monitor="roc_auc",
        optimizer_params={"lr": args.lr},
    )
    scores = trainer.evaluate(loaders["test"])
    return {"variant": name, "n_test": len(parts["test"]),
            "auroc": scores["roc_auc"], "auprc": scores["pr_auc"]}

def run(args: argparse.Namespace, root: str, cache_dir: str | None) -> list[dict]:
    dataset = PhysioNet2012Dataset(
        root=root, cache_dir=cache_dir, download=args.download and not args.synthetic
    )
    split = split_patients(dataset, args.seed)
    rows = []
    for name, options in VARIANTS.items():
        row = run_variant(dataset, split, name, options, args)
        print(f"{row['variant']:<12} n_test {row['n_test']:>5} "
              f"AUROC {row['auroc']:.4f} AUPRC {row['auprc']:.4f}", flush=True)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data/physionet2012")
    p.add_argument("--download", action="store_true", help="fetch the data if absent")
    p.add_argument("--synthetic", action="store_true",
                   help="use a small random cohort instead of the real data")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--output-dir", default="output/physionet2012_logreg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            write_synthetic_root(Path(tmp) / "raw")
            run(args, str(Path(tmp) / "raw"), str(Path(tmp) / "cache"))
    else:
        run(args, args.root, args.cache_dir)


if __name__ == "__main__":  # required: the dataset uses multiprocessing on Windows
    main()
