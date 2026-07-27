"""End-to-end example: loading a MEDS dataset with PyHealth.

This example uses the public *MIMIC-IV demo data in the Medical Event Data
Standard (MEDS)* (PhysioNet, v0.0.1, ODbL v1.0, ~100 subjects):
https://doi.org/10.13026/t2y8-ea41

Download it once (open access, ~a few MB):

    wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/

Then run:

    python examples/meds_demo.py \\
        --root physionet.org/files/mimic-iv-demo-meds/0.0.1

Any dataset following the MEDS layout (``data/**.parquet`` +
``metadata/subject_splits.parquet``) works the same way. See the MEDS
specification: https://github.com/Medical-Event-Data-Standard/meds
"""

import argparse
import tempfile
from pathlib import Path
from typing import Any, List

import polars as pl
import pyhealth.datasets.configs as meds_configs

from pyhealth.datasets import MEDSDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import InHospitalMortalityMEDS
from pyhealth.trainer import Trainer

# Full-stay MEDS code sequences are often thousands of events long. Keeping
# every code makes a vanilla RNN prohibitively slow on CPU, so the training
# block below keeps only the *most recent* codes per stay. This is a demo
# ergonomics choice, not a benchmark configuration: production runs should
# use the unmodified task (``InHospitalMortalityMEDS()``) and an appropriate
# model/window for the sequence lengths involved.
_DEMO_MAX_SEQ_LEN = 256


class _DemoMortalityTask(InHospitalMortalityMEDS):
    """Demo-only wrapper that tail-truncates ``codes`` before ``set_task``."""

    def __init__(self, max_seq_len: int = _DEMO_MAX_SEQ_LEN, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len

    def __call__(self, patient: Any) -> List[dict]:
        samples = super().__call__(patient)
        for sample in samples:
            codes = sample["codes"]
            if len(codes) > self.max_seq_len:
                sample["codes"] = codes[-self.max_seq_len :]
        return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Root of the MEDS dataset (directory containing data/ and metadata/)",
    )
    parser.add_argument(
        "--subset",
        default="train",
        help="Split to load as a subset (default: train)",
    )
    args = parser.parse_args()

    # 1) Load the full dataset: every Parquet shard under data/ is read,
    #    including nested split directories (data/<split>/<shard>.parquet).
    dataset = MEDSDataset(root=args.root)
    dataset.stats()

    # 2) Peek at the canonical event frame (typed straight from Parquet:
    #    string patient ids, datetime64[ms] timestamps, float values).
    events = dataset.global_event_df
    print(events.head(5).collect())

    # 3) Load a split-restricted subset. Subjects are selected through the
    #    metadata/subject_splits.parquet assignment; each subset uses its
    #    own processing cache.
    subset = MEDSDataset(root=args.root, subset=args.subset)
    n_subset = len(subset.unique_patient_ids)
    n_total = len(dataset.unique_patient_ids)
    print(f"Subjects in subset '{args.subset}': {n_subset} / {n_total}")

    # 4) Static (null-time) MEDS events, e.g. demographics, are preserved.
    n_static = (
        events.filter(pl.col("timestamp").is_null()).select(pl.len()).collect().item()
    )
    print(f"Static (null-time) events: {n_static}")

    # 5) In-hospital mortality task + minimal RNN training loop (1 epoch).
    #    Sequences are tail-truncated via _DemoMortalityTask (see module note).
    cfg = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
    task_cache = tempfile.mkdtemp(prefix="meds_demo_task_")
    cohort = MEDSDataset(
        root=args.root,
        config_path=str(cfg),
        subset=args.subset,
        cache_dir=task_cache,
    )
    samples = cohort.set_task(_DemoMortalityTask())
    print(
        f"Mortality task samples ({args.subset}, codes tail-truncated to "
        f"{_DEMO_MAX_SEQ_LEN}): {len(samples)}"
    )

    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    model = RNN(dataset=samples, embedding_dim=64, hidden_dim=64)
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="roc_auc",
    )
    metrics = trainer.evaluate(test_dataloader)
    print(
        "Test metrics (smoke test only — 1 epoch, tail-truncated sequences; "
        "not interpretable as model quality):"
    )
    print(metrics)


if __name__ == "__main__":
    # BaseDataset spawns Dask worker processes; keep the main-module guard.
    main()
