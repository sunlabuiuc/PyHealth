"""EHRMamba on full MIMIC-IV: length-of-stay benchmark.

Run on full MIMIC-IV (dev=False) with caching. Task is 10-class LOS classification.
Requires MIMIC-IV 2.2 from PhysioNet. Set EHR_ROOT and optionally CACHE_BASE.

Usage:
    python examples/length_of_stay/length_of_stay_mimic4_ehrmamba.py
    python examples/length_of_stay/length_of_stay_mimic4_ehrmamba.py --quick-test  # ~5 min smoke test
    EHR_ROOT=/path/to/mimiciv/2.2 python examples/length_of_stay/length_of_stay_mimic4_ehrmamba.py
"""

import argparse
import os
import time

# Parse --gpu before importing torch so CUDA_VISIBLE_DEVICES takes effect
_parser = argparse.ArgumentParser()
_parser.add_argument("--gpu", type=int, default=None, help="GPU index (e.g. 1). Sets CUDA_VISIBLE_DEVICES.")
_parser.add_argument("--quick-test", action="store_true", help="Dev mode, 2 epochs, ~5 min.")
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import torch

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_sample
from pyhealth.models import EHRMamba
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer

# Config: set via env or override here
EHR_ROOT = os.environ.get("EHR_ROOT", "/srv/local/data/physionet.org/files/mimiciv/2.2/")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_BASE = os.environ.get(
    "CACHE_BASE",
    os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "benchmark_cache")),
)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20

DATASET_CACHE = os.path.join(CACHE_BASE, "mimic4_ehr_los")
TASK_CACHE = os.path.join(CACHE_BASE, "mimic4_los_ehrmamba")


def main():
    args = _parser.parse_args()
    quick_test = args.quick_test
    gpu_id = args.gpu

    dev = quick_test
    epochs = 2 if quick_test else EPOCHS
    dataset_cache = os.path.join(
        CACHE_BASE, "mimic4_ehr_los_quick" if quick_test else "mimic4_ehr_los"
    )
    task_cache = os.path.join(
        CACHE_BASE, "mimic4_los_ehrmamba_quick" if quick_test else "mimic4_los_ehrmamba"
    )
    num_workers = 1 if quick_test else 4

    print("EHRMamba – Length of stay (full MIMIC-IV)")
    if quick_test:
        print("*** QUICK TEST MODE (dev=True, 2 epochs) ***")
    if gpu_id is not None:
        print("gpu:", gpu_id, "(CUDA_VISIBLE_DEVICES)")
    print("device:", DEVICE)
    print("ehr_root:", EHR_ROOT)
    print("cache: dataset", dataset_cache, "| task", task_cache)
    print("seed:", SEED, "| batch_size:", BATCH_SIZE, "| epochs:", epochs)

    t0 = time.perf_counter()
    dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=dev,
        cache_dir=dataset_cache,
    )
    print(f"Base dataset loaded in {time.perf_counter() - t0:.1f}s")

    task = LengthOfStayPredictionMIMIC4()
    t1 = time.perf_counter()
    sample_dataset = dataset.set_task(
        task,
        num_workers=num_workers,
        cache_dir=task_cache,
    )
    print(f"Task set in {time.perf_counter() - t1:.1f}s | samples: {len(sample_dataset)}")

    train_dataset, val_dataset, test_dataset = split_by_sample(
        sample_dataset, ratios=[0.7, 0.1, 0.2], seed=SEED
    )
    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EHRMamba(
        dataset=sample_dataset,
        embedding_dim=128,
        num_layers=2,
        state_size=16,
        conv_kernel=4,
        dropout=0.1,
    )
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
        device=DEVICE,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="accuracy",
    )

    results = trainer.evaluate(test_loader)
    total_time = time.perf_counter() - t0

    print("\nTest Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY: EHRMamba Length of Stay (MIMIC-IV)")
    print("=" * 60)
    print(f"Model: EHRMamba")
    print(f"Dataset: MIMIC-IV")
    print(f"Task: Length of Stay Prediction (10-class)")
    print(f"Mode: {'quick-test (dev)' if quick_test else 'full'}")
    print(f"Seed: {SEED}  |  Batch size: {BATCH_SIZE}  |  Epochs: {epochs}")
    print(f"Train/val/test samples: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print(f"Total samples: {len(sample_dataset)}")
    print(f"Test accuracy:      {results.get('accuracy', float('nan')):.4f}")
    print(f"Test F1 (weighted): {results.get('f1_weighted', float('nan')):.4f}")
    print(f"Total wall time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
