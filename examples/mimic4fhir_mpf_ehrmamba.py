"""EHRMambaCEHR on MIMIC-IV FHIR NDJSON: MPF mortality task with ablations.

Requires credentialled MIMIC-IV on FHIR from PhysioNet (not bundled here).

Usage:
    cd PyHealth && PYTHONPATH=. python examples/mimic4fhir_mpf_ehrmamba.py --quick-test
    export MIMIC4_FHIR_ROOT=/path/to/fhir
    PYTHONPATH=. python examples/mimic4fhir_mpf_ehrmamba.py --fhir-root "$MIMIC4_FHIR_ROOT"
    python examples/mimic4fhir_mpf_ehrmamba.py --fhir-root /path/to/fhir --max-len 1024 --no-mpf
"""

from __future__ import annotations

import argparse
import os
import time

_parser = argparse.ArgumentParser(description="EHRMambaCEHR on MIMIC-IV FHIR (MPF)")
_parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="GPU index; sets CUDA_VISIBLE_DEVICES.",
)
_parser.add_argument(
    "--fhir-root",
    type=str,
    default=None,
    help="Root directory with NDJSON (default: MIMIC4_FHIR_ROOT env).",
)
_parser.add_argument(
    "--max-len",
    type=int,
    default=512,
    help="Sequence length ablation (512 / 1024 / 2048 per proposal).",
)
_parser.add_argument(
    "--no-mpf",
    action="store_true",
    help="Ablation: use generic CLS/REG specials instead of task MPF tokens.",
)
_parser.add_argument(
    "--hidden-dim",
    type=int,
    default=128,
    help="Embedding / hidden size (proposal ablation).",
)
_parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Adam learning rate (passed to Trainer via monitor only if extended).",
)
_parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Use synthetic in-memory FHIR lines only (no disk root).",
)
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import torch

from pyhealth.datasets import (
    MIMIC4FHIRDataset,
    build_fhir_sample_dataset_from_lines,
    create_sample_dataset,
    get_dataloader,
    split_by_sample,
    synthetic_ndjson_lines_two_class,
)
from pyhealth.models import EHRMambaCEHR
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask
from pyhealth.trainer import Trainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_BASE = os.environ.get(
    "CACHE_BASE",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "benchmark_cache")),
)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20


def main() -> None:
    args = _parser.parse_args()
    fhir_root = args.fhir_root or os.environ.get("MIMIC4_FHIR_ROOT")
    quick = args.quick_test
    epochs = 2 if quick else EPOCHS

    print("EHRMambaCEHR – MIMIC-IV FHIR (MPF clinical prediction)")
    print("device:", DEVICE)
    print("max_len:", args.max_len, "| use_mpf:", not args.no_mpf)
    print("hidden_dim:", args.hidden_dim, "| lr:", args.lr)

    task = MPFClinicalPredictionTask(
        max_len=args.max_len,
        use_mpf=not args.no_mpf,
    )

    if quick:
        lines = synthetic_ndjson_lines_two_class()
        _, vocab, samples = build_fhir_sample_dataset_from_lines(lines, task)
        print("quick-test: synthetic samples", len(samples))
    else:
        if not fhir_root or not os.path.isdir(fhir_root):
            raise SystemExit(
                "Set MIMIC4_FHIR_ROOT or pass --fhir-root to a directory of NDJSON files."
            )
        ds = MIMIC4FHIRDataset(root=fhir_root, max_patients=500)
        task.vocab = ds.vocab
        task._specials = None
        samples = ds.gather_samples(task)
        vocab = ds.vocab
        print("fhir_root:", fhir_root, "| samples:", len(samples))

    sample_ds = create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name="mimic4_fhir_mpf",
    )
    vocab_size = max(max(s["concept_ids"]) for s in samples) + 1

    if len(sample_ds) < 8:
        train_ds = val_ds = test_ds = sample_ds
    else:
        train_ds, val_ds, test_ds = split_by_sample(
            sample_ds, ratios=[0.7, 0.1, 0.2], seed=SEED
        )
    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = EHRMambaCEHR(
        dataset=sample_ds,
        vocab_size=vocab_size,
        embedding_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.1,
    )
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"], device=DEVICE)

    t0 = time.perf_counter()
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="roc_auc",
    )
    results = trainer.evaluate(test_loader)
    print("Test:", {k: float(v) for k, v in results.items()})
    print("wall_s:", round(time.perf_counter() - t0, 1))
    _ = vocab  # vocab may be saved: vocab.save(os.path.join(CACHE_BASE, "fhir_vocab.json"))


if __name__ == "__main__":
    main()
