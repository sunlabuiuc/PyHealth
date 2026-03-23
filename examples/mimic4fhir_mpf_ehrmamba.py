"""EHRMambaCEHR on MIMIC-IV FHIR NDJSON with MPF clinical prediction (ablations).

Replication target: EHRMamba / CEHR-style modeling on tokenized FHIR timelines
(e.g. `arXiv:2405.14567 <https://arxiv.org/abs/2405.14567>`_). This script is
runnable end-to-end on **synthetic** NDJSON (``--quick-test``) or on
credentialled MIMIC-IV on FHIR from PhysioNet.

Experimental setup (for write-ups / PR):
    * **Data**: Synthetic two-patient NDJSON (``--quick-test``) or disk NDJSON
      under ``MIMIC4_FHIR_ROOT`` / ``--fhir-root``.
    * **Task ablations**: ``max_len`` (context window), ``use_mpf`` vs generic
      ``<cls>``/``<reg>`` boundaries (``--no-mpf``).
    * **Model ablations**: ``hidden_dim`` (embedding width); optional dropout
      fixed at 0.1 in this script.
    * **Train**: Adam via :class:`~pyhealth.trainer.Trainer`, monitor ROC-AUC,
      report test ROC-AUC / PR-AUC.

**Ablation mode** (``--ablation``): sweeps a small grid on synthetic data only,
trains 1 epoch per config, and prints a comparison table. Use this to document
how task/model knobs affect metrics on the minimal fixture before scaling to
real FHIR.

**Findings** (fill in after your runs; synthetic runs are noisy):
    On ``--quick-test`` data, longer ``max_len`` and MPF specials typically
    change logits enough to move AUC slightly; real MIMIC-IV FHIR runs are
    needed for conclusive comparisons. Paste your table from ``--ablation``
    into the PR description.

**Known limitation (full FHIR tree):** :class:`~pyhealth.datasets.MIMIC4FHIRDataset`
loads **every** resource from **every** file matching ``glob_pattern`` into
memory before grouping by patient. A complete PhysioNet export is **not** expected
to fit comfortably on a laptop without a **restricted** ``glob_pattern`` (subset
of ``*.ndjson.gz`` files) or future streaming ingest. See dataset API docs.

**Approximate minimum specs** (``--quick-test``, CPU, synthetic 2-patient
fixture; measured once on macOS/arm64 with ``/usr/bin/time -l``): peak RSS
~**600–700 MiB**, wall **~10–15 s** for two short epochs. Real NDJSON/GZ at scale
needs proportionally more RAM, disk, and time; GPU helps training, not the
current all-in-RAM parse.

Usage:
    cd PyHealth && PYTHONPATH=. python examples/mimic4fhir_mpf_ehrmamba.py --quick-test
    PYTHONPATH=. python examples/mimic4fhir_mpf_ehrmamba.py --quick-test --ablation
    export MIMIC4_FHIR_ROOT=/path/to/fhir
    pixi run -e base python examples/mimic4fhir_mpf_ehrmamba.py --fhir-root "$MIMIC4_FHIR_ROOT"
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

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
    "--glob-pattern",
    type=str,
    default=None,
    help=(
        "Override glob for NDJSON/NDJSON.GZ (default: yaml **/*.ndjson.gz). "
        "Use a narrow pattern (e.g. MimicPatient*.ndjson.gz) to limit RAM—the "
        "loader reads every matching file fully before grouping patients."
    ),
)
_parser.add_argument(
    "--max-len",
    type=int,
    default=512,
    help="Sequence length ablation (e.g. 512 / 1024 / 2048 per proposal).",
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
    help="Embedding / hidden size (model ablation).",
)
_parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Adam learning rate (trainer.train optimizer_params).",
)
_parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Use synthetic in-memory FHIR lines only (no disk root).",
)
_parser.add_argument(
    "--ablation",
    action="store_true",
    help="Run a small max_len × MPF × hidden_dim grid on synthetic data; print table.",
)
_parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Training epochs (default: 2 with --quick-test, else 20).",
)
_parser.add_argument(
    "--max-patients",
    type=int,
    default=500,
    help="Max grouped patients after full parse (disk FHIR only); lower to save RAM.",
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
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20


def _build_loaders(
    samples: List[Dict[str, Any]],
    task: MPFClinicalPredictionTask,
) -> tuple[Any, Any, Any, Any, int]:
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
    return sample_ds, train_loader, val_loader, test_loader, vocab_size


def run_single_train(
    *,
    lines: List[str],
    max_len: int,
    use_mpf: bool,
    hidden_dim: int,
    epochs: int,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Train/eval one configuration; returns test metrics (floats)."""

    task = MPFClinicalPredictionTask(max_len=max_len, use_mpf=use_mpf)
    _, _, samples = build_fhir_sample_dataset_from_lines(lines, task)
    sample_ds, train_l, val_l, test_l, vocab_size = _build_loaders(samples, task)
    model = EHRMambaCEHR(
        dataset=sample_ds,
        vocab_size=vocab_size,
        embedding_dim=hidden_dim,
        num_layers=2,
        dropout=0.1,
    )
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"], device=DEVICE)
    trainer.train(
        train_dataloader=train_l,
        val_dataloader=val_l,
        epochs=epochs,
        monitor="roc_auc",
        optimizer_params={"lr": lr},
    )
    results = trainer.evaluate(test_l)
    return {k: float(v) for k, v in results.items()}


def run_ablation_table(*, lr: float = 1e-3) -> None:
    """Task × model grid on synthetic NDJSON (short runs for comparison)."""

    # Ablations: context length, MPF vs CLS/REG, plus one hidden_dim pair.
    grid = [
        (32, True, 32),
        (32, False, 32),
        (96, True, 64),
        (96, False, 64),
    ]
    lines = synthetic_ndjson_lines_two_class()
    print(
        "Ablation (synthetic, 1 epoch each): max_len, use_mpf, hidden_dim, lr="
        f"{lr} -> test roc_auc, pr_auc"
    )
    rows = []
    t0 = time.perf_counter()
    for max_len, use_mpf, hidden_dim in grid:
        metrics = run_single_train(
            lines=lines,
            max_len=max_len,
            use_mpf=use_mpf,
            hidden_dim=hidden_dim,
            epochs=1,
            lr=lr,
        )
        rows.append((max_len, use_mpf, hidden_dim, metrics))
        print(
            f"  max_len={max_len} mpf={use_mpf} hid={hidden_dim} -> "
            f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
        )
    print("ablation_wall_s:", round(time.perf_counter() - t0, 2))
    best = max(rows, key=lambda r: r[3]["roc_auc"])
    print(
        "best_by_roc_auc:",
        {
            "max_len": best[0],
            "use_mpf": best[1],
            "hidden_dim": best[2],
            "metrics": best[3],
        },
    )


def main() -> None:
    args = _parser.parse_args()
    fhir_root = args.fhir_root or os.environ.get("MIMIC4_FHIR_ROOT")
    quick = args.quick_test
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 2 if quick else EPOCHS

    if args.ablation:
        if not quick:
            raise SystemExit("--ablation requires --quick-test (synthetic data only).")
        run_ablation_table(lr=args.lr)
        return

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
        ds = MIMIC4FHIRDataset(
            root=fhir_root,
            max_patients=args.max_patients,
            glob_pattern=args.glob_pattern,
        )
        print("glob_pattern:", ds.glob_pattern)
        task.vocab = ds.vocab
        task._specials = None
        samples = ds.gather_samples(task)
        vocab = ds.vocab
        print("fhir_root:", fhir_root, "| samples:", len(samples))

    if not samples:
        raise SystemExit(
            "No training samples (0 patients or empty sequences). "
            "PhysioNet MIMIC-IV FHIR uses *.ndjson.gz (default glob **/*.ndjson.gz). "
            "If your tree is plain *.ndjson, construct MIMIC4FHIRDataset with "
            "glob_pattern='**/*.ndjson'."
        )

    sample_ds, train_loader, val_loader, test_loader, vocab_size = _build_loaders(
        samples, task
    )

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
        optimizer_params={"lr": args.lr},
    )
    results = trainer.evaluate(test_loader)
    print("Test:", {k: float(v) for k, v in results.items()})
    print("wall_s:", round(time.perf_counter() - t0, 1))
    print("concept_vocab_size:", vocab.vocab_size)


if __name__ == "__main__":
    main()
