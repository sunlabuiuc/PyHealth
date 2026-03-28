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

**Scaling:** :class:`~pyhealth.datasets.MIMIC4FHIRDataset` streams NDJSON to
hash-sharded Parquet (bounded RAM during ingest). This example trains via
``dataset.set_task(MPFClinicalPredictionTask)`` → LitData-backed
:class:`~pyhealth.datasets.sample_dataset.SampleDataset` →
:class:`~pyhealth.trainer.Trainer` (PyHealth’s standard path), instead of
materializing all samples with ``gather_samples()``. Prefer ``--max-patients`` to
 bound ingest when possible. Very large cohorts still need RAM/disk for task
caches and MPF vocabulary warmup.

**Offline Parquet (NDJSON → Parquet already done):** pass
``--prebuilt-global-event-dir`` pointing at a directory of ``shard-*.parquet``
(from ingest / ``stream_fhir_ndjson_root_to_sharded_parquet``). The example seeds
``global_event_df.parquet/`` under the usual PyHealth cache UUID so
``BaseDataset.global_event_df`` skips re-ingest — the downstream path is still
``global_event_df`` → :class:`~pyhealth.data.Patient` →
:class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask` →
:class:`~pyhealth.trainer.Trainer``. Use ``--fhir-root`` / ``--glob-pattern`` /
``--ingest-num-shards`` / ``--max-patients -1`` matching the ingest fingerprint.
``--train-patient-cap`` restricts task transforms via ``task.pre_filter`` using a
label-aware deterministic patient subset. The full ``unique_patient_ids`` scan and MPF vocab warmup
in the dataset still walk the cached cohort.

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

    # Prebuilt Parquet shards (skip NDJSON re-ingest); cap patients for a smoke train
    pixi run -e base python examples/mimic4fhir_mpf_ehrmamba.py \\
      --prebuilt-global-event-dir /path/to/shard_parquet_dir \\
      --fhir-root /same/as/ndjson/ingest/root \\
      --glob-pattern 'Mimic*.ndjson.gz' --ingest-num-shards 16 --max-patients -1 \\
      --train-patient-cap 2048 --epochs 2 \\
      --ntfy-url 'https://ntfy.sh/your-topic'
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "Use a narrow pattern to limit ingest time and cache size."
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
    help=(
        "Fingerprint for cache dir: cap patients during ingest (-1 = full cohort, "
        "match an uncapped NDJSON→Parquet export)."
    ),
)
_parser.add_argument(
    "--prebuilt-global-event-dir",
    type=str,
    default=None,
    help=(
        "Directory with shard-*.parquet from NDJSON ingest. Seeds "
        "cache/global_event_df.parquet/ so training skips re-ingest (downstream "
        "unchanged: Patient + MPF + Trainer)."
    ),
)
_parser.add_argument(
    "--ingest-num-shards",
    type=int,
    default=None,
    help="Fingerprint only: must match NDJSON→Parquet ingest (default: dataset YAML / heuristic).",
)
_parser.add_argument(
    "--train-patient-cap",
    type=int,
    default=None,
    help=(
        "After cache is ready, only build samples from a deterministic label-aware "
        "patient subset of size N (reduces train time; unique-id scan of "
        "global_event_df still runs once)."
    ),
)
_parser.add_argument(
    "--ntfy-url",
    type=str,
    default=None,
    help="POST notification when main() finishes (e.g. https://ntfy.sh/topic).",
)
_parser.add_argument(
    "--loss-plot-path",
    type=str,
    default=None,
    help="Write loss curve PNG here (default: alongside Trainer log under output/).",
)
_parser.add_argument(
    "--cache-dir",
    type=str,
    default=None,
    help="PyHealth dataset cache parent (UUID subdir added by MIMIC4FHIRDataset).",
)
_parser.add_argument(
    "--task-num-workers",
    type=int,
    default=None,
    help=(
        "Workers for LitData task/processor transforms (default: dataset "
        "``num_workers``, usually 1)."
    ),
)
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import torch

import polars as pl

from pyhealth.datasets import MIMIC4FHIRDataset, get_dataloader
from pyhealth.datasets.mimic4_fhir import fhir_patient_from_patient, infer_mortality_label
from pyhealth.models import EHRMambaCEHR
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask
from pyhealth.trainer import Trainer


class PatientCappedMPFTask(MPFClinicalPredictionTask):
    """Example-only: limit task transform to an explicit patient_id allow-list."""

    def __init__(
        self,
        *,
        max_len: int,
        use_mpf: bool,
        patient_ids_allow: List[str],
    ) -> None:
        super().__init__(max_len=max_len, use_mpf=use_mpf)
        self.patient_ids_allow = patient_ids_allow

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col("patient_id").is_in(self.patient_ids_allow))


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
SPLIT_RATIOS = (0.7, 0.1, 0.2)


def _max_patients_arg(v: int) -> Optional[int]:
    return None if v is not None and v < 0 else v


def _seed_global_event_cache_from_shards(prebuilt_dir: Path, ds: MIMIC4FHIRDataset) -> None:
    """Link shard-*.parquet into the dataset cache as part-*.parquet (PyHealth layout)."""

    shards = sorted(prebuilt_dir.glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No shard-*.parquet under {prebuilt_dir} — use ingest output directory."
        )
    ge = ds.cache_dir / "global_event_df.parquet"
    if ge.exists() and any(ge.glob("*.parquet")):
        return
    ge.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(shards):
        dest = ge / f"part-{i:05d}.parquet"
        if dest.exists():
            continue
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)


def _parse_train_losses_from_log(log_path: Path) -> List[float]:
    """Mean training loss per epoch from Trainer file log."""

    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    losses: List[float] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "--- Train epoch-" in line and i + 1 < len(lines):
            m = re.search(r"loss:\s*([0-9.eE+-]+)", lines[i + 1])
            if m:
                losses.append(float(m.group(1)))
    return losses


def _write_loss_plot(losses: List[float], out_path: Path) -> None:
    if not losses:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        csv_path = out_path.with_suffix(".csv")
        csv_path.write_text(
            "epoch,train_loss_mean\n"
            + "\n".join(f"{i},{v}" for i, v in enumerate(losses)),
            encoding="utf-8",
        )
        print(
            "matplotlib not installed; wrote", csv_path, "(pip install matplotlib for PNG)"
        )
        return
    plt.figure(figsize=(6, 3.5))
    plt.plot(range(len(losses)), losses, marker="o", linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("mean train loss")
    plt.title("EHRMambaCEHR training loss (MPF)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("loss plot:", out_path)


def _ntfy(url: str, title: str, message: str) -> None:
    try:
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            method="POST",
        )
        req.add_header("Title", title[:200])
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status >= 400:
                print("ntfy HTTP", resp.status, file=sys.stderr)
    except urllib.error.URLError as e:
        print("ntfy failed:", e, file=sys.stderr)


def _quick_test_ndjson_dir() -> str:
    """Write two-patient synthetic NDJSON; returns temp directory (caller cleans up)."""

    from pyhealth.datasets.mimic4_fhir import synthetic_mpf_two_patient_ndjson_text

    tmp = tempfile.mkdtemp(prefix="pyhealth_mimic4_fhir_quick_")
    Path(tmp, "fixture.ndjson").write_text(
        synthetic_mpf_two_patient_ndjson_text(),
        encoding="utf-8",
    )
    return tmp


def _patient_label(ds: MIMIC4FHIRDataset, patient_id: str) -> int:
    patient = ds.get_patient(patient_id)
    return int(infer_mortality_label(fhir_patient_from_patient(patient)))


def _ensure_binary_label_coverage(ds: MIMIC4FHIRDataset) -> None:
    found: Dict[int, str] = {}
    scanned = 0
    for patient_id in ds.unique_patient_ids:
        label = _patient_label(ds, patient_id)
        scanned += 1
        found.setdefault(label, patient_id)
        if len(found) == 2:
            print(
                "label preflight:",
                {"scanned_patients": scanned, "example_patient_ids": found},
            )
            return
    raise SystemExit(
        "Binary mortality example found only one label in the available cohort; "
        "cannot build a valid binary training set from this cache."
    )


def _select_patient_ids_for_cap(
    ds: MIMIC4FHIRDataset, requested_cap: int
) -> List[str]:
    patient_ids = ds.unique_patient_ids
    if not patient_ids:
        return []

    desired = max(2, requested_cap)
    desired = min(desired, len(patient_ids))
    if desired < requested_cap:
        print(
            f"train_patient_cap requested {requested_cap}, but only {desired} patients are available."
        )
    elif requested_cap < 2:
        print(
            f"train_patient_cap={requested_cap} is too small for binary labels; using {desired}."
        )

    encountered: List[str] = []
    label_by_patient_id: Dict[str, int] = {}
    first_by_label: Dict[int, str] = {}
    for patient_id in patient_ids:
        label = _patient_label(ds, patient_id)
        encountered.append(patient_id)
        label_by_patient_id[patient_id] = label
        first_by_label.setdefault(label, patient_id)
        if len(encountered) >= desired and len(first_by_label) == 2:
            break

    if len(first_by_label) < 2:
        raise SystemExit(
            "Unable to satisfy --train-patient-cap with both binary labels from the "
            "available cohort. Use a different cache/export or remove the cap."
        )

    selected = encountered[:desired]
    selected_labels = {label_by_patient_id[pid] for pid in selected}
    if len(selected_labels) == 1:
        missing_label = 1 - next(iter(selected_labels))
        replacement = first_by_label[missing_label]
        for idx in range(len(selected) - 1, -1, -1):
            if label_by_patient_id[selected[idx]] != missing_label:
                selected[idx] = replacement
                break

    counts = {
        0: sum(1 for pid in selected if label_by_patient_id[pid] == 0),
        1: sum(1 for pid in selected if label_by_patient_id[pid] == 1),
    }
    print(
        "train_patient_cap selection:",
        {
            "requested": requested_cap,
            "selected": len(selected),
            "scanned_patients": len(encountered),
            "label_counts": counts,
        },
    )
    return selected


def _sample_label(sample: Dict[str, Any]) -> int:
    label = sample["label"]
    if isinstance(label, torch.Tensor):
        return int(label.reshape(-1)[0].item())
    return int(label)


def _split_counts(n: int) -> List[int]:
    if n < 3:
        raise ValueError("Need at least 3 samples for three-way stratified split.")
    counts = [1, 1, 1]
    remaining = n - 3
    raw = [ratio * remaining for ratio in SPLIT_RATIOS]
    floors = [int(x) for x in raw]
    for i, floor in enumerate(floors):
        counts[i] += floor
    assigned = sum(counts)
    order = sorted(
        range(3),
        key=lambda i: raw[i] - floors[i],
        reverse=True,
    )
    for i in order:
        if assigned >= n:
            break
        counts[i] += 1
        assigned += 1
    counts[0] += n - assigned
    return counts


def _split_sample_dataset_for_binary_metrics(sample_ds: Any) -> tuple[Any, Any, Any]:
    if len(sample_ds) < 8:
        print("sample count < 8; reusing the full dataset for train/val/test.")
        return sample_ds, sample_ds, sample_ds

    label_to_indices: Dict[int, List[int]] = {0: [], 1: []}
    for idx in range(len(sample_ds)):
        label_to_indices[_sample_label(sample_ds[idx])].append(idx)

    label_counts = {label: len(indices) for label, indices in label_to_indices.items()}
    min_count = min(label_counts.values())
    if min_count < 3:
        print(
            "label distribution too small for disjoint binary train/val/test splits; "
            "reusing the full dataset for train/val/test.",
            label_counts,
        )
        return sample_ds, sample_ds, sample_ds

    rng = random.Random(SEED)
    split_indices: List[List[int]] = [[], [], []]
    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_train, n_val, n_test = _split_counts(len(shuffled))
        split_indices[0].extend(shuffled[:n_train])
        split_indices[1].extend(shuffled[n_train : n_train + n_val])
        split_indices[2].extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    for indices in split_indices:
        indices.sort()

    split_counts = []
    for indices in split_indices:
        split_counts.append(
            {
                0: sum(1 for idx in indices if _sample_label(sample_ds[idx]) == 0),
                1: sum(1 for idx in indices if _sample_label(sample_ds[idx]) == 1),
                "n": len(indices),
            }
        )
    print(
        "binary stratified split counts:",
        {"train": split_counts[0], "val": split_counts[1], "test": split_counts[2]},
    )
    return (
        sample_ds.subset(split_indices[0]),
        sample_ds.subset(split_indices[1]),
        sample_ds.subset(split_indices[2]),
    )


def _build_loaders_from_sample_dataset(
    sample_ds: Any,
    vocab_size: int,
) -> tuple[Any, Any, Any, Any, int]:
    train_ds, val_ds, test_ds = _split_sample_dataset_for_binary_metrics(sample_ds)
    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return sample_ds, train_loader, val_loader, test_loader, vocab_size


def run_single_train(
    *,
    fhir_root: str,
    max_len: int,
    use_mpf: bool,
    hidden_dim: int,
    epochs: int,
    lr: float = 1e-3,
    glob_pattern: str = "*.ndjson",
    cache_dir: Optional[str] = None,
    dataset_max_patients: Optional[int] = 500,
    ingest_num_shards: Optional[int] = None,
    prebuilt_global_event_dir: Optional[str] = None,
    train_patient_cap: Optional[int] = None,
) -> Dict[str, float]:
    """Train/eval one configuration; returns test metrics (floats)."""

    ds_kw: Dict[str, Any] = {
        "root": fhir_root,
        "glob_pattern": glob_pattern,
        "cache_dir": cache_dir,
        "max_patients": dataset_max_patients,
    }
    if ingest_num_shards is not None:
        ds_kw["ingest_num_shards"] = ingest_num_shards
    ds = MIMIC4FHIRDataset(**ds_kw)
    if prebuilt_global_event_dir:
        _seed_global_event_cache_from_shards(
            Path(prebuilt_global_event_dir).expanduser().resolve(), ds
        )
    if train_patient_cap is not None:
        allow = _select_patient_ids_for_cap(ds, train_patient_cap)
        task: MPFClinicalPredictionTask = PatientCappedMPFTask(
            max_len=max_len,
            use_mpf=use_mpf,
            patient_ids_allow=allow,
        )
    else:
        _ensure_binary_label_coverage(ds)
        task = MPFClinicalPredictionTask(max_len=max_len, use_mpf=use_mpf)
    sample_ds = ds.set_task(task, num_workers=1)
    vocab_size = ds.vocab.vocab_size
    sample_ds, train_l, val_l, test_l, vocab_size = _build_loaders_from_sample_dataset(
        sample_ds, vocab_size
    )
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
    tmp = _quick_test_ndjson_dir()
    try:
        print(
            "Ablation (synthetic, 1 epoch each): max_len, use_mpf, hidden_dim, lr="
            f"{lr} -> test roc_auc, pr_auc"
        )
        rows = []
        t0 = time.perf_counter()
        for max_len, use_mpf, hidden_dim in grid:
            metrics = run_single_train(
                fhir_root=tmp,
                max_len=max_len,
                use_mpf=use_mpf,
                hidden_dim=hidden_dim,
                epochs=1,
                lr=lr,
                cache_dir=tmp,
                dataset_max_patients=500,
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
    except Exception:
        print(f"ablation: leaving scratch directory for debugging: {tmp}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    args = _parser.parse_args()
    status = "abort"
    ntfy_detail = ""
    try:
        _main_train(args)
        status = "ok"
        ntfy_detail = "Training finished successfully."
    except SystemExit as e:
        status = "exit"
        ntfy_detail = f"SystemExit {e.code!r}"
        raise
    except Exception as e:
        status = "error"
        ntfy_detail = f"{type(e).__name__}: {e}"[:3800]
        raise
    finally:
        if args.ntfy_url and status in ("ok", "error"):
            _ntfy(
                args.ntfy_url,
                "mimic-fhir-train OK" if status == "ok" else "mimic-fhir-train FAIL",
                ntfy_detail,
            )


def _main_train(args: argparse.Namespace) -> None:
    fhir_root = args.fhir_root or os.environ.get("MIMIC4_FHIR_ROOT")
    quick = args.quick_test
    quick_test_tmp: Optional[str] = None
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

    sample_ds: Any
    vocab: Any

    if quick:
        quick_test_tmp = _quick_test_ndjson_dir()
        ds = MIMIC4FHIRDataset(
            root=quick_test_tmp,
            glob_pattern="*.ndjson",
            cache_dir=quick_test_tmp,
            max_patients=500,
        )
        try:
            print(
                "pipeline: synthetic NDJSON → ingest Parquet → set_task → "
                "SampleDataset → Trainer"
            )
            task = MPFClinicalPredictionTask(
                max_len=args.max_len,
                use_mpf=not args.no_mpf,
            )
            print("set_task (quick-test, num_workers=1)...")
            t_task0 = time.perf_counter()
            sample_ds = ds.set_task(task, num_workers=1)
            print(
                "set_task done: n_samples=",
                len(sample_ds),
                "wall_s=",
                round(time.perf_counter() - t_task0, 2),
            )
            vocab = ds.vocab
        except Exception:
            print(
                f"quick-test: leaving NDJSON/Parquet scratch at {quick_test_tmp}",
                file=sys.stderr,
            )
            raise
    else:
        mp = _max_patients_arg(args.max_patients)
        if not fhir_root or not os.path.isdir(fhir_root):
            raise SystemExit(
                "Set MIMIC4_FHIR_ROOT or pass --fhir-root to an existing directory "
                "(NDJSON tree for ingest fingerprint, even when using --prebuilt-global-event-dir)."
            )
        ds_kw: Dict[str, Any] = {
            "root": fhir_root,
            "max_patients": mp,
            "cache_dir": args.cache_dir,
        }
        if args.glob_pattern is not None:
            ds_kw["glob_pattern"] = args.glob_pattern
        if args.ingest_num_shards is not None:
            ds_kw["ingest_num_shards"] = args.ingest_num_shards
        ds = MIMIC4FHIRDataset(**ds_kw)
        if args.prebuilt_global_event_dir:
            pb = Path(args.prebuilt_global_event_dir).expanduser().resolve()
            if not pb.is_dir():
                raise SystemExit(f"--prebuilt-global-event-dir not a directory: {pb}")
            print(
                "pipeline: offline NDJSON→Parquet shards → seed global_event_df cache → "
                "set_task → SampleDataset → Trainer (no NDJSON re-ingest)"
            )
            _seed_global_event_cache_from_shards(pb, ds)
        else:
            print(
                "pipeline: NDJSON root → MIMIC4FHIRDataset ingest → Parquet cache → "
                "set_task → SampleDataset → Trainer"
            )
        print("glob_pattern:", ds.glob_pattern, "| max_patients fingerprint:", mp)
        if args.train_patient_cap is not None:
            print("train_patient_cap:", args.train_patient_cap)
            allow = _select_patient_ids_for_cap(ds, args.train_patient_cap)
            mpf_task: MPFClinicalPredictionTask = PatientCappedMPFTask(
                max_len=args.max_len,
                use_mpf=not args.no_mpf,
                patient_ids_allow=allow,
            )
            print("task patient allow-list size:", len(allow))
        else:
            _ensure_binary_label_coverage(ds)
            mpf_task = MPFClinicalPredictionTask(
                max_len=args.max_len,
                use_mpf=not args.no_mpf,
            )
        nw = args.task_num_workers
        if nw is None:
            nw = ds.num_workers
        print(f"set_task (LitData task cache, num_workers={nw})...")
        t_task0 = time.perf_counter()
        sample_ds = ds.set_task(mpf_task, num_workers=nw)
        print(
            "set_task done: n_samples=",
            len(sample_ds),
            "wall_s=",
            round(time.perf_counter() - t_task0, 2),
        )
        vocab = ds.vocab
        print("fhir_root:", fhir_root)

    try:
        if len(sample_ds) == 0:
            raise SystemExit(
                "No training samples (0 patients or empty sequences). "
                "PhysioNet MIMIC-IV FHIR uses *.ndjson.gz (default glob **/*.ndjson.gz). "
                "If your tree is plain *.ndjson, construct MIMIC4FHIRDataset with "
                "glob_pattern='**/*.ndjson'."
            )

        sample_ds, train_loader, val_loader, test_loader, vocab_size = (
            _build_loaders_from_sample_dataset(sample_ds, vocab.vocab_size)
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

        log_txt = (
            Path(trainer.exp_path) / "log.txt" if trainer.exp_path else None
        )
        if log_txt and log_txt.is_file():
            losses = _parse_train_losses_from_log(log_txt)
            print("train_loss_per_epoch:", losses)
            plot_path = (
                Path(args.loss_plot_path)
                if args.loss_plot_path
                else Path(trainer.exp_path) / "train_loss.png"
            )
            if trainer.exp_path:
                _write_loss_plot(losses, plot_path)
    finally:
        if quick_test_tmp is not None:
            shutil.rmtree(quick_test_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
