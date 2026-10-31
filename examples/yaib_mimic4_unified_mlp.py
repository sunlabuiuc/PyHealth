"""Train Unified_MLP or Unified_LogisticRegression on the five MIMIC-IV YAIB tasks.

Run from the repository root with PyHealth and its dependencies installed::

    .venv/Scripts/python.exe -B examples/yaib_mimic4_unified_mlp.py
    .venv/Scripts/python.exe -B examples/yaib_mimic4_unified_mlp.py --check
    .venv/Scripts/python.exe -B examples/yaib_mimic4_unified_mlp.py --model logistic_regression

The default root is D:/physionet.org/files/mimiciv/3.1, the local MIMIC-IV
installation. Override it with --root on other machines. This uses version
3.1; the original YAIB benchmark used version 2.2.
The root must contain the MIMIC-IV hosp/ and icu/ directories. Development
mode uses fewer patients; some task cohorts or splits may be too small.
This is a training example using PyHealth's YAIB task definitions, not a
reproduction of the original benchmark's tuning and cross-validation protocol.
"""

import argparse
import gc
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import cast

# Bound native thread pools before importing numerical libraries. Windows
# preprocessing workers inherit these limits instead of each using 32 threads.
for variable in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import torch

from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit, SampleDataset, get_dataloader
from pyhealth.models import Unified_LogisticRegression, Unified_MLP
from pyhealth.tasks import (
    YAIBAKIMIMIC4,
    YAIBKidneyFunctionMIMIC4,
    YAIBLengthOfStayMIMIC4,
    YAIBMortalityMIMIC4,
    YAIBSepsisMIMIC4,
)
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed


TASKS = {
    "mortality": YAIBMortalityMIMIC4,
    "kidney_function": YAIBKidneyFunctionMIMIC4,
    "length_of_stay": YAIBLengthOfStayMIMIC4,
    "aki": YAIBAKIMIMIC4,
    "sepsis": YAIBSepsisMIMIC4,
}
REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "hosp": ["patients", "admissions", "labevents", "transfers", "microbiologyevents"],
    "icu": ["icustays", "chartevents", "outputevents", "inputevents", "procedureevents"],
}


def preflight(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Check source files and exercise the chosen device before preprocessing."""
    args.root = str(Path(args.root).expanduser().resolve())
    missing = [
        str(Path(args.root) / folder / f"{table}.csv.gz")
        for folder, tables in TABLES.items()
        for table in tables
        if not (Path(args.root) / folder / f"{table}.csv.gz").is_file()
    ]
    if missing:
        parser.error("Missing MIMIC-IV files:\n" + "\n".join(missing))
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    # PyTorch uses the cuda device API for AMD ROCm too. Exercise the linear
    # forward/backward operations used by both models before preprocessing.
    try:
        device = torch.device(args.device)
        probe = torch.nn.Linear(args.embedding_dim, args.hidden_dim).to(device)
        values = torch.zeros(2, 25, args.embedding_dim, device=device)
        probe(values).square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            print(f"GPU: {torch.cuda.get_device_name(device)} (HIP {torch.version.hip})")
        del probe, values
    except (RuntimeError, ValueError, AssertionError) as error:
        parser.error(f"Device preflight failed for {args.device}: {error}")
    print(f"Data: {args.root}\nCache: {args.cache_dir}\nOutput: {args.output_dir}")
    print(
        f"Device: {args.device}; preprocessing workers: {args.num_workers}; "
        f"PyTorch CPU threads: {args.cpu_threads}; batch size: {args.batch_size}",
        flush=True,
    )


def run_task(
    base_dataset: MIMIC4EHRDataset, name: str, args: argparse.Namespace, output_dir: Path
) -> dict[str, object]:
    """Fit preprocessing on training patients, train, and evaluate once."""
    set_seed(args.seed)
    task = TASKS[name]()
    # Split before fitting imputation/normalization statistics. Cohorts differ
    # between tasks, so the same seed need not assign shared patients identically.
    train_data, val_data, test_data = cast(
        tuple[SampleDataset, SampleDataset, SampleDataset],
        base_dataset.set_task(
            task,
            split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=args.seed),
            num_workers=args.num_workers,
        ),
    )
    sizes = dict(train=len(train_data), validation=len(val_data), test=len(test_data))
    print(f"\n{name}: {sizes}", flush=True)
    if not all(sizes.values()):
        raise ValueError(
            f"{name} has an empty split: {sizes}. Use more patients (omit --dev)."
        )
    train_loader = get_dataloader(
        train_data, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = get_dataloader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=args.batch_size, shuffle=False)

    # All tasks share hourly numeric dynamics and static demographics. Both
    # models use cumulative means for packed hourly LOS, AKI, and sepsis labels.
    # MLP transforms records before pooling and flattened statics before fusion;
    # logistic regression pools learned embeddings and uses raw static vectors.
    model = (
        Unified_MLP(
            dataset=train_data,
            embedding_dim={"dynamic": [{"values": args.embedding_dim}]},
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.3,
            mask="concat",
        )
        if args.model == "mlp"
        else Unified_LogisticRegression(
            dataset=train_data,
            embedding_dim={"dynamic": [{"values": args.embedding_dim}]},
            mask="concat",
        )
    )
    metrics = (
        ["roc_auc", "pr_auc", "accuracy", "f1"]
        if model.mode in ("binary", "timeseries_binary")
        else ["mae", "mse"]
    )
    trainer = Trainer(
        model=model,
        metrics=metrics,
        device=args.device,
        output_path=str(output_dir),
        exp_name=name,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
        max_grad_norm=5.0,
    )
    # Time-series metrics pool valid hourly labels across stays. Kidney MAE
    # is in mg/dL; remaining-length-of-stay MAE is in hours.
    scores = {key: float(value) for key, value in trainer.evaluate(test_loader).items()}
    print(f"{name} test metrics: {scores}", flush=True)
    return {"mode": model.mode, "samples": sizes, "test_metrics": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="D:/physionet.org/files/mimiciv/3.1",
        help="MIMIC-IV directory (default: D:/physionet.org/files/mimiciv/3.1)",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model", choices=["mlp", "logistic_regression"], default="mlp")
    parser.add_argument("--num-layers", type=int, default=2, help="MLP layers per level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4, help="Dataset preprocessing workers")
    parser.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--device", default=None, help="E.g. cpu or cuda:0; defaults to automatic")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("D:/physionet.org/pyhealth-cache/mimiciv-3.1"),
        help="Persistent preprocessing cache, reused across runs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "output" / "yaib_unified_mlp",
    )
    parser.add_argument("--check", action="store_true", help="Check data files and linear device support, then exit")
    args = parser.parse_args()
    for field in ("epochs", "batch_size", "embedding_dim", "hidden_dim", "num_workers", "cpu_threads"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    preflight(args, parser)
    if args.check:
        print("Preflight passed; no dataset preprocessing or training started.")
        return

    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    set_seed(args.seed)
    # Union of tables required by the five tasks; core patient/admission/stay
    # tables are added automatically by MIMIC4EHRDataset.
    base_dataset = MIMIC4EHRDataset(
        root=args.root,
        tables=[
            "labevents", "chartevents", "outputevents", "transfers",
            "inputevents", "procedureevents", "microbiologyevents",
        ],
        dev=args.dev,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    results: dict[str, dict[str, object]] = {}
    for name in args.tasks:
        results[name] = run_task(base_dataset, name, args, output_dir)
        # Preserve completed results even if a later task fails.
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        # Release the previous task's objects and GPU allocator cache before
        # preparing the next cohort.
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nCheckpoints, logs, configuration, and test metrics: {output_dir}")


if __name__ == "__main__":
    main()
