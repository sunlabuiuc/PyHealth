"""Prepare or run TPC day-one experiments on authorized local ICU data.

Clinical status: DATA ACCESS PENDING; EXPERIMENTS NOT RUN; RESULTS PENDING.
No clinical results are bundled. See examples/tpc/README.md for the protocol,
pending results table, limitations, and commands. This script never downloads
data. Without --run, it only writes a plan and PENDING table; no dataset is
instantiated and no training is performed.

    python -m examples.eicu_los_tpc --prepare-only --all-configs
    python -m examples.eicu_los_tpc --run --dataset eicu \
        --root /authorized/eicu/2.0 --data-version 2.0 \
        --output-dir output/tpc_eicu_baseline

MIMIC-IV is also supported with --dataset mimic4. All clinical-data execution
remains unverified until authorized data are available. Synthetic unit tests
exercise the adapter and runner separately; they are not clinical experiments.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import pyhealth
from examples.synthetic_icu_los_tpc import BASE, METRICS, ablation_configs, los_metrics
from pyhealth.datasets import (
    MIMIC4EHRDataset,
    PatientSplit,
    SampleDataset,
    eICUDataset,
    get_dataloader,
)
from pyhealth.models import TPC
from pyhealth.models.tpc import msle_loss
from pyhealth.tasks import TPCDayOne

TABLES = {
    "eicu": [
        "patient",
        "lab",
        "vitalperiodic",
        "vitalaperiodic",
        "intakeoutput",
        "respiratorycharting",
        "nursecharting",
    ],
    "mimic4": [
        "patients",
        "admissions",
        "icustays",
        "labevents",
        "chartevents",
        "outputevents",
    ],
}


def clinical_configs() -> dict[str, dict[str, Any]]:
    """Reuses the synthetic one-factor sweep without dataset-size experiments."""
    return {
        name: override
        for name, override in ablation_configs().items()
        if "n_train" not in override
    }


def results_markdown(
    results: dict[str, list[dict[str, Any]]],
    configurations: list[str],
    seeds: list[int],
) -> str:
    """Renders measured complete rows or explicit PENDING placeholders."""
    lines = [
        "# TPC clinical results",
        "",
        "Only configurations with every planned seed completed are summarized.",
        "PENDING means no complete clinical result; it is not a measured zero.",
        "",
        "| Configuration | " + " | ".join(METRICS) + " | LongN |",
        "| --- | " + " | ".join(["---:"] * (len(METRICS) + 1)) + " |",
    ]
    for name in configurations:
        runs = results.get(name, [])
        if {run["seed"] for run in runs} != set(seeds):
            cells = ["PENDING"] * (len(METRICS) + 1)
        else:
            ordered = sorted(runs, key=lambda run: seeds.index(run["seed"]))
            cells = []
            for metric in METRICS:
                values = np.array([run["test"][metric] for run in ordered])
                cells.append(f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}")
            cells.append("/".join(str(int(run["test"]["LongN"])) for run in ordered))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Writes a new local plan; never reads source data or invents results."""
    available = clinical_configs()
    configurations = list(available) if args.all_configs else args.configs
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "status": "PENDING",
        "clinical_data_access": "USER_PROVIDED_LOCAL_DATA_REQUIRED",
        "clinical_experiments": "NOT_RUN",
        "clinical_results": "PENDING",
        "dataset": args.dataset,
        "data_version": args.data_version,
        "source_root": str(args.root) if args.root is not None else "PENDING",
        "versions": {
            "python": sys.version.split()[0],
            "pyhealth": pyhealth.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "split_ratios": [0.7, 0.15, 0.15],
        "prediction_hour": 24,
        "baseline": BASE,
        "baseline_lr": 0.0023,
        "configurations": {name: available[name] for name in configurations},
        "checkpoint_selection": "minimum validation MSLE",
        "dev": args.dev,
        "device": args.device,
        "results": {},
    }
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "results.md").write_text(
        results_markdown({}, configurations, args.seeds), encoding="utf-8"
    )
    return plan


def preflight(root: Path, dataset: str) -> None:
    """Checks local source CSVs and their configured join files without loading."""
    config_dir = Path(__file__).resolve().parents[1] / "pyhealth/datasets/configs"
    config_file = config_dir / ("eicu.yaml" if dataset == "eicu" else "mimic4_ehr.yaml")
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))["tables"]
    required = set()
    for table in TABLES[dataset]:
        required.add(config[table]["file_path"])
        required.update(join["file_path"] for join in config[table].get("join", []))
    missing = []
    for relative in sorted(required):
        source = root / relative
        twin = (
            source.with_suffix("") if source.suffix == ".gz" else Path(f"{source}.gz")
        )
        if not source.is_file() and not twin.is_file():
            missing.append(relative)
    if missing:
        raise FileNotFoundError(
            "Clinical data not available; no download or training attempted. "
            f"Missing under {root}: {', '.join(missing)}"
        )


def _predict(
    model: TPC, dataset: SampleDataset, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for batch in get_dataloader(dataset, batch_size=batch_size, shuffle=False):
            out = model(**batch)
            truth.append(out["y_true"].flatten().cpu())
            predictions.append(out["y_prob"].flatten().cpu())
    return torch.cat(truth).numpy(), torch.cat(predictions).numpy()


def train_configuration(
    datasets: tuple[SampleDataset, SampleDataset, SampleDataset],
    task: TPCDayOne,
    override: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    checkpoint: Path,
) -> dict[str, Any]:
    """Fits one configuration, selecting on validation only, then tests once."""
    train, validation, test = datasets
    params = {**BASE, **override}
    lr = params.pop("lr", 0.0023)
    decay_base = params.pop("decay_base", 0.75)
    fill_labs = params.pop("forward_fill_labs", True)
    torch.manual_seed(seed)
    model = TPC(
        train,
        decay_key=None,
        observed_key="observed",
        static_key="static",
        decay_base=decay_base,
        forward_fill_labs=fill_labs,
        lab_indices=task.lab_indices,
        **params,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True, seed=seed)
    best_msle = math.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
        y_true, y_pred = _predict(model, validation, args.batch_size)
        score = msle_loss(torch.from_numpy(y_pred), torch.from_numpy(y_true)).item()
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation MSLE.")
        if score < best_msle:
            best_msle, best_epoch = score, epoch + 1
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device, weights_only=True)
    )
    y_true, y_pred = _predict(model, test, args.batch_size)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_msle": best_msle,
        "samples": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "test": los_metrics(y_true, y_pred),
    }


def run_experiments(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    """Runs only after explicit --run against already-present local data."""
    torch.set_num_threads(args.cpu_threads)
    if args.dataset == "eicu":
        base = eICUDataset(
            root=str(args.root),
            tables=TABLES["eicu"][1:],
            cache_dir=args.cache_dir,
            dev=args.dev,
            num_workers=args.num_workers,
        )
    else:
        base = MIMIC4EHRDataset(
            root=str(args.root),
            tables=TABLES["mimic4"][3:],
            cache_dir=args.cache_dir,
            dev=args.dev,
            num_workers=args.num_workers,
        )
    task = TPCDayOne(args.dataset)
    plan.update(
        status="RUNNING",
        clinical_data_access="LOCAL_DATA_USED",
        clinical_experiments="IN_PROGRESS",
    )
    configurations = list(plan["configurations"])
    _save_progress(args.output_dir, plan, configurations, args.seeds)
    for seed in args.seeds:
        splits = base.set_task(
            task,
            split=PatientSplit((0.7, 0.15, 0.15), seed=seed),
            num_workers=args.num_workers,
        )
        train, validation, test = splits
        if len(train) < 2 or not len(validation) or len(test) < 2:
            raise ValueError(
                "Clinical cohort is too small: need >=2 train/test stays and "
                "a nonempty validation split. No valid clinical result is claimed."
            )
        preprocessing = {
            "feature_names": task.feature_names,
            "lab_indices": task.lab_indices,
            "statistics_fit_on": "training patients only",
            "inputs": {
                key: {
                    "center": train.input_processors[key].center,
                    "scale": train.input_processors[key].scale,
                }
                for key in ("time_series", "static")
            },
        }
        (args.output_dir / f"preprocessing_seed{seed}.json").write_text(
            json.dumps(preprocessing, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name, override in plan["configurations"].items():
            filename = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            checkpoint = args.output_dir / f"{filename}_seed{seed}.pt"
            result = train_configuration(
                (train, validation, test), task, override, seed, args, checkpoint
            )
            plan["results"].setdefault(name, []).append(result)
            _save_progress(args.output_dir, plan, configurations, args.seeds)
    plan.update(
        status="COMPLETED",
        clinical_data_access="LOCAL_DATA_USED",
        clinical_experiments="COMPLETED",
        clinical_results="MEASURED",
    )
    _save_progress(args.output_dir, plan, configurations, args.seeds)


def _save_progress(
    directory: Path, plan: dict[str, Any], configurations: list[str], seeds: list[int]
) -> None:
    (directory / "results.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (directory / "results.md").write_text(
        results_markdown(plan["results"], configurations, seeds), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    """Defaults to preparation only; requires --run to touch clinical data."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--dataset", choices=TABLES, default="eicu")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-version", default="PENDING")
    parser.add_argument("--output-dir", type=Path, default=Path("output/tpc_clinical"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("output/tpc_clinical_cache")
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--configs", nargs="+", choices=clinical_configs(), default=["baseline"]
    )
    selection.add_argument("--all-configs", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds requires at least three distinct seeds.")
    if len(args.configs) != len(set(args.configs)):
        parser.error("--configs must not repeat a configuration.")
    if any(
        getattr(args, field) <= 0
        for field in ("epochs", "batch_size", "num_workers", "cpu_threads")
    ):
        parser.error("Epochs, batch size, workers and CPU threads must be positive.")
    if args.run:
        if args.root is None:
            parser.error("--run requires --root pointing to authorized local data.")
        if args.data_version == "PENDING":
            parser.error(
                "--run requires --data-version to document the source release."
            )
        args.root = args.root.expanduser().resolve()
        preflight(args.root, args.dataset)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable; choose --device cpu.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    plan = write_plan(args)
    if args.run:
        try:
            run_experiments(args, plan)
        except (ValueError, RuntimeError, OSError) as exc:
            plan.update(
                status="FAILED",
                clinical_experiments="INCOMPLETE",
                clinical_results="PARTIAL" if plan["results"] else "PENDING",
                error=str(exc),
            )
            _save_progress(
                args.output_dir, plan, list(plan["configurations"]), args.seeds
            )
            raise
    else:
        print(
            f"Preparation only: {args.output_dir}\n"
            "Clinical data access, experiments and results remain PENDING. "
            "No dataset was loaded and no model was trained."
        )


if __name__ == "__main__":
    main()
