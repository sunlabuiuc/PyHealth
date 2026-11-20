"""Run 12-month SEER lung survival benchmarks and the tumor-extent ablation.

Models are a training-majority classifier, PyHealth LogisticRegression, and
a two-layer PyHealth MLP. Both learned models run on numeric and one-hot
inputs. A fixed 80/10/10 patient split is shared by all configurations, and
only training patients shape preprocessing. Validation ROC-AUC plus F1
selects the learning rate and checkpoint. Test F1 uses survivors as the
positive class and a probability threshold of 0.5.

The ablation retrains the one-hot MLP with and without stage and extent
fields for three model seeds. Both versions use the learning rate and epoch
count chosen for the full one-hot MLP. The current SEER release, feature
subset, ordinal codes, architecture and tuning grid differ from the paper.

Run without arguments for a synthetic smoke run. For real data, supply
--data-dir and --cache-dir, both pointing to approved local storage.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import PatientSplit, SEERDataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MLP, LogisticRegression
from pyhealth.tasks import SEERSurvivalPrediction
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

SPLIT_NAMES = ("train", "validation", "test")
METRICS = ("roc_auc", "f1", "accuracy")
MONITOR = "roc_auc_plus_f1"
MODEL_SETTINGS = {
    "logistic_regression": {"embedding_dim": 64},
    "mlp": {"embedding_dim": 64, "hidden_dim": 64, "n_layers": 2},
}


class ValidationTrainer(Trainer):
    """Record the epoch selected by the trainer's validation monitor."""

    def __init__(self, model: torch.nn.Module, device: str | None) -> None:
        super().__init__(
            model, metrics=list(METRICS), device=device, enable_logging=False
        )
        self.best_epoch = 0
        self.best_score = -math.inf
        self.validation_epochs = 0

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        scores = super().evaluate(dataloader)
        scores[MONITOR] = scores["roc_auc"] + scores["f1"]
        self.validation_epochs += 1
        if scores[MONITOR] > self.best_score:
            self.best_epoch = self.validation_epochs
            self.best_score = scores[MONITOR]
        return scores


def write_synthetic_export(root: Path) -> None:
    """Generate independent example tumors with a learnable stage effect."""
    rng = np.random.default_rng(735)
    rows = []
    for patient in range(320):
        advanced = bool(rng.integers(2))
        survived = rng.random() < (0.24 if advanced else 0.74)
        months = int(rng.integers(12, 61) if survived else rng.integers(0, 12))
        cause = "Alive or dead of other cause" if survived else (
            "Dead (attributable to this cancer dx)"
        )
        rows.append({
            "Patient ID": f"synthetic-{patient:04d}",
            "Record number recode": "01",
            "Year of diagnosis": 2004 + patient % 6,
            "Site recode ICD-O-3/WHO 2008": "Lung and Bronchus",
            "Primary Site": rng.choice(["341", "342", "349"]),
            "Sex": rng.choice(["Male", "Female"]),
            "Age recode with <1 year olds and 90+": rng.choice(["60-64", "70-74"]),
            "Histologic Type ICD-O-3": rng.choice(["8070", "8140"]),
            "Grade Recode (thru 2017)": rng.choice(["II", "III"]),
            "Derived AJCC Stage Group, 6th ed (2004-2015)": "IV" if advanced else "I",
            "Laterality": rng.choice(["Left", "Right"]),
            "Marital status at diagnosis": rng.choice(["Married", "Single"]),
            "CS tumor size (2004-2015)": "999" if patient % 11 == 0 else (
                f"{rng.integers(45, 91) if advanced else rng.integers(8, 40):03d}"
            ),
            "CS extension (2004-2015)": "400" if advanced else "100",
            "CS lymph nodes (2004-2015)": "100" if advanced else "000",
            "CS mets at dx (2004-2015)": "10" if advanced else "00",
            "Survival months": months,
            "SEER cause-specific death classification": cause,
        })
    with (root / "seer_export.txt").open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(
    dataset: SEERDataset, split: PatientSplit, stack: ExitStack,
    encoding: str, include_extent: bool = True,
) -> tuple:
    task = SEERSurvivalPrediction(encoding=encoding, include_extent=include_extent)
    parts = dataset.set_task(task, split=split)
    for part in parts:
        stack.callback(part.close)
    return parts


def labels(part: Any, batch_size: int) -> np.ndarray:
    """Read the labels in a split without retaining feature tensors or IDs."""
    loader = get_dataloader(part, batch_size=batch_size)
    return np.concatenate([batch["label"].numpy().reshape(-1) for batch in loader])


def model_run(
    name: str, parts: tuple, args: argparse.Namespace, *,
    seed: int, lr: float, epochs: int, tune: bool,
) -> ValidationTrainer:
    set_seed(seed)
    train, validation, _ = parts
    model_class = MLP if name == "mlp" else LogisticRegression
    model = model_class(dataset=train, **MODEL_SETTINGS[name])
    trainer = ValidationTrainer(model, args.device)
    trainer.train(
        train_dataloader=get_dataloader(
            train, batch_size=args.batch_size, shuffle=True, seed=seed
        ),
        val_dataloader=(
            get_dataloader(validation, batch_size=args.batch_size) if tune else None
        ),
        epochs=epochs,
        optimizer_params={"lr": lr},
        monitor=MONITOR if tune else None,
    )
    return trainer


def test_scores(
    trainer: ValidationTrainer, test: Any, batch_size: int,
) -> dict[str, float]:
    loader = get_dataloader(test, batch_size=batch_size)
    scores = Trainer.evaluate(trainer, loader)
    return {metric: float(scores[metric]) for metric in METRICS}


def tune_and_test(name: str, parts: tuple, args: argparse.Namespace) -> dict:
    """Select a configuration using validation, then evaluate it once on test."""
    best_trainer = None
    candidates = []
    selected = None
    for lr in args.learning_rates:
        trainer = model_run(
            name, parts, args, seed=args.model_seeds[0], lr=lr,
            epochs=args.epochs, tune=True,
        )
        candidate = {
            "learning_rate": lr,
            "epochs": trainer.best_epoch,
            "validation_roc_auc_plus_f1": float(trainer.best_score),
        }
        candidates.append(candidate)
        if best_trainer is None or trainer.best_score > best_trainer.best_score:
            best_trainer, selected = trainer, candidate
    return {
        "selected": selected,
        "validation_candidates": candidates,
        "test": test_scores(best_trainer, parts[2], args.batch_size),
    }


def run_experiment(
    data_dir: Path, cache_dir: Path, args: argparse.Namespace,
) -> dict:
    dataset = SEERDataset(data_dir, cache_dir=cache_dir, num_workers=args.workers)
    split = PatientSplit(ratios=(0.8, 0.1, 0.1), seed=args.split_seed)
    with ExitStack() as stack:
        one_hot = prepare_data(dataset, split, stack, "one_hot")
        patient_sets = [set(part.patient_to_index) for part in one_hot]
        for idx, patients in enumerate(patient_sets):
            if not patients or any(patients & prev for prev in patient_sets[:idx]):
                raise RuntimeError("The patient split is empty or overlaps")
        observed = [labels(part, args.batch_size) for part in one_hot]
        if any(np.unique(y).size != 2 for y in observed):
            raise ValueError("ROC-AUC requires both labels in each partition")

        majority = int(observed[0].mean() > 0.5)
        baseline = binary_metrics_fn(
            observed[2], np.full(observed[2].shape, float(majority)), list(METRICS)
        )
        numeric = prepare_data(dataset, split, stack, "numeric")
        reduced = prepare_data(dataset, split, stack, "one_hot", include_extent=False)
        for parts in (numeric, reduced):
            for reference, variant in zip(one_hot, parts):
                same_records = (
                    reference.record_to_index.keys() == variant.record_to_index.keys()
                )
                if len(reference) != len(variant) or not same_records:
                    raise RuntimeError(
                        "Model configurations produced different cohort splits"
                    )

        models = {
            f"{name}_{encoding}": tune_and_test(name, parts, args)
            for encoding, parts in (("numeric", numeric), ("one_hot", one_hot))
            for name in MODEL_SETTINGS
        }
        selected = models["mlp_one_hot"]["selected"]
        ablation = []
        for seed in args.model_seeds:
            row = {"seed": seed}
            for name, parts in (("full", one_hot), ("without_extent", reduced)):
                trainer = model_run(
                    "mlp", parts, args, seed=seed, lr=selected["learning_rate"],
                    epochs=selected["epochs"], tune=False,
                )
                row[name] = test_scores(trainer, parts[2], args.batch_size)
            row["delta_full_minus_without_extent"] = {
                metric: row["full"][metric] - row["without_extent"][metric]
                for metric in METRICS
            }
            ablation.append(row)
        return {
            "data": "synthetic" if args.synthetic else "local_seer",
            "horizon_months": 12,
            "split_ratios": split.ratios,
            "split_seed": args.split_seed,
            "model_seeds": args.model_seeds,
            "architecture": MODEL_SETTINGS,
            "optimizer": {"name": "Adam", "batch_size": args.batch_size},
            "selection_metric": MONITOR,
            "f1_probability_threshold": 0.5,
            "partitions": {
                name: {
                    "patients": len(patients),
                    "tumors": len(y),
                    "positive_fraction": float(y.mean()),
                }
                for name, patients, y in zip(SPLIT_NAMES, patient_sets, observed)
            },
            "majority_baseline": {
                "predicted_label": majority,
                "test": {metric: float(value) for metric, value in baseline.items()},
            },
            "models": models,
            "ablation": {
                "training": selected,
                "runs": ablation,
                "mean_delta_full_minus_without_extent": {
                    metric: float(np.mean([
                        row["delta_full_minus_without_extent"][metric]
                        for row in ablation
                    ]))
                    for metric in METRICS
                },
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory with seer_export.txt")
    parser.add_argument("--cache-dir", type=Path, help="Approved local PyHealth cache")
    parser.add_argument("--results", type=Path, help="Optional aggregate JSON output")
    parser.add_argument(
        "--epochs", type=int, help="Maximum epochs per tuning candidate"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seeds", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.001, 0.003]
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Dataset processing workers"
    )
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--synthetic-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.data_dir is not None and args.cache_dir is None:
        parser.error("--cache-dir is required with --data-dir")
    args.synthetic = args.data_dir is None or args.synthetic_worker
    if args.epochs is None:
        args.epochs = 3 if args.synthetic else 10
    if args.batch_size is None:
        args.batch_size = 64 if args.synthetic else 1024
    if min(args.epochs, args.batch_size, args.workers, args.threads) < 1:
        parser.error("epochs, batch size, workers, and threads must be positive")
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("learning rates must be finite and positive")
    if min(args.split_seed, *args.model_seeds) < 0 or len(set(args.model_seeds)) != 3:
        parser.error("seeds must be nonnegative and the three model seeds distinct")
    torch.set_num_threads(args.threads)

    if args.data_dir is None:
        with TemporaryDirectory(prefix="sr") as tmp:
            root = Path(tmp)
            write_synthetic_export(root)
            cache = args.cache_dir or root / "c"
            if sys.platform == "win32":
                # LitData readers can keep cache files mapped until process exit.
                command = [
                    sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                    "--data-dir", str(root), "--cache-dir", str(cache),
                    "--synthetic-worker",
                ]
                raise SystemExit(subprocess.run(command, check=False).returncode)
            report = run_experiment(root, cache, args)
    else:
        report = run_experiment(args.data_dir, args.cache_dir, args)
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.results is not None:
        args.results.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
