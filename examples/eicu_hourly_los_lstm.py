"""
Contributors: Manling Tan, Yining Jin, Xiao Luo
NetIDs: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483

Hourly remaining ICU length-of-stay training and 2x3 missingness ablation.

Trains PyHealth ``RNN_V2(rnn_type="LSTM")`` on ``TPCLengthOfStayEICU`` using
patient-disjoint splits (70% train / 15% validation / 15% test) and evaluates
the six paper metrics (MAD, MAPE, MSE, MSLE, R^2, and Cohen's linear weighted
Kappa) plus training-set mean/median baselines.

Novel Missingness Robustness Ablation (2 x 3 design):
    * Two training conditions with identical model architecture and splits:
      Model A (0% extra observation removal) and Model B (10% extra random
      observation removal before forward-fill).
    * Three held-out test conditions (0%, 10%, 20% extra observation removal),
      evaluated with deterministic per-stay seeds so both models see the exact
      same masked test cells, while reusing each model's own training-fitted
      imputation/normalization processors.

Metric conventions (Section 5.3.1 & Appendix A/D of Rocheteau et al.):
    * Targets and predictions are converted together from hours to days
      (divided by 24.0) before scoring.
    * MAPE clamps the true denominator below at 4/24 days (4 hours) to
      prevent explosion near discharge.
    * Zero-target policy & MSLE: exact-discharge labels from
      ``TPCLengthOfStayEICU`` can equal 0.0 hours. Following Appendix A
      of the paper (HardTanh bounds ``[1/48, 100]`` days), both targets
      and predictions are clipped to ``[1/48, 100.0]`` days before taking
      the natural logarithm for MSLE.
    * Cohen's linear weighted Kappa discretizes days into 10 ordered bins:
      ``[0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-14, 14+]``.

Experimental Setup & Measured Results (15,000-patient eICU-CRD subset):
    * Subset & split: 15,000 unique patients sampled uniformly at random
      (seed=42), split 70% / 15% / 15% by ``patient_id``.
    * Model & training: ``RNN_V2(rnn_type="LSTM")``, ``embedding_dim=64``,
      ``hidden_dim=128``, ``num_layers=2``, ``dropout=0.2``, ``mask="concat"``,
      ``include_decay=True``, ``lr=0.00129``, ``batch_size=128``, ``epochs=8``,
      MSE loss on min-max normalized hours (PyHealth default for
      ``timeseries_regression``).
    * Results (in days):

      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | Model / Condition     | Train Drop | Test Drop | MAD ↓ | MAPE (%) | MSE ↓ | MSLE  | R^2 ↑  | Kappa ↑ |
      +=======================+============+===========+=======+==========+=======+=======+========+=========+
      | Mean Baseline         | 0%         | 0%        | 3.336 | 413.3    | 28.24 | 2.750 | -0.002 | 0.000   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | Median Baseline       | 0%         | 0%        | 2.837 | 187.8    | 31.32 | 2.008 | -0.111 | 0.000   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | LSTM (Train 0% drop)  | 0%         | 0%        | 2.859 | 260.7    | 24.72 | 2.266 | 0.123  | 0.253   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | LSTM (Train 0% drop)  | 0%         | 10%       | 2.839 | 255.6    | 24.79 | 2.256 | 0.120  | 0.247   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | LSTM (Train 0% drop)  | 0%         | 20%       | 2.819 | 251.1    | 24.84 | 2.223 | 0.118  | 0.243   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | LSTM (Train 10% drop) | 10%        | 0%        | 2.893 | 271.6    | 24.74 | 2.205 | 0.122  | 0.252   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | LSTM (Train 10% drop) | 10%        | 10%       | 2.869 | 266.3    | 24.76 | 2.189 | 0.121  | 0.247   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+
      | LSTM (Train 10% drop) | 10%        | 20%       | 2.844 | 261.1    | 24.77 | 2.153 | 0.121  | 0.243   |
      +-----------------------+------------+-----------+-------+----------+-------+-------+--------+---------+

    * Findings:
      1. Baseline alignment: our 15k subset baselines closely match the full
         eICU paper baselines (Table 2: Mean MAD 3.21 / MSE 29.5; Median MAD
         2.76 / MSE 32.6), and ``LSTM (Train 0% drop)`` at 0% test drop
         closely tracks the paper's ``Standard LSTM (MSE)`` (MSE 24.72 vs
         24.5, Kappa 0.253 vs 0.28). Remaining gaps reflect using ~10% of
         patients, 48 YAIB dynamic + 4 static features (vs 87 + 17), and
         MSE loss rather than log-space MSLE loss.
      2. Robustness ablation: as test missingness increases from 0% to 20%,
         Model A (0% train drop) degrades by +0.12 MSE and -0.005 R^2, whereas
         Model B (10% train drop) degrades by only +0.03 MSE and -0.001 R^2,
         outperforming Model A on MSE, R^2, and MSLE under 10% and 20%
         missingness.

Examples:
    Smoke check on five synthetic patients::

        python examples/eicu_hourly_los_lstm.py --synthetic --epochs 2

    Run full 2x3 ablation on a local eICU-CRD or eICU Demo directory::

        python examples/eicu_hourly_los_lstm.py --root /path/to/eicu-crd
"""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Bound native thread pools on Windows before importing torch/polars.
for _var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, r2_score

from examples.tpc_support.generate_synthetic import TABLES, generate
from pyhealth.datasets import PatientSplit, eICUDataset, get_dataloader
from pyhealth.models import RNN_V2
from pyhealth.tasks import TPCLengthOfStayEICU
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

HOURS_PER_DAY = 24.0
MAPE_FLOOR_DAYS = 4.0 / 24.0
MIN_LOS_DAYS = 1.0 / 48.0
MAX_LOS_DAYS = 100.0
KAPPA_BIN_EDGES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 14.0)
KAPPA_LABELS = list(range(len(KAPPA_BIN_EDGES) + 1))

DEFAULT_TRAIN_RATES = (0.0, 0.1)
DEFAULT_TEST_RATES = (0.0, 0.1, 0.2)


def compute_los_metrics(
    y_true_hours: np.ndarray,
    y_pred_hours: np.ndarray,
) -> dict[str, float]:
    """Compute the 6 paper LoS evaluation metrics in days.

    Args:
        y_true_hours: 1D array of ground-truth remaining LoS in hours.
        y_pred_hours: 1D array of predicted remaining LoS in hours.

    Returns:
        Dictionary with keys ``mad``, ``mape``, ``mse``, ``msle``, ``r2``,
        and ``kappa``.
    """
    y_true = np.asarray(y_true_hours, dtype=np.float64).reshape(-1) / HOURS_PER_DAY
    y_pred = np.asarray(y_pred_hours, dtype=np.float64).reshape(-1) / HOURS_PER_DAY
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(
            f"Expected non-empty matching shapes, got {y_true.shape} and {y_pred.shape}."
        )

    abs_err = np.abs(y_true - y_pred)
    mad = float(np.mean(abs_err))
    mape = float(np.mean(abs_err / np.maximum(y_true, MAPE_FLOOR_DAYS)) * 100.0)
    mse = float(np.mean((y_true - y_pred) ** 2))

    # Clip both targets and predictions to [1/48, 100] days for log-space MSLE.
    log_true = np.log(np.clip(y_true, MIN_LOS_DAYS, MAX_LOS_DAYS))
    log_pred = np.log(np.clip(y_pred, MIN_LOS_DAYS, MAX_LOS_DAYS))
    msle = float(np.mean((log_true - log_pred) ** 2))

    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan")

    true_bins = np.digitize(y_true, KAPPA_BIN_EDGES, right=False)
    pred_bins = np.digitize(y_pred, KAPPA_BIN_EDGES, right=False)
    if len(np.unique(np.concatenate([true_bins, pred_bins]))) < 2:
        kappa = 0.0
    else:
        kappa = float(
            cohen_kappa_score(
                true_bins,
                pred_bins,
                labels=KAPPA_LABELS,
                weights="linear",
            )
        )
        if np.isnan(kappa):
            kappa = 0.0

    return {
        "mad": mad,
        "mape": mape,
        "mse": mse,
        "msle": msle,
        "r2": r2,
        "kappa": kappa,
    }


def _extract_target_hours(dataset: Any, batch_size: int) -> np.ndarray:
    """Extract unscaled ground-truth hourly targets from a SampleDataset."""
    label_proc = dataset.output_processors["los"]
    loader = get_dataloader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
    )
    parts: list[np.ndarray] = []
    for batch in loader:
        true_list, _ = label_proc.to_numpy(batch["los"], batch["los"][0])
        parts.extend(true_list)
    return np.concatenate(parts, axis=0)


def compute_constant_baselines(
    train_true_hours: np.ndarray,
    test_true_hours: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate training-set mean and median baselines on test targets."""
    mean_hours = float(np.mean(train_true_hours))
    median_hours = float(np.median(train_true_hours))
    return {
        "mean_baseline": compute_los_metrics(
            test_true_hours,
            np.full_like(test_true_hours, mean_hours, dtype=np.float64),
        ),
        "median_baseline": compute_los_metrics(
            test_true_hours,
            np.full_like(test_true_hours, median_hours, dtype=np.float64),
        ),
    }


def _build_embedding_dim(
    embedding_dim: int,
    include_decay: bool,
) -> dict[str, list[dict[str, int]]]:
    """Build per-field LevelEmbedding config matching TPCLengthOfStayEICU."""
    config: dict[str, list[dict[str, int]]] = {
        "dynamic": [{"values": embedding_dim}],
        "static": [{"values": embedding_dim}],
    }
    if include_decay:
        config["decay"] = [{"values": embedding_dim}]
    return config


def _train_single_model(
    base_dataset: eICUDataset,
    *,
    train_missing_rate: float,
    include_decay: bool,
    split: PatientSplit,
    epochs: int,
    batch_size: int,
    lr: float,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    seed: int,
    num_workers: int,
    device: str,
) -> tuple[RNN_V2, Trainer, Any, Any, dict[int, Any], np.ndarray, dict[str, int]]:
    """Train one LSTM model at ``train_missing_rate`` using training-only stats."""
    set_seed(seed)
    train_task = TPCLengthOfStayEICU(
        missing_rate=train_missing_rate,
        seed=seed,
        include_decay=include_decay,
    )
    train_data, val_data, test_data = base_dataset.set_task(
        train_task,
        split=split,
        num_workers=num_workers,
    )
    sizes = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }
    if not all(sizes.values()):
        raise ValueError(f"Empty dataset split encountered: {sizes}")

    train_loader = get_dataloader(
        train_data,
        batch_size=min(batch_size, len(train_data)),
        shuffle=True,
        seed=seed,
    )
    val_loader = get_dataloader(
        val_data,
        batch_size=min(batch_size, len(val_data)),
        shuffle=False,
    )

    model = RNN_V2(
        dataset=train_data,
        embedding_dim=_build_embedding_dim(embedding_dim, include_decay),
        hidden_dim=hidden_dim,
        rnn_type="LSTM",
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
        mask="concat",
    )
    trainer = Trainer(
        model=model,
        metrics=["mae", "mse"],
        device=device,
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
        max_grad_norm=5.0,
    )

    train_true_hours = _extract_target_hours(train_data, batch_size)
    processors = (train_data.input_processors, train_data.output_processors)
    # Free training/validation loader references while keeping fitted processors.
    del train_loader, val_loader, train_data, val_data
    return (
        model,
        trainer,
        processors[0],
        processors[1],
        {0: test_data},
        train_true_hours,
        sizes,
    )


def _format_results_table(rows: list[dict[str, Any]]) -> str:
    """Format experiment results into a readable Markdown comparison table."""
    header = (
        "| Model / Condition | Train Drop | Test Drop | "
        "MAD ↓ | MAPE (%) ↓ | MSE ↓ | MSLE ↓ | R^2 ↑ | Kappa ↑ |"
    )
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    lines = [header, sep]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['label']} | {row['train_missing_rate']:.0%} | "
            f"{row['test_missing_rate']:.0%} | "
            f"{m['mad']:.3f} | {m['mape']:.1f} | {m['mse']:.2f} | "
            f"{m['msle']:.3f} | {m['r2']:.3f} | {m['kappa']:.3f} |"
        )
    return "\n".join(lines)


def train_and_evaluate(
    root: str | Path | None = None,
    *,
    synthetic: bool = False,
    train_rates: tuple[float, ...] = DEFAULT_TRAIN_RATES,
    test_rates: tuple[float, ...] = DEFAULT_TEST_RATES,
    include_decay: bool = True,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 0.00129,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    seed: int = 42,
    num_workers: int = 1,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run the 2x3 missingness ablation study and return all metric rows."""
    if not synthetic and root is None:
        raise ValueError("Provide --root <eicu_dir> or pass --synthetic.")

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _run_with_root(
        data_root: Path,
        effective_cache: Path | None,
        split_ratios: tuple[float, float, float],
    ) -> list[dict[str, Any]]:
        dataset_kwargs: dict[str, Any] = {
            "root": str(data_root),
            "tables": list(TABLES),
            "num_workers": num_workers,
        }
        if effective_cache is not None:
            dataset_kwargs["cache_dir"] = str(effective_cache)

        base_dataset = eICUDataset(**dataset_kwargs)
        split = PatientSplit(ratios=split_ratios, seed=seed)
        results: list[dict[str, Any]] = []
        baselines_recorded = False

        for train_rate in train_rates:
            (
                model,
                trainer,
                in_procs,
                out_procs,
                cached_test_by_rate,
                train_true_hours,
                sizes,
            ) = _train_single_model(
                base_dataset,
                train_missing_rate=train_rate,
                include_decay=include_decay,
                split=split,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                seed=seed,
                num_workers=num_workers,
                device=chosen_device,
            )

            model_label = f"LSTM (Train {train_rate:.0%} drop)"
            for test_rate in test_rates:
                if np.isclose(test_rate, train_rate):
                    test_data = cached_test_by_rate[0]
                else:
                    test_task = TPCLengthOfStayEICU(
                        missing_rate=test_rate,
                        seed=seed,
                        include_decay=include_decay,
                    )
                    # Reuse training-fitted processors so test stats never leak.
                    _, _, test_data = base_dataset.set_task(
                        test_task,
                        input_processors=in_procs,
                        output_processors=out_procs,
                        split=split,
                        num_workers=num_workers,
                    )

                test_loader = get_dataloader(
                    test_data,
                    batch_size=min(batch_size, len(test_data)),
                    shuffle=False,
                )
                y_true_stays, y_pred_stays, test_loss = trainer.inference(test_loader)
                test_true_hours = np.concatenate(y_true_stays, axis=0)
                test_pred_hours = np.concatenate(y_pred_stays, axis=0)
                metrics = compute_los_metrics(test_true_hours, test_pred_hours)

                if not baselines_recorded and np.isclose(test_rate, 0.0):
                    baselines = compute_constant_baselines(
                        train_true_hours, test_true_hours
                    )
                    for base_name, base_label in (
                        ("mean_baseline", "Mean Baseline"),
                        ("median_baseline", "Median Baseline"),
                    ):
                        results.append(
                            {
                                "label": base_label,
                                "train_missing_rate": 0.0,
                                "test_missing_rate": test_rate,
                                "include_decay": include_decay,
                                "splits": sizes,
                                "n_test_hours": len(test_true_hours),
                                "metrics": baselines[base_name],
                            }
                        )
                    baselines_recorded = True

                results.append(
                    {
                        "label": model_label,
                        "train_missing_rate": float(train_rate),
                        "test_missing_rate": float(test_rate),
                        "include_decay": include_decay,
                        "splits": sizes,
                        "test_loss_scaled_mse": float(test_loss),
                        "n_test_hours": len(test_true_hours),
                        "metrics": metrics,
                    }
                )
                del test_loader, test_data

            del trainer, model, in_procs, out_procs, cached_test_by_rate
            gc.collect()
            if torch.device(chosen_device).type == "cuda":
                torch.cuda.empty_cache()

        del base_dataset
        gc.collect()
        return results

    if synthetic:
        with tempfile.TemporaryDirectory(
            prefix="tpc-lstm-ablation-", ignore_cleanup_errors=True
        ) as tmp:
            tmp_path = Path(tmp)
            csv_dir = tmp_path / "csv"
            generate(csv_dir)
            # Only 3 of the 5 synthetic stays pass cohort criteria;
            # (0.4, 0.3, 0.3) guarantees 1 train / 1 val / 1 test patient.
            return _run_with_root(csv_dir, tmp_path / "cache", (0.4, 0.3, 0.3))

    assert root is not None
    data_root = Path(root).expanduser().resolve()
    effective_cache = (
        Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    )
    return _run_with_root(data_root, effective_cache, (0.7, 0.15, 0.15))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Path to eICU-CRD or eICU Demo CSV directory.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a fast smoke ablation on generated synthetic patients.",
    )
    parser.add_argument(
        "--train-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_TRAIN_RATES),
        help="Training missingness rates (default: 0.0 0.1).",
    )
    parser.add_argument(
        "--test-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEST_RATES),
        help="Test missingness rates (default: 0.0 0.1 0.2).",
    )
    parser.add_argument(
        "--no-decay",
        dest="include_decay",
        action="store_false",
        help="Disable exponential decay indicators (0.75^hours_since_obs).",
    )
    parser.set_defaults(include_decay=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00129)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full JSON metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = train_and_evaluate(
        root=args.root,
        synthetic=args.synthetic,
        train_rates=tuple(args.train_rates),
        test_rates=tuple(args.test_rates),
        include_decay=args.include_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    print("\n=== Hourly Remaining ICU LoS (2x3 Missingness Ablation) ===")
    print(_format_results_table(rows))
    if args.output_json is not None:
        out_path = args.output_json
        if out_path.is_dir():
            out_path = out_path / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
