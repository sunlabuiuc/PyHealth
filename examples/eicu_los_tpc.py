"""Author: Tamjit Kumar, Yi Wang
NetID: tkumar70, ywang4715
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Temporal Pointwise Convolution (TPC) Length of Stay Prediction Pipeline
             and reproducible ablation study for eICU.

Primary Dataset:
    eICU Collaborative Research Database (v2.0)
    https://eicu-crd.mit.edu/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import time
from itertools import chain
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from pyhealth.datasets import (
    SampleDataset,
    collate_fn_dict_with_padding,
    eICUDataset,
)
from pyhealth.models.tpc import TPC
from pyhealth.tasks import (
    BucketBatchSampler,
    LengthOfStayTPCeICU,
    MapDataset,
    compute_los_metrics,
    length_of_stay_metrics_fn,
)
from pyhealth.tasks.length_of_stay_tpc import LOS_FLOOR_DAYS
from pyhealth.tasks.length_of_stay_tpc_eicu import (
    CORE_VITALS_MAP,
    get_valid_eicu_stay_ids,
    load_eicu_vitals_cache,
)

__all__ = [
    "AblationConfig",
    "BucketBatchSampler",
    "MapDataset",
    "compute_los_metrics",
    "length_of_stay_metrics_fn",
    "run_ablation_study",
]


class AblationConfig(TypedDict):
    """Configuration dictionary defining a specific TPC model ablation."""

    name: str
    loss_fn: str
    use_decay: bool
    aux_label_key: str | None
    share_weights: bool
    use_skip: bool


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Replace a checkpoint only after its replacement is fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        clean = {
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in payload.items()
        }
        temporary.write_text(
            json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_signature(
    *,
    eicu_root: str,
    train_pids: set[str],
    val_pids: set[str],
    test_pids: set[str],
    task: LengthOfStayTPCeICU,
) -> str:
    """Identify the cohort and fitted feature pipeline without storing patient IDs."""
    root = Path(eicu_root)
    source_sizes = {
        path.name: path.stat().st_size
        for path in sorted(root.glob("*.csv"))
        if path.name in {
            "patient.csv", "lab.csv", "physicalExam.csv",
            "vitalPeriodic.csv", "vitalAperiodic.csv",
        }
    }
    payload = {
        "source_sizes": source_sizes,
        "splits": {
            name: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
            for name, ids in (
                ("train", train_pids), ("val", val_pids), ("test", test_pids)
            )
        },
        "lab_names": task.lab_names,
        "norm_stats": task.norm_stats,
        "static_stats": task.static_stats,
        "cat_vocabs": task.cat_vocabs,
        "time_series_features": task.time_series_features,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_sampler: BucketBatchSampler,
    continuous_dim: int,
    static_dim: int,
    time_dim: int = 0,
    n_layers: int = 9,
    kernel_size: int = 4,
    temp_dim: int = 12,
    point_dim: int = 13,
    hidden_dim: int = 17,
    dropout: float = 0.45,
    temp_dropout: float = 0.05,
    batchnorm: str = "mybatchnorm",
    aux_weight: float = 100.0,
    epochs: int = 15,
    lr: float = 0.00226,
    device: str | torch.device = "cpu",
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    data_signature: str | None = None,
    log_every_batches: int = 100,
) -> list[dict[str, Any]]:
    """Runs the architectural ablation study replicating the paper's methodology.

    Ablations Evaluated:
    1. TPC (Proposed): MSLE Loss, Exponential Decay, Multi-Modal Inputs
    2. TPC (No Decay): Ablates missing data decay channels (use_decay=False)
    3. TPC (MSE Loss): Ablates logarithmic MSLE loss in favor of standard MSE loss
    4. TPC (Multitask): Joint hourly LoS regression and in-hospital mortality prediction

    Experimental Control:
    - Same random seed and batch order across configs.
    - Sampler reset ensures identical mini-batch sequences across all configs.
    - Configurable training epochs (default 15 matching paper eICU experiments).

    Args:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for test set.
        train_sampler: BucketBatchSampler used on train_loader.
        continuous_dim: Number of dynamic time-series features.
        static_dim: Number of static baseline features.
        time_dim: Number of non-convolved temporal context features (default: 2).
        n_layers: Number of TPC blocks (default: 9, Rocheteau best eICU).
        kernel_size: Filter size across time (default: 4, Rocheteau best eICU).
        temp_dim: Temporal channels per feature (default: 12, Rocheteau best eICU).
        point_dim: Pointwise channels (default: 13, Rocheteau best eICU).
        hidden_dim: Prediction head hidden dimension (default: 17, Rocheteau best eICU).
        dropout: Pointwise & head dropout rate (default: 0.45, Rocheteau best eICU).
        temp_dropout: Temporal conv dropout rate (default: 0.05, Rocheteau best eICU).
        batchnorm: Normalization variant ('mybatchnorm', 'default', 'none', etc.).
        aux_weight: Weight parameter for auxiliary mortality loss (default: 100.0,
            matching alpha in Rocheteau et al., arXiv:2007.09483 Appendix C).
        epochs: Number of training epochs per ablation config. Default: 15.
        lr: Learning rate for Adam optimizer. Default: 0.00226.
        device: Device to train on ('cpu' or 'cuda'). Default: 'cpu'.
        seed: Random seed for initialization. Default: 42.
        checkpoint_dir: Directory for per-configuration latest/best checkpoints.
        resume: Resume each configuration from its latest checkpoint if present.
        data_signature: Fingerprint of cohort, sources, and fitted preprocessing.
        log_every_batches: Print training progress every N batches (default: 100).
    """
    if log_every_batches < 1:
        raise ValueError("log_every_batches must be at least 1")
    dev = torch.device(device)
    print("\n" + "=" * 90)
    print("HELD-OUT TEST ABLATION STUDY: MULTI-MODAL TPC PIPELINE VERIFICATION (eICU)")
    print(
        f"Features: {continuous_dim} Continuous TS + {static_dim} Static + "
        f"{time_dim} Temporal Context | Device: {dev}"
    )
    print(
        f"Architecture: n_layers={n_layers}, k={kernel_size}, temp_dim={temp_dim}, "
        f"point_dim={point_dim}, hidden_dim={hidden_dim}, bn={batchnorm}, "
        f"aux_weight={aux_weight}"
    )
    print("=" * 90)

    configs: list[AblationConfig] = [
        {
            "name": "TPC (Proposed)",
            "loss_fn": "msle",
            "use_decay": True,
            "aux_label_key": None,
            "share_weights": False,
            "use_skip": True,
        },
        {
            "name": "TPC (No Decay)",
            "loss_fn": "msle",
            "use_decay": False,
            "aux_label_key": None,
            "share_weights": False,
            "use_skip": True,
        },
        {
            "name": "TPC (MSE Loss)",
            "loss_fn": "mse",
            "use_decay": True,
            "aux_label_key": None,
            "share_weights": False,
            "use_skip": True,
        },
        {
            "name": "TPC (Multitask)",
            "loss_fn": "msle",
            "use_decay": True,
            "aux_label_key": "mortality",
            "share_weights": False,
            "use_skip": True,
        },
        {
            "name": "TPC (Temp. WS)",
            "loss_fn": "msle",
            "use_decay": True,
            "aux_label_key": None,
            "share_weights": True,
            "use_skip": True,
        },
        {
            "name": "TPC (no skip)",
            "loss_fn": "msle",
            "use_decay": True,
            "aux_label_key": None,
            "share_weights": False,
            "use_skip": False,
        },
    ]

    def evaluate_loader(
        model: TPC, loader: DataLoader
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        model.eval()
        all_yp: list[torch.Tensor] = []
        all_yt: list[torch.Tensor] = []
        all_mask: list[torch.Tensor] = []
        sum_mort_bce = 0.0
        total_mort_batches = 0
        mort_preds_24: list[float] = []
        mort_labels_24: list[float] = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    k: v.to(dev) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
                out = model(**batch)
                y_pred = out["y_prob"]
                y_true = batch["los"]
                raw_mask = batch.get("mask")
                if raw_mask is not None:
                    mask = raw_mask.bool()
                else:
                    mask = (y_true > 0).squeeze(-1).bool()
                if mask.dim() == 3:
                    mask = mask.squeeze(-1)

                all_yp.append(y_pred.detach().cpu())
                all_yt.append(y_true.detach().cpu())
                all_mask.append(mask.detach().cpu())

                if model.has_aux_task and "aux_prob" in out and "mortality" in batch:
                    raw_mort = (
                        batch["mortality"]
                        .float()
                        .view(-1, 1, 1)
                        .expand_as(out["aux_prob"])
                    )
                    m_eval = out.get("mask")
                    if m_eval is None:
                        m_eval = mask.unsqueeze(-1)
                    elif m_eval.dim() == 2:
                        m_eval = m_eval.unsqueeze(-1)
                    if m_eval.sum() > 0:
                        mort_loss = torch.nn.functional.binary_cross_entropy(
                            out["aux_prob"][m_eval],
                            raw_mort[m_eval],
                        ).item()
                        sum_mort_bce += mort_loss
                        total_mort_batches += 1

                    # 24h mortality on stays reaching >= 24h (t = 23 index)
                    mort_1d = batch["mortality"].view(-1).tolist()
                    for b_i, y_m in enumerate(mort_1d):
                        s_len = int(mask[b_i].sum().item())
                        if s_len >= 24 and out["aux_prob"].shape[1] >= 24:
                            p24 = out["aux_prob"][b_i, 23, 0].item()
                            mort_preds_24.append(p24)
                            mort_labels_24.append(float(y_m))

        if len(all_yp) == 0:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float("nan"),
                float("nan"),
                float("nan"),
            )

        max_T = max(yp.shape[1] for yp in all_yp)
        padded_yp = [
            torch.nn.functional.pad(yp, (0, 0, 0, max_T - yp.shape[1]))
            if yp.shape[1] < max_T
            else yp
            for yp in all_yp
        ]
        padded_yt = [
            torch.nn.functional.pad(yt, (0, 0, 0, max_T - yt.shape[1]))
            if yt.shape[1] < max_T
            else yt
            for yt in all_yt
        ]
        padded_mask = [
            torch.nn.functional.pad(m.float(), (0, max_T - m.shape[1])).bool()
            if m.shape[1] < max_T
            else m
            for m in all_mask
        ]
        yp_cat = torch.cat(padded_yp, dim=0)
        yt_cat = torch.cat(padded_yt, dim=0)
        mask_cat = torch.cat(padded_mask, dim=0)

        los_metrics = length_of_stay_metrics_fn(
            y_true=yt_cat,
            y_pred=yp_cat,
            mask=mask_cat,
            floor_days=LOS_FLOOR_DAYS,
            min_eval_hours=4,
        )

        avg_mort = (
            sum_mort_bce / max(total_mort_batches, 1)
            if total_mort_batches > 0
            else float("nan")
        )
        auroc_24 = float("nan")
        auprc_24 = float("nan")
        if len(mort_labels_24) > 0 and len(set(mort_labels_24)) > 1:
            try:
                auroc_24 = float(roc_auc_score(mort_labels_24, mort_preds_24))
                auprc_24 = float(average_precision_score(mort_labels_24, mort_preds_24))
            except ValueError:
                pass

        return (
            los_metrics["msle"],
            los_metrics["mse"],
            los_metrics["mae"],
            los_metrics["mape"],
            los_metrics["r2"],
            los_metrics["binned_kappa"],
            avg_mort,
            auroc_24,
            auprc_24,
        )

    # 1. Compute Simple Empirical Baselines
    all_train_y: list[torch.Tensor] = []
    for raw_batch in train_loader:
        m = raw_batch.get("mask")
        yt = raw_batch["los"]
        if m is not None:
            m_bool = m.bool()
            if m_bool.dim() == 3:
                m_bool = m_bool.squeeze(-1)
            all_train_y.append(yt.squeeze(-1)[m_bool].cpu())
        else:
            all_train_y.append(yt.squeeze(-1).cpu())
    if all_train_y:
        train_y_cat = torch.cat(all_train_y)
        mean_los = float(train_y_cat.mean().item())
        median_los = float(train_y_cat.median().item())
        print(
            f"\nEmpirical Baselines: Train Mean LoS = {mean_los:.2f} days, "
            f"Median LoS = {median_los:.2f} days"
        )

    results = []

    for cfg in configs:
        print(f"\nTraining configuration: {cfg['name']}...")

        config_dir = None
        latest_path = None
        best_path = None
        config_signature = None
        if checkpoint_dir is not None:
            slug = re.sub(r"[^a-z0-9]+", "_", cfg["name"].lower()).strip("_")
            config_dir = Path(checkpoint_dir) / slug
            latest_path = config_dir / "latest.pt"
            best_path = config_dir / "best.pt"
            signature_payload = {
                "version": 1,
                "data_signature": data_signature,
                "config": cfg,
                "continuous_dim": continuous_dim,
                "static_dim": static_dim,
                "time_dim": time_dim,
                "n_layers": n_layers,
                "kernel_size": kernel_size,
                "temp_dim": temp_dim,
                "point_dim": point_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "temp_dropout": temp_dropout,
                "batchnorm": batchnorm,
                "aux_weight": aux_weight,
                "lr": lr,
                "seed": seed,
            }
            config_signature = hashlib.sha256(
                json.dumps(signature_payload, sort_keys=True).encode()
            ).hexdigest()
            if latest_path.exists() and not resume:
                raise FileExistsError(
                    f"Checkpoint already exists: {latest_path}. "
                    "Use --resume or choose another --checkpoint_dir."
                )

        # Reset sampler epoch so each configuration sees identical batch order
        train_sampler.reset()

        # Reset seeds for reproducible parameter initialization
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TPC(
            dataset=None,
            continuous_dim=continuous_dim,
            static_dim=static_dim,
            time_dim=time_dim,
            n_layers=n_layers,
            temp_dim=temp_dim,
            point_dim=point_dim,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temp_dropout=temp_dropout,
            batchnorm=batchnorm,
            loss_fn=cfg["loss_fn"],
            min_time=4,
            floor_days=LOS_FLOOR_DAYS,
            use_decay=cfg["use_decay"],
            share_weights=cfg["share_weights"],
            use_skip=cfg["use_skip"],
            aux_label_key=cfg["aux_label_key"],
            aux_weight=aux_weight,
        )
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_msle = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        start_epoch = 0
        if resume and latest_path is not None and latest_path.exists():
            # Checkpoints are created by this script; never load untrusted files.
            # RNG state tensors must stay on CPU for torch.set_rng_state().
            # load_state_dict moves model and optimizer state to their devices.
            saved = torch.load(latest_path, map_location="cpu", weights_only=False)
            if saved.get("signature") != config_signature:
                raise ValueError(
                    f"Checkpoint settings or data do not match this run: {latest_path}"
                )
            if saved.get("batch_size", train_sampler.batch_size) != train_sampler.batch_size:
                raise ValueError(f"Checkpoint batch size does not match: {latest_path}")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            start_epoch = int(saved["completed_epoch"])
            if start_epoch > epochs:
                raise ValueError(
                    f"Checkpoint has {start_epoch} completed epochs, "
                    f"but --epochs={epochs}. Increase --epochs to resume."
                )
            best_val_msle = float(saved["best_val_msle"])
            train_sampler.epoch = start_epoch
            _restore_rng_state(saved["rng_state"])
            if best_path is None or not best_path.exists():
                raise FileNotFoundError(
                    f"Best model checkpoint is missing for resumed run: {best_path}"
                )
            print(f"  Resumed after epoch {start_epoch} from {latest_path}")
        elif resume and latest_path is not None:
            print("  No latest checkpoint found; starting this configuration fresh.")

        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            recent_loss_sum = 0.0
            recent_batches = 0
            epoch_start = time.perf_counter()
            total_batches = len(train_loader)
            for raw_batch in train_loader:
                batch = {
                    k: v.to(dev) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
                optimizer.zero_grad()
                out = model(**batch)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                batch_loss = loss.item()
                train_loss_sum += batch_loss
                train_batches += 1
                recent_loss_sum += batch_loss
                recent_batches += 1
                if train_batches % log_every_batches == 0 or train_batches == total_batches:
                    elapsed = time.perf_counter() - epoch_start
                    print(
                        f"    Epoch {epoch + 1}/{epochs}, batch "
                        f"{train_batches}/{total_batches}: "
                        f"recent train loss = {recent_loss_sum / recent_batches:.4f}, "
                        f"elapsed = {elapsed / 60:.1f} min",
                        flush=True,
                    )
                    recent_loss_sum = 0.0
                    recent_batches = 0

            val_metrics = evaluate_loader(model, val_loader)
            val_msle_ep = val_metrics[0]
            avg_train_loss = train_loss_sum / max(train_batches, 1)
            print(
                f"    Epoch {epoch + 1:2d}/{epochs:2d}: "
                f"Train Loss = {avg_train_loss:.4f}, "
                f"Val MSLE = {val_msle_ep:.4f}"
            )
            if val_msle_ep < best_val_msle:
                best_val_msle = val_msle_ep
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                if best_path is not None:
                    _atomic_torch_save(
                        {
                            "signature": config_signature,
                            "completed_epoch": epoch + 1,
                            "best_val_msle": best_val_msle,
                            "batch_size": train_sampler.batch_size,
                            "model_state": best_state,
                        },
                        best_path,
                    )
            if latest_path is not None:
                _atomic_torch_save(
                    {
                        "signature": config_signature,
                        "completed_epoch": epoch + 1,
                        "best_val_msle": best_val_msle,
                        "batch_size": train_sampler.batch_size,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "rng_state": _rng_state(),
                    },
                    latest_path,
                )

        if best_path is not None and best_path.exists():
            saved_best = torch.load(best_path, map_location=dev, weights_only=False)
            if saved_best.get("signature") != config_signature:
                raise ValueError(f"Best checkpoint does not match this run: {best_path}")
            model.load_state_dict(saved_best["model_state"])
        elif best_state is not None:
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

        val_msle = evaluate_loader(model, val_loader)[0]
        (
            test_msle,
            test_mse,
            test_mae,
            test_mape,
            test_r2,
            test_kappa,
            _test_mort,
            test_auroc_24,
            test_auprc_24,
        ) = evaluate_loader(model, test_loader)

        results.append(
            {
                "Configuration": cfg["name"],
                "Val MSLE": val_msle,
                "Test MSLE": test_msle,
                "Test MSE": test_mse,
                "Test MAD": test_mae,
                "Test MAPE": test_mape,
                "Test R2": test_r2,
                "Test Kappa": test_kappa,
                "Mort 24h AUROC": test_auroc_24,
                "Mort 24h AUPRC": test_auprc_24,
            }
        )
        if config_dir is not None:
            _atomic_json_save(results[-1], config_dir / "metrics.json")

    print("\n" + "=" * 138)
    print(f"HELD-OUT TEST ABLATION STUDY RESULTS (eICU, {epochs} epochs):")
    print(
        f"{'Configuration':<26} | {'Val MSLE':<9} | {'Test MSLE':<9} | "
        f"{'Test MSE':<9} | {'Test MAD':<9} | {'Test MAPE':<10} | "
        f"{'Test R2':<9} | {'Test Kappa':<10} | "
        f"{'24h AUROC':<10} | {'24h AUPRC':<10}"
    )
    print("-" * 138)
    for r in results:
        auroc_s = (
            f"{r['Mort 24h AUROC']:<10.4f}"
            if not np.isnan(r["Mort 24h AUROC"])
            else "      N/A "
        )
        auprc_s = (
            f"{r['Mort 24h AUPRC']:<10.4f}"
            if not np.isnan(r["Mort 24h AUPRC"])
            else "      N/A "
        )
        print(
            f"{r['Configuration']:<26} | {r['Val MSLE']:<9.4f} | "
            f"{r['Test MSLE']:<9.4f} | {r['Test MSE']:<9.4f} | "
            f"{r['Test MAD']:<9.4f} | {r['Test MAPE']:<9.2f}% | "
            f"{r['Test R2']:<9.4f} | {r['Test Kappa']:<10.4f} | "
            f"{auroc_s} | {auprc_s}"
        )
    print("=" * 138)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eICU Length of Stay Prediction with TPC"
    )
    default_demo_dir = str(
        Path(__file__).resolve().parents[1] / "test-resources" / "core" / "eicudemo"
    )
    if not os.path.exists(default_demo_dir):
        default_demo_dir = "./test-resources/core/eicudemo"

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("EICU_ROOT", default_demo_dir),
        help=(
            "Path to eICU directory containing CSV files "
            "(default: $EICU_ROOT or ./test-resources/core/eicudemo)"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help=("Chunk size (number of rows) for reading large CSVs (default: 250000)"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per ablation config (default: 15)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32)",
    )
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=4,
        help="CPU workers for the training DataLoader (default: 4)",
    )
    parser.add_argument(
        "--log_every_batches",
        type=int,
        default=100,
        help="Print training progress every N batches (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00226,
        help="Learning rate for Adam optimizer (default: 0.00226)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=9,
        help="Number of TPC blocks (default: 9, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=4,
        help="Temporal filter size (default: 4, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--temp_dim",
        type=int,
        default=12,
        help="Temporal output channels per feature (default: 12, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--point_dim",
        type=int,
        default=13,
        help="Pointwise output channels (default: 13, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=17,
        help="Prediction head hidden dimension (default: 17, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.45,
        help="Pointwise & head dropout rate (default: 0.45, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--temp_dropout",
        type=float,
        default=0.05,
        help="Temporal conv dropout rate (default: 0.05, Rocheteau best eICU)",
    )
    parser.add_argument(
        "--batchnorm",
        type=str,
        default="mybatchnorm",
        choices=[
            "default",
            "mybatchnorm",
            "pointonly",
            "temponly",
            "low_momentum",
            "none",
        ],
        help=(
            "BatchNorm variant to use (default: 'mybatchnorm', reproducing "
            "Rocheteau et al. CHIL 2021; use 'default' for standard running statistics)"
        ),
    )
    parser.add_argument(
        "--aux_weight",
        type=float,
        default=100.0,
        help=(
            "Auxiliary loss weight for multitask training (default: 100.0, "
            "matching Rocheteau et al. arXiv:2007.09483 Appendix C)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training (default: 42)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Controlled directory for processed PyHealth data (recommended for eICU).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint root (default: checkpoints under the PyHealth cache).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each ablation from its latest matching checkpoint.",
    )
    parser.add_argument(
        "--use_native_physicalexam",
        action="store_true",
        help=(
            "Extract bedside vitals and admission baselines directly from "
            "eICUDataset(tables=['patient', 'lab', 'physicalexam']) without "
            "using out-of-core vitals cache (recommended for demo/smaller cohorts)."
        ),
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help=(
            "Force re-extraction of vitals cache, empirical statistics, and "
            "task samples from scratch, bypassing existing cache files."
        ),
    )
    args = parser.parse_args()

    eicu_root = args.data_dir
    if not os.path.exists(eicu_root):
        raise FileNotFoundError(
            f"eICU data directory not found: '{eicu_root}'. "
            "Please specify a valid path containing eICU CSV files using "
            "--data_dir, or ensure './test-resources/core/eicudemo' exists."
        )

    print("1. Initializing eICUDataset...")
    # On full eICU-CRD (~200k stays), physicalExam is ~1.5 GB and vitalPeriodic
    # is >10 GB. Instantiating in-memory Event objects for all physicalexam rows
    # on full eICU exhausts RAM. Therefore, by default we load ['patient', 'lab']
    # into eICUDataset and stream vitals out-of-core via load_eicu_vitals_cache().
    # For smaller cohorts or demo data, --use_native_physicalexam loads 'physicalexam'
    # directly into eICUDataset and extracts vitals natively without any cache file.
    tables = ["patient", "lab"]
    if args.use_native_physicalexam:
        tables.append("physicalexam")
    dataset = eICUDataset(
        root=eicu_root,
        tables=tables,
        cache_dir=args.cache_dir,
    )

    # Partition patients strictly BEFORE task processing to prevent any data leakage
    print("\n2. Partitioning patient cohort (70% Train, 15% Val, 15% Test)...")
    all_pids = sorted(dataset.unique_patient_ids)
    rng = random.Random(args.seed)
    perm_pids = list(all_pids)
    rng.shuffle(perm_pids)
    n_p = len(perm_pids)
    n_train = int(n_p * 0.70)
    n_val = int(n_p * 0.15)
    train_pids = set(perm_pids[:n_train])
    val_pids = set(perm_pids[n_train : n_train + n_val])
    test_pids = set(perm_pids[n_train + n_val :])
    print(
        f"  Train patients: {len(train_pids)}, Val patients: {len(val_pids)}, "
        f"Test patients: {len(test_pids)}"
    )

    if args.use_native_physicalexam:
        print(
            "\n3. Native extraction enabled (--use_native_physicalexam): "
            "vitals and baseline items will be extracted directly from "
            "dataset 'physicalexam' events (no cache file)."
        )
        vitals_by_stay = None
        baseline_static_by_stay = None
    else:
        print(
            "\n3. Loading bedside vital signs & admission baseline items "
            "from physicalExam (out-of-core cache)..."
        )
        vitals_cache_file = os.path.join(
            str(dataset.cache_dir), "tpc_eicu_vitals_cache.pkl"
        )
        if args.refresh_cache and os.path.exists(vitals_cache_file):
            print("  --refresh_cache specified: removing existing vitals cache...")
            try:
                os.remove(vitals_cache_file)
            except OSError:
                pass
        if os.path.exists(vitals_cache_file):
            valid_stay_ids = None
        else:
            valid_stay_ids = get_valid_eicu_stay_ids(dataset)
            print(f"  Identified {len(valid_stay_ids)} adult ICU stays (>= 5h).")
        vitals_by_stay, baseline_static_by_stay = load_eicu_vitals_cache(
            eicu_root,
            dataset,
            chunksize=args.chunksize,
            valid_stay_ids=valid_stay_ids,
            output_path=vitals_cache_file,
        )
        print(
            f"  Loaded vitals for {len(vitals_by_stay)} stays, "
            f"baseline items for {len(baseline_static_by_stay)} stays."
        )

    mode_tag = "native" if args.use_native_physicalexam else "cached"
    stats_cache_file = os.path.join(
        str(dataset.cache_dir), f"tpc_eicu_train_stats_{mode_tag}_s{args.seed}.pkl"
    )
    if args.refresh_cache and os.path.exists(stats_cache_file):
        print("  --refresh_cache specified: removing existing stats cache...")
        try:
            os.remove(stats_cache_file)
        except OSError:
            pass
    loaded_stats = False
    task: LengthOfStayTPCeICU | None = None
    if os.path.exists(stats_cache_file):
        print(
            "\n4 & 5. Loading cached training lab selection & empirical "
            f"statistics from {stats_cache_file}..."
        )
        try:
            with open(stats_cache_file, "rb") as f:
                (
                    qualifying_labs,
                    _thresh_met,
                    emp_stats,
                    feat_counts,
                    static_stats,
                    cat_vocabs,
                ) = pickle.load(f)
            task = LengthOfStayTPCeICU(
                lab_names=qualifying_labs,
                norm_stats=emp_stats,
                vitals_by_stay=vitals_by_stay,
                baseline_static_by_stay=baseline_static_by_stay,
                static_stats=static_stats,
                cat_vocabs=cat_vocabs,
                feat_counts=feat_counts,
            )
            loaded_stats = True
        except (EOFError, pickle.UnpicklingError, OSError) as err:
            print(
                f"  Cached stats file was corrupt or incomplete ({err}). Re-fitting..."
            )
            try:
                os.remove(stats_cache_file)
            except OSError:
                pass

    if not loaded_stats or task is None:
        print(
            "\n4 & 5. Fitting task on training cohort "
            "(lab selection + empirical stats)..."
        )
        task = LengthOfStayTPCeICU.fit(
            dataset=dataset,
            train_pids=train_pids,
            vitals_by_stay=vitals_by_stay,
            baseline_static_by_stay=baseline_static_by_stay,
        )
        tmp_stats = f"{stats_cache_file}.tmp"
        with open(tmp_stats, "wb") as f:
            pickle.dump(
                (
                    task.lab_names,
                    True,
                    task.norm_stats,
                    task.feat_counts,
                    task.static_stats,
                    task.cat_vocabs,
                ),
                f,
            )
        os.replace(tmp_stats, stats_cache_file)

    total_ts_features = len(CORE_VITALS_MAP) + len(task.lab_names)
    print(
        f"  Computed normalization parameters across {total_ts_features} "
        "time-series features."
    )
    print(f"  Core Vital Signs ({len(CORE_VITALS_MAP)}):")
    for vid, vname in CORE_VITALS_MAP.items():
        stats_item = task.norm_stats.get(vid, (0.0, 1.0))
        mean, std = float(stats_item[0]), float(stats_item[1])
        cnt = task.feat_counts.get(vid, 0)
        print(
            f"    {vname} ({vid}): {cnt:5d} measurements, "
            f"mean={mean:7.2f}, std={std:6.2f}"
        )

    print(
        "\n6. Applying ICU Length of Stay Task "
        "(Multi-Modal Time Series + Static Features)..."
    )
    # PyHealth includes task_name in deeply nested cache paths. Keep it short
    # enough for Windows while retaining the same task implementation.
    task.task_name = "eicu_tpc_los"
    print(f"  Derived static feature dimension: {task.static_dim}")
    if args.refresh_cache:
        tasks_dir = Path(dataset.cache_dir) / "tasks"
        if tasks_dir.exists():
            print("  --refresh_cache specified: removing existing task samples cache...")
            import shutil
            shutil.rmtree(tasks_dir, ignore_errors=True)
    sample_dataset = cast(SampleDataset, dataset.set_task(task))
    print(f"Total valid ICU stays created: {len(sample_dataset)}")

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else Path(dataset.cache_dir) / "checkpoints"
    )
    data_signature = _checkpoint_signature(
        eicu_root=eicu_root,
        train_pids=train_pids,
        val_pids=val_pids,
        test_pids=test_pids,
        task=task,
    )
    print(f"Checkpoints: {checkpoint_dir}")

    if len(sample_dataset) > 0:
        sample0 = sample_dataset[0]
        print("\nSample 0 summary:")
        print(f"  Patient ID: {sample0['patient_id']}")
        print(f"  Visit/Stay ID: {sample0['visit_id']}")
        print(f"  Vitals shape (T x F): {sample0['vitals'].shape}")
        print(f"  Vitals decay shape (T x F): {sample0['vitals_decay'].shape}")
        print(f"  Static features shape: {sample0['static'].shape}")
        print(f"  Mask shape: {sample0['mask'].shape}")
        print(f"  Hourly remaining LoS shape: {sample0['los'].shape}")
        print(f"  Remaining LoS at hour 0: {sample0['los'][0].item():.2f} days")
        print(f"  Remaining LoS at final hour: {sample0['los'][-1].item():.2f} days")

        mort_count = sum(s["mortality"].item() for s in sample_dataset)
        mort_rate = mort_count / len(sample_dataset) * 100.0
        print(
            f"  Cohort In-Hospital Mortality Prevalence: "
            f"{mort_count:.0f}/{len(sample_dataset)} ({mort_rate:.1f}%)"
        )

        p2idx = sample_dataset.patient_to_index
        cohort_pids = set(p2idx.keys())
        train_cohort_pids = cohort_pids & train_pids
        val_cohort_pids = cohort_pids & val_pids
        test_cohort_pids = cohort_pids & test_pids

        train_indices = list(chain(*[p2idx[pid] for pid in train_cohort_pids]))
        val_indices = list(chain(*[p2idx[pid] for pid in val_cohort_pids]))
        test_indices = list(chain(*[p2idx[pid] for pid in test_cohort_pids]))

        print("\n" + "=" * 40)
        print("Cohort summaries.")
        print("Number of patients")
        print(f"    Train:      {len(train_cohort_pids):>8,d}")
        print(f"    Validation: {len(val_cohort_pids):>8,d}")
        print(f"    Test:       {len(test_cohort_pids):>8,d}")
        print("Number of stays")
        print(f"    Train:      {len(train_indices):>8,d}")
        print(f"    Validation: {len(val_indices):>8,d}")
        print(f"    Test:       {len(test_indices):>8,d}")
        print("=" * 40 + "\n")

        # Sequence lengths for bucket batching without materializing
        # all tensors into RAM
        train_lengths = [sample_dataset[i]["vitals"].shape[0] for i in train_indices]
        val_lengths = [sample_dataset[i]["vitals"].shape[0] for i in val_indices]
        test_lengths = [sample_dataset[i]["vitals"].shape[0] for i in test_indices]
        train_sampler = BucketBatchSampler(
            train_lengths,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        val_sampler = BucketBatchSampler(
            val_lengths,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        test_sampler = BucketBatchSampler(
            test_lengths,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        train_loader = DataLoader(
            MapDataset(sample_dataset, train_indices),
            batch_sampler=train_sampler,
            collate_fn=collate_fn_dict_with_padding,
            num_workers=args.train_num_workers,
        )
        val_loader = DataLoader(
            MapDataset(sample_dataset, val_indices),
            batch_sampler=val_sampler,
            collate_fn=collate_fn_dict_with_padding,
        )
        test_loader = DataLoader(
            MapDataset(sample_dataset, test_indices),
            batch_sampler=test_sampler,
            collate_fn=collate_fn_dict_with_padding,
        )

        run_ablation_study(
            train_loader,
            val_loader,
            test_loader,
            train_sampler=train_sampler,
            continuous_dim=task.continuous_dim,
            static_dim=task.static_dim,
            time_dim=task.time_dim,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            temp_dim=args.temp_dim,
            point_dim=args.point_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            temp_dropout=args.temp_dropout,
            batchnorm=args.batchnorm,
            aux_weight=args.aux_weight,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
            data_signature=data_signature,
            log_every_batches=args.log_every_batches,
        )

        print(
            "\nEnd-to-end eICU TPC pipeline & unconfounded ablation "
            "successfully completed!"
        )
    else:
        print("No valid ICU samples generated.")
