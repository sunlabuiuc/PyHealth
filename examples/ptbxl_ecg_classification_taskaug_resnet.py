"""PTB-XL ECG binary classification with TaskAug + 1-D ResNet-18.

Demonstrates full pipeline replication of Raghu et al. (2022):
    "Data Augmentation for Electrocardiograms", CHIL 2022.
    https://proceedings.mlr.press/v174/raghu22a.html

This script contains four sections:

1. **Standard training** — joint optimisation of backbone + policy with a
   single Adam optimiser (fast baseline).

2. **Bi-level training (BiLevelTrainer)** — inner loop updates the backbone
   on augmented training data; outer loop updates the policy on clean
   validation loss.  Uses a first-order DARTS-style approximation.

3. **Ablation study** — compares six configurations on synthetic data:
   (a) no augmentation, (b) fixed random augmentation, (c) TaskAug 1-stage,
   (d) TaskAug 2-stage (default), (e) frozen policy (random init, never
   updated), (f) shared magnitudes (class-agnostic mu_0 = mu_1).

4. **Learning-rate sweep** — evaluates TaskAug K=2 with three outer-loop
   learning rates {1e-2, 1e-3, 1e-4} to show sensitivity to this
   hyperparameter.

Usage
-----
Real PTB-XL data (requires download from PhysioNet)::

    python ptbxl_ecg_classification_taskaug_resnet.py \
        --data_root /path/to/ptb-xl/ \
        --task MI --mode bilevel --epochs 20

Synthetic data (no download needed, for testing/CI)::

    python ptbxl_ecg_classification_taskaug_resnet.py --synthetic

Ablation results (synthetic, default)
-------------------------------------
Expected relative ordering: D >= C >= F >= B >= E >= A (AUROC).
Configs D and C (learned policy) should outperform A (no augmentation).
Config E (frozen policy) isolates the benefit of *learning* the policy.
Config F (shared magnitudes) tests the class-specific magnitude hypothesis.
The lr sweep should show lr_outer=1e-3 outperforms 1e-2 (too aggressive)
and 1e-4 (too slow to converge in few epochs).
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Synthetic dataset (no real data required)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n_train: int = 200,
    n_val: int = 50,
    leads: int = 12,
    length: int = 1000,
    pos_rate: float = 0.3,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """Generate synthetic ECG-like tensors for offline testing.

    Positive-class signals have a higher-amplitude sinusoidal component
    injected into lead 0 to give the model a learnable signal.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        leads: Number of ECG leads (channels).
        length: Time-series length.
        pos_rate: Fraction of positive-class samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    rng = np.random.default_rng(seed)

    def _make_split(n: int) -> Tuple[np.ndarray, np.ndarray]:
        labels = (rng.random(n) < pos_rate).astype(np.int64)
        signals = rng.standard_normal((n, leads, length)).astype(np.float32)
        # Inject discriminative signal into lead 0 for positive class
        t = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
        for i, lbl in enumerate(labels):
            if lbl == 1:
                signals[i, 0] += 0.5 * np.sin(5 * t)
        return signals, labels

    x_tr, y_tr = _make_split(n_train)
    x_val, y_val = _make_split(n_val)

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Real PTB-XL dataset (optional — skipped if --synthetic)
# ---------------------------------------------------------------------------

def make_ptbxl_dataset(
    data_root: str,
    task_label: str = "MI",
    sampling_rate: int = 100,
) -> Tuple[TensorDataset, TensorDataset]:
    """Load PTB-XL via PyHealth and return TensorDatasets.

    Performs an 80/20 train/val split on the first 5000 records (N=5000
    regime from Raghu et al. Table 2).

    Args:
        data_root: Path to the PTB-XL root directory.
        task_label: One of ``"MI"``, ``"HYP"``, ``"STTC"``, ``"CD"``.
        sampling_rate: Waveform sampling rate (100 or 500 Hz).

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    from pyhealth.datasets.ptbxl import PTBXLDataset
    from pyhealth.tasks.ecg_classification import ECGBinaryClassification

    dataset = PTBXLDataset(
        root=data_root,
        sampling_rate=sampling_rate,
    )
    sample_ds = dataset.set_task(ECGBinaryClassification(task_label=task_label))

    # Collect all samples
    ecgs, labels = [], []
    for sample in sample_ds:
        ecgs.append(sample["ecg"].clone().detach())
        labels.append(sample["label"])

    ecgs = torch.stack(ecgs)                        # (N, 12, T)
    labels = torch.stack(labels).squeeze(-1).long()  # (N,) not (N, 1)

    # 80/20 split (up to 5000 samples)
    n = min(len(ecgs), 5000)
    ecgs, labels = ecgs[:n], labels[:n]
    split = int(0.8 * n)
    train_ds = TensorDataset(ecgs[:split], labels[:split])
    val_ds = TensorDataset(ecgs[split:], labels[split:])
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUROC without sklearn (trapezoidal rule)."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    return float(np.trapezoid(tpr, fpr))


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUPRC (average precision) without sklearn (trapezoidal rule)."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    # Prepend (recall=0, precision=1) so the curve starts at the top-left
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Return loss, accuracy, and AUROC on *loader*."""
    model.eval()
    all_loss, all_prob, all_true = [], [], []

    for ecg, label in loader:
        ecg, label = ecg.to(device), label.to(device)
        out = model(ecg=ecg, label=label)
        all_loss.append(out["loss"].item())
        all_prob.extend(out["y_prob"].squeeze(-1).cpu().numpy())
        all_true.extend(label.cpu().numpy())

    y_prob = np.array(all_prob)
    y_true = np.array(all_true)
    acc = ((y_prob > 0.5).astype(int) == y_true).mean()
    return {
        "loss": float(np.mean(all_loss)),
        "accuracy": float(acc),
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_auprc(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Standard training loop
# ---------------------------------------------------------------------------

def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> List[Dict]:
    """Standard joint optimisation of backbone + policy.

    Args:
        model: :class:`TaskAugResNet` instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Compute device.

    Returns:
        List of per-epoch metric dicts.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for ecg, label in train_loader:
            ecg, label = ecg.to(device), label.to(device)
            optimizer.zero_grad()
            model(ecg=ecg, label=label)["loss"].backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"[Standard] Epoch {epoch:3d} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_acc={metrics['accuracy']:.3f} | "
            f"val_auroc={metrics['auroc']:.3f}"
        )

    return history


# ---------------------------------------------------------------------------
# Bi-level trainer (DARTS first-order approximation)
# ---------------------------------------------------------------------------

class BiLevelTrainer:
    """First-order bi-level optimiser for TaskAug (Raghu et al., 2022).

    The inner loop updates the ResNet backbone on augmented training batches
    using Adam.  The outer loop updates the augmentation policy on clean
    validation batches — approximating implicit differentiation with a
    single-step unrolling (DARTS-style first-order approximation).

    Args:
        model: :class:`TaskAugResNet` instance.
        lr_inner: Inner-loop (backbone) learning rate.
        lr_outer: Outer-loop (policy) learning rate.
        device: Compute device.
    """

    def __init__(
        self,
        model: "TaskAugResNet",  # noqa: F821
        lr_inner: float = 1e-3,
        lr_outer: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.inner_opt = optim.Adam(model.backbone_parameters(), lr=lr_inner)
        self.outer_opt = optim.RMSprop(model.policy_parameters(), lr=lr_outer)

    def step(
        self,
        ecg_train: torch.Tensor,
        label_train: torch.Tensor,
        ecg_val: torch.Tensor,
        label_val: torch.Tensor,
    ) -> Tuple[float, float]:
        """Execute one inner + outer update step.

        Args:
            ecg_train: Training ECG batch ``(B, 12, T)``.
            label_train: Training labels ``(B,)``.
            ecg_val: Validation ECG batch ``(B, 12, T)`` (clean, no augment).
            label_val: Validation labels ``(B,)``.

        Returns:
            Tuple of ``(train_loss, val_loss)`` floats.
        """
        ecg_train = ecg_train.to(self.device)
        label_train = label_train.to(self.device)
        ecg_val = ecg_val.to(self.device)
        label_val = label_val.to(self.device)

        # Inner step: update backbone on augmented training data
        self.model.train()
        self.inner_opt.zero_grad()
        train_loss = self.model(ecg=ecg_train, label=label_train)["loss"]
        train_loss.backward()
        self.inner_opt.step()

        # Outer step: update policy on clean validation data
        # (first-order approximation — no Neumann series unrolling)
        self.outer_opt.zero_grad()
        val_loss = self.model(ecg=ecg_val, label=label_val)["loss"]
        val_loss.backward()
        self.outer_opt.step()

        return float(train_loss.item()), float(val_loss.item())

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
    ) -> List[Dict]:
        """Train for *epochs* epochs with bi-level optimisation.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (also used for outer loop).
            epochs: Number of epochs.

        Returns:
            List of per-epoch metric dicts.
        """
        val_iter = iter(val_loader)
        history: List[Dict] = []

        for epoch in range(1, epochs + 1):
            for ecg_tr, lbl_tr in train_loader:
                # Sample a validation batch for the outer step
                try:
                    ecg_val, lbl_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    ecg_val, lbl_val = next(val_iter)

                self.step(ecg_tr, lbl_tr, ecg_val, lbl_val)

            metrics = evaluate(self.model, val_loader, self.device)
            metrics["epoch"] = epoch
            history.append(metrics)
            print(
                f"[BiLevel]  Epoch {epoch:3d} | "
                f"val_loss={metrics['loss']:.4f} | "
                f"val_acc={metrics['accuracy']:.3f} | "
                f"val_auroc={metrics['auroc']:.3f}"
            )

        return history


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
) -> Dict[str, Dict]:
    """Compare six augmentation configurations on the same data split.

    Configurations
    --------------
    A. **No augmentation** — backbone-only, no policy.
    B. **Fixed random augmentation** — Gaussian noise (sigma=0.1), no learning.
    C. **TaskAug 1-stage** — learned policy, K=1.
    D. **TaskAug 2-stage** — learned policy, K=2 (paper default).
    E. **Frozen policy** — policy initialized at random but never updated;
       only backbone trains.  Isolates the benefit of *learning* the policy.
    F. **Shared magnitudes** — class-agnostic magnitudes (mu_0 = mu_1);
       tests the asymmetric augmentation hypothesis from Section 3 of the paper.

    Args:
        train_ds: Training TensorDataset.
        val_ds: Validation TensorDataset.
        device: Compute device.
        epochs: Training epochs per configuration.
        batch_size: Mini-batch size.

    Returns:
        Dict mapping config keys to metric dicts.
    """
    from pyhealth.models.taskaug_resnet import TaskAugResNet, _ResNet1D, TaskAugPolicy

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    results: Dict[str, Dict] = {}

    # ---- Configuration A: no augmentation (backbone only) ----
    print("\n Ablation A: No Augmentation ")

    class BackboneOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _ResNet1D(in_channels=12, num_classes=1)

        def forward(self, ecg: torch.Tensor, label: Optional[torch.Tensor] = None):
            logits = self.backbone(ecg)
            y_prob = torch.sigmoid(logits)
            out = {"logit": logits, "y_prob": y_prob}
            if label is not None:
                out["loss"] = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), label.float()
                )
                out["y_true"] = label
            return out

    torch.manual_seed(42)
    model_a = BackboneOnly().to(device)
    opt_a = optim.Adam(model_a.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model_a.train()
        for ecg, lbl in train_loader:
            ecg, lbl = ecg.to(device), lbl.to(device)
            opt_a.zero_grad()
            model_a(ecg=ecg, label=lbl)["loss"].backward()
            opt_a.step()
    results["A_no_aug"] = evaluate(model_a, val_loader, device)
    print(f"  AUROC={results['A_no_aug']['auroc']:.3f}  "
          f"AUPRC={results['A_no_aug']['auprc']:.3f}  "
          f"ACC={results['A_no_aug']['accuracy']:.3f}")

    # ---- Configuration B: fixed Gaussian noise ----
    print("\n Ablation B: Fixed Gaussian Noise ")

    class FixedNoiseModel(nn.Module):
        def __init__(self, noise_std: float = 0.1) -> None:
            super().__init__()
            self.noise_std = noise_std
            self.backbone = _ResNet1D(in_channels=12, num_classes=1)

        def forward(self, ecg: torch.Tensor, label: Optional[torch.Tensor] = None):
            if self.training:
                ecg = ecg + self.noise_std * torch.randn_like(ecg)
            logits = self.backbone(ecg)
            y_prob = torch.sigmoid(logits)
            out = {"logit": logits, "y_prob": y_prob}
            if label is not None:
                out["loss"] = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), label.float()
                )
                out["y_true"] = label
            return out

    torch.manual_seed(42)
    model_b = FixedNoiseModel().to(device)
    opt_b = optim.Adam(model_b.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model_b.train()
        for ecg, lbl in train_loader:
            ecg, lbl = ecg.to(device), lbl.to(device)
            opt_b.zero_grad()
            model_b(ecg=ecg, label=lbl)["loss"].backward()
            opt_b.step()
    results["B_fixed_noise"] = evaluate(model_b, val_loader, device)
    print(f"  AUROC={results['B_fixed_noise']['auroc']:.3f}  "
          f"AUPRC={results['B_fixed_noise']['auprc']:.3f}  "
          f"ACC={results['B_fixed_noise']['accuracy']:.3f}")

    # ---- Configurations C & D: TaskAug 1-stage and 2-stage ----
    for stages, key, label in [(1, "C_taskaug_1stage", "TaskAug 1-stage"),
                                (2, "D_taskaug_2stage", "TaskAug 2-stage (paper)")]:
        print(f"\n Ablation {key[0]}: {label} ")
        torch.manual_seed(42)
        mock_ds = _make_mock_dataset()
        model_x = TaskAugResNet(mock_ds, policy_stages=stages).to(device)
        trainer = BiLevelTrainer(model_x, lr_inner=1e-3, lr_outer=1e-3, device=device)
        trainer.fit(train_loader, val_loader, epochs=epochs)
        results[key] = evaluate(model_x, val_loader, device)
        print(f"  AUROC={results[key]['auroc']:.3f}  "
              f"AUPRC={results[key]['auprc']:.3f}  "
              f"ACC={results[key]['accuracy']:.3f}")

    # ---- Configuration E: frozen policy (random init, never updated) ----
    print("\nAblation E: Frozen Policy")
    torch.manual_seed(42)
    mock_ds = _make_mock_dataset()
    model_e = TaskAugResNet(mock_ds, policy_stages=2).to(device)
    for p in model_e.policy.parameters():
        p.requires_grad_(False)
    opt_e = optim.Adam(model_e.backbone_parameters(), lr=1e-3)
    for epoch in range(epochs):
        model_e.train()
        for ecg, lbl in train_loader:
            ecg, lbl = ecg.to(device), lbl.to(device)
            opt_e.zero_grad()
            model_e(ecg=ecg, label=lbl)["loss"].backward()
            opt_e.step()
    results["E_frozen_policy"] = evaluate(model_e, val_loader, device)
    print(f"  AUROC={results['E_frozen_policy']['auroc']:.3f}  "
          f"AUPRC={results['E_frozen_policy']['auprc']:.3f}  "
          f"ACC={results['E_frozen_policy']['accuracy']:.3f}")

    # ---- Configuration F: shared magnitudes (no class-specific mu) ----
    print("\nAblation F: Shared Magnitudes")
    torch.manual_seed(42)
    mock_ds = _make_mock_dataset()
    model_f = TaskAugResNet(mock_ds, policy_stages=2, shared_magnitudes=True).to(device)
    trainer_f = BiLevelTrainer(model_f, lr_inner=1e-3, lr_outer=1e-3, device=device)
    trainer_f.fit(train_loader, val_loader, epochs=epochs)
    results["F_shared_mag"] = evaluate(model_f, val_loader, device)
    print(f"  AUROC={results['F_shared_mag']['auroc']:.3f}  "
          f"AUPRC={results['F_shared_mag']['auprc']:.3f}  "
          f"ACC={results['F_shared_mag']['accuracy']:.3f}")

    # ---- Summary table ----
    print("\n" + "=" * 72)
    print(f"{'Configuration':<30}  {'AUROC':>7}  {'AUPRC':>7}  {'Accuracy':>9}")
    print("-" * 72)
    names = {
        "A_no_aug": "A. No augmentation",
        "B_fixed_noise": "B. Fixed Gaussian noise",
        "C_taskaug_1stage": "C. TaskAug K=1",
        "D_taskaug_2stage": "D. TaskAug K=2 (paper)",
        "E_frozen_policy": "E. Frozen policy",
        "F_shared_mag": "F. Shared magnitudes",
    }
    for key, display in names.items():
        r = results[key]
        print(f"  {display:<28}  {r['auroc']:>7.3f}  {r['auprc']:>7.3f}  {r['accuracy']:>9.3f}")
    print("=" * 72)

    return results


# ---------------------------------------------------------------------------
# Learning-rate sweep
# ---------------------------------------------------------------------------

def run_lr_sweep(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
) -> Dict[str, Dict]:
    """Sweep outer-loop learning rate for TaskAug K=2.

    Tests lr_outer in {1e-2, 1e-3, 1e-4} while keeping lr_inner=1e-3 fixed.
    Demonstrates sensitivity to the outer-loop learning rate — the paper
    uses 1e-3 as the default.

    Args:
        train_ds: Training TensorDataset.
        val_ds: Validation TensorDataset.
        device: Compute device.
        epochs: Training epochs per configuration.
        batch_size: Mini-batch size.

    Returns:
        Dict mapping lr description to metric dicts.
    """
    from pyhealth.models.taskaug_resnet import TaskAugResNet

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    results: Dict[str, Dict] = {}
    for lr_outer in [1e-2, 1e-3, 1e-4]:
        key = f"lr_outer={lr_outer}"
        print(f"\n LR Sweep: {key} ")
        torch.manual_seed(42)
        mock_ds = _make_mock_dataset()
        model = TaskAugResNet(mock_ds, policy_stages=2).to(device)
        trainer = BiLevelTrainer(model, lr_inner=1e-3, lr_outer=lr_outer, device=device)
        trainer.fit(train_loader, val_loader, epochs=epochs)
        results[key] = evaluate(model, val_loader, device)
        print(f"  AUROC={results[key]['auroc']:.3f}  "
              f"AUPRC={results[key]['auprc']:.3f}  "
              f"ACC={results[key]['accuracy']:.3f}")

    print("\n" + "=" * 62)
    print(f"{'lr_outer':<20}  {'AUROC':>7}  {'AUPRC':>7}  {'Accuracy':>9}")
    print("-" * 62)
    for key, r in results.items():
        print(f"  {key:<18}  {r['auroc']:>7.3f}  {r['auprc']:>7.3f}  {r['accuracy']:>9.3f}")
    print("=" * 62)

    return results


# ---------------------------------------------------------------------------
# Helper — mock dataset for TaskAugResNet in ablation
# ---------------------------------------------------------------------------

def _make_mock_dataset():
    from unittest.mock import MagicMock

    ds = MagicMock()
    ds.input_schema = {"ecg": "tensor"}
    ds.output_schema = {"label": "binary"}
    proc = MagicMock()
    proc.size.return_value = 1
    ds.output_processors = {"label": proc}
    return ds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PTB-XL ECG classification with TaskAug + ResNet-18"
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to PTB-XL root directory (omit to use synthetic data)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data regardless of --data_root"
    )
    parser.add_argument(
        "--task", choices=["MI", "HYP", "STTC", "CD"], default="MI"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "bilevel", "ablation", "lr_sweep"],
        default="ablation",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--policy_stages", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    if args.synthetic or args.data_root is None:
        print("Using synthetic data (200 train / 50 val samples).")
        train_ds, val_ds = make_synthetic_dataset()
    else:
        print(f"Loading PTB-XL from {args.data_root}, task={args.task}")
        train_ds, val_ds = make_ptbxl_dataset(args.data_root, task_label=args.task)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- Run ----
    if args.mode == "ablation":
        run_ablation(train_ds, val_ds, device, epochs=args.epochs,
                     batch_size=args.batch_size)
        return

    if args.mode == "lr_sweep":
        run_lr_sweep(train_ds, val_ds, device, epochs=args.epochs,
                     batch_size=args.batch_size)
        return

    mock_ds = _make_mock_dataset()
    from pyhealth.models.taskaug_resnet import TaskAugResNet

    model = TaskAugResNet(mock_ds, policy_stages=args.policy_stages)

    t0 = time.time()
    if args.mode == "standard":
        train_standard(model, train_loader, val_loader,
                       epochs=args.epochs, lr=args.lr, device=device)
    else:  # bilevel
        trainer = BiLevelTrainer(model, lr_inner=args.lr, lr_outer=args.lr,
                                  device=device)
        trainer.fit(train_loader, val_loader, epochs=args.epochs)

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
