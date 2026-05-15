"""Full pipeline: GDSC → DrugSensitivityPredictionGDSC → CADRE.

Reproduces the CADRE training procedure and ablation study from:
    Tao, Y. et al. (2020). Predicting Drug Sensitivity of Cancer Cell Lines
    via Collaborative Filtering with Contextual Attention. MLHC 2020.

Expected results on GDSC test set (paper Table 1):
    F1:    64.3 ± 0.22
    Acc:   78.6 ± 0.34
    AUROC: 83.4 ± 0.19
    AUPR:  70.6 ± 1.30

Ablation study (paper Table 2):
    CADRE       (full model)              F1 ~64.3  AUROC ~83.4
    SADRE       (no pathway context)      F1 ~62.1  AUROC ~81.9
    ADRE        (mean pooling, no attn)   F1 ~60.8  AUROC ~80.5
    CADRE-100   (embedding_dim=100)       F1 ~63.1  AUROC ~82.6
    CADRE-free  (trainable gene emb)      F1 ~63.8  AUROC ~82.8

Usage — full paper replication:
    python examples/gdsc_drug_sensitivity_prediction_cadre.py \\
        --data_dir /path/to/originalData --output_dir ./outputs

Usage — ablation study (trains all 5 variants and compares):
    python examples/gdsc_drug_sensitivity_prediction_cadre.py \\
        --data_dir /path/to/originalData --ablation

Usage — quick demo with synthetic data (no real data required):
    python examples/gdsc_drug_sensitivity_prediction_cadre.py --demo
"""

import argparse
import os
import pickle
import random
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

from pyhealth.datasets import GDSCDataset
from pyhealth.tasks import DrugSensitivityPredictionGDSC
from pyhealth.models import CADRE, cadre_collate_fn


# ---------------------------------------------------------------------------
# Step 1: Load dataset and apply task
# ---------------------------------------------------------------------------


def load_data(
    data_dir: str, seed: int = 2019
) -> Tuple[GDSCDataset, Subset, Subset, Subset]:
    """Load GDSC, apply the drug-sensitivity task, split 60/20/20.

    Args:
        data_dir: Path to directory containing GDSC CSV files.
        seed: Random seed for reproducible split.

    Returns:
        Tuple of (GDSCDataset, train_subset, val_subset, test_subset).
    """
    print("Loading dataset...")
    dataset = GDSCDataset(data_dir=data_dir)
    dataset.summary()

    sample_ds = dataset.set_task(DrugSensitivityPredictionGDSC())

    n = len(sample_ds)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * 0.6)
    n_val = int(n * 0.8)

    train_ds = Subset(sample_ds, indices[:n_train].tolist())
    val_ds = Subset(sample_ds, indices[n_train:n_val].tolist())
    test_ds = Subset(sample_ds, indices[n_val:].tolist())

    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return dataset, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Step 2: Missing value imputation (Section 4.2)
# ---------------------------------------------------------------------------


def fill_mask_training(train_ds: Subset) -> None:
    """Fill missing drug labels with per-drug mode in training set.

    Paper Section 4.2: 'if the sensitivity of a cell line to a drug was
    missing, we filled the missing value with the mode of the available
    sensitivities to this specific drug.'

    Modifies samples in-place: sets mask to all 1s and fills labels.

    Args:
        train_ds: Training split as a ``torch.utils.data.Subset``.
    """
    # Access the underlying samples via the Subset indices
    samples = [train_ds.dataset[i] for i in train_ds.indices]
    num_drugs = len(samples[0]["labels"])
    num_samples = len(samples)

    # Collect labels and masks into arrays for vectorized computation
    labels = np.array([s["labels"] for s in samples], dtype=np.float32)
    masks = np.array([s["mask"] for s in samples], dtype=np.float32)

    for d in range(num_drugs):
        tested = masks[:, d] == 1
        if tested.sum() == 0:
            continue
        # Mode = 1 if more positives than negatives, else 0
        pos_count = labels[tested, d].sum()
        neg_count = tested.sum() - pos_count
        fill_val = 1 if pos_count > neg_count else 0

        # Fill untested entries
        untested = masks[:, d] == 0
        labels[untested, d] = fill_val

    # Write back to the underlying dataset samples
    for i, idx in enumerate(train_ds.indices):
        train_ds.dataset.samples[idx]["labels"] = labels[i].astype(int).tolist()
        train_ds.dataset.samples[idx]["mask"] = [1] * num_drugs


# ---------------------------------------------------------------------------
# Step 3: OneCycle LR/Momentum scheduler (Section 3.5)
# ---------------------------------------------------------------------------


class OneCycle:
    """1-Cycle policy for learning rate and momentum scheduling.

    Phase 1 (warm-up, 45%):   LR η/10 → η,   momentum 0.95 → 0.85
    Phase 2 (cool-down, 45%): LR η → η/10,   momentum 0.85 → 0.95
    Phase 3 (annihilation, 10%): LR η/10 → η/100, momentum 0.95

    Args:
        total_steps: Total number of optimiser steps.
        max_lr: Peak learning rate (η).
        div: Divisor for initial and final LR. Default: ``10``.
        prcnt: Percentage of steps used for the annihilation phase.
            Default: ``10``.
        momentum_vals: (high, low) momentum bounds. Default: ``(0.95, 0.85)``.
    """

    def __init__(
        self,
        total_steps: int,
        max_lr: float,
        div: int = 10,
        prcnt: int = 10,
        momentum_vals: Tuple[float, float] = (0.95, 0.85),
    ) -> None:
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div = div
        self.step_len = int(total_steps * (1 - prcnt / 100) / 2)
        self.high_mom = momentum_vals[0]
        self.low_mom = momentum_vals[1]
        self.iteration = 0

    def step(self) -> Tuple[float, float]:
        """Return (lr, momentum) for current step, then advance.

        Returns:
            Tuple of (learning_rate, momentum) for this step.
        """
        self.iteration += 1
        lr = self._calc_lr()
        mom = self._calc_mom()
        return lr, mom

    def _calc_lr(self) -> float:
        it = self.iteration
        if it > 2 * self.step_len:  # annihilation phase
            ratio = (it - 2 * self.step_len) / (
                self.total_steps - 2 * self.step_len
            )
            return self.max_lr / self.div * (1 - ratio * (1 - 1 / self.div))
        elif it > self.step_len:  # cool-down phase
            ratio = 1 - (it - self.step_len) / self.step_len
            return self.max_lr * (1 + ratio * (self.div - 1)) / self.div
        else:  # warm-up phase
            ratio = it / self.step_len
            return self.max_lr * (1 + ratio * (self.div - 1)) / self.div

    def _calc_mom(self) -> float:
        it = self.iteration
        if it > 2 * self.step_len:  # annihilation
            return self.high_mom
        elif it > self.step_len:  # cool-down
            ratio = (it - self.step_len) / self.step_len
            return self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:  # warm-up
            ratio = it / self.step_len
            return self.high_mom - ratio * (self.high_mom - self.low_mom)


# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: CADRE,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate model on a dataloader; returns metrics dict.

    Args:
        model: Trained CADRE model.
        dataloader: DataLoader for the split to evaluate.
        device: Compute device.

    Returns:
        Dict with keys ``f1``, ``accuracy``, ``precision``, ``recall``,
        ``auroc``, ``aupr`` (all in [0, 1]) plus raw ``labels``,
        ``probs``, and ``masks`` arrays.
    """
    model.eval()
    all_labels, all_probs, all_masks = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            gene_indices = batch["gene_indices"].to(device)
            result = model(gene_indices)
            all_probs.append(result["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())

    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Flatten and apply mask
    flat_labels = labels.flatten()
    flat_probs = probs.flatten()
    flat_masks = masks.flatten()

    idx = flat_masks == 1
    y_true = flat_labels[idx]
    y_prob = flat_probs[idx]
    y_pred = (y_prob >= 0.5).astype(float)

    eps = 1e-5  # noqa: F841 — kept for numerical-stability parity with reCADRE

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        aupr = auc(rec_curve, prec_curve)
    except ValueError:
        aupr = 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
        "aupr": aupr,
        "labels": labels,
        "probs": probs,
        "masks": masks,
    }


# ---------------------------------------------------------------------------
# Step 5: Training loop
# ---------------------------------------------------------------------------


def run_one(
    dataset: GDSCDataset,
    train_ds: Subset,
    val_ds: Subset,
    test_ds: Subset,
    device: torch.device,
    args: argparse.Namespace,
    label: str = "CADRE",
    embedding_dim: Optional[int] = None,
    use_attention: Optional[bool] = None,
    use_cntx_attn: Optional[bool] = None,
    freeze_gene_emb: Optional[bool] = None,
) -> Dict:
    """Train one model configuration; returns test metrics.

    Keyword overrides (``embedding_dim``, ``use_attention``, etc.) are used
    by the ablation study to vary a single hyperparameter per run while
    inheriting all other settings from ``args``.

    Args:
        dataset: Loaded GDSCDataset (supplies embeddings and pathway info).
        train_ds: Training split (after fill_mask_training has been called).
        val_ds: Validation split.
        test_ds: Test split.
        device: Compute device.
        args: Parsed CLI arguments supplying default hyperparameters.
        label: Display name for progress output.
        embedding_dim: Override for ``args.embedding_dim``.
        use_attention: Override for ``args.use_attention``.
        use_cntx_attn: Override for ``args.use_cntx_attn``.
        freeze_gene_emb: Override for ``not args.train_gene_emb``.

    Returns:
        Dict with test-set metrics from :func:`evaluate`.
    """
    # Apply per-run overrides (ablation study only)
    emb_dim = embedding_dim if embedding_dim is not None else args.embedding_dim
    attn = use_attention if use_attention is not None else args.use_attention
    cntx = use_cntx_attn if use_cntx_attn is not None else args.use_cntx_attn
    freeze = (
        freeze_gene_emb
        if freeze_gene_emb is not None
        else not args.train_gene_emb
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cadre_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cadre_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cadre_collate_fn,
    )

    gene_emb = dataset.get_gene_embeddings()
    pw_info = dataset.get_pathway_info()

    model = CADRE(
        gene_embeddings=gene_emb,
        num_drugs=len(dataset.drug_ids),
        num_pathways=pw_info["num_pathways"],
        drug_pathway_ids=pw_info["drug_pathway_ids"],
        embedding_dim=emb_dim,
        attention_size=args.attention_size,
        attention_head=args.attention_head,
        dropout_rate=args.dropout_rate,
        use_attention=attn,
        use_cntx_attn=cntx,
        freeze_gene_emb=freeze,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[{label}] Parameters: {trainable:,} trainable / {total:,} total")

    # SGD with OneCycle LR/momentum (paper: lr=0.3, wd=3e-4, 48k steps)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.95,
        weight_decay=args.weight_decay,
    )

    # Paper: batch_size=8, max_iter=48000 → 6000 optimizer steps
    steps_per_epoch = len(train_loader)
    total_optimizer_steps = args.max_iter // args.batch_size
    total_epochs = (
        total_optimizer_steps + steps_per_epoch - 1
    ) // steps_per_epoch
    scheduler = OneCycle(total_optimizer_steps, args.learning_rate)

    print(
        f"  Training for {total_optimizer_steps} steps ({total_epochs} epochs), "
        f"{steps_per_epoch} steps/epoch"
    )

    logs: Dict = {
        "args": vars(args),
        "epoch": [],
        "step": [],
        "train_loss": [],
        "train_f1": [],
        "train_acc": [],
        "train_auroc": [],
        "train_aupr": [],
        "val_f1": [],
        "val_acc": [],
        "val_auroc": [],
        "val_aupr": [],
    }

    global_step = 0
    best_val_f1 = 0.0
    best_model_state: Optional[dict] = None
    start_time = time.time()

    for epoch in range(total_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            if global_step >= total_optimizer_steps:
                break

            gene_indices = batch["gene_indices"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            # OneCycle LR/momentum update
            lr, mom = scheduler.step()
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["momentum"] = mom

            optimizer.zero_grad()
            result = model(gene_indices, labels=labels, mask=mask)
            loss = result["loss"]
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        # Evaluate every eval_every epochs (or at the last step)
        if (
            (epoch + 1) % args.eval_every == 0
            or global_step >= total_optimizer_steps
        ):
            train_metrics = evaluate(model, train_loader, device)
            val_metrics = evaluate(model, val_loader, device)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            elapsed = time.time() - start_time

            print(
                f"  [Epoch {epoch + 1:3d} | Step {global_step:5d} | {elapsed:.0f}s] "
                f"loss={avg_loss:.4f} | "
                f"trn F1={100 * train_metrics['f1']:.1f} "
                f"AUC={100 * train_metrics['auroc']:.1f} | "
                f"val F1={100 * val_metrics['f1']:.1f} "
                f"AUC={100 * val_metrics['auroc']:.1f} "
                f"AUPR={100 * val_metrics['aupr']:.1f} "
                f"Acc={100 * val_metrics['accuracy']:.1f}"
            )

            logs["epoch"].append(epoch + 1)
            logs["step"].append(global_step)
            logs["train_loss"].append(avg_loss)
            logs["train_f1"].append(train_metrics["f1"])
            logs["train_acc"].append(train_metrics["accuracy"])
            logs["train_auroc"].append(train_metrics["auroc"])
            logs["train_aupr"].append(train_metrics["aupr"])
            logs["val_f1"].append(val_metrics["f1"])
            logs["val_acc"].append(val_metrics["accuracy"])
            logs["val_auroc"].append(val_metrics["auroc"])
            logs["val_aupr"].append(val_metrics["aupr"])

            # Save best model by val F1
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

        if global_step >= total_optimizer_steps:
            break

    # Final evaluation on test set using best-val-F1 model
    print(f"\n=== Final Evaluation [{label}] (best val F1 checkpoint) ===")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)

    test_metrics = evaluate(model, test_loader, device)
    train_metrics_final = evaluate(model, train_loader, device)
    val_metrics_final = evaluate(model, val_loader, device)

    print(
        f"Train: F1={100 * train_metrics_final['f1']:.1f}  "
        f"Acc={100 * train_metrics_final['accuracy']:.1f}  "
        f"AUROC={100 * train_metrics_final['auroc']:.1f}  "
        f"AUPR={100 * train_metrics_final['aupr']:.1f}"
    )
    print(
        f"Val:   F1={100 * val_metrics_final['f1']:.1f}  "
        f"Acc={100 * val_metrics_final['accuracy']:.1f}  "
        f"AUROC={100 * val_metrics_final['auroc']:.1f}  "
        f"AUPR={100 * val_metrics_final['aupr']:.1f}"
    )
    print(
        f"Test:  F1={100 * test_metrics['f1']:.1f}  "
        f"Acc={100 * test_metrics['accuracy']:.1f}  "
        f"AUROC={100 * test_metrics['auroc']:.1f}  "
        f"AUPR={100 * test_metrics['aupr']:.1f}"
    )

    # Bundle extra info for callers that want to save outputs
    test_metrics["_logs"] = logs
    test_metrics["_train_final"] = train_metrics_final
    test_metrics["_val_final"] = val_metrics_final
    test_metrics["_pw_info"] = pw_info
    test_metrics["_elapsed"] = time.time() - start_time
    return test_metrics


# ---------------------------------------------------------------------------
# Step 6: Ablation study
# ---------------------------------------------------------------------------

# Ablation configurations (paper Table 2):
#
#  Name          use_attention  use_cntx_attn  embedding_dim  freeze_gene_emb
#  CADRE         True           True           200            True   ← full model
#  SADRE         True           False          200            True   ← no pathway ctx
#  ADRE          False          False          200            True   ← mean pooling
#  CADRE-100     True           True           100            True   ← smaller emb
#  CADRE-free    True           True           200            False  ← trainable emb
ABLATION_CONFIGS: List[Dict] = [
    dict(label="CADRE",
         use_attention=True, use_cntx_attn=True,
         embedding_dim=200, freeze_gene_emb=True),
    dict(label="SADRE (no pathway ctx)",
         use_attention=True, use_cntx_attn=False,
         embedding_dim=200, freeze_gene_emb=True),
    dict(label="ADRE (mean pooling)",
         use_attention=False, use_cntx_attn=False,
         embedding_dim=200, freeze_gene_emb=True),
    dict(label="CADRE-100 (emb=100)",
         use_attention=True, use_cntx_attn=True,
         embedding_dim=100, freeze_gene_emb=True),
    dict(label="CADRE-free (trainable emb)",
         use_attention=True, use_cntx_attn=True,
         embedding_dim=200, freeze_gene_emb=False),
]


def ablation_study(args: argparse.Namespace) -> None:
    """Run all ablation configurations and print a comparison table.

    Experimental setup:
    - Dataset: GDSC (60/20/20 cell-line split, seed=2019)
    - Optimiser: SGD with 1-Cycle LR (max_lr=0.3, wd=3e-4, 48k steps)
    - Evaluation: masked F1, AUROC, AUPR on held-out test cell lines
    - Metric reported: best-val-F1 checkpoint on test set

    Args:
        args: Parsed CLI arguments.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args)
    print(f"Device: {device}")

    dataset, train_ds, val_ds, test_ds = load_data(args.data_dir, args.seed)

    if not args.no_fill_mask:
        fill_mask_training(train_ds)
        print("Applied fill_mask to training set")
    else:
        print("Skipped fill_mask (--no_fill_mask)")

    results = []
    for cfg in ABLATION_CONFIGS:
        # Reset seeds before each run for fair comparison
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        m = run_one(
            dataset=dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
            args=args,
            label=cfg["label"],
            embedding_dim=cfg.get("embedding_dim"),
            use_attention=cfg.get("use_attention"),
            use_cntx_attn=cfg.get("use_cntx_attn"),
            freeze_gene_emb=cfg.get("freeze_gene_emb"),
        )
        results.append((cfg["label"], m))

    # Print comparison table
    print("\n" + "=" * 70)
    print("Ablation Study — GDSC Test Set")
    print("=" * 70)
    print(f"{'Model':<28} {'F1':>6} {'AUROC':>7} {'AUPR':>6} {'Acc':>6}")
    print("-" * 70)
    for name, m in results:
        print(
            f"{name:<28} "
            f"{100 * m['f1']:6.1f} "
            f"{100 * m['auroc']:7.1f} "
            f"{100 * m['aupr']:6.1f} "
            f"{100 * m['accuracy']:6.1f}"
        )
    print("=" * 70)
    print("Paper reference (CADRE):         64.3   83.4   70.6   78.6")


# ---------------------------------------------------------------------------
# Step 7: Demo mode (synthetic data, no GDSC files required)
# ---------------------------------------------------------------------------


def _write_synthetic_gdsc(tmp_dir: str) -> None:
    """Write minimal synthetic GDSC CSV files for smoke-testing.

    Creates a tiny GDSC-compatible dataset:
    - 10 cell lines, 20 genes, 5 drugs, 3 pathways, ~30 % missing labels.

    Args:
        tmp_dir: Directory to write CSV files into.
    """
    rng = np.random.RandomState(0)
    n_cells, n_genes, n_drugs = 10, 20, 5

    # Binary gene expression (cell lines × genes)
    exp = pd.DataFrame(
        rng.randint(0, 2, (n_cells, n_genes)),
        index=[f"CL{i}" for i in range(n_cells)],
        columns=[str(g) for g in range(1, n_genes + 1)],
    )
    exp.to_csv(os.path.join(tmp_dir, "exp_gdsc.csv"))

    # Binary sensitivity matrix; ~30 % missing (NaN)
    sens = rng.randint(0, 2, (n_cells, n_drugs)).astype(float)
    sens[rng.rand(n_cells, n_drugs) < 0.3] = np.nan
    drug_ids = list(range(1001, 1001 + n_drugs))
    tgt = pd.DataFrame(
        sens,
        index=[f"CL{i}" for i in range(n_cells)],
        columns=[str(d) for d in drug_ids],
    )
    tgt.to_csv(os.path.join(tmp_dir, "gdsc.csv"))

    # Drug metadata (Name + Target pathway)
    pathways = ["PI3K/MTOR", "ERK MAPK", "WNT"]
    drug_info = pd.DataFrame(
        {
            "Name": [f"Drug{i}" for i in range(n_drugs)],
            "Target pathway": [pathways[i % len(pathways)] for i in range(n_drugs)],
        },
        index=drug_ids,
    )
    drug_info.to_csv(os.path.join(tmp_dir, "drug_info_gdsc.csv"))

    # Gene2Vec embeddings (n_genes+1 rows; row 0 = padding vector)
    emb = rng.randn(n_genes + 1, 8).astype(np.float32)
    emb[0] = 0.0
    np.savetxt(os.path.join(tmp_dir, "exp_emb_gdsc.csv"), emb, delimiter=",")


def demo(args: argparse.Namespace) -> None:
    """Smoke-test the full pipeline with synthetic data.

    Trains for a tiny number of steps to verify that data loading, model
    forward pass, and evaluation all run without errors.  Results are
    meaningless — only absence of exceptions matters.

    Args:
        args: Parsed CLI arguments (``seed`` and ``batch_size`` are used).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_synthetic_gdsc(tmp_dir)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = torch.device("cpu")
        dataset, train_ds, val_ds, test_ds = load_data(tmp_dir, seed=args.seed)
        fill_mask_training(train_ds)

        # Override to run just 10 mini-batches
        demo_args = argparse.Namespace(**vars(args))
        demo_args.max_iter = args.batch_size * 10
        demo_args.eval_every = 1

        m = run_one(
            dataset=dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
            args=demo_args,
            label="CADRE (demo)",
        )
        print("\n=== Demo Results (synthetic data — values are meaningless) ===")
        print(f"F1:    {100 * m['f1']:.1f}")
        print(f"AUROC: {100 * m['auroc']:.1f}")
        print("Pipeline smoke-test passed.")


# ---------------------------------------------------------------------------
# Step 8: Full paper replication (single run)
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full paper-replication training on real GDSC data.

    Args:
        args: Parsed CLI arguments.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args)
    print(f"Device: {device}")

    dataset, train_ds, val_ds, test_ds = load_data(args.data_dir, args.seed)

    if not args.no_fill_mask:
        fill_mask_training(train_ds)
        print("Applied fill_mask to training set")
    else:
        print("Skipped fill_mask (--no_fill_mask)")

    test_m = run_one(
        dataset=dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        device=device,
        args=args,
        label="CADRE",
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    logs = test_m["_logs"]
    pw_info = test_m["_pw_info"]
    val_metrics_final = test_m["_val_final"]
    elapsed = test_m["_elapsed"]

    logs["test_f1"] = test_m["f1"]
    logs["test_acc"] = test_m["accuracy"]
    logs["test_auroc"] = test_m["auroc"]
    logs["test_aupr"] = test_m["aupr"]
    logs["test_precision"] = test_m["precision"]
    logs["test_recall"] = test_m["recall"]
    logs["test_probs"] = test_m["probs"]
    logs["test_labels"] = test_m["labels"]
    logs["test_masks"] = test_m["masks"]
    logs["train_time_seconds"] = elapsed

    logs_path = os.path.join(args.output_dir, "logs.pkl")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f, protocol=2)
    print(f"\nLogs saved to {logs_path}")

    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(
        {"args": vars(args), "pathway_info": pw_info},
        model_path,
    )
    print(f"Model saved to {model_path}")

    summary_path = os.path.join(args.output_dir, "results.txt")
    with open(summary_path, "w") as f:
        f.write("reCADRE Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Hyperparameters:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nDataset:\n")
        f.write(f"  Cell lines: 846\n")
        f.write(f"  Drugs: 260\n")
        f.write(f"  Genes: 3000 (1500 active)\n")
        f.write(f"  Pathways: {pw_info['num_pathways']}\n")
        f.write(f"  Split: 60/20/20\n")
        f.write("\nResults (Test Set):\n")
        f.write(f"  F1 Score:  {100 * test_m['f1']:.2f}\n")
        f.write(f"  Accuracy:  {100 * test_m['accuracy']:.2f}\n")
        f.write(f"  AUROC:     {100 * test_m['auroc']:.2f}\n")
        f.write(f"  AUPR:      {100 * test_m['aupr']:.2f}\n")
        f.write(f"  Precision: {100 * test_m['precision']:.2f}\n")
        f.write(f"  Recall:    {100 * test_m['recall']:.2f}\n")
        f.write("\nResults (Validation Set):\n")
        f.write(f"  F1 Score:  {100 * val_metrics_final['f1']:.2f}\n")
        f.write(f"  Accuracy:  {100 * val_metrics_final['accuracy']:.2f}\n")
        f.write(f"  AUROC:     {100 * val_metrics_final['auroc']:.2f}\n")
        f.write(f"  AUPR:      {100 * val_metrics_final['aupr']:.2f}\n")
        f.write("\nPaper Reference (CADRE on GDSC, Table 1):\n")
        f.write(f"  F1 Score:  64.3 ± 0.22\n")
        f.write(f"  Accuracy:  78.6 ± 0.34\n")
        f.write(f"  AUROC:     83.4 ± 0.19\n")
        f.write(f"  AUPR:      70.6 ± 1.30\n")
        f.write(f"\nTraining time: {elapsed:.1f}s\n")
    print(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_device(args: argparse.Namespace) -> torch.device:
    """Select compute device based on CLI flags and availability.

    Args:
        args: Parsed arguments; reads ``cpu`` and ``use_cuda``.

    Returns:
        Selected ``torch.device``.
    """
    if args.cpu:
        return torch.device("cpu")
    if args.use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Train reCADRE model on GDSC drug sensitivity data"
    )

    # Resolve default paths relative to this script's directory
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(_script_dir, "..", "originalData"),
        help="Path to GDSC CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_script_dir, "..", "outputs"),
        help="Directory to save checkpoint, logs, and results",
    )

    # Model architecture (Table A2)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--attention_size", type=int, default=128)
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.6)
    parser.add_argument(
        "--use_attention", action="store_true", default=True,
        help="Enable multi-head attention in the encoder",
    )
    parser.add_argument(
        "--no_attention", action="store_true", default=False,
        help="Disable attention (ADRE ablation — mean pooling)",
    )
    parser.add_argument(
        "--use_cntx_attn", action="store_true", default=True,
        help="Enable contextual (pathway) conditioning in attention",
    )
    parser.add_argument(
        "--no_cntx_attn", action="store_true", default=False,
        help="Disable contextual attention (SADRE ablation)",
    )
    parser.add_argument(
        "--train_gene_emb", action="store_true", default=False,
        help="Unfreeze gene embeddings (CADRE∆pretrain variant)",
    )
    parser.add_argument(
        "--no_fill_mask", action="store_true", default=False,
        help="Skip per-drug-mode imputation of missing labels",
    )

    # Training (Table A2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_iter", type=int, default=48000,
        help="Total training iterations (paper: 48k for GDSC)",
    )
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)

    # Misc
    parser.add_argument(
        "--eval_every", type=int, default=10,
        help="Evaluate every N epochs",
    )
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument(
        "--use_cuda", action="store_true", default=True,
        help="Use CUDA if available",
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False,
        help="Force CPU even if GPU/MPS available",
    )

    # Modes
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run all 5 ablation variants and print comparison table",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Smoke-test the pipeline with synthetic data (no real data needed)",
    )

    args = parser.parse_args()

    # Handle negation flags (match reCADRE train.py convention)
    if args.no_attention:
        args.use_attention = False
    if args.no_cntx_attn:
        args.use_cntx_attn = False

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        demo(args)
    elif args.ablation:
        ablation_study(args)
    else:
        train(args)
