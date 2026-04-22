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
    python examples/gdsc_drug_sensitivity_prediction_cadre.py \\
        --demo
"""

import argparse
import os
import random
import tempfile
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
    dataset = GDSCDataset(data_dir=data_dir)
    dataset.summary()

    sample_ds = dataset.set_task(DrugSensitivityPredictionGDSC())

    n = len(sample_ds)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.6)
    n_val = int(n * 0.8)

    train_ds = Subset(sample_ds, idx[:n_train].tolist())
    val_ds = Subset(sample_ds, idx[n_train:n_val].tolist())
    test_ds = Subset(sample_ds, idx[n_val:].tolist())

    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return dataset, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Step 2: Missing value imputation (paper Section 4.2)
# ---------------------------------------------------------------------------


def fill_mask_training(subset: Subset) -> None:
    """Fill missing drug labels with per-drug mode on the training split.

    Per Section 4.2: 'if the sensitivity of a cell line to a drug was missing,
    we filled the missing value with the mode of the available sensitivities
    to this specific drug.'

    Modifies samples in-place; sets mask to all 1s after filling.

    Args:
        subset: Training split as a ``torch.utils.data.Subset``.
    """
    samples = [subset.dataset[i] for i in subset.indices]
    num_drugs = len(samples[0]["labels"])

    labels = np.array([s["labels"] for s in samples], dtype=np.float32)
    masks = np.array([s["mask"] for s in samples], dtype=np.float32)

    for d in range(num_drugs):
        tested = masks[:, d] == 1
        if tested.sum() == 0:
            continue
        fill_val = 1 if labels[tested, d].sum() > (tested.sum() / 2) else 0
        labels[~tested, d] = fill_val

    for i, idx in enumerate(subset.indices):
        subset.dataset.samples[idx]["labels"] = labels[i].astype(int).tolist()
        subset.dataset.samples[idx]["mask"] = [1] * num_drugs


# ---------------------------------------------------------------------------
# Step 3: OneCycle LR/momentum scheduler (paper Section 3.5)
# ---------------------------------------------------------------------------


class OneCycle:
    """1-Cycle policy: warm-up (45%) → cool-down (45%) → annihilation (10%).

    LR schedule:       η/10  →  η  →  η/10  →  η/100
    Momentum schedule: 0.95  → 0.85  → 0.95

    Args:
        total_steps: Total number of optimiser steps.
        max_lr: Peak learning rate (η).
        div: Divisor for initial and final LR.  Default: ``10``.
        prcnt: Percentage of steps used for the annihilation phase.
            Default: ``10``.
        momentum_vals: (high, low) momentum bounds.  Default: ``(0.95, 0.85)``.
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
        self.high_mom, self.low_mom = momentum_vals
        self.iteration = 0

    def step(self) -> Tuple[float, float]:
        """Return (lr, momentum) for current step, then advance.

        Returns:
            Tuple of (learning_rate, momentum) for this step.
        """
        self.iteration += 1
        return self._calc_lr(), self._calc_mom()

    def _calc_lr(self) -> float:
        it = self.iteration
        if it > 2 * self.step_len:
            ratio = (it - 2 * self.step_len) / (
                self.total_steps - 2 * self.step_len
            )
            return self.max_lr / self.div * (1 - ratio * (1 - 1 / self.div))
        elif it > self.step_len:
            ratio = 1 - (it - self.step_len) / self.step_len
            return self.max_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = it / self.step_len
            return self.max_lr * (1 + ratio * (self.div - 1)) / self.div

    def _calc_mom(self) -> float:
        it = self.iteration
        if it > 2 * self.step_len:
            return self.high_mom
        elif it > self.step_len:
            ratio = (it - self.step_len) / self.step_len
            return self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = it / self.step_len
            return self.high_mom - ratio * (self.high_mom - self.low_mom)


# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: CADRE,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataloader; returns dict with F1, Acc, AUROC, AUPR.

    Args:
        model: Trained CADRE model.
        dataloader: DataLoader for the split to evaluate.
        device: Compute device.

    Returns:
        Dict with keys ``f1``, ``accuracy``, ``precision``, ``recall``,
        ``auroc``, ``aupr`` (all in [0, 1]).
    """
    model.eval()
    all_labels, all_probs, all_masks = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            out = model(batch["gene_indices"].to(device))
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())

    labels = np.concatenate(all_labels).flatten()
    probs = np.concatenate(all_probs).flatten()
    masks = np.concatenate(all_masks).flatten()

    y_true = labels[masks == 1]
    y_prob = probs[masks == 1]
    y_pred = (y_prob >= 0.5).astype(float)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "aupr": auc(rec_curve, prec_curve),
    }


# ---------------------------------------------------------------------------
# Step 5: Single training run
# ---------------------------------------------------------------------------


def run_one(
    dataset: GDSCDataset,
    train_ds: Subset,
    val_ds: Subset,
    test_ds: Subset,
    device: torch.device,
    batch_size: int,
    max_iter: int,
    embedding_dim: int = 200,
    attention_size: int = 128,
    attention_head: int = 8,
    dropout_rate: float = 0.6,
    use_attention: bool = True,
    use_cntx_attn: bool = True,
    freeze_gene_emb: bool = True,
    label: str = "CADRE",
) -> Dict[str, float]:
    """Train one model configuration; returns test metrics.

    Args:
        dataset: Loaded GDSCDataset (supplies embeddings and pathway info).
        train_ds: Training split (after fill_mask_training has been called).
        val_ds: Validation split.
        test_ds: Test split.
        device: Compute device.
        batch_size: Mini-batch size.
        max_iter: Total number of training samples (steps × batch_size).
        embedding_dim: Gene and drug embedding dimension.
        attention_size: Attention hidden dimension.
        attention_head: Number of attention heads.
        dropout_rate: Dropout probability.
        use_attention: Enable multi-head attention in the encoder.
        use_cntx_attn: Enable contextual (pathway) conditioning.
        freeze_gene_emb: Freeze Gene2Vec weights during training.
        label: Display name for progress output.

    Returns:
        Dict with test-set ``f1``, ``accuracy``, ``auroc``, ``aupr``.
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=cadre_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=cadre_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=cadre_collate_fn
    )

    pw_info = dataset.get_pathway_info()
    model = CADRE(
        gene_embeddings=dataset.get_gene_embeddings(),
        num_drugs=len(dataset.drug_ids),
        num_pathways=pw_info["num_pathways"],
        drug_pathway_ids=pw_info["drug_pathway_ids"],
        embedding_dim=embedding_dim,
        attention_size=attention_size,
        attention_head=attention_head,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        use_cntx_attn=use_cntx_attn,
        freeze_gene_emb=freeze_gene_emb,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{label}] Trainable parameters: {trainable:,}")

    optimizer = optim.SGD(
        model.parameters(), lr=0.3, momentum=0.95, weight_decay=3e-4
    )
    total_steps = max_iter // batch_size
    scheduler = OneCycle(total_steps, max_lr=0.3)
    total_epochs = (total_steps + len(train_loader) - 1) // len(train_loader)

    global_step = 0
    best_val_f1 = 0.0
    best_state: Optional[dict] = None

    for epoch in range(total_epochs):
        model.train()
        losses = []

        for batch in train_loader:
            if global_step >= total_steps:
                break

            lr, mom = scheduler.step()
            for pg in optimizer.param_groups:
                pg["lr"], pg["momentum"] = lr, mom

            optimizer.zero_grad()
            out = model(
                batch["gene_indices"].to(device),
                labels=batch["labels"].to(device),
                mask=batch["mask"].to(device),
            )
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())
            global_step += 1

        if (epoch + 1) % 10 == 0 or global_step >= total_steps:
            val_m = evaluate(model, val_loader, device)
            print(
                f"  [Epoch {epoch + 1:3d} | Step {global_step:5d}] "
                f"loss={np.mean(losses):.4f}  "
                f"val F1={100 * val_m['f1']:.1f}  "
                f"AUROC={100 * val_m['auroc']:.1f}"
            )
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

        if global_step >= total_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    test_m = evaluate(model, test_loader, device)
    return test_m


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
    dict(
        label="CADRE",
        use_attention=True,
        use_cntx_attn=True,
        embedding_dim=200,
        freeze_gene_emb=True,
    ),
    dict(
        label="SADRE (no pathway ctx)",
        use_attention=True,
        use_cntx_attn=False,
        embedding_dim=200,
        freeze_gene_emb=True,
    ),
    dict(
        label="ADRE (mean pooling)",
        use_attention=False,
        use_cntx_attn=False,
        embedding_dim=200,
        freeze_gene_emb=True,
    ),
    dict(
        label="CADRE-100 (emb=100)",
        use_attention=True,
        use_cntx_attn=True,
        embedding_dim=100,
        freeze_gene_emb=True,
    ),
    dict(
        label="CADRE-free (trainable emb)",
        use_attention=True,
        use_cntx_attn=True,
        embedding_dim=200,
        freeze_gene_emb=False,
    ),
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

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    dataset, train_ds, val_ds, test_ds = load_data(args.data_dir, args.seed)
    fill_mask_training(train_ds)

    results = []
    for cfg in ABLATION_CONFIGS:
        # Reset seeds before each run for fair comparison.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        m = run_one(
            dataset=dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            embedding_dim=cfg.get("embedding_dim", 200),
            use_attention=cfg["use_attention"],
            use_cntx_attn=cfg["use_cntx_attn"],
            freeze_gene_emb=cfg.get("freeze_gene_emb", True),
            label=cfg["label"],
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
    - 10 cell lines, 20 genes, 5 drugs, 3 pathways
    - ~30 % missing drug labels

    Args:
        tmp_dir: Directory to write CSV files into.
    """
    rng = np.random.RandomState(0)
    n_cells, n_genes, n_drugs = 10, 20, 5

    # Binary gene expression matrix (cell lines × genes)
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
    emb[0] = 0.0  # padding row
    np.savetxt(os.path.join(tmp_dir, "exp_emb_gdsc.csv"), emb, delimiter=",")


def demo(args: argparse.Namespace) -> None:
    """Smoke-test the full pipeline with synthetic data (fast, no real data).

    Trains for a small number of steps to verify that the data loading,
    model forward pass, and evaluation code all run without errors.

    Args:
        args: Parsed CLI arguments (only ``seed`` and ``batch_size`` are used).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_synthetic_gdsc(tmp_dir)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = torch.device("cpu")
        dataset, train_ds, val_ds, test_ds = load_data(tmp_dir, seed=args.seed)
        fill_mask_training(train_ds)

        m = run_one(
            dataset=dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
            batch_size=args.batch_size,
            # 10 mini-batches total — fast smoke test
            max_iter=args.batch_size * 10,
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

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    dataset, train_ds, val_ds, test_ds = load_data(args.data_dir, args.seed)
    fill_mask_training(train_ds)

    pw_info = dataset.get_pathway_info()
    test_m = run_one(
        dataset=dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        device=device,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        label="CADRE",
    )

    print("\n=== Test Results ===")
    print(f"F1:        {100 * test_m['f1']:.2f}  (paper: 64.3)")
    print(f"Accuracy:  {100 * test_m['accuracy']:.2f}  (paper: 78.6)")
    print(f"AUROC:     {100 * test_m['auroc']:.2f}  (paper: 83.4)")
    print(f"AUPR:      {100 * test_m['aupr']:.2f}  (paper: 70.6)")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            {"pathway_info": pw_info},
            os.path.join(args.output_dir, "cadre_gdsc.pt"),
        )
        print(f"\nCheckpoint saved to {args.output_dir}/cadre_gdsc.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GDSC drug sensitivity prediction with CADRE"
    )
    parser.add_argument(
        "--data_dir",
        default="originalData",
        help="Path to GDSC CSV files",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save checkpoint (single-run mode only)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_iter",
        type=int,
        default=48000,
        help="Total training iterations (paper: 48k)",
    )
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all 5 ablation variants and print comparison table",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Smoke-test the pipeline with synthetic data (no real data needed)",
    )
    args = parser.parse_args()

    if args.demo:
        demo(args)
    elif args.ablation:
        ablation_study(args)
    else:
        train(args)
