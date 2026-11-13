"""End-to-end demo: TCGA RNA-seq survival prediction with BulkRNABert.

Reproduces the downstream survival prediction pipeline of Gélard et al.
(2025), "BulkRNABert: Cancer prognosis from bulk RNA-seq based language
models" (ML4H 2025). This example runs the full PyHealth contribution:

    TCGARNASeqDataset -> TCGASurvivalPrediction -> BulkRNABert -> Cox loss

Two modes:

1. **Synthetic mode (default, no arguments).** Generates 60 synthetic
   patients in a temporary directory and trains on them. Runs in seconds
   on CPU with no downloads, no TCGA data, no Hugging Face weights.

2. **Real mode (``--artifacts-dir``).** Points at the artifacts produced by
   the reference repo (``pull_data.py`` + ``train.py``) and trains on the
   real 256-d BulkRNABert embeddings for all of TCGA.

**Ablation.** Compares the original global Cox partial likelihood (all
patients in one risk set) against our proposed cohort-restricted version
(risk sets restricted to each cancer cohort), using identical splits,
seeds, and hyperparameters. Reports overall Harrell C-index, weighted
per-cohort C-index, and macro C-index. Both modes run through the same
PyHealth pipeline, so the comparison is pipeline-level, not per-component.

Usage:
    python examples/tcga_rna_seq_tcga_survival_prediction_bulk_rna_bert.py
    python examples/tcga_rna_seq_tcga_survival_prediction_bulk_rna_bert.py \
        --artifacts-dir /path/to/train/artifacts
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import (
    PatientSplit,
    TCGARNASeqDataset,
    get_dataloader,
)
from pyhealth.metrics.survival import concordance_index
from pyhealth.models import BulkRNABert
from pyhealth.tasks import TCGASurvivalPrediction


# ---------------------------------------------------------------------------
# Synthetic data (used in default mode)
# ---------------------------------------------------------------------------

def _write_synthetic_csv(root: Path, n_per_cohort: int = 20, seed: int = 0) -> None:
    """Writes a minimal tcga_rna_seq.csv with 3 cohorts and random embeddings."""
    rng = np.random.default_rng(seed)
    cohorts = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC"]
    rows = []
    for ci, cohort in enumerate(cohorts):
        for i in range(n_per_cohort):
            emb = rng.standard_normal(256).astype(np.float32)
            # Give the model a little signal: survival time weakly tied to emb[0].
            duration = float(
                200.0 + 100.0 * emb[0] + 50.0 * ci + 10.0 * rng.random()
            )
            event = int(rng.random() < 0.6)
            rows.append({
                "patient_id": f"TCGA-{cohort[5:]}-{i:04d}",
                "cohort": cohort,
                "duration": duration,
                "event": event,
                "embedding": json.dumps(emb.tolist()),
            })
    pd.DataFrame(rows).to_csv(root / "tcga_rna_seq.csv", index=False)


# ---------------------------------------------------------------------------
# Per-cohort aggregation
# ---------------------------------------------------------------------------

def per_cohort_c_index(duration, risk, event, cohort):
    """Per-cohort C-index plus overall, macro, and weighted aggregates.

    Wraps ``pyhealth.metrics.survival.concordance_index``. The aggregation
    is ablation-specific and lives here rather than in the library, which
    returns only the overall C-index.
    """
    overall = concordance_index(duration, event, risk)
    per_cohort = {}
    for c in np.unique(cohort):
        mask = cohort == c
        key = int(c)
        if mask.sum() < 2:
            per_cohort[key] = float("nan")
            continue
        per_cohort[key] = concordance_index(
            duration[mask], event[mask], risk[mask]
        )
    valid = {k: v for k, v in per_cohort.items() if not np.isnan(v)}
    if not valid:
        return {
            "c_index_overall": overall,
            "c_index_macro": float("nan"),
            "c_index_weighted": float("nan"),
            "per_cohort": per_cohort,
        }
    sizes = np.array([np.sum(cohort == k) for k in valid], dtype=float)
    scores = np.array(list(valid.values()))
    return {
        "c_index_overall": overall,
        "c_index_macro": float(scores.mean()),
        "c_index_weighted": float((sizes * scores).sum() / sizes.sum()),
        "per_cohort": per_cohort,
    }


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _unpack(v):
    """Unwrap ``(tensor, mask)`` tuples that some processors return."""
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


def _feats(batch: dict) -> dict:
    """Drop non-tensor fields and unwrap processor tuples."""
    return {k: _unpack(v) for k, v in batch.items() if k != "patient_id"}


# ---------------------------------------------------------------------------
# Training loop (shared by both ablation arms)
# ---------------------------------------------------------------------------

def train(
    train_ds,
    val_ds,
    test_ds,
    loss_mode: str,
    epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 256,
    patience: int = 20,
    seed: int = 0,
) -> dict:
    """Train a fresh BulkRNABert head and return test-set C-indices.

    Args:
        train_ds, val_ds, test_ds: Split SampleDatasets from ``set_task``.
        loss_mode: ``"global"`` (paper default) or ``"cohort"`` (extension).
        epochs, lr, batch_size, patience, seed: Training hyperparameters.
            Defaults match the reference repo's training configuration.

    Returns:
        Dict with ``c_index_overall``, ``c_index_weighted``, ``c_index_macro``,
        and ``per_cohort``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BulkRNABert(
        dataset=train_ds,
        use_cached_embeddings=True,
        cox_loss_mode=loss_mode,
    )
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    def _collect(model, loader):
        model.eval()
        risks, dur, events, cohorts = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                feats = _feats(batch)
                out = model.forward(**feats)
                risks.append(out["logit"].squeeze(-1).cpu().numpy())
                dur.append(feats["duration"].reshape(-1).cpu().numpy())
                events.append(feats["event"].reshape(-1).cpu().numpy())
                if "cohort" in feats:
                    cohorts.append(feats["cohort"].reshape(-1).cpu().numpy())
        return (
            np.concatenate(dur),
            np.concatenate(risks),
            np.concatenate(events),
            np.concatenate(cohorts).astype(int) if cohorts else None,
        )
    best_val = -1.0
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            feats = _feats(batch)
            out = model.forward(**feats)
            loss = out["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()

        t_val, r_val, e_val, _ = _collect(model, val_loader)
        val_ci = concordance_index(t_val, e_val, r_val)
        if val_ci > best_val:
            best_val = val_ci
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    t, r, e, c = _collect(model, test_loader)
    return per_cohort_c_index(t, r, e, c)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Directory with samples.tsv, survival_labels.tsv, and "
            "embeddings_repo.npy. Omit for the synthetic demo."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    # Windows cannot unlink litdata's memory-mapped files while this process
    # still holds handles, so cleanup errors are silenced for the demo only.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        if args.artifacts_dir is None:
            print("Synthetic mode: generating 60 synthetic patients.")
            _write_synthetic_csv(tmp)
            dataset = TCGARNASeqDataset(
                root=str(tmp), cache_dir=str(tmp / "cache"),
            )
        else:
            print(f"Real mode: reading artifacts from {args.artifacts_dir}")
            dataset = TCGARNASeqDataset(
                root=str(args.artifacts_dir),
                artifacts_dir=str(args.artifacts_dir),
                cache_dir=str(tmp / "cache"),
            )

        task = TCGASurvivalPrediction()
        # Patient-level 70/10/20 split. Validation drives early stopping.
        train_ds, val_ds, test_ds = dataset.set_task(
            task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=args.seed),
        )
        print(
            f"Train: {len(train_ds)} samples, "
            f"Val: {len(val_ds)} samples, "
            f"Test: {len(test_ds)} samples "
            f"across {len(dataset.unique_patient_ids)} patients."
        )

        results = {}
        for mode in ("global", "cohort"):
            print(f"\nTraining with {mode} Cox loss...")
            res = train(
                train_ds, val_ds, test_ds, loss_mode=mode,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                patience=args.patience, seed=args.seed,
            )
            results[mode] = res
            print(f"  overall C-index:  {res['c_index_overall']:.3f}")
            print(f"  weighted C-index: {res['c_index_weighted']:.3f}")
            print(f"  macro C-index:    {res['c_index_macro']:.3f}")
            for cohort_id, ci in res["per_cohort"].items():
                name = TCGASurvivalPrediction.INT_TO_COHORT.get(cohort_id, str(cohort_id))
                print(f"    {name}: {ci:.3f}")

        print("\n--- Ablation summary ---")
        print(f"{'metric':<20}{'global':>10}{'cohort':>10}")
        for key in ("c_index_overall", "c_index_weighted", "c_index_macro"):
            g = results["global"][key]
            c = results["cohort"][key]
            print(f"{key:<20}{g:>10.3f}{c:>10.3f}")

        print(
            "\nNote: synthetic mode tests pipeline integration and the "
            "ablation plumbing, not BulkRNABert's published performance. "
            "Run with --artifacts-dir pointing at the reference repo's "
            "artifacts for a meaningful comparison against the paper."
        )


if __name__ == "__main__":
    main()