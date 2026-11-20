"""End-to-end demo: TCGA RNA-seq survival prediction with BulkRNABert.

Reproduces the downstream survival prediction pipeline of Gélard et al.
(2025), "BulkRNABert: Cancer prognosis from bulk RNA-seq based language
models" (ML4H 2025). Full PyHealth contribution:

    TCGARNASeqDataset -> TCGASurvivalPrediction -> BulkRNABert -> Cox loss

Three modes:

1. Synthetic (default): generates 120 synthetic patients in a temp dir.
2. Local CSV (``--data-dir DIR``): reads ``tcga_rna_seq.csv`` from DIR.
3. Download (``--download``): fetches ``tcga_rna_seq.csv`` (~56 MB) from
   the Hugging Face Dataset ``cstein2/TCGABulkBertRNAEmbeddings``.

**Ablation.** Compares global Cox partial likelihood (paper default) against
cohort-restricted Cox risk sets (our extension). Both arms use identical
splits, seeds, hyperparameters, and initialization.

**Cox risk sets.** Both arms use the entire training split to construct
their partial-likelihood risk sets. The global configuration includes
patients from all cohorts in one risk set; the cohort-restricted
configuration builds separate risk sets within each cancer cohort. This
implements the full-training-set Cox objective rather than a minibatch
approximation.

Note: this differs from the reference repo's batch-size-256 training.
Both objectives are valid; ours matches the paper's mathematical
definition of the partial likelihood.

**Evaluation.** Overall C-index uses
``pyhealth.metrics.survival.concordance_index``. The per-cohort aggregation
(weighted and macro C-index) is ablation-specific and lives here rather
than in ``pyhealth.metrics.survival``, which is cohort-agnostic.

**Real-data results (3 seeds, full-batch Cox, 300 epochs, patience 50):**

    ==========  ==================  ====================  ==========
    metric      paper (mean±std)    global (mean±std)     cohort
    ==========  ==================  ====================  ==========
    overall     0.765 ± 0.011       0.757 ± 0.007         0.690 ± 0.022
    weighted    0.642 ± 0.014       0.643 ± 0.004         0.636 ± 0.009
    macro       0.656 ± 0.025       0.629 ± 0.015         0.625 ± 0.022
    ==========  ==================  ====================  ==========

Global Cox reproduces the paper's overall and weighted C-index within
one standard deviation. The cohort-restricted extension does not
improve weighted or macro C-index (both deltas within seed noise),
and it significantly hurts overall C-index (−0.067), suggesting the
global objective already learns good within-cohort rankings and the
restriction sacrifices useful cross-cohort signal.

This negative result is consistent with the reference repo's earlier
observation under minibatch Cox.
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
# Synthetic data
# ---------------------------------------------------------------------------

def _write_synthetic_csv(root: Path, n_per_cohort: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cohorts = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC"]
    rows = []
    for ci, cohort in enumerate(cohorts):
        for i in range(n_per_cohort):
            emb = rng.standard_normal(256).astype(np.float32)
            duration = float(max(1.0, 200.0 + 100.0 * emb[0] + 50.0 * ci + 10.0 * rng.random()))
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
# Per-cohort aggregation (ablation-specific; kept local to the example)
# ---------------------------------------------------------------------------

def per_cohort_c_index(duration, risk, event, cohort):
    """Per-cohort C-index plus overall, macro, and weighted aggregates.

    Wraps ``pyhealth.metrics.survival.concordance_index`` once per cohort.
    The aggregation is ablation-specific and not part of the library, which
    returns only the overall C-index.

    Weighted C-index weights each cohort by its number of test samples.
    Macro C-index is the unweighted mean. Cohorts with fewer than two
    samples or no comparable pairs are excluded from both aggregates.
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
            "c_index": overall,
            "c_index_macro": float("nan"),
            "c_index_weighted": float("nan"),
            "per_cohort": per_cohort,
        }
    sizes = np.array([np.sum(cohort == k) for k in valid], dtype=float)
    scores = np.array(list(valid.values()))
    return {
        "c_index": overall,
        "c_index_macro": float(scores.mean()),
        "c_index_weighted": float((sizes * scores).sum() / sizes.sum()),
        "per_cohort": per_cohort,
    }


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _unpack(v):
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


def _feats(batch: dict) -> dict:
    return {k: _unpack(v) for k, v in batch.items() if k != "patient_id"}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_ds,
    val_ds,
    test_ds,
    loss_mode: str,
    epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    eval_batch_size: int = 256,
    seed: int = 0,
) -> dict:
    """Train with full-batch Cox, matching the paper's objective.

    The paper defines the Cox partial likelihood over the full training
    risk set: R_i = {j | T_j >= T_i}. We load all training patients as a
    single batch and take one optimizer step per epoch, which reproduces
    that objective exactly. Hyperparameters (lr, epochs, patience) are not
    specified by the paper; defaults are chosen for stability with ~10k
    samples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BulkRNABert(
        dataset=train_ds,
        cox_loss_mode=loss_mode,
    )
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )

    # Preload the full training set as one batch for full-batch Cox.
    train_loader = get_dataloader(
        train_ds, batch_size=len(train_ds), shuffle=False,
    )
    train_feats = _feats(next(iter(train_loader)))

    val_loader = get_dataloader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
    )
    test_loader = get_dataloader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
    )

    def evaluate(loader):
        model.eval()
        risks, dur, events, cohorts = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                feats = _feats(batch)
                out = model(embedding=feats["embedding"])
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
        optim.zero_grad()
        out = model(**train_feats)
        out["loss"].backward()
        optim.step()

        t_val, r_val, e_val, _ = evaluate(val_loader)
        if len(t_val) < 2 or not np.any(e_val):
            raise ValueError(
                f"Validation set has too few events or samples "
                f"(n={len(t_val)}, events={int(e_val.sum())}) to compute "
                "a C-index. Increase synthetic size or adjust the split."
            )
        val_ci = concordance_index(t_val, e_val, r_val)
        if not np.isfinite(val_ci):
            raise ValueError(
                f"Validation C-index is undefined (NaN) with "
                f"n={len(t_val)}, events={int(e_val.sum())}."
            )

        if val_ci > best_val:
            best_val = val_ci
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    t, r, e, c = evaluate(test_loader)
    if c is not None:
        return per_cohort_c_index(t, r, e, c)
    return {"c_index": concordance_index(t, e, r), "per_cohort": {}}

# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _print_result(seed: int, mode: str, res: dict) -> None:
    print(
        f"  [{seed}/{mode}] overall {res['c_index']:.3f}  "
        f"weighted {res.get('c_index_weighted', float('nan')):.3f}  "
        f"macro {res.get('c_index_macro', float('nan')):.3f}"
    )
    for cohort_id, ci in res.get("per_cohort", {}).items():
        name = TCGASurvivalPrediction.INT_TO_COHORT.get(
            int(cohort_id), str(cohort_id)
        )
        print(f"    {name}: {ci:.3f}")


def _print_summary(all_results: dict[str, list[dict]]) -> None:
    n_seeds = len(all_results["global"])
    if n_seeds < 2:
        return
    print(f"\n{'=' * 62}\nSummary across {n_seeds} seeds\n{'=' * 62}")
    print(f"{'metric':<12}{'global (mean±std)':>24}{'cohort (mean±std)':>24}")
    for key, label in [
        ("c_index", "overall"),
        ("c_index_weighted", "weighted"),
        ("c_index_macro", "macro"),
    ]:
        g = [r.get(key, float("nan")) for r in all_results["global"]]
        c = [r.get(key, float("nan")) for r in all_results["cohort"]]
        gm, gs = np.nanmean(g), np.nanstd(g, ddof=1)
        cm, cs = np.nanmean(c), np.nanstd(c, ddof=1)
        print(f"{label:<12}{gm:>10.3f} ± {gs:>6.3f}   {cm:>10.3f} ± {cs:>6.3f}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing tcga_rna_seq.csv.",
    )
    data_group.add_argument(
        "--download",
        action="store_true",
        help="Download tcga_rna_seq.csv from Hugging Face.",
    )
    parser.add_argument(
        "--hf-repo",
        default=TCGARNASeqDataset.HF_REPO_ID,
        help="Hugging Face Dataset repo ID.",
    )
    parser.add_argument(
        "--hf-revision",
        default=TCGARNASeqDataset.HF_REVISION,
        help="Hugging Face revision to pin.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="One or more random seeds; results are averaged across all "
             "when multiple are given (default: [0]).",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="If given, save per-seed results to this JSON file "
             "(saved incrementally so partial runs survive a crash).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)

        if args.data_dir is not None:
            print(f"Real mode (local CSV): {args.data_dir}")
            dataset = TCGARNASeqDataset(
                root=str(args.data_dir),
                cache_dir=str(tmp_path / "cache"),
            )
        elif args.download:
            print(
                f"Real mode (download): "
                f"{args.hf_repo}@{args.hf_revision}"
            )
            dataset = TCGARNASeqDataset.from_huggingface(
                repo_id=args.hf_repo,
                revision=args.hf_revision,
                cache_dir=str(tmp_path / "cache"),
            )
        else:
            print("Synthetic mode: generating 120 synthetic patients.")
            _write_synthetic_csv(tmp_path)
            dataset = TCGARNASeqDataset(
                root=str(tmp_path),
                cache_dir=str(tmp_path / "cache"),
            )

        task = TCGASurvivalPrediction()
        all_results: dict[str, list[dict]] = {"global": [], "cohort": []}

        for seed in args.seeds:
            if len(args.seeds) > 1:
                print(f"\n{'=' * 62}\nSeed {seed}\n{'=' * 62}")
            train_ds, val_ds, test_ds = dataset.set_task(
                task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=seed),
            )
            print(
                f"Train: {len(train_ds)}  Val: {len(val_ds)}  "
                f"Test: {len(test_ds)}"
            )

            for mode in ("global", "cohort"):
                if len(args.seeds) == 1:
                    print(f"\nTraining with {mode} Cox loss...")
                else:
                    print(f"\nTraining [{seed}/{mode}] ...")
                res = train(
                    train_ds, val_ds, test_ds, loss_mode=mode,
                    epochs=args.epochs, lr=args.lr,
                    patience=args.patience,
                    eval_batch_size=args.eval_batch_size,
                    seed=seed,
                )
                res["_seed"] = seed
                all_results[mode].append(res)
                _print_result(seed, mode, res)

                if args.json_out is not None:
                    args.json_out.parent.mkdir(parents=True, exist_ok=True)
                    with open(args.json_out, "w") as fh:
                        json.dump(all_results, fh, indent=2, default=float)

        if len(args.seeds) == 1:
            print("\n--- Ablation summary ---")
            print(f"{'metric':<20}{'global':>10}{'cohort':>10}")
            for key in ("c_index", "c_index_weighted", "c_index_macro"):
                g = all_results["global"][0].get(key, float("nan"))
                c = all_results["cohort"][0].get(key, float("nan"))
                print(f"{key:<20}{g:>10.3f}{c:>10.3f}")
        else:
            _print_summary(all_results)

        if args.json_out is not None:
            print(f"\nFull per-seed results saved to {args.json_out}")


if __name__ == "__main__":
    main()
