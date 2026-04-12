"""AMA Prediction -- LogisticRegression Ablation with Fairness Analysis.

Reproduces the Against-Medical-Advice discharge prediction from:

    Boag, W.; Suresh, H.; Celi, L. A.; Szolovits, P.; Ghassemi, M.
    "Racial Disparities and Mistrust in End-of-Life Care."
    Machine Learning for Healthcare Conference, PMLR, 2018.

For each baseline the script reports:
  1. Overall AUROC / PR-AUC averaged over N random 60/40 splits.
  2. Subgroup performance (AUROC, PR-AUC) sliced by Race, Age Group,
     and Insurance Type.
  3. Fairness metrics per subgroup:
     - Demographic Parity  = % predicted AMA  (P(Y_hat=1 | Group=g))
     - Equal Opportunity   = True Positive Rate (P(Y_hat=1 | Y=1, Group=g))

Usage (synthetic demo data -- illustrative only, likely no AMA positives):
    python examples/mimic3_ama_prediction_logistic_regression.py

Usage (real MIMIC-III):
    python examples/mimic3_ama_prediction_logistic_regression.py \\
        --root /path/to/mimic-iii/1.4 --splits 100 --epochs 10
"""

import argparse
import tempfile
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import LogisticRegression
from pyhealth.tasks import AMAPredictionMIMIC3
from pyhealth.trainer import Trainer

SYNTHETIC_ROOT = (
    "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
)

BASELINES = {
    "BASELINE": ["demographics", "age", "los"],
    "BASELINE+RACE": ["demographics", "age", "los", "race"],
    "BASELINE+RACE+SUBSTANCE": [
        "demographics", "age", "los", "race", "has_substance_use",
    ],
}


# ------------------------------------------------------------------
# Helpers -- demographics lookup
# ------------------------------------------------------------------

def _build_demographics_lookup(dataset, task):
    """Run the task on every patient and collect raw demographic info.

    Returns a dict mapping ``(patient_id, visit_id)`` to a dict with
    keys ``race``, ``age``, and ``insurance``.
    """
    lookup = {}
    for patient in dataset.iter_patients():
        for sample in task(patient):
            pid = str(sample["patient_id"])
            vid = str(sample["visit_id"])
            race = sample["race"][0].split(":", 1)[1]
            age = sample["age"][0]
            insurance = "Other"
            for t in sample["demographics"]:
                if t.startswith("insurance:"):
                    insurance = t.split(":", 1)[1]
                    break
            lookup[(pid, vid)] = {
                "race": race, "age": age, "insurance": insurance,
            }
    return lookup


def _age_group(age):
    if age < 45:
        return "Young (18-44)"
    if age < 65:
        return "Middle (45-64)"
    return "Senior (65+)"


# ------------------------------------------------------------------
# Helpers -- inference with demographic labels
# ------------------------------------------------------------------

def _get_predictions(model, dataloader, lookup):
    """Run model on *dataloader*, return predictions + subgroup labels."""
    model.eval()
    all_probs, all_labels = [], []
    all_races, all_ages, all_ins = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            output = model(**batch)
            all_probs.append(output["y_prob"].detach().cpu())
            all_labels.append(output["y_true"].detach().cpu())

            pids = batch["patient_id"]
            vids = batch["visit_id"]
            if isinstance(vids, torch.Tensor):
                vids = vids.tolist()
            if isinstance(pids, torch.Tensor):
                pids = pids.tolist()

            for pid, vid in zip(pids, vids):
                info = lookup.get((str(pid), str(vid)), {})
                all_races.append(info.get("race", "Other"))
                all_ages.append(_age_group(info.get("age", 0.0)))
                all_ins.append(info.get("insurance", "Other"))

    y_prob = torch.cat(all_probs).numpy().ravel()
    y_true = torch.cat(all_labels).numpy().ravel()
    groups = {
        "Race": np.array(all_races),
        "Age Group": np.array(all_ages),
        "Insurance": np.array(all_ins),
    }
    return y_prob, y_true, groups


# ------------------------------------------------------------------
# Helpers -- safe metrics
# ------------------------------------------------------------------

def _safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


def _safe_prauc(y, p):
    if np.sum(y) == 0:
        return float("nan")
    try:
        return average_precision_score(y, p)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------
# Single split
# ------------------------------------------------------------------

def _create_model(sample_dataset, feature_keys, embedding_dim=128):
    """Create a LogisticRegression with the requested feature subset."""
    model = LogisticRegression(
        dataset=sample_dataset, embedding_dim=embedding_dim,
    )
    model.feature_keys = list(feature_keys)
    output_size = model.get_output_size()
    model.fc = torch.nn.Linear(
        len(feature_keys) * embedding_dim, output_size,
    )
    return model


def _run_single_split(sample_dataset, feature_keys, lookup,
                      seed, epochs, batch_size=32):
    """Train + evaluate one 60/40 split.  Returns metrics dict or None."""
    train_ds, _, test_ds = split_by_patient(
        sample_dataset, [0.6, 0.0, 0.4], seed=seed,
    )
    train_dl = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    model = _create_model(sample_dataset, feature_keys)
    trainer = Trainer(model=model)
    try:
        trainer.train(
            train_dataloader=train_dl, val_dataloader=None,
            epochs=epochs, monitor=None,
        )
    except Exception as exc:
        print(f"    train failed: {exc}")
        return None

    y_prob, y_true, groups = _get_predictions(model, test_dl, lookup)
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    overall_auroc = _safe_auroc(y_true, y_prob)
    overall_prauc = _safe_prauc(y_true, y_prob)

    subgroup = {}
    for attr_name, attr_vals in groups.items():
        subgroup[attr_name] = {}
        for grp in sorted(set(attr_vals)):
            mask = attr_vals == grp
            n = int(mask.sum())
            if n < 2:
                continue
            yt, yp, yd = y_true[mask], y_prob[mask], y_pred[mask]
            pos = yt.sum()
            subgroup[attr_name][grp] = {
                "auroc": _safe_auroc(yt, yp),
                "pr_auc": _safe_prauc(yt, yp),
                "pct_pred": float(yd.mean()) * 100,
                "tpr": float(yd[yt == 1].mean()) * 100 if pos > 0
                       else float("nan"),
                "n": n,
            }

    return {
        "auroc": overall_auroc,
        "pr_auc": overall_prauc,
        "subgroups": subgroup,
    }


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def _nanmean(lst):
    v = [x for x in lst if not np.isnan(x)]
    return np.mean(v) if v else float("nan")


def _nanstd(lst):
    v = [x for x in lst if not np.isnan(x)]
    return np.std(v) if v else float("nan")


def _aggregate(results):
    """Aggregate per-split metrics into means and stds."""
    valid = [r for r in results if r is not None]
    if not valid:
        return None

    agg = {
        "n": len(valid),
        "auroc_mean": _nanmean([r["auroc"] for r in valid]),
        "auroc_std": _nanstd([r["auroc"] for r in valid]),
        "pr_auc_mean": _nanmean([r["pr_auc"] for r in valid]),
        "pr_auc_std": _nanstd([r["pr_auc"] for r in valid]),
    }

    all_attrs = set()
    for r in valid:
        all_attrs.update(r["subgroups"].keys())

    agg["subgroups"] = {}
    for attr in sorted(all_attrs):
        agg["subgroups"][attr] = {}
        all_grps = set()
        for r in valid:
            if attr in r["subgroups"]:
                all_grps.update(r["subgroups"][attr].keys())

        for grp in sorted(all_grps):
            aurocs, praucs, pcts, tprs, ns = [], [], [], [], []
            for r in valid:
                m = r["subgroups"].get(attr, {}).get(grp)
                if m is None:
                    continue
                aurocs.append(m["auroc"])
                praucs.append(m["pr_auc"])
                pcts.append(m["pct_pred"])
                tprs.append(m["tpr"])
                ns.append(m["n"])

            agg["subgroups"][attr][grp] = {
                "auroc_mean": _nanmean(aurocs),
                "auroc_std": _nanstd(aurocs),
                "pr_auc_mean": _nanmean(praucs),
                "pr_auc_std": _nanstd(praucs),
                "pct_pred_mean": _nanmean(pcts),
                "tpr_mean": _nanmean(tprs),
                "n_avg": int(np.mean(ns)) if ns else 0,
            }
    return agg


# ------------------------------------------------------------------
# Pretty-printing
# ------------------------------------------------------------------

def _fmt(val, digits=4):
    return "N/A" if np.isnan(val) else f"{val:.{digits}f}"


def _print_results(name, feature_keys, agg):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {name}  (LogisticRegression)")
    print(f"  Features: {feature_keys}")
    print(f"{'=' * w}")

    if agg is None:
        print("  No valid results.\n")
        return

    ci_lo = agg["auroc_mean"] - 1.96 * agg["auroc_std"]
    ci_hi = agg["auroc_mean"] + 1.96 * agg["auroc_std"]
    print(f"\n  1. Overall Performance ({agg['n']} splits)")
    print(f"     AUROC:  {_fmt(agg['auroc_mean'])} +/- {_fmt(agg['auroc_std'])}"
          f"  95% CI ({_fmt(ci_lo)}, {_fmt(ci_hi)})")
    print(f"     PR-AUC: {_fmt(agg['pr_auc_mean'])} +/- {_fmt(agg['pr_auc_std'])}")

    print(f"\n  2. Subgroup Performance")
    for attr, grps in agg["subgroups"].items():
        print(f"     {attr}:")
        print(f"       {'Group':<20} {'AUROC':>15} {'PR-AUC':>15} {'n_avg':>7}")
        print(f"       {'-'*58}")
        for grp, m in grps.items():
            a_str = f"{_fmt(m['auroc_mean'])}+/-{_fmt(m['auroc_std'])}"
            p_str = f"{_fmt(m['pr_auc_mean'])}+/-{_fmt(m['pr_auc_std'])}"
            print(f"       {grp:<20} {a_str:>15} {p_str:>15} {m['n_avg']:>7}")

    print(f"\n  3. Fairness Metrics")
    print(f"     Demographic Parity (% Predicted AMA):")
    for attr, grps in agg["subgroups"].items():
        parts = [f"{g}: {_fmt(m['pct_pred_mean'],2)}%"
                 for g, m in grps.items()]
        print(f"       {attr}: {',  '.join(parts)}")

    print(f"     Equal Opportunity (True Positive Rate):")
    for attr, grps in agg["subgroups"].items():
        parts = [f"{g}: {_fmt(m['tpr_mean'],2)}%"
                 for g, m in grps.items()]
        print(f"       {attr}: {',  '.join(parts)}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AMA prediction ablation -- LogisticRegression",
    )
    parser.add_argument("--root", default=SYNTHETIC_ROOT,
                        help="MIMIC-III root (local path or URL)")
    parser.add_argument("--splits", type=int, default=100,
                        help="Number of random 60/40 splits (default 100)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per split")
    parser.add_argument("--dev", action="store_true",
                        help="Use dev mode (1000 patients)")
    args = parser.parse_args()

    cache_dir = tempfile.mkdtemp(prefix="ama_lr_")
    print(f"Cache: {cache_dir}")
    print(f"Root:  {args.root}")
    print(f"Splits: {args.splits}  |  Epochs: {args.epochs}")

    print("\n[1/4] Loading dataset...")
    t0 = time.time()
    dataset = MIMIC3Dataset(
        root=args.root, tables=[],
        cache_dir=cache_dir, dev=args.dev,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    dataset.stats()

    print("\n[2/4] Applying AMA task...")
    task = AMAPredictionMIMIC3()
    try:
        sample_dataset = dataset.set_task(task)
    except ValueError as exc:
        if "unique labels" not in str(exc).lower():
            raise
        print(f"\n  {exc}")
        print("  The dataset contains no AMA-positive cases.")
        print("  AMA prevalence is ~2% so small/synthetic data")
        print("  often lacks positives.  Demonstrating the task")
        print("  on raw patients instead:\n")
        total = 0
        for patient in dataset.iter_patients():
            samples = task(patient)
            total += len(samples)
        print(f"  Task produced {total} samples (all label=0)")
        print("\n  Re-run with real MIMIC-III for ablation:")
        print("    python examples/"
              "mimic3_ama_prediction_logistic_regression.py \\")
        print("        --root /path/to/mimic-iii/1.4")
        print("\nDone.")
        return
    print(f"  Samples: {len(sample_dataset)}")

    print("\n[3/4] Building demographics lookup...")
    t0 = time.time()
    lookup = _build_demographics_lookup(dataset, task)
    print(f"  {len(lookup)} entries in {time.time()-t0:.1f}s")

    print(f"\n[4/4] Running ablation ({len(BASELINES)} baselines "
          f"x {args.splits} splits)...\n")

    t_total = time.time()
    for name, feature_keys in BASELINES.items():
        split_results = []
        for i in range(args.splits):
            t0 = time.time()
            res = _run_single_split(
                sample_dataset, feature_keys, lookup,
                seed=i, epochs=args.epochs,
            )
            elapsed = time.time() - t0
            if res is not None:
                print(f"  [{name}] split {i+1:3d}/{args.splits}: "
                      f"AUROC={_fmt(res['auroc'])}  ({elapsed:.1f}s)")
            else:
                print(f"  [{name}] split {i+1:3d}/{args.splits}: FAILED")
            split_results.append(res)

        agg = _aggregate(split_results)
        _print_results(name, feature_keys, agg)

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
