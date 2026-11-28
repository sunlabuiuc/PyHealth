"""Concept bottleneck model for vasopressor onset on MIMIC-III: full
pipeline and ablation study.

Paper: Wu, Parbhoo, Havasi, Doshi-Velez. "Learning Optimal Summaries of
Clinical Time-series with Concept Bottleneck Models." MLHC 2022.
https://proceedings.mlr.press/v182/wu22a.html
Original code: https://github.com/dtak/optimal-summaries-public

Pipeline (all PyHealth): ``MIMIC3Dataset`` -> ``VasopressorOnsetMIMIC3``
task -> ``ConceptBottleneckModel`` and its two baselines
(``LogisticRegressionWithSummaries``, ``LSTMNextStateBaseline``) ->
``Trainer`` -> AUROC / AUPRC / Brier / ECE, then greedy feature selection
and fine-tuning of the bottleneck.

How to run
----------
No data needed (default): writes a small synthetic MIMIC-III-format
dataset to a temporary directory and runs everything on it::

    python examples/mimic3_vasopressor_onset_concept_bottleneck.py

Real data: point ``--root`` at MIMIC-III v1.4 (credentialed) or at the
open 100-patient demo (https://physionet.org/content/mimiciii-demo/1.4/)::

    python examples/mimic3_vasopressor_onset_concept_bottleneck.py --root DIR

Experimental setup
------------------
- Cohort and inputs: see ``VasopressorOnsetMIMIC3``. First 6 ICU hours,
  28 vitals/labs + indicators + 10 statics; label = vasopressor started
  later in the stay.
- Splits: 68/17/15 train/val/test, the sizes the original scripts use
  (test 0.15, then 0.20 of the rest), over seeds 1-6. Scaling statistics
  are fitted on the training part only (the original uses the whole
  cohort).
- Models: k=4 concepts, L1 0.001 and cosine penalty 0.01 (the values
  the original greedy script selects), Adam, class-weighted BCE. The
  best epoch by validation AUROC is kept.

Ablations (none of these is in the paper)
-----------------------------------------
1. Input scaling: z-score ``(x - mean) / std`` (PyHealth's default)
   against the original notebook's ``(x - mean) / variance``.
2. Split: stratified by label (as in the original bottleneck scripts)
   against unstratified (as in the original LSTM script).
3. LSTM axis order: ``batch_first=True`` (this port) against
   ``batch_first=False``, which reproduces the original's LSTM being
   fed batch-first tensors while reading them time-first.
4. Metrics beyond AUROC: AUPRC, Brier score and expected calibration
   error for every model.

Also run, as in the paper: greedy (concept, feature) selection on the
trained bottleneck followed by fine-tuning with only the selected
weights. Selection here scores on the validation part; the original
scores on the test part.

Results on the synthetic data
-----------------------------
These numbers come from the default run of this file (400 random
synthetic patients, 335 in the cohort, 70 positive; seeds 1-6; 30
epochs; about 3.5 minutes on CPU). The data is random with a planted
signal, so they show that the pipeline runs and what the ablations
measure. They say nothing about MIMIC-III or about the paper's results.

Test set, mean +- std over the 6 seeds::

    configuration           AUROC          AUPRC          Brier          ECE
    cbm                     0.806 +- 0.057 0.563 +- 0.128 0.227 +- 0.012 0.285 +- 0.022
    logreg_summaries        0.800 +- 0.056 0.554 +- 0.119 0.201 +- 0.033 0.257 +- 0.054
    lstm                    0.797 +- 0.055 0.541 +- 0.110 0.196 +- 0.028 0.279 +- 0.031
    lstm_axis_swap          0.800 +- 0.047 0.543 +- 0.099 0.205 +- 0.024 0.277 +- 0.028
    cbm_variance_scaling    0.765 +- 0.067 0.516 +- 0.120 0.209 +- 0.030 0.265 +- 0.029
    cbm_unstratified        0.765 +- 0.065 0.536 +- 0.075 0.203 +- 0.021 0.211 +- 0.034

Findings on the synthetic data:

- Scaling: variance scaling scored 0.041 AUROC lower than z-score for
  the bottleneck, within one standard deviation across seeds.
- Split: unstratified test parts held 7 to 15 positives (12, 12, 12, 15,
  12, 7) where stratified ones hold a fixed share; the two rows use
  different test patients, so their AUROCs are not directly comparable.
- LSTM axis order: no measurable difference (0.797 against 0.800).
- All three models were poorly calibrated (ECE 0.26-0.29) with the
  class-weighted loss, which AUROC alone does not show.
- Greedy selection (seed 1): fine-tuning on the 1, 2, 4 and 8 selected
  (concept, feature) pairs gave test AUROC 0.770, 0.784, 0.764 and
  0.768, against 0.770 with all 402 features per concept.
"""

import argparse
import csv
import logging
import os
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pyhealth.datasets import MIMIC3Dataset, create_sample_dataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import (
    ConceptBottleneckModel,
    LogisticRegressionWithSummaries,
    LSTMNextStateBaseline,
)
from pyhealth.models.concept_bottleneck import (
    apply_topk_mask,
    greedy_select_features,
    top_features_by_weight_mass,
)
from pyhealth.tasks import VasopressorOnsetMIMIC3
from pyhealth.tasks.vasopressor_onset_mimic3 import (
    apply_standardizer,
    fit_standardizer,
)
from pyhealth.trainer import Trainer

TABLES = ["chartevents", "labevents", "outputevents", "inputevents_mv"]
METRICS = ["roc_auc", "pr_auc", "brier", "ECE"]
SCHEMA = {"values": "tensor", "indicators": "tensor", "statics": "tensor"}
LABEL = "vasopressor_onset"


# ---------------------------------------------------------------------------
# Synthetic MIMIC-III-format data (so the example runs without credentials)
# ---------------------------------------------------------------------------
def write_synthetic_mimic3(root: str, num_patients: int = 400, seed: int = 0) -> None:
    """Writes a small synthetic MIMIC-III v1.4-format dataset to ``root``.

    Every number is random. Each patient has one ICU stay and a hidden
    severity; sicker patients have lower blood pressure, higher heart
    rate and lactate, less urine, and are more likely to be started on a
    vasopressor. A few patients fall outside the cohort on purpose
    (CareVue stays, short stays, vasopressor in the first 6 hours).
    """
    rng = np.random.default_rng(seed)
    fmt = "%Y-%m-%d %H:%M:%S"
    tables: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "PATIENTS", "ADMISSIONS", "ICUSTAYS", "CHARTEVENTS",
            "LABEVENTS", "OUTPUTEVENTS", "INPUTEVENTS_MV",
        )
    }
    # (itemid, mean, slope per unit severity, noise, P(measured in an hour))
    vitals = [
        (220045, 85, 10, 6, 0.9), (220052, 80, -8, 5, 0.9),
        (220050, 120, -10, 8, 0.9), (220051, 65, -6, 5, 0.9),
        (224689, 18, 2, 2, 0.8), (220277, 97, -1, 1, 0.9),
        (223762, 37, 0.2, 0.4, 0.4), (220739, 3.5, -0.5, 0.5, 0.4),
        (223901, 5.5, -0.6, 0.5, 0.4), (223900, 4.5, -0.6, 0.5, 0.4),
        (223835, 40, 8, 5, 0.3),
    ]
    labs = [
        (50813, 1.6, 0.9, 0.4, 0.25), (50912, 1.0, 0.3, 0.2, 0.2),
        (51006, 18, 5, 4, 0.2), (51222, 11, -0.5, 1, 0.2),
        (51301, 9, 2, 2, 0.2), (50983, 139, 0, 3, 0.2),
        (50971, 4.1, 0.1, 0.4, 0.2), (50882, 24, -2, 2, 0.2),
        (50931, 120, 10, 20, 0.2), (51265, 220, -20, 50, 0.2),
    ]
    start0 = np.datetime64("2150-01-01T08:00:00")
    for i in range(num_patients):
        sid, hadm, stay = 1000 + i, 20000 + i, 30000 + i
        severity = rng.normal()
        intime = start0 + np.timedelta64(int(i * 24 * 3600), "s")
        los = float(rng.uniform(12, 120)) if rng.random() > 0.05 else 4.0
        outtime = intime + np.timedelta64(int(los * 3600), "s")

        def ts(hours: float, base=intime) -> str:
            stamp = base + np.timedelta64(int(hours * 3600), "s")
            return stamp.astype("datetime64[s]").item().strftime(fmt)

        age_years = float(rng.uniform(25, 95))
        dob = intime - np.timedelta64(int(age_years * 365.25 * 24 * 3600), "s")
        tables["PATIENTS"].append({
            "subject_id": sid, "gender": rng.choice(["F", "M"]),
            "dob": ts(0, dob), "dod": "", "dod_hosp": "", "dod_ssn": "",
            "expire_flag": 0,
        })
        tables["ADMISSIONS"].append({
            "subject_id": sid, "hadm_id": hadm, "admittime": ts(-2),
            "dischtime": ts(los + 24), "deathtime": "",
            "admission_type": rng.choice(["EMERGENCY", "URGENT", "ELECTIVE"]),
            "admission_location": "", "discharge_location": "",
            "insurance": "", "language": "", "religion": "",
            "marital_status": "", "ethnicity": "", "edregtime": "",
            "edouttime": "", "diagnosis": "", "hospital_expire_flag": 0,
        })
        unit = rng.choice(["MICU", "SICU", "TSICU", "CSRU", "CCU"])
        tables["ICUSTAYS"].append({
            "subject_id": sid, "hadm_id": hadm, "icustay_id": stay,
            "dbsource": "metavision" if rng.random() > 0.05 else "carevue",
            "first_careunit": unit, "last_careunit": unit,
            "intime": ts(0), "outtime": ts(0, outtime),
        })
        if rng.random() > 0.03:
            tables["CHARTEVENTS"].append({
                "subject_id": sid, "hadm_id": hadm, "icustay_id": stay,
                "itemid": 226512, "charttime": ts(0.3),
                "value": "", "valuenum": round(float(rng.normal(80, 15)), 1),
                "valueuom": "kg",
            })
        for hour in range(8):
            for itemid, mean, slope, noise, rate in vitals:
                if rng.random() < rate:
                    value = mean + slope * severity + rng.normal(0, noise)
                    tables["CHARTEVENTS"].append({
                        "subject_id": sid, "hadm_id": hadm, "icustay_id": stay,
                        "itemid": itemid, "charttime": ts(hour + rng.random()),
                        "value": "", "valuenum": round(float(value), 2),
                        "valueuom": "",
                    })
            for itemid, mean, slope, noise, rate in labs:
                if rng.random() < rate:
                    value = mean + slope * max(severity, -1) + rng.normal(0, noise)
                    tables["LABEVENTS"].append({
                        "subject_id": sid, "hadm_id": hadm, "itemid": itemid,
                        "charttime": ts(hour + rng.random()), "value": "",
                        "valuenum": round(float(value), 2), "valueuom": "",
                        "flag": "",
                    })
            if rng.random() < 0.8:
                volume = max(80 - 25 * severity + rng.normal(0, 20), 0)
                tables["OUTPUTEVENTS"].append({
                    "subject_id": sid, "hadm_id": hadm, "icustay_id": stay,
                    "itemid": 226559, "charttime": ts(hour + rng.random()),
                    "value": round(float(volume), 1), "valueuom": "mL",
                })
        if rng.random() < 1 / (1 + np.exp(1.8 - 1.6 * severity)):
            onset = float(rng.uniform(1.0, 0.8 * los))
            tables["INPUTEVENTS_MV"].append({
                "subject_id": sid, "hadm_id": hadm, "icustay_id": stay,
                "itemid": 221906, "starttime": ts(onset),
                "endtime": ts(onset + 2), "amount": 1, "amountuom": "mg",
                "rate": 0.1, "rateuom": "mcg/kg/min", "patientweight": 80,
                "orderid": i, "linkorderid": i,
                "ordercategoryname": "", "statusdescription": "",
            })
    lab_items = [
        {"itemid": itemid, "label": f"lab {itemid}", "fluid": "Blood",
         "category": "Chemistry"}
        for itemid, *_ in labs
    ]
    tables["D_LABITEMS"] = lab_items
    for name, rows in tables.items():
        with open(os.path.join(root, f"{name}.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Data: load once, then split / scale per seed
# ---------------------------------------------------------------------------
def load_raw_samples(root: str, cache_dir: str) -> list[dict[str, Any]]:
    """MIMIC3Dataset -> VasopressorOnsetMIMIC3 -> list of raw samples.

    The task runs with ``normalization=None`` so that each seed can fit
    its own scaling on its own training part.
    """
    dataset = MIMIC3Dataset(root=root, tables=TABLES, cache_dir=cache_dir)
    raw = dataset.set_task(VasopressorOnsetMIMIC3(normalization=None))
    samples = []
    for sample in raw:
        samples.append({
            "patient_id": str(sample["patient_id"]),
            "values": np.asarray(sample["values"], dtype=float).tolist(),
            "indicators": np.asarray(sample["indicators"], dtype=float).tolist(),
            "statics": np.asarray(sample["statics"], dtype=float).tolist(),
            LABEL: int(np.asarray(sample[LABEL]).reshape(-1)[0]),
        })
    return samples


def make_split(
    samples: Sequence[dict[str, Any]],
    seed: int,
    stratify: bool = True,
    scaling: str = "zscore",
) -> tuple[Any, Any, Any, float]:
    """68/17/15 split, scaling fitted on train; returns datasets + pos_weight."""
    labels = np.array([sample[LABEL] for sample in samples])
    index = np.arange(len(samples))
    rest, test = train_test_split(
        index, test_size=0.15, random_state=seed,
        stratify=labels if stratify else None,
    )
    train, val = train_test_split(
        rest, test_size=0.20, random_state=seed,
        stratify=labels[rest] if stratify else None,
    )
    train_samples = [samples[i] for i in train]
    stats = fit_standardizer(train_samples, method=scaling)
    parts: list[Any] = []
    for part in (train, val, test):
        scaled = apply_standardizer([samples[i] for i in part], stats)
        # Processors are fitted on the training part and reused, so a
        # part with a single class (possible without stratification)
        # still loads.
        fitted = parts[0] if parts else None
        parts.append(
            create_sample_dataset(
                samples=scaled,
                input_schema=SCHEMA,
                output_schema={LABEL: "binary"},
                dataset_name="mimic3",
                task_name="VasopressorOnsetMIMIC3",
                input_processors=fitted.input_processors if fitted else None,
                output_processors=fitted.output_processors if fitted else None,
            )
        )
    prevalence = float(labels[train].mean())
    pos_weight = (1.0 - prevalence) / max(prevalence, 1e-6)
    return parts[0], parts[1], parts[2], pos_weight


def loaders(train: Any, val: Any, test: Any, batch_size: int) -> tuple[Any, Any, Any]:
    """Dataloaders for the three parts."""
    return (
        get_dataloader(train, batch_size=batch_size, shuffle=True),
        get_dataloader(val, batch_size=batch_size),
        get_dataloader(test, batch_size=batch_size),
    )


def fit_and_score(
    model: torch.nn.Module, train_loader: Any, val_loader: Any, test_loader: Any,
    epochs: int, lr: float,
) -> dict[str, float]:
    """Trains with the PyHealth Trainer; returns the test metrics."""
    trainer = Trainer(model=model, metrics=METRICS, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
        monitor="roc_auc",
    )
    try:
        return trainer.evaluate(test_loader)
    except ValueError:
        # AUROC is undefined when the test part has a single class.
        return {metric: float("nan") for metric in METRICS}


def predict(model: torch.nn.Module, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    """Labels and predicted probabilities over a dataloader."""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            y_true.append(out["y_true"].cpu().numpy().ravel())
            y_prob.append(out["y_prob"].cpu().numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_seed(
    samples: Sequence[dict[str, Any]], seed: int, args: argparse.Namespace
) -> dict[str, dict[str, float]]:
    """All model / ablation configurations for one seed."""
    results: dict[str, dict[str, float]] = {}

    def cbm(dataset: Any, pos_weight: float) -> ConceptBottleneckModel:
        return ConceptBottleneckModel(
            dataset=dataset, num_concepts=args.num_concepts,
            l1_lambda=args.l1_lambda, cos_sim_lambda=args.cos_sim_lambda,
            pos_weight=pos_weight,
        )

    # Main setting: stratified split, z-score scaling.
    train, val, test, pos_weight = make_split(samples, seed)
    data = loaders(train, val, test, args.batch_size)
    torch.manual_seed(seed)
    results["cbm"] = fit_and_score(
        cbm(train, pos_weight), *data, args.epochs, args.lr
    )
    torch.manual_seed(seed)
    results["logreg_summaries"] = fit_and_score(
        LogisticRegressionWithSummaries(dataset=train, pos_weight=pos_weight),
        *data, args.epochs, args.lr,
    )
    # Ablation 3: LSTM axis order (same initial weights for both).
    for name, batch_first in (("lstm", True), ("lstm_axis_swap", False)):
        torch.manual_seed(seed)
        results[name] = fit_and_score(
            LSTMNextStateBaseline(
                dataset=train, pos_weight=pos_weight, batch_first=batch_first
            ),
            *data, args.epochs, args.lr,
        )
    # Ablation 1: the original's variance scaling.
    train, val, test, pos_weight = make_split(samples, seed, scaling="variance")
    torch.manual_seed(seed)
    results["cbm_variance_scaling"] = fit_and_score(
        cbm(train, pos_weight), *loaders(train, val, test, args.batch_size),
        args.epochs, args.lr,
    )
    # Ablation 2: unstratified split.
    train, val, test, pos_weight = make_split(samples, seed, stratify=False)
    torch.manual_seed(seed)
    results["cbm_unstratified"] = fit_and_score(
        cbm(train, pos_weight), *loaders(train, val, test, args.batch_size),
        args.epochs, args.lr,
    )
    labels = [int(np.asarray(sample[LABEL]).reshape(-1)[0]) for sample in test]
    results["cbm_unstratified"]["test_positives"] = float(sum(labels))
    return results


def run_greedy(
    samples: Sequence[dict[str, Any]], seed: int, args: argparse.Namespace
) -> list[dict[str, float]]:
    """Greedy (concept, feature) selection, then fine-tuning (paper Sec. 4).

    Candidates are the features covering 90% of each concept's weight
    mass (capped at ``--greedy-candidates`` per concept). Each round adds
    the pair that most raises validation AUROC. The model is then
    retrained from scratch with only the selected weights free, for
    growing prefixes of the selection.
    """
    train, val, test, pos_weight = make_split(samples, seed)
    train_loader, val_loader, test_loader = loaders(train, val, test, args.batch_size)

    def new_model() -> ConceptBottleneckModel:
        torch.manual_seed(seed)
        return ConceptBottleneckModel(
            dataset=train, num_concepts=args.num_concepts,
            l1_lambda=args.l1_lambda, cos_sim_lambda=args.cos_sim_lambda,
            pos_weight=pos_weight,
        )

    model = new_model()
    full = fit_and_score(
        model, train_loader, val_loader, test_loader, args.epochs, args.lr
    )
    trained_weight = model.bottleneck.weight.detach().clone()
    candidates = [
        inds[: args.greedy_candidates]
        for inds in top_features_by_weight_mass(trained_weight, mass=0.90)
    ]

    def score_fn(mask: torch.Tensor) -> float:
        with torch.no_grad():
            model.bottleneck.weight.copy_(trained_weight * mask)
        y_true, y_prob = predict(model, val_loader)
        return binary_metrics_fn(y_true, y_prob, metrics=["roc_auc"])["roc_auc"]

    selections, val_aucs = greedy_select_features(
        candidates, score_fn, budget=args.greedy_budget,
        encoding_dim=model.encoder.encoding_dim,
    )
    rows = [{"num_features": float("nan"), "val_auroc_greedy": float("nan"),
             "test_auroc_finetuned": full["roc_auc"]}]
    sizes = {k for k in (1, 2, 4, len(selections)) if 1 <= k <= len(selections)}
    for top_k in sorted(sizes):
        finetuned = new_model()
        apply_topk_mask(finetuned, selections[:top_k])
        scores = fit_and_score(
            finetuned, train_loader, val_loader, test_loader, args.epochs, args.lr
        )
        rows.append({"num_features": float(top_k),
                     "val_auroc_greedy": val_aucs[top_k - 1],
                     "test_auroc_finetuned": scores["roc_auc"]})
    return rows


def summarize(all_results: list[dict[str, dict[str, float]]]) -> None:
    """Prints mean +- std over seeds for every configuration."""
    print(f"\n{'configuration':<24}" + "".join(f"{m:>18}" for m in METRICS))
    for name in all_results[0]:
        cells = []
        for metric in METRICS:
            scores = np.array([r[name][metric] for r in all_results], dtype=float)
            cells.append(f"{np.nanmean(scores):.3f} +- {np.nanstd(scores):.3f}")
        print(f"{name:<24}" + "".join(f"{c:>18}" for c in cells))
    positives = [r["cbm_unstratified"]["test_positives"] for r in all_results]
    print(f"\nunstratified test positives per seed: {[int(p) for p in positives]}")


def main() -> None:
    """Runs the pipeline and the ablations; prints the result tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=None,
                        help="MIMIC-III v1.4 (or demo) folder; default: synthetic")
    parser.add_argument("--num-synthetic-patients", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-concepts", type=int, default=4)
    parser.add_argument("--l1-lambda", type=float, default=0.001)
    parser.add_argument("--cos-sim-lambda", type=float, default=0.01)
    parser.add_argument("--greedy-budget", type=int, default=8)
    parser.add_argument("--greedy-candidates", type=int, default=15)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        root = args.root
        if root is None:
            root = os.path.join(workdir, "synthetic_mimic3")
            os.makedirs(root)
            write_synthetic_mimic3(root, args.num_synthetic_patients)
            print(f"Synthetic MIMIC-III-format data written to {root}")
        samples = load_raw_samples(root, os.path.join(workdir, "cache"))
    labels = np.array([sample[LABEL] for sample in samples])
    print(f"Cohort: {len(samples)} patients, {int(labels.sum())} positive "
          f"({100 * labels.mean():.1f}%)")

    # The Trainer logs every epoch; keep the output to the result tables.
    logging.getLogger("pyhealth").setLevel(logging.WARNING)
    all_results = [run_seed(samples, seed, args) for seed in args.seeds]
    greedy_rows = run_greedy(samples, args.seeds[0], args)
    summarize(all_results)

    print("\nGreedy selection + fine-tuning (seed "
          f"{args.seeds[0]}; first row = all features):")
    for row in greedy_rows:
        print(f"  features={row['num_features']:>4.0f}  "
              f"val AUROC (greedy)={row['val_auroc_greedy']:.3f}  "
              f"test AUROC (fine-tuned)={row['test_auroc_finetuned']:.3f}")


if __name__ == "__main__":
    main()
