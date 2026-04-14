"""Ablation study for AMA discharge prediction on MIMIC-III.

This script demonstrates the AMAPredictionMIMIC3 task with three feature
ablations and evaluates model fairness using AUROC across demographic
subgroups (race, age, insurance). A logistic regression classifier is
trained on the extracted features to analyze how demographic information
affects prediction of against-medical-advice (AMA) discharge.

Paper: Boag, W.; Suresh, H.; Celi, L. A.; Szolovits, P.; and Ghassemi, M.
"Racial Disparities and Mistrust in End-of-Life Care." Machine Learning
for Healthcare Conference, PMLR 106:211-235, 2018.

Ablation configurations tested:
    1. BASELINE: demographics (gender, insurance) + age + los
    2. BASELINE+RACE: adds normalized ethnicity feature
    3. BASELINE+RACE+SUBSTANCE: adds substance use diagnosis flag

Results:
    For each baseline, we report:
    - Overall AUROC averaged over N random 60/40 train/test splits
    - Subgroup performance (AUROC) stratified by:
      * Race (White, Black, Hispanic, Asian, Native American, Other)
      * Age Group (Young 18-44, Middle 45-64, Senior 65+)
      * Insurance (Public, Private, Self Pay)
    - Fairness metrics per subgroup:
      * Demographic Parity: % predicted AMA per group
      * Equal Opportunity: True Positive Rate per group
    These reveal disparities in model behavior across demographics.

Usage (synthetic exhaustive grid -- default when ``--root`` is omitted):
    python examples/mimic3_ama_prediction_logistic_regression.py

Usage (synthetic random demo):
    python examples/mimic3_ama_prediction_logistic_regression.py \\
        --data-source synthetic --synthetic-mode random --patients 200

Usage (real MIMIC-III; same as ``--root /path`` with ``--data-source auto``):
    python examples/mimic3_ama_prediction_logistic_regression.py \\
        --data-source real --root /path/to/mimic-iii/1.4 --splits 100 --epochs 10
"""

import argparse
import gzip
import itertools
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import LogisticRegression
from pyhealth.tasks import AMAPredictionMIMIC3
from pyhealth.trainer import Trainer


def generate_synthetic_mimic3(
    root: str,
    n_patients: int = 50,
    avg_admissions_per_patient: int = 2,
    seed: int = 42,
    mode: str = "exhaustive",
) -> None:
    """Write gzipped PATIENTS, ADMISSIONS, and ICUSTAYS CSVs for local demos.

    ``mode="exhaustive"`` (default) emits one patient per element of the
    Cartesian product of task-relevant factors so every combination appears
    at least once: gender × MIMIC ethnicity string × raw insurance × age
    band (maps to Young / Middle / Senior) × AMA vs non-AMA discharge ×
    substance vs non-substance diagnosis text.  A few extra rows cover
    NEWBORN filtering, EXPIRED, SNF, and missing insurance (``Other``).

    ``mode="random"`` reproduces the legacy stochastic generator (use for
    quick tests via ``--synthetic-mode random``).

    Args:
        root: Directory to write CSV files to.
        n_patients: Used only when ``mode="random"`` (patient count).
        avg_admissions_per_patient: Poisson mean per patient (random mode).
        seed: RNG seed (random mode only).
        mode: ``"exhaustive"`` or ``"random"``.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    patients_data: List[dict] = []
    admissions_data: List[dict] = []
    icustays_data: List[dict] = []

    genders = ["M", "F"]
    ethnicities = [
        "WHITE",
        "BLACK/AFRICAN AMERICAN",
        "HISPANIC OR LATINO",
        "ASIAN - CHINESE",
        "AMERICAN INDIAN/ALASKA NATIVE",
        "UNKNOWN/NOT SPECIFIED",
    ]
    # Raw insurance values (normalize to Public / Private / Self Pay / Other)
    insurances_raw: List[Optional[str]] = [
        "Medicare",
        "Medicaid",
        "Government",
        "Private",
        "Self Pay",
        None,
    ]
    admission_types = ["EMERGENCY", "URGENT", "NEWBORN", "ELECTIVE"]
    discharge_locations = [
        "HOME",
        "SKILLED NURSING FACILITY",
        "LONG TERM CARE",
        "LEFT AGAINST MEDICAL ADVI",
        "EXPIRED",
    ]
    diagnoses_substance = [
        "ALCOHOL WITHDRAWAL",
        "OPIOID DEPENDENCE",
        "HEROIN OVERDOSE",
        "COCAINE INTOXICATION",
        "DRUG WITHDRAWAL SEIZURE",
        "ETOH ABUSE",
        "SUBSTANCE ABUSE",
        "OVERDOSE - ACCIDENTAL",
    ]
    diagnoses_other = [
        "PNEUMONIA",
        "ACUTE MYOCARDIAL INFARCTION",
        "CHEST PAIN",
        "CONGESTIVE HEART FAILURE",
        "SEPSIS",
        "ACUTE KIDNEY INJURY",
        "ACUTE RESPIRATORY FAILURE",
        "ASPIRATION",
    ]

    def write_csv_gz(filename: str, data: List[dict]) -> None:
        df = pd.DataFrame(data)
        filepath = root_path / f"{filename}.gz"
        with gzip.open(filepath, "wt") as f:
            df.to_csv(f, index=False)
        print(f"  Created {filename}.gz ({len(data)} rows)")

    def append_visit(
        subject_id: int,
        hadm_id: int,
        icustay_id: int,
        *,
        gender: str,
        age_years: int,
        ethnicity: str,
        insurance_raw: Optional[str],
        admission_type: str,
        discharge_loc: str,
        diagnosis: str,
        day_offset: int,
    ) -> int:
        """Append one patient + admission + icustay; return next icustay_id."""
        dob = datetime(2000, 1, 1) - timedelta(days=int(age_years * 365))
        patients_data.append({
            "subject_id": subject_id,
            "gender": gender,
            "dob": dob.strftime("%Y-%m-%d %H:%M:%S"),
            "dod": None,
            "dod_hosp": None,
            "dod_ssn": None,
            "expire_flag": 0,
        })
        admit_time = datetime(2150, 1, 1) + timedelta(days=day_offset)
        discharge_time = admit_time + timedelta(days=7)
        admissions_data.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "admission_type": admission_type,
            "admission_location": "EMERGENCY ROOM ADMIT",
            "insurance": insurance_raw,
            "language": "ENGLISH",
            "religion": "CHRISTIAN",
            "marital_status": "SINGLE",
            "ethnicity": ethnicity,
            "edregtime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "edouttime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnosis": diagnosis,
            "discharge_location": discharge_loc,
            "dischtime": discharge_time.strftime("%Y-%m-%d %H:%M:%S"),
            "admittime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "hospital_expire_flag": 1 if discharge_loc == "EXPIRED" else 0,
        })
        icu_intime = admit_time + timedelta(hours=2)
        icu_outtime = discharge_time - timedelta(hours=2)
        if icu_intime < icu_outtime:
            icustays_data.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "icustay_id": icustay_id,
                "first_careunit": "MICU",
                "last_careunit": "MICU",
                "dbsource": "metavision",
                "intime": icu_intime.strftime("%Y-%m-%d %H:%M:%S"),
                "outtime": icu_outtime.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return icustay_id + 1
        return icustay_id

    if mode == "exhaustive":
        # Ages map to Young (18-44) / Middle (45-64) / Senior (65+).
        age_bands = [30, 52, 72]
        discharge_ama = ["HOME", "LEFT AGAINST MEDICAL ADVI"]
        diagnosis_texts = ["PNEUMONIA", "ALCOHOL WITHDRAWAL"]
        combos = itertools.product(
            genders,
            ethnicities,
            insurances_raw,
            age_bands,
            discharge_ama,
            diagnosis_texts,
        )
        subject_id = 1
        hadm_id = 100
        icustay_id = 1000
        for idx, (gender, eth, ins_raw, age_y, disch, diag) in enumerate(combos):
            icustay_id = append_visit(
                subject_id,
                hadm_id,
                icustay_id,
                gender=gender,
                age_years=age_y,
                ethnicity=eth,
                insurance_raw=ins_raw,
                admission_type="EMERGENCY",
                discharge_loc=disch,
                diagnosis=diag,
                day_offset=idx % 500,
            )
            subject_id += 1
            hadm_id += 1

        # Extra coverage: SNF discharge, EXPIRED, NEWBORN (skipped by task).
        exhaustive_grid_n = (
            len(genders)
            * len(ethnicities)
            * len(insurances_raw)
            * len(age_bands)
            * len(discharge_ama)
            * len(diagnosis_texts)
        )
        for k, extra in enumerate(
            (
                ("M", 45, "WHITE", "Private", "EMERGENCY", "SKILLED NURSING FACILITY", "SEPSIS"),
                ("F", 55, "BLACK/AFRICAN AMERICAN", "Medicaid", "EMERGENCY", "EXPIRED", "CHEST PAIN"),
                ("M", 28, "HISPANIC OR LATINO", "Private", "NEWBORN", "HOME", "PNEUMONIA"),
            ),
        ):
            g, age_y, eth, ins, adm_type, disch, diag = extra
            icustay_id = append_visit(
                subject_id,
                hadm_id,
                icustay_id,
                gender=g,
                age_years=age_y,
                ethnicity=eth,
                insurance_raw=ins,
                admission_type=adm_type,
                discharge_loc=disch,
                diagnosis=diag,
                day_offset=(exhaustive_grid_n + k) % 500,
            )
            subject_id += 1
            hadm_id += 1

        print(
            f"Generating exhaustive synthetic MIMIC-III in {root_path} "
            f"({len(patients_data)} patients, cross-product + edge rows)...",
        )
    elif mode == "random":
        np.random.seed(seed)
        subject_id = 1
        hadm_id = 100
        icustay_id = 1000
        insurances = ["Medicare", "Medicaid", "Private", "Self Pay", "Government"]
        for i in range(n_patients):
            gender = genders[i % len(genders)]
            ethnicity = ethnicities[i % len(ethnicities)]
            insurance = insurances[i % len(insurances)]

            age_at_visit = int(
                np.random.choice([25, 45, 65, 85]) + np.random.randint(-5, 5)
            )
            dob = datetime(2000, 1, 1) - timedelta(days=age_at_visit * 365)

            patients_data.append({
                "subject_id": subject_id,
                "gender": gender,
                "dob": dob.strftime("%Y-%m-%d %H:%M:%S"),
                "dod": None,
                "dod_hosp": None,
                "dod_ssn": None,
                "expire_flag": 0,
            })

            n_admissions = max(
                1, int(np.random.poisson(avg_admissions_per_patient)),
            )
            for j in range(n_admissions):
                admit_time = datetime(2150, 1, 1) + timedelta(days=int(j * 100))
                discharge_time = admit_time + timedelta(
                    days=int(np.random.randint(1, 30)),
                )

                admission_type = admission_types[(i + j) % len(admission_types)]

                if np.random.random() < 0.15:
                    discharge_loc = "LEFT AGAINST MEDICAL ADVI"
                elif np.random.random() < 0.05:
                    discharge_loc = "EXPIRED"
                else:
                    discharge_loc = discharge_locations[
                        (i + j) % (len(discharge_locations) - 2)
                    ]

                if np.random.random() < 0.2:
                    diagnosis = diagnoses_substance[
                        np.random.randint(0, len(diagnoses_substance))
                    ]
                else:
                    diagnosis = diagnoses_other[
                        np.random.randint(0, len(diagnoses_other))
                    ]

                admissions_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "admission_type": admission_type,
                    "admission_location": "EMERGENCY ROOM ADMIT",
                    "insurance": insurance,
                    "language": "ENGLISH",
                    "religion": "CHRISTIAN",
                    "marital_status": "SINGLE",
                    "ethnicity": ethnicity,
                    "edregtime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "edouttime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "diagnosis": diagnosis,
                    "discharge_location": discharge_loc,
                    "dischtime": discharge_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "admittime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hospital_expire_flag": 1
                    if discharge_loc == "EXPIRED"
                    else 0,
                })

                icu_intime = admit_time + timedelta(
                    hours=int(np.random.randint(0, 12)),
                )
                icu_outtime = discharge_time - timedelta(
                    hours=int(np.random.randint(0, 12)),
                )

                if icu_intime < icu_outtime:
                    icustays_data.append({
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "icustay_id": icustay_id,
                        "first_careunit": "MICU",
                        "last_careunit": "MICU",
                        "dbsource": "metavision",
                        "intime": icu_intime.strftime("%Y-%m-%d %H:%M:%S"),
                        "outtime": icu_outtime.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    icustay_id += 1

                hadm_id += 1

            subject_id += 1

        print(f"Generating random synthetic MIMIC-III in {root_path}...")
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'exhaustive' or 'random'.")

    write_csv_gz("PATIENTS.csv", patients_data)
    write_csv_gz("ADMISSIONS.csv", admissions_data)
    write_csv_gz("ICUSTAYS.csv", icustays_data)
    print("Done.")


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
                "pct_pred": float(yd.mean()) * 100,
                "tpr": float(yd[yt == 1].mean()) * 100 if pos > 0
                       else float("nan"),
                "n": n,
            }

    return {
        "auroc": overall_auroc,
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
            aurocs, pcts, tprs, ns = [], [], [], []
            for r in valid:
                m = r["subgroups"].get(attr, {}).get(grp)
                if m is None:
                    continue
                aurocs.append(m["auroc"])
                pcts.append(m["pct_pred"])
                tprs.append(m["tpr"])
                ns.append(m["n"])

            agg["subgroups"][attr][grp] = {
                "auroc_mean": _nanmean(aurocs),
                "auroc_std": _nanstd(aurocs),
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

    print(f"\n  2. Subgroup Performance")
    for attr, grps in agg["subgroups"].items():
        print(f"     {attr}:")
        print(f"       {'Group':<20} {'AUROC':>15} {'n_avg':>7}")
        print(f"       {'-'*42}")
        for grp, m in grps.items():
            a_str = f"{_fmt(m['auroc_mean'])}+/-{_fmt(m['auroc_std'])}"
            print(f"       {grp:<20} {a_str:>15} {m['n_avg']:>7}")

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
    parser.add_argument(
        "--data-source",
        choices=("auto", "synthetic", "real"),
        default="auto",
        help="auto: use --root if set, else synthetic. synthetic: always local CSVs. "
        "real: require --root to MIMIC-III on disk.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=("exhaustive", "random"),
        default="exhaustive",
        help="exhaustive: full cross-product of demographics×AMA×substance (default). "
        "random: stochastic demo (--patients applies).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="MIMIC-III root directory (required for --data-source real, or sets "
        "auto mode to real when provided).",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=100,
        help="Synthetic patient count (random mode only; exhaustive ignores this).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random synthetic mode.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of random 60/40 splits (default 5 for speed with synthetic data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs per split (default 3 for speed with synthetic data)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="With random synthetic: only 10 patients. Exhaustive grid unchanged.",
    )
    args = parser.parse_args()

    if args.data_source == "real" and not args.root:
        parser.error("--data-source real requires --root /path/to/mimic-iii/1.4")

    use_synthetic = args.data_source == "synthetic" or (
        args.data_source == "auto" and args.root is None
    )
    if args.data_source == "synthetic" and args.root:
        print(
            "Note: --root is ignored with --data-source synthetic "
            "(data is written to a temporary directory).\n",
        )

    cache_dir = tempfile.mkdtemp(prefix="ama_lr_")

    if use_synthetic:
        print("[Setup] Generating synthetic MIMIC-III dataset...")
        data_dir = tempfile.mkdtemp(prefix="synthetic_mimic3_")
        n_patients = (
            10 if args.dev and args.synthetic_mode == "random" else args.patients
        )
        generate_synthetic_mimic3(
            data_dir,
            n_patients=n_patients,
            avg_admissions_per_patient=2,
            seed=args.seed,
            mode=args.synthetic_mode,
        )
        args.root = str(data_dir)
        print(f"        Synthetic data: {data_dir}")
        print(f"        Mode: {args.synthetic_mode}\n")
    else:
        assert args.root is not None
        print(f"Using real MIMIC-III from: {args.root}\n")

    print(f"Cache: {cache_dir}")
    print(f"Root:  {args.root}")
    print(f"Splits: {args.splits}  |  Epochs: {args.epochs}")

    print("\n[1/4] Loading dataset...")
    t0 = time.time()
    dataset = MIMIC3Dataset(
        root=args.root,
        tables=[],
        cache_dir=cache_dir,
        dev=args.dev,
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
        print("  For synthetic data: this is expected if AMA rate is low.")
        print("  For synthetic random mode with more patients:")
        print("    python examples/"
              "mimic3_ama_prediction_logistic_regression.py \\")
        print("        --data-source synthetic --synthetic-mode random "
              "--patients 500\n")
        print("  For real MIMIC-III:")
        print("    python examples/"
              "mimic3_ama_prediction_logistic_regression.py \\")
        print("        --data-source real --root /path/to/mimic-iii/1.4")
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
