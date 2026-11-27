"""CaliForest on MIMIC-Extract: reproduction and task-setup ablations.

Reproduces the evaluation of

    Park, Yubin, and Joyce C. Ho. "CaliForest: Calibrated Random Forest for
    Health Data." ACM CHIL 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC8299436/

with PyHealth's ``MIMICExtractDataset``, the ``MIMICExtract*`` tasks, the
existing ``CaliForest`` model, and the paper's calibration metrics.

Models (as in the paper):
    CF-Iso / CF-Logit  PyHealth ``CaliForest`` (isotonic / logistic).
    RF-NoCal           Random forest on the full training split.
    RC-Iso / RC-Logit  Random forest on 70% of the training split, calibrated
                       (isotonic / Platt) on the held-out 30%.

Metrics: AUROC plus the paper's six calibration metrics (Brier, scaled Brier,
Hosmer-Lemeshow p, Spiegelhalter p, reliability-in-the-small/large), reported
as mean ± SD over Monte Carlo 70/30 train/test splits.

Experiments:
    1. Reproduction: the paper's four tasks at the default setup (first 24 h,
       6 h gap): in-hospital mortality, ICU mortality, LOS > 3 d, LOS > 7 d.
    2. Ablation 1 (observation window): ``window_hours`` = 6, 12, 24 for
       in-hospital mortality. Gap and minimum stay are held at 24 + 6 h, so
       the cohort is identical across windows and only the input changes.
    3. Ablation 2 (label definition): ``threshold_days`` = 3, 5, 7, 14 for ICU
       length of stay, which changes prevalence from common to rare.

Usage:
    # Synthetic data (default): runs anywhere in a few minutes.
    python examples/mimic_extract_mortality_califorest.py

    # Real MIMIC-Extract (folder with all_hourly_data.h5; needs PhysioNet
    # credentialed access and ``pip install tables``):
    python examples/mimic_extract_mortality_califorest.py \
        --root /path/to/mimic_extract --cache-dir /path/to/cache \
        --splits 10 --trees 300 --depth 10

Results:
    Synthetic data (default run: 300 stays, 265 after the 30 h cohort filter,
    6 variables, 3 splits, 50 trees, depth 5; about 4 minutes). These show
    the pipeline runs end to end; outcomes come from a simple severity score,
    so they say nothing about MIMIC. Mean over splits:

    In-hospital mortality, 24 h window:
        CF-Iso    AUROC 0.847   HL p 0.00   reliability-in-the-small 0.011
        RF-NoCal  AUROC 0.854   HL p 0.25   reliability-in-the-small 0.012
        RC-Iso    AUROC 0.833   HL p 0.04   reliability-in-the-small 0.023
    Ablation 1, CF-Iso AUROC for windows 6 / 12 / 24 h:
        0.853 / 0.837 / 0.847
    Ablation 2, prevalence for LOS > 3 / 5 / 7 / 14 d:
        58.5% / 32.8% / 16.2% / 3.4%
        (CF-Iso AUROC at 14 d: 0.624, unstable with ~3 test positives)

    Findings so far:

    - Window ablation: on synthetic data the severity signal is present from
      hour 0, so AUROC is flat across 6-24 h; on MIMIC, shorter windows are
      expected to lose information.
    - Threshold ablation: as the outcome gets rarer (LOS > 14 d, 3.4%), all
      metrics become unstable across splits.
    - PyHealth's ``CaliForest`` was worse calibrated than the plain random
      forest here. On a separate 3,000-sample synthetic check, the authors'
      reference CaliForest reached Hosmer-Lemeshow p = 0.49 while PyHealth's
      reached p = 0.00 with similar AUROC. The likely cause is that PyHealth
      weights samples by 1 / variance over all trees, while the paper uses
      out-of-bag trees with a Bayesian prior (Sec. 3.1). This affects how far
      PyHealth's CaliForest can reproduce the paper's calibration results.

    MIMIC-Extract: results on the real cohort are not included here, since
    the data needs credentialed PhysioNet access. Running the script with
    ``--root`` on MIMIC-Extract's ``all_hourly_data.h5`` reproduces them
    (the paper uses 10 splits, 300 trees, depth 10; depth 5 for LOS > 7 d).
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from pyhealth.datasets import MIMICExtractDataset
from pyhealth.datasets.mimicextract import build_mimic_extract_tables
from pyhealth.metrics.calibration import (
    hosmer_lemeshow,
    reliability_in_the_large,
    reliability_in_the_small,
    scaled_brier,
    spiegelhalter,
)
from pyhealth.models import CaliForest
from pyhealth.tasks import (
    MIMICExtractLengthOfStay,
    MIMICExtractMortalityHospital,
    MIMICExtractMortalityICU,
)

MODELS = ["CF-Iso", "CF-Logit", "RF-NoCal", "RC-Iso", "RC-Logit"]
METRICS = ["auroc", "brier", "scaled_brier", "hl_p", "spiegel_p", "rel_small",
           "rel_large"]  # fmt: skip


# ---------------------------------------------------------------------------
# Synthetic MIMIC-Extract data
# ---------------------------------------------------------------------------

SYNTHETIC_VARIABLES = [
    "Heart Rate",
    "Glucose",
    "Systolic blood pressure",
    "Respiratory rate",
    "Oxygen saturation",
    "Temperature",
]


def make_synthetic_mimic_extract(
    root: Path, n_stays: int = 300, n_hours: int = 30, seed: int = 0
) -> None:
    """Writes a fake MIMIC-Extract cohort in the dataset's CSV format.

    Each stay has a hidden severity score that shifts its vitals and drives
    its outcomes, so the tasks are learnable. About 30% of hourly values are
    missing, like real ICU data.
    """
    rng = np.random.default_rng(seed)
    severity = rng.normal(size=n_stays)
    stay_ids = np.arange(n_stays) + 1000
    mort_hosp = rng.random(n_stays) < 1 / (1 + np.exp(-(-2.0 + 1.5 * severity)))
    mort_icu = mort_hosp & (rng.random(n_stays) < 0.7)
    los_icu = np.exp(rng.normal(1.2 + 0.5 * severity, 0.6))  # days
    intime = pd.Timestamp("2150-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n_stays), unit="D"
    )
    patients = pd.DataFrame(
        {
            "subject_id": stay_ids,
            "hadm_id": stay_ids + 50_000,
            "icustay_id": stay_ids,
            "los_icu": los_icu,
            "mort_icu": mort_icu.astype(int),
            "mort_hosp": mort_hosp.astype(int),
            "gender": rng.choice(["F", "M"], n_stays),
            "age": rng.uniform(18, 90, n_stays).round(1),
            "intime": intime,
            "outtime": intime + pd.to_timedelta(los_icu, unit="D"),
        }
    ).set_index(["subject_id", "hadm_id", "icustay_id"])

    rows, index = [], []
    for i, stay in enumerate(stay_ids):
        for hour in range(n_hours):
            values = []
            for j in range(len(SYNTHETIC_VARIABLES)):
                if rng.random() < 0.3:
                    values += [0, np.nan, np.nan]  # not measured this hour
                else:
                    mean = (j + 1) * 10 + 3 * severity[i] + rng.normal()
                    values += [1, mean, np.nan]
            rows.append(values)
            index.append((stay, stay + 50_000, stay, hour))
    vitals_labs = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(
            index, names=["subject_id", "hadm_id", "icustay_id", "hours_in"]
        ),
        columns=pd.MultiIndex.from_product(
            [SYNTHETIC_VARIABLES, ["count", "mean", "std"]],
            names=["LEVEL2", "Aggregation Function"],
        ),
    )
    build_mimic_extract_tables(patients, vitals_labs, root)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_predict(
    name: str, samples, x_tr, y_tr, x_te, trees: int, depth: int, seed: int
) -> np.ndarray:
    """Trains one of the paper's five models and returns test probabilities."""
    if name.startswith("CF-"):
        calibration = "isotonic" if name == "CF-Iso" else "logistic"
        model = CaliForest(
            dataset=samples,
            n_estimators=trees,
            max_depth=depth,
            calibration=calibration,
            random_state=seed,
        )
        model.fit_model(features=x_tr, labels=y_tr)
        return model.predict_proba_numpy(features=x_te).ravel()

    def forest():
        return RandomForestClassifier(
            n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=-1
        )

    if name == "RF-NoCal":
        return forest().fit(x_tr, y_tr).predict_proba(x_te)[:, 1]

    # RC-*: forest on 70% of the training split, calibrator on the other 30%.
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=seed
    )
    rf = forest().fit(x_fit, y_fit)
    p_cal, p_te = rf.predict_proba(x_cal)[:, 1], rf.predict_proba(x_te)[:, 1]
    if name == "RC-Iso":
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
        return iso.predict(p_te)
    platt = LogisticRegression().fit(p_cal.reshape(-1, 1), y_cal)
    return platt.predict_proba(p_te.reshape(-1, 1))[:, 1]


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """AUROC and the paper's six calibration metrics."""
    p = np.clip(p, 1e-8, 1 - 1e-8)  # keep Spiegelhalter defined
    return {
        "auroc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "scaled_brier": scaled_brier(p, y),
        "hl_p": hosmer_lemeshow(p, y)[1],
        "spiegel_p": spiegelhalter(p, y)[1],
        "rel_small": reliability_in_the_small(p, y),
        "rel_large": reliability_in_the_large(p, y),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_task(dataset, task, models, args) -> pd.DataFrame:
    """Runs ``task`` and evaluates ``models`` over Monte Carlo 70/30 splits."""
    samples = dataset.set_task(task, num_workers=1)
    x = np.stack([s["features"].numpy().ravel() for s in samples])
    y = np.array([int(s["label"].item()) for s in samples])
    print(f"  {len(y)} stays, {x.shape[1]} features, prevalence {y.mean():.1%}")

    records = []
    for seed in range(args.splits):
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=seed
        )
        for name in models:
            p = fit_predict(name, samples, x_tr, y_tr, x_te, args.trees,
                            args.depth, seed)  # fmt: skip
            records.append({"model": name, "split": seed, **evaluate(y_te, p)})
    return pd.DataFrame(records)


def summarize(results: pd.DataFrame, by: list) -> pd.DataFrame:
    """Formats results as 'mean ± sd' per group."""
    grouped = results.groupby(by, sort=False)[METRICS]
    mean, sd = grouped.mean(), grouped.std().fillna(0)
    return mean.combine(
        sd, lambda m, s: m.map("{:.3f}".format) + " +/- " + s.map("{:.3f}".format)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="MIMIC-Extract folder (default: synthetic)")
    parser.add_argument("--cache-dir", help="PyHealth cache folder")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    tmp = None
    if args.root is None:
        tmp = Path(tempfile.mkdtemp())
        root, cache = tmp / "data", tmp / "cache"
        root.mkdir()
        make_synthetic_mimic_extract(root)
        dataset = MIMICExtractDataset(root=str(root), cache_dir=str(cache))
    else:
        # max_hours=30 keeps the hourly table small: the window plus the gap.
        dataset = MIMICExtractDataset(
            root=args.root, max_hours=30, cache_dir=args.cache_dir
        )

    # Use every hourly variable in the data, in a fixed order, like the paper.
    variables = sorted(
        dataset.global_event_df.filter(pl.col("event_type") == "vitals_labs")
        .select(pl.col("vitals_labs/variable").unique())
        .collect()["vitals_labs/variable"]
        .to_list()
    )
    print(f"{len(variables)} hourly variables")

    tables = {}

    print("\n1. Reproduction (24 h window, 6 h gap)")
    paper_tasks = {
        "Mortality (hospital)": MIMICExtractMortalityHospital(variables=variables),
        "Mortality (ICU)": MIMICExtractMortalityICU(variables=variables),
        "LOS > 3 d": MIMICExtractLengthOfStay(variables=variables, threshold_days=3),
        "LOS > 7 d": MIMICExtractLengthOfStay(variables=variables, threshold_days=7),
    }
    frames = []
    for label, task in paper_tasks.items():
        print(f"- {label}")
        frames.append(run_task(dataset, task, MODELS, args).assign(task=label))
    tables["reproduction"] = summarize(pd.concat(frames), ["task", "model"])

    ablation_models = ["CF-Iso", "RF-NoCal", "RC-Iso"]

    print("\n2. Ablation 1: observation window (in-hospital mortality)")
    frames = []
    for hours in (6, 12, 24):
        # Hold window + gap at 30 h so every window uses the same cohort.
        task = MIMICExtractMortalityHospital(
            variables=variables, window_hours=hours, gap_hours=30 - hours
        )
        print(f"- window {hours} h")
        frames.append(
            run_task(dataset, task, ablation_models, args).assign(window_h=hours)
        )
    tables["ablation_window"] = summarize(pd.concat(frames), ["window_h", "model"])

    print("\n3. Ablation 2: LOS threshold")
    frames = []
    for days in (3, 5, 7, 14):
        task = MIMICExtractLengthOfStay(variables=variables, threshold_days=days)
        print(f"- LOS > {days} d")
        frames.append(
            run_task(dataset, task, ablation_models, args).assign(threshold_d=days)
        )
    tables["ablation_threshold"] = summarize(
        pd.concat(frames), ["threshold_d", "model"]
    )

    for name, table in tables.items():
        print(f"\n== {name} ==")
        print(table.to_string())

    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)  # cache files may stay locked


if __name__ == "__main__":
    main()
