# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Deep Cox Mixtures on SUPPORT2 with a small ablation over K, width and indicators.

Pipeline: Support2Dataset -> dataset.set_task(Support2SurvivalTask()) ->
samples -> per-split preprocessing -> DeepCoxMixtures.fit -> evaluation on a
held-out split. The ablation trains one model per configuration:

- K (number of Cox components): 1, 3, 6. K=1 is a single deep Cox model, so it
  shows what the mixture adds.
- Hidden width: 50, 100.
- Missing-value indicators for eight laboratory values: off/on. This is an
  extension, not part of the paper.

Each configuration is scored on the validation split (used to pick the
configuration) and on the test split (reported only). A linear Cox model is
included for reference. Metrics: mean IPCW Brier score at the 25/50/75%
quantiles of training event times (lower is better) and IPCW concordance at the
median horizon (higher is better).

Run on generated data (the default; about 30 s on a laptop CPU):

    python examples/support2_survival_dcm.py --synthetic

Run on the public SUPPORT2 CSV (https://hbiostat.org/data/repo/support2csv.zip):

    python examples/support2_survival_dcm.py --data /path/to/support2.csv \
        --epochs 100 --output /path/to/output

Generated-data numbers only show that the pipeline runs; they say nothing
about clinical performance. The five-fold SUPPORT2 study reported in
``examples/support2_dcm/RESULTS.md`` is a separate driver (see its README).

Needs ``pip install 'pyhealth[dcm]'`` plus the authors' auton-survival package;
see :class:`pyhealth.models.DeepCoxMixtures`.
"""

import argparse
import itertools
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_ipcw
from sksurv.util import Surv

from pyhealth.datasets import Support2Dataset, create_sample_dataset, get_dataloader
from pyhealth.models import DeepCoxMixtures
from pyhealth.tasks import Support2SurvivalTask
from pyhealth.tasks.support2_survival import (
    SUPPORT2_DCM_CATEGORICAL,
    SUPPORT2_DCM_NUMERIC,
)

CATEGORICAL = list(SUPPORT2_DCM_CATEGORICAL)
NUMERIC = list(SUPPORT2_DCM_NUMERIC)
LABS = ["wblc", "alb", "bili", "crea", "sod", "ph", "glucose", "bun"]
CONFIG = Path(__file__).parents[1] / "pyhealth/datasets/configs/support2.yaml"

# Ablation grid: (K, hidden width, missing-value indicators).
GRID = list(itertools.product([1, 3, 6], [50, 100], [False, True]))


def write_synthetic_support2(path: Path, n: int, seed: int) -> None:
    """Write generated SUPPORT2-shaped rows with two latent risk groups.

    Args:
        path: Destination CSV.
        n: Number of generated pseudo-patients.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({key: rng.normal(10, 2, n) for key in NUMERIC})
    raw["age"] = rng.normal(65, 10, n)
    for key in CATEGORICAL:
        raw[key] = rng.choice(["a", "b", "c"], n)
    # Two latent groups with different baseline hazards; group membership
    # depends on dzgroup, risk within a group on age and creatinine.
    group = (raw["dzgroup"] == "a") | (rng.random(n) < 0.2)
    scale = np.where(group, 40.0, 200.0)
    risk = 0.03 * (raw["age"] - 65) + 0.3 * (raw["crea"] - 10)
    event_time = rng.exponential(scale * np.exp(-risk)) + 1
    censor_time = rng.exponential(250, n) + 1
    raw["d.time"] = np.minimum(event_time, censor_time).round(1)
    raw["death"] = (event_time <= censor_time).astype(int)
    raw[LABS] = raw[LABS].mask(rng.random((n, len(LABS))) < 0.15)
    attributes = yaml.safe_load(CONFIG.read_text())["tables"]["support2"][
        "attributes"
    ]
    for key in attributes:
        if key not in raw:
            raw[key] = np.nan
    raw.insert(0, "sno", np.arange(1, n + 1))
    raw.to_csv(path, index=False)


def task_samples(data_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """Load SUPPORT2 with PyHealth and run Support2SurvivalTask.

    Args:
        data_dir: Directory containing ``support2.csv``.
        cache_dir: PyHealth cache directory.

    Returns:
        One row per sample with the 24 covariates, ``time`` and ``event``.
    """
    dataset = Support2Dataset(
        root=str(data_dir), tables=["support2"], cache_dir=cache_dir
    )
    samples = dataset.set_task(Support2SurvivalTask(), num_workers=1)
    rows = [
        {
            "patient_id": str(sample["patient_id"]),
            **sample["covariates"],
            "time": float(sample["time"]),
            "event": int(sample["event"]),
        }
        for sample in samples
    ]
    frame = pd.DataFrame(rows)
    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
    for key in CATEGORICAL:
        frame[key] = frame[key].map(lambda v: np.nan if v is None else str(v))
    return frame.sort_values("patient_id", kind="stable").reset_index(drop=True)


def covariates(frame: pd.DataFrame, indicators: bool) -> pd.DataFrame:
    """Select the 24 covariates, optionally adding lab missingness indicators.

    Args:
        frame: Task samples.
        indicators: Whether to append one 0/1 column per lab.

    Returns:
        Covariate frame for the preprocessor.
    """
    out = frame[CATEGORICAL + NUMERIC].copy()
    if indicators:
        for key in LABS:
            out[key + "__missing"] = out[key].isna().astype(float)
    return out


def preprocessor(indicators: bool) -> ColumnTransformer:
    """Median-impute and scale numbers, one-hot encode categories.

    Args:
        indicators: Pass the lab missingness indicators through unchanged.

    Returns:
        An unfitted transformer; fit it on the training split only.
    """
    parts = [
        (
            "numeric",
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
            ),
            NUMERIC,
        ),
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            CATEGORICAL,
        ),
    ]
    if indicators:
        parts.append(("missing", "passthrough", [k + "__missing" for k in LABS]))
    return ColumnTransformer(parts, sparse_threshold=0)


def to_dataset(x: np.ndarray, frame: pd.DataFrame, name: str) -> Any:
    """Wrap a dense matrix and labels as a PyHealth SampleDataset.

    Args:
        x: Preprocessed covariates.
        frame: Matching task samples (patient id, time, event).
        name: Dataset name.

    Returns:
        SampleDataset with ``features``, ``time`` and ``event`` fields.
    """
    samples = [
        {
            "patient_id": pid,
            "features": row.tolist(),
            "time": [t],
            "event": [float(e)],
        }
        for pid, row, t, e in zip(frame.patient_id, x, frame.time, frame.event)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"time": "tensor", "event": "tensor"},
        dataset_name=name,
    )


def score(
    train: np.ndarray, test: np.ndarray, curves: np.ndarray, horizons: np.ndarray
) -> tuple[float, float]:
    """Mean IPCW Brier over horizons and IPCW concordance at the median one.

    Args:
        train: Structured training outcomes (censoring weights come from here).
        test: Structured outcomes being scored.
        curves: Survival probabilities ``[patients, horizons]``.
        horizons: Evaluation times inside the follow-up of both sets.

    Returns:
        ``(mean_brier, concordance)``.
    """
    _, brier = brier_score(train, test, curves, horizons)
    middle = len(horizons) // 2
    concordance = concordance_index_ipcw(
        train, test, 1 - curves[:, middle], tau=horizons[middle]
    )[0]
    return float(np.mean(brier)), float(concordance)


def run(args: argparse.Namespace, work: Path) -> pd.DataFrame:
    """Run the pipeline and the ablation; return one row per configuration.

    Args:
        args: Parsed command-line arguments.
        work: Directory for the generated CSV and PyHealth cache.

    Returns:
        Results table.
    """
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.data is None:
        write_synthetic_support2(data_dir / "support2.csv", args.n, args.seed)
    else:
        # The public CSV has no patient id column; number rows in a copy.
        raw = pd.read_csv(args.data)
        if "sno" not in raw:
            raw.insert(0, "sno", np.arange(1, len(raw) + 1))
        raw.to_csv(data_dir / "support2.csv", index=False)
    frame = task_samples(data_dir, work / "cache")
    print(f"Task samples: {len(frame)}, events: {int(frame.event.sum())}")

    # Event-stratified 60/20/20 split; preprocessing is fit on train only.
    index = np.arange(len(frame))
    train, test = train_test_split(
        index, test_size=0.2, random_state=args.seed, stratify=frame.event
    )
    train, val = train_test_split(
        train, test_size=0.25, random_state=args.seed, stratify=frame.event[train]
    )
    y = Surv.from_arrays(frame.event.astype(bool), frame.time)
    event_times = y[train]["time"][y[train]["event"]]
    horizons = np.quantile(event_times, [0.25, 0.5, 0.75])
    print(f"Horizons (days): {np.round(horizons, 1).tolist()}")

    rows = []
    for k, width, indicators in GRID:
        prep = preprocessor(indicators).fit(covariates(frame.iloc[train], indicators))
        x = prep.transform(covariates(frame, indicators)).astype(np.float32)
        parts = {
            name: to_dataset(x[ix], frame.iloc[ix], f"support2_{name}")
            for name, ix in [("train", train), ("val", val), ("test", test)]
        }
        loaders = {
            name: get_dataloader(ds, batch_size=256, shuffle=False)
            for name, ds in parts.items()
        }
        model = DeepCoxMixtures(parts["train"], k=k, width=width, seed=args.seed)
        model.fit(
            loaders["train"],
            loaders["val"],
            horizons=horizons,
            epochs=args.epochs,
            capture_output=True,
        )
        row = {"model": "DCM", "K": k, "width": width, "indicators": indicators}
        for name, ix in [("val", val), ("test", test)]:
            # forward() returns y_prob: survival at the horizons, per patient.
            with torch.no_grad():
                curves = np.concatenate(
                    [model(**batch)["y_prob"].numpy() for batch in loaders[name]]
                )
            brier, concordance = score(y[train], y[ix], curves, horizons)
            row.update({f"{name}_brier": brier, f"{name}_cindex": concordance})
        row["epochs"] = model.fit_audit["epochs"]
        rows.append(row)
        print(", ".join(f"{key}={value}" for key, value in row.items()))

    # Linear Cox reference on the same split and preprocessing.
    prep = preprocessor(False).fit(covariates(frame.iloc[train], False))
    x = prep.transform(covariates(frame, False))
    cox = CoxPHSurvivalAnalysis(alpha=0.1).fit(x[train], y[train])
    row = {"model": "Cox PH", "K": "-", "width": "-", "indicators": False}
    for name, ix in [("val", val), ("test", test)]:
        curves = np.array([f(horizons) for f in cox.predict_survival_function(x[ix])])
        brier, concordance = score(y[train], y[ix], curves, horizons)
        row.update({f"{name}_brier": brier, f"{name}_cindex": concordance})
    rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> pd.DataFrame:
    """Parse arguments, run the ablation and print the results table.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Results table.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true", help="use generated data (default)"
    )
    source.add_argument("--data", help="SUPPORT2 CSV instead of generated data")
    parser.add_argument("--n", type=int, default=600, help="generated rows")
    parser.add_argument("--epochs", type=int, default=30, help="max EM epochs")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", help="directory for results.csv and the cache; default: temp"
    )
    args = parser.parse_args(argv)
    if args.output is None:
        with tempfile.TemporaryDirectory() as tmp:
            table = run(args, Path(tmp))
    else:
        output = Path(args.output)
        table = run(args, output)
        table.to_csv(output / "results.csv", index=False)
    dcm = table[table.model == "DCM"]
    best = dcm.loc[dcm.val_brier.astype(float).idxmin()]
    pd.set_option("display.width", 120)
    print(table.round(4).to_string(index=False))
    print(
        f"Selected on validation Brier: K={best.K}, width={best.width}, "
        f"indicators={best.indicators}; test Brier {best.test_brier:.4f}"
    )
    return table


if __name__ == "__main__":
    main()
