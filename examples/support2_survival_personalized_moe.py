"""SUPPORT2 personalized MoE dropout ablation following Morrill et al.

    pip install SurvivalEVAL
    python examples/support2_survival_personalized_moe.py --root data/support2
    python examples/support2_survival_personalized_moe.py --synthetic --epochs 2 --seeds 42

Downloads missing data from UCI; loads patients through Support2Dataset in CSV
order. Uses the reference's 42 features, physiological/median imputation, missing
indicators, scaling, categorical embeddings and 20-quantile hday buckets. All
preprocessing fits the fixed seed-42 80/10/10 training split only.

Defaults: seeds 42–46, one 128-unit layer, eight experts, 4-dimensional embeddings,
Adam (lr=0.0005, eps=1e-7), batch 64, up to 1,000 epochs with patience 10 on
validation NLL. Each seed pairs initialization and batch order across dropout
0/0.1/0.3/0.5. The grid has 100 training-range points plus a final overflow point.

Reports median-survival Harrell C-index, grid-averaged IPCW Brier/ECE (ten equal-
mass risk bins, reverse KM floored at 0.01), NLL gaps, mean/sample SD and paired
changes from dropout=0. Synthetic mode checks execution only. Repeated training
seeds measure optimization variability on one split, not population uncertainty.
SUPPORT2 results (7283/911/911 patients; mean +/- sample SD over five seeds):
    Dropout  C-index          IPCW Brier       IPCW ECE         NLL gap
    0.0      0.8085+/-0.0025  0.1433+/-0.0031  0.0444+/-0.0043  0.1949+/-0.0403
    0.1      0.8105+/-0.0027  0.1409+/-0.0027  0.0413+/-0.0029  0.1881+/-0.0069
    0.3      0.8075+/-0.0037  0.1424+/-0.0017  0.0426+/-0.0059  0.2046+/-0.0411
    0.5      0.8043+/-0.0022  0.1431+/-0.0016  0.0473+/-0.0034  0.1871+/-0.0204
Early stopping ended runs after 14–38 epochs. Dropout 0.1 had the best mean
discrimination and calibration; larger dropout did not consistently improve them.
Reference: https://github.com/ToddMorrill/survival-moe (7ce58fa).
"""

import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from torch.utils.data import DataLoader

from pyhealth.datasets import Support2Dataset
from pyhealth.models import PersonalizedSurvivalMoE

PHYSIOLOGICAL = {
    "alb": 3.5,
    "pafi": 333.3,
    "bili": 1.01,
    "crea": 1.01,
    "bun": 6.51,
    "wblc": 9.0,
    "urine": 2502,
}
NUMERIC = list(PHYSIOLOGICAL) + [
    "age",
    "num.co",
    "scoma",
    "charges",
    "totcst",
    "totmcst",
    "avtisst",
    "sps",
    "aps",
    "surv2m",
    "surv6m",
    "diabetes",
    "dementia",
    "prg2m",
    "dnrday",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "sod",
    "ph",
    "glucose",
    "adls",
    "adlsc",
]
CATEGORICAL = [
    "sex",
    "dzgroup",
    "dzclass",
    "edu",
    "income",
    "race",
    "ca",
    "prg6m",
    "dnr",
    "adlp",
]


def preprocess(frame: pd.DataFrame, train: np.ndarray):
    """Fit reference preprocessing on training rows; return tensors and vocab sizes."""
    numeric, categorical, cardinalities = [], [], []
    for col in NUMERIC:
        values = (
            pd.to_numeric(frame[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy()
            .reshape(-1, 1)
        )
        imputer = SimpleImputer(
            strategy="constant" if col in PHYSIOLOGICAL else "median",
            fill_value=PHYSIOLOGICAL.get(col),
            keep_empty_features=True,
        )
        transform = make_pipeline(imputer, StandardScaler()).fit(values[train])
        numeric.extend([transform.transform(values), np.isnan(values)])
    for col in CATEGORICAL:
        values = frame[col].fillna("missing").astype(str)
        vocabulary = ["missing"] + sorted(set(values.iloc[train]) - {"missing"})
        codes = values.map({v: i for i, v in enumerate(vocabulary)}).fillna(0)
        categorical.append(codes.to_numpy(int))
        cardinalities.append(len(vocabulary))
    hday = pd.to_numeric(frame.hday, errors="coerce").to_numpy().reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="quantile")
    if "quantile_method" in discretizer.get_params():
        discretizer.set_params(quantile_method="linear")
    bucket = make_pipeline(SimpleImputer(strategy="median"), discretizer)
    bucket.fit(hday[train])
    categorical.append(bucket.transform(hday).ravel().astype(int))
    cardinalities.append(len(bucket[-1].bin_edges_[0]))
    return (
        torch.tensor(np.concatenate(numeric, axis=1), dtype=torch.float32),
        torch.tensor(np.stack(categorical, axis=1), dtype=torch.long),
        tuple(cardinalities),
    )


def censoring_km(time: np.ndarray, event: np.ndarray):
    """Reverse KM: tied events leave the risk set before censoring; floor at .01."""
    grid, probabilities, current = np.unique(time), [], 1.0
    for t in grid:
        censored = np.sum((time == t) & (event == 0))
        at_risk = np.sum(time >= t) - np.sum((time == t) & (event == 1))
        if censored:
            current *= 1 - censored / at_risk
        probabilities.append(max(current, 0.01))
    return grid, np.asarray(probabilities)


def km_at(km, t):
    """Evaluate the right-continuous censoring distribution."""
    grid, probability = km
    index = np.searchsorted(grid, t, side="right") - 1
    return np.where(index < 0, 1.0, probability[np.maximum(index, 0)])


def calibration_metrics(train_time, train_event, time, event, survival, grid):
    """Reference IPCW Brier/ECE, averaged uniformly over the full time grid."""
    km = censoring_km(train_time, train_event)
    clamped = np.minimum(time, train_time.max())
    event_weight = event / km_at(km, clamped)
    horizon_weight = 1 / km_at(km, np.minimum(grid, train_time.max()))
    brier, ece = [], []
    for j, horizon in enumerate(grid):
        risk = 1 - survival[:, j]
        target = time <= horizon
        weights = np.where(clamped <= horizon, event_weight, horizon_weight[j])
        brier.append(np.mean((risk - target) ** 2 * weights))
        events = event_weight * target
        survivors = (time > horizon) * horizon_weight[j]
        error = 0.0
        for idx in np.array_split(np.argsort(risk, kind="stable"), min(10, len(time))):
            numerator = events[idx].sum()
            denominator = numerator + survivors[idx].sum()
            empirical = numerator / denominator if denominator else 0.0
            error += len(idx) / len(time) * abs(risk[idx].mean() - empirical)
        ece.append(error)
    return {"ipcw_brier": float(np.mean(brier)), "ipcw_ece": float(np.mean(ece))}


def ensure_support2(path: Path) -> None:
    """Download a missing CSV and normalize its ID column for Support2Dataset."""
    if path.exists():
        return
    url = "https://archive.ics.uci.edu/static/public/880/data.csv"
    with urlopen(url, timeout=30) as response:
        frame = pd.read_csv(response).rename(columns={"id": "sno"})
    if not {"sno", "d.time", "death"}.issubset(frame.columns) or frame.empty:
        raise ValueError(
            "Downloaded SUPPORT2 CSV is missing patients or outcome columns"
        )
    if frame.sno.isna().any() or frame.sno.duplicated().any():
        raise ValueError("Downloaded SUPPORT2 CSV has invalid patient IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        pending = Path(tmp) / "support2.csv"
        frame.to_csv(pending, index=False)
        pending.replace(path)


def load_data(root: str | None, synthetic: bool, seed: int) -> pd.DataFrame:
    """Load baseline fields via PyHealth, or generate independent synthetic data."""
    if synthetic:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            rng.normal(size=(240, len(NUMERIC) + 1)), columns=NUMERIC + ["hday"]
        )
        for col in CATEGORICAL:
            frame[col] = rng.choice(["a", "b"], len(frame))
        event_time = rng.exponential(100 * np.exp(-0.6 * frame.age)) + 1
        censor_time = rng.uniform(20, 250, len(frame))
        frame["d.time"] = np.minimum(event_time, censor_time)
        frame["death"] = (event_time <= censor_time).astype(int)
        return frame
    ensure_support2(Path(root) / "support2.csv")
    dataset = Support2Dataset(
        root=root, tables=["support2"], cache_dir=Path(root) / ".pyhealth_cache"
    )
    rows = []
    for pid in pd.read_csv(Path(root) / "support2.csv", dtype={"sno": str}).sno:
        events = dataset.get_patient(pid).get_events(event_type="support2")
        if len(events) != 1:
            raise ValueError(f"Expected exactly one SUPPORT2 row for {pid}")
        rows.append(
            {
                key: getattr(events[0], key, None)
                for key in NUMERIC + CATEGORICAL + ["hday", "d.time", "death"]
            }
        )
    return pd.DataFrame(rows)


def summarize(rows):
    """Summarize seed variability and paired changes from the no-dropout arm."""
    metrics = ["c_index", "ipcw_brier", "ipcw_ece", "gap"]
    baseline = {row["seed"]: row for row in rows if row["dropout"] == 0}
    summary = []
    for dropout in [0.0, 0.1, 0.3, 0.5]:
        arm = [row for row in rows if row["dropout"] == dropout]
        record = {"dropout": dropout}
        for metric in metrics:
            values = np.array([row[metric] for row in arm], dtype=float)
            delta = values - [baseline[row["seed"]][metric] for row in arm]
            record[metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "paired_delta_mean": float(delta.mean()),
                "paired_delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else None,
            }
        summary.append(record)
    return summary


def run(args: argparse.Namespace) -> dict:
    """Run paired dropout arms and report held-out results for every seed."""
    from SurvivalEVAL import LifelinesEvaluator

    if min(args.epochs, args.batch_size, args.patience) < 1 or args.time_bins < 3:
        raise ValueError("epochs/batch_size/patience must be positive; time_bins >= 3")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide distinct training seeds")
    torch.set_num_threads(1)
    frame = load_data(args.root, args.synthetic, 42)
    times = pd.to_numeric(frame["d.time"]).to_numpy(float)
    events = pd.to_numeric(frame.death).to_numpy(float)
    if (
        not np.isfinite(times).all()
        or (times < 0).any()
        or not np.isin(events, [0, 1]).all()
    ):
        raise ValueError("Invalid survival times or event indicators")
    train, test = train_test_split(
        np.arange(len(frame)), test_size=0.1, random_state=42
    )
    train, val = train_test_split(train, test_size=1 / 9, random_state=42)
    features, categories, cardinalities = preprocess(frame, train)
    boundaries = torch.linspace(0, times[train].max(), args.time_bins)
    grid = torch.cat([boundaries, boundaries[-1:] + 1]).numpy()
    time_bin = torch.tensor(np.searchsorted(grid, times).clip(max=len(grid) - 1))
    event_tensor = torch.tensor(events, dtype=torch.float32)
    rows = []
    for seed in args.seeds:
        for dropout in [0.0, 0.1, 0.3, 0.5]:
            torch.manual_seed(seed)
            model = PersonalizedSurvivalMoE(
                None,
                features.shape[1],
                len(grid),
                hidden_dim=args.hidden_dim,
                num_experts=args.experts,
                dropout=dropout,
                num_hidden_layers=1,
                categorical_cardinalities=cardinalities,
                embedding_dim=4,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
            loader = DataLoader(
                train,
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(42),
            )

            def predict(idx, model=model):
                return model(
                    features=features[idx],
                    categories=categories[idx],
                    time_bin=time_bin[idx],
                    event=event_tensor[idx],
                )

            best, state, best_epoch, stale = float("inf"), None, 0, 0
            for epoch in range(args.epochs):
                model.train()
                for idx in loader:
                    optimizer.zero_grad()
                    predict(idx)["loss"].backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    score = float(predict(val)["nll"])
                if score < best:
                    best, state, best_epoch, stale = (
                        score,
                        copy.deepcopy(model.state_dict()),
                        epoch + 1,
                        0,
                    )
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                out = predict(test)
                train_nll = float(predict(train)["nll"])
            survival = np.column_stack([out["survival"].numpy(), np.zeros(len(test))])
            evaluator = LifelinesEvaluator(
                pd.DataFrame(survival.T, index=grid),
                times[test],
                events[test],
                times[train],
                events[train],
                predict_time_method="Median",
                interpolation="Linear",
            )
            try:
                c_index = float(evaluator.concordance()[0])
            except ZeroDivisionError:
                c_index = float("nan")
            rows.append(
                dict(
                    seed=seed,
                    dropout=dropout,
                    best_epoch=best_epoch,
                    epochs_run=epoch + 1,
                    train_nll=train_nll,
                    val_nll=best,
                    test_nll=float(out["nll"]),
                    gap=best - train_nll,
                    c_index=c_index,
                    **calibration_metrics(
                        times[train],
                        events[train],
                        times[test],
                        events[test],
                        survival,
                        grid,
                    ),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
    result = {
        "config": vars(args),
        "sizes": list(map(len, [train, val, test])),
        "time_grid": grid.tolist(),
        "results": rows,
        "summary": summarize(rows),
    }

    def finite(value):
        if isinstance(value, dict):
            return {key: finite(v) for key, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value

    return finite(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--root", help="Data directory; downloads support2.csv if absent"
    )
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", default="personalized_moe_results.json")
    args = parser.parse_args()
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results["summary"], indent=2, allow_nan=False))
