# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Run the five-fold SUPPORT2 protocol; each completed fold is checkpointed.

This driver works on NumPy arrays loaded from the CSV, not on a PyHealth
dataset, so it uses ``DCMReference`` (the EM engine behind
``DeepCoxMixtures.fit_survival``) directly. The same fixed settings are used.
"""

import argparse
import importlib.metadata
import itertools
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from pyhealth.models._dcm_reference import DCMReference

from .data import features, load_data, mask_labs, preprocessing, sha256
from .evaluation import choose_horizons, metrics

Model = Union[DCMReference, CoxPHSurvivalAnalysis, RandomSurvivalForest]


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def curve(model: Model, x: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """Return survival probabilities ``[patients, horizons]`` for any model."""
    if isinstance(model, DCMReference):
        return model.predict_survival(x, horizons)
    return np.array([fn(horizons) for fn in model.predict_survival_function(x)])


def new_model(kind: str, config: dict[str, Any], seed: int) -> Model:
    """Create an unfitted model of the given family and configuration."""
    if kind == "dcm":
        return DCMReference(**config, seed=seed)
    if kind == "cox":
        return CoxPHSurvivalAnalysis(**config, ties="breslow", n_iter=200)
    return RandomSurvivalForest(**config, n_estimators=100, n_jobs=1, random_state=seed)


def main(argv: Optional[list[str]] = None) -> None:
    """Run the selected folds and write fold records under ``--output``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="SUPPORT2 CSV")
    parser.add_argument("--output", required=True, help="new results directory")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for d in ["models", "predictions"]:
        (out / d).mkdir(exist_ok=True)
    frame, y = load_data(args.data)
    if args.smoke:
        ix, _ = train_test_split(
            np.arange(len(y)),
            train_size=600,
            random_state=20261102,
            stratify=y["event"],
        )
        frame, y = frame.iloc[ix].reset_index(drop=True), y[ix]
    grids = {
        "dcm": [dict(k=k, width=w) for k, w in itertools.product([3, 4, 6], [50, 100])],
        "cox": [dict(alpha=float(a)) for a in np.logspace(-3, 2, 6)],
        "rsf": [
            dict(min_samples_leaf=leaf, max_features=share)
            for leaf, share in itertools.product([5, 10, 20], ["sqrt", 1.0])
        ],
    }
    if args.smoke:
        grids = {k: v[:1] for k, v in grids.items()}
    manifest = {
        "dataset_sha256": sha256(args.data),
        "n": len(y),
        "events": int(y["event"].sum()),
        "seed": 20261102,
        "mask_seeds": [71, 172, 273],
        "grids": grids,
        "args": vars(args),
        "python": platform.python_version(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "numpy",
                "scipy",
                "pandas",
                "scikit-survival",
                "scikit-learn",
            ]
        },
        "protocol": (
            "smoke only"
            if args.smoke
            else "five outer folds; train-only inner validation search; paired masking"
        ),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if (out / "manifest.json").exists():
        existing = json.loads((out / "manifest.json").read_text())
        for key in [
            "dataset_sha256",
            "n",
            "events",
            "seed",
            "mask_seeds",
            "grids",
            "versions",
        ]:
            if existing[key] != manifest[key]:
                raise ValueError(
                    f"Existing results use different {key}; use a new output directory"
                )
        for key in ["epochs", "smoke"]:
            if existing["args"][key] != manifest["args"][key]:
                raise ValueError(
                    f"Existing results use different {key}; use a new output directory"
                )
    else:
        write_json(out / "manifest.json", manifest)
    for fold, (outer, test) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=20261102).split(frame, y["event"])
    ):
        if fold not in args.folds:
            continue
        if (out / f"fold_{fold}.json").exists():
            print(f"Fold {fold} already complete; preserving checkpoint", flush=True)
            continue
        record = {"fold": fold, "search": [], "failures": [], "evaluation": []}
        training, validation = train_test_split(
            outer, test_size=0.2, random_state=1000 + fold, stratify=y[outer]["event"]
        )
        write_json(
            out / f"splits_{fold}.json",
            {
                "train": training.tolist(),
                "validation": validation.tolist(),
                "outer_train": outer.tolist(),
                "test": test.tolist(),
                "test_patient_ids": frame.iloc[test].patient_id.tolist(),
            },
        )
        prep = preprocessing().fit(features(frame.iloc[training]))
        xt = prep.transform(features(frame.iloc[training])).astype(np.float32)
        xv = prep.transform(features(frame.iloc[validation])).astype(np.float32)
        val_times, omitted = choose_horizons(y[training], y[validation])
        record["validation_horizons"] = val_times.tolist()
        record["omitted_validation_horizons"] = omitted
        selected = {}
        for kind, configs in grids.items():
            best = None
            for index, config in enumerate(configs):
                started = time.monotonic()
                entry = {"model": kind, "config": config, "index": index}
                try:
                    model = new_model(kind, config, 2000 + fold)
                    if kind == "dcm":
                        model.fit(
                            xt,
                            y[training],
                            validation=(xv, y[validation]),
                            epochs=min(args.epochs, 3) if args.smoke else args.epochs,
                            capture_output=True,
                        )
                        entry["audit"] = model.audit
                    else:
                        model.fit(xt, y[training])
                    pred = curve(model, xv, val_times)
                    entry.update(metrics(y[training], y[validation], pred, val_times))
                    if kind == "dcm":
                        entry["prediction_audit"] = model.last_prediction_audit
                    entry["status"] = "ok"
                    if best is None or entry["mean_brier"] < best["mean_brier"]:
                        best = entry
                except Exception as error:
                    entry.update(
                        status="failed",
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                    record["failures"].append(entry)
                entry["seconds"] = time.monotonic() - started
                record["search"].append(entry)
                write_json(out / f"progress_{fold}.json", record)
                print(
                    f"fold={fold} {kind} {index} {entry['status']} "
                    f"brier={entry.get('mean_brier')} "
                    f"seconds={entry['seconds']:.1f}",
                    flush=True,
                )
            if best is None:
                raise RuntimeError(
                    f"All {kind} configurations failed; see progress_{fold}.json"
                )
            selected[kind] = best
        record["selected"] = selected
        horizons, omitted = choose_horizons(y[outer], y[test])
        record["horizons"] = horizons.tolist()
        record["omitted_horizons"] = omitted
        models = {}
        for kind, indicators in [
            ("cox", False),
            ("rsf", False),
            ("dcm", False),
            ("dcm", True),
        ]:
            name = kind + ("_indicators" if indicators else "")
            p = preprocessing(indicators).fit(features(frame.iloc[outer], indicators))
            x = p.transform(features(frame.iloc[outer], indicators)).astype(np.float32)
            model = new_model(kind, selected[kind]["config"], 2000 + fold)
            if kind == "dcm":
                model.fit(
                    x,
                    y[outer],
                    epochs=selected[kind]["audit"]["epochs"],
                    refit=True,
                    capture_output=True,
                )
                record.setdefault("refit_audits", {})[name] = model.audit
            else:
                model.fit(x, y[outer])
            payload = {
                "model": model,
                "preprocessing": p,
                "horizons": horizons,
                "versions": manifest["versions"],
                "indicators": indicators,
            }
            joblib.dump(
                payload, out / "models" / f"fold{fold}_{name}.joblib", compress=3
            )
            # Prove actual bundle reload before accepting it.
            restored = joblib.load(out / "models" / f"fold{fold}_{name}.joblib")
            probe = x[:10]
            np.testing.assert_array_equal(
                curve(model, probe, horizons), curve(restored["model"], probe, horizons)
            )
            models[name] = (model, p, indicators)
        for rate in [0, 0.1, 0.3]:
            seeds = [71] if rate == 0 else [71, 172, 273]
            for seed in seeds:
                corrupted, mask_info = mask_labs(frame.iloc[test], rate, seed)
                for name, (model, p, indicators) in models.items():
                    x = p.transform(features(corrupted, indicators)).astype(np.float32)
                    pred = curve(model, x, horizons)
                    score = metrics(y[outer], y[test], pred, horizons)
                    entry = {
                        "model": name,
                        "mask_rate": rate,
                        "mask_seed": seed,
                        "mask_info": mask_info,
                        **score,
                    }
                    if isinstance(model, DCMReference):
                        entry["prediction_audit"] = model.last_prediction_audit
                    record["evaluation"].append(entry)
                    np.savez_compressed(
                        out
                        / "predictions"
                        / f"fold{fold}_{name}_mask{rate}_seed{seed}.npz",
                        patient_id=frame.iloc[test].patient_id.to_numpy(dtype=str),
                        time=y[test]["time"],
                        event=y[test]["event"],
                        horizons=horizons,
                        survival=pred,
                    )
        record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(out / f"fold_{fold}.json", record)
        print(f"Completed fold {fold}", flush=True)


if __name__ == "__main__":
    main()
