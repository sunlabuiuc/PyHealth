"""Authors: Murphy Crosby

Deep Cox Mixtures on SUPPORT2: reproduction and an ablation of the mixture.

Paper: Deep Cox Mixtures for Survival Regression, Nagpal et al., MLHC 2021.
https://proceedings.mlr.press/v149/nagpal21a.html

This script runs the whole pipeline: load SUPPORT2 with ``Support2Dataset``,
build samples with the ``SurvivalDCMSupport2`` task, train ``SurvivalDCM``
with cross-validation, and evaluate discrimination and calibration.

How to run
----------
With no arguments it generates a small synthetic SUPPORT2-like file, so it
runs anywhere in a few minutes. That run only shows that the pipeline works:
its numbers come from made-up data and say nothing about the models::

    python examples/support2_survival_survival_dcm.py

To use the real data, download ``support2.csv`` from the UCI Machine Learning
Repository (https://archive.ics.uci.edu/dataset/880/support2) and pass it::

    python examples/support2_survival_survival_dcm.py --data /path/support2.csv

The UCI file names its patient column ``id``; ``Support2Dataset`` expects
``sno``. The script writes a working copy with the column renamed.

What is compared
----------------
Every model is the same ``SurvivalDCM`` class with different settings:

* ``CPH``: one subgroup and no hidden layer, a linear Cox model.
* ``Deep Cox (K=1)``: one subgroup with a hidden-layer encoder. This is a
  single deep Cox model in the style of DeepSurv.
* ``DCM (K=3)``, ``DCM (K=4)``, ``DCM (K=6)``: the mixture, at the three
  numbers of subgroups the paper tuned over.

Ablation: the paper reports only its best ``K``. Comparing ``K=1`` with
``K=3, 4, 6`` on the same folds shows how much of the result comes from the
mixture itself rather than from the neural encoder.

Evaluation follows the paper: 5-fold cross-validation, with predictions from
all folds pooled, at the 25th, 50th and 75th percentiles of the event times,
for all patients and for non-White patients (a missing race counts as
non-White, which reproduces the paper's 21.02%). The metrics are the
time-dependent concordance index (higher is better), the Brier score and the
expected calibration error (lower is better). Standard errors come from
bootstrapping the pooled predictions.

Differences from the paper
--------------------------
* Imputation and scaling statistics are fitted on each training fold only.
  The authors fit them on all patients, which leaks test-fold information.
* Hyperparameters are fixed instead of tuned. As in the authors' code,
  training stops early: about one training patient in seven is held out, and
  training stops when the loss on them has not improved for 3 epochs.
* The concordance index is truncated at each horizon but not weighted for
  censoring. The Brier score is censoring-weighted. The calibration error
  uses a Kaplan-Meier estimate within each bin of predicted survival.

Results
-------
Full run on the real SUPPORT2 file (9,105 patients, 31.89% censored, 21.02%
non-White; horizons 14, 58 and 252 days), with default settings. Each cell is
the 25th / 50th / 75th percentile horizon.

All patients
model            C_td                   Brier                  ECE
CPH              0.689 / 0.673 / 0.670  0.134 / 0.200 / 0.213  0.020 / 0.024 / 0.020
Deep Cox (K=1)   0.696 / 0.680 / 0.676  0.133 / 0.197 / 0.211  0.022 / 0.023 / 0.016
DCM (K=3)        0.753 / 0.713 / 0.679  0.124 / 0.189 / 0.211  0.018 / 0.022 / 0.029
DCM (K=4)        0.725 / 0.695 / 0.668  0.128 / 0.194 / 0.214  0.017 / 0.017 / 0.021
DCM (K=6)        0.742 / 0.702 / 0.673  0.125 / 0.192 / 0.213  0.015 / 0.021 / 0.027

Non-White patients
model            C_td                   Brier                  ECE
CPH              0.716 / 0.700 / 0.692  0.133 / 0.193 / 0.205  0.031 / 0.038 / 0.038
Deep Cox (K=1)   0.719 / 0.704 / 0.697  0.132 / 0.192 / 0.203  0.021 / 0.028 / 0.028
DCM (K=3)        0.763 / 0.729 / 0.700  0.124 / 0.187 / 0.205  0.033 / 0.049 / 0.056
DCM (K=4)        0.740 / 0.714 / 0.687  0.129 / 0.190 / 0.206  0.029 / 0.040 / 0.034
DCM (K=6)        0.755 / 0.718 / 0.691  0.126 / 0.190 / 0.206  0.020 / 0.038 / 0.043

Bootstrap standard errors are about 0.004 to 0.007 for C_td, 0.002 for the
Brier score and 0.003 to 0.005 for ECE over all patients, and roughly twice
that for non-White patients. They reflect which patients are in the test
data, not the variation from retraining a model.

Findings. On C_td and the Brier score the baselines reproduce the paper's
Table 8 to within about 0.003. DCM discriminates better than both baselines
at the 25th and 50th percentile horizons (paper, K tuned: 0.743 / 0.706 /
0.675), and the three are close at the 75th. DCM is at least as well
calibrated as the baselines at the first two horizons but not at the 75th,
and for non-White patients it does not have the lowest calibration error, so
that claim of the paper is not reproduced.

Ablation. Every mixture (K=3, 4, 6) beats the single deep Cox model (K=1) on
C_td at the first two horizons: by 0.03 to 0.06 at the 25th percentile and by
0.015 to 0.033 at the 50th. So the gain comes from the mixture and not from
the encoder. The order among K=3, 4 and 6 changed between two training runs,
so there is no evidence that more than three subgroups helps.
"""

import argparse
import copy
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import pyhealth.datasets
from pyhealth.datasets import Support2Dataset, get_dataloader
from pyhealth.metrics import concordance_index, ipcw_brier_score
from pyhealth.models import SurvivalDCM
from pyhealth.tasks import SurvivalDCMSupport2
from pyhealth.tasks.survival_dcm_support2 import (
    DCM_SUPPORT2_CATEGORIES,
    DCM_SUPPORT2_FEATURES,
    DCM_SUPPORT2_NUMERIC,
)

# The models to compare. Each entry is the settings passed to SurvivalDCM.
MODELS = {
    "CPH": {"num_components": 1, "num_hidden_layers": 0},
    "Deep Cox (K=1)": {"num_components": 1},
    "DCM (K=3)": {"num_components": 3},
    "DCM (K=4)": {"num_components": 4},
    "DCM (K=6)": {"num_components": 6},
}
QUANTILES = (0.25, 0.50, 0.75)


def write_synthetic_support2(path: Path, num_patients: int, seed: int) -> None:
    """Write a made-up file with SUPPORT2's columns, for running without data.

    Survival depends on age, on mean blood pressure and on the disease class,
    so the models have something to learn. Some values are left missing.

    Args:
        path: Where to write the CSV.
        num_patients: Number of rows.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    config = Path(pyhealth.datasets.__file__).parent / "configs" / "support2.yaml"
    columns = ["sno"] + yaml.safe_load(config.read_text())["tables"]["support2"][
        "attributes"
    ]
    categories = dict(DCM_SUPPORT2_CATEGORIES)

    rows = []
    for i in range(num_patients):
        row = {"sno": str(i + 1)}
        for name in DCM_SUPPORT2_NUMERIC:
            # About one value in ten is missing, written as in the UCI file.
            row[name] = "NaN" if rng.random() < 0.1 else f"{rng.normal(50, 10):.2f}"
        for name, levels in categories.items():
            row[name] = str(rng.choice(levels))
        # Most patients are White, as in the real data.
        if rng.random() < 0.79:
            row["race"] = "white"
        age = rng.normal(63, 15)
        blood_pressure = rng.normal(85, 25)
        row["age"] = f"{age:.2f}"
        row["meanbp"] = f"{blood_pressure:.2f}"

        # Cancer patients die on a different time pattern, so the hazards of
        # the two groups are not proportional.
        risk = np.exp(0.03 * (age - 63) - 0.02 * (blood_pressure - 85))
        shape = 0.7 if row["dzclass"] == "Cancer" else 1.8
        death_time = 200 * rng.weibull(shape) / risk ** (1 / shape)
        censor_time = rng.exponential(600)
        row["d.time"] = str(int(max(1, np.ceil(min(death_time, censor_time)))))
        row["death"] = "1" if death_time <= censor_time else "0"
        rows.append(row)

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(data: Path | None, root: Path, seed: int) -> None:
    """Put a ``support2.csv`` that ``Support2Dataset`` can read into ``root``.

    Args:
        data: The real SUPPORT2 file, or None to generate synthetic data.
        root: Working directory to write into.
        seed: Random seed for the synthetic data.
    """
    target = root / "support2.csv"
    if data is None:
        print("No --data given: using synthetic SUPPORT2-like data.")
        write_synthetic_support2(target, num_patients=600, seed=seed)
        return
    # Read everything as text so the values are written back unchanged.
    frame = pd.read_csv(data, dtype=str, keep_default_na=False)
    if "sno" not in frame.columns:
        frame = frame.rename(columns={"id": "sno"})
    frame.to_csv(target, index=False)


def load_all(samples) -> dict:
    """Read a whole sample dataset into memory as one batch.

    Args:
        samples: A PyHealth sample dataset.

    Returns:
        ``patient_id`` (list), and ``features``, ``duration`` and ``event``
        as tensors with one row per patient.
    """
    loader = get_dataloader(samples, batch_size=len(samples), shuffle=False)
    batch = next(iter(loader))
    return {
        "patient_id": list(batch["patient_id"]),
        "features": batch["features"].float(),
        "duration": batch["duration"].float().reshape(-1),
        "event": batch["event"].float().reshape(-1),
    }


def fit_standardizer(features: torch.Tensor) -> tuple[list[float], list[float]]:
    """Fit the paper's imputation and scaling on raw training features.

    The mean of each column is taken over its observed values. The standard
    deviation is taken after the missing values are replaced by that mean,
    which is what imputing first and scaling second does.

    Args:
        features: Raw features of the training patients, NaN where missing.

    Returns:
        The mean and the scale of each column.
    """
    mean = torch.nanmean(features, dim=0)
    filled = torch.where(torch.isnan(features), mean, features)
    scale = filled.std(dim=0, correction=0)
    scale[scale == 0] = 1.0  # a constant column is left unscaled
    return mean.tolist(), scale.tolist()


def validation_loss(model: SurvivalDCM, loader) -> float:
    """Average loss of a model on held-out patients, without training on them."""
    model.eval()
    with torch.no_grad():
        losses = [model(**batch)["loss"].item() for batch in loader]
    return sum(losses) / len(losses)


def train_model(
    train_samples, settings: dict, max_epochs: int, patience: int, seed: int
) -> tuple[SurvivalDCM, int]:
    """Train one model with the EM loop, stopping early on a validation split.

    Some training patients are held out. After every epoch the loss on them
    is measured, and training stops once it has not improved for ``patience``
    epochs in a row. The model is then put back to its best epoch. This keeps
    the model from overfitting without ever looking at the test fold.

    Args:
        train_samples: Standardized training samples.
        settings: Keyword arguments for ``SurvivalDCM``.
        max_epochs: Most passes over the training data.
        patience: Epochs to wait for the validation loss to improve.
        seed: Random seed.

    Returns:
        The trained model, in evaluation mode, and the epoch it was taken from.
    """
    torch.manual_seed(seed)

    # Hold out about one training patient in seven for validation.
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(train_samples))
    num_validation = len(train_samples) // 7
    validation_samples = train_samples.subset(shuffled[:num_validation].tolist())
    fit_samples = train_samples.subset(shuffled[num_validation:].tolist())

    loader = get_dataloader(fit_samples, batch_size=128, shuffle=True, seed=seed)
    validation_loader = get_dataloader(
        validation_samples, batch_size=len(validation_samples), shuffle=False
    )
    model = SurvivalDCM(fit_samples, **settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    model.fit_baselines(loader)  # once, before training
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        model.fit_baselines(loader)  # again after every epoch

        current_loss = validation_loss(model, validation_loader)
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def concordance_at(
    duration: np.ndarray, event: np.ndarray, survival: np.ndarray, time: float
) -> float:
    """Time-dependent concordance index at one horizon.

    Among pairs where one patient died by ``time`` and the other outlived
    them, this is the share where the model gave the patient who died the
    lower survival probability at ``time``.
    """
    # Only deaths up to the horizon start a comparison.
    counted = event * (duration <= time)
    return concordance_index(duration, counted, 1.0 - survival)


def kaplan_meier_at(duration: np.ndarray, event: np.ndarray, time: float) -> float:
    """Kaplan-Meier estimate of the share of a group surviving past ``time``."""
    survival = 1.0
    for death_time in np.unique(duration[(event == 1) & (duration <= time)]):
        at_risk = (duration >= death_time).sum()
        deaths = ((duration == death_time) & (event == 1)).sum()
        survival = survival * (1 - deaths / at_risk)
    return survival


def calibration_error(
    duration: np.ndarray,
    event: np.ndarray,
    survival: np.ndarray,
    time: float,
    num_bins: int = 10,
) -> float:
    """Expected calibration error of survival probabilities at one horizon.

    Patients are sorted by predicted survival and cut into equal-sized bins.
    In each bin the average prediction is compared with the share that
    actually survived, estimated with Kaplan-Meier to allow for censoring.
    The result is the average gap over the bins.
    """
    order = np.argsort(survival)
    total_gap = 0.0
    for members in np.array_split(order, num_bins):
        predicted = survival[members].mean()
        observed = kaplan_meier_at(duration[members], event[members], time)
        total_gap = total_gap + len(members) * abs(observed - predicted)
    return total_gap / len(survival)


METRICS = {
    "C_td": concordance_at,
    "Brier": ipcw_brier_score,
    "ECE": calibration_error,
}


def score(
    duration: np.ndarray,
    event: np.ndarray,
    survival: np.ndarray,
    times: np.ndarray,
    num_bootstrap: int,
    seed: int,
) -> dict:
    """Compute every metric at every horizon, with bootstrap standard errors.

    Args:
        duration: Follow-up times of the scored patients.
        event: Their event indicators.
        survival: Predicted survival, one column per horizon.
        times: The horizons.
        num_bootstrap: Number of bootstrap resamples; 0 skips them.
        seed: Random seed for the resampling.

    Returns:
        ``{metric: [(value, standard_error), ...]}`` with one pair per
        horizon.
    """
    rng = np.random.default_rng(seed)
    resamples = [
        rng.integers(0, len(duration), len(duration)) for _ in range(num_bootstrap)
    ]
    results = {}
    for name, metric in METRICS.items():
        results[name] = []
        for column, time in enumerate(times):
            value = metric(duration, event, survival[:, column], float(time))
            repeats = [
                metric(duration[rows], event[rows], survival[rows, column], float(time))
                for rows in resamples
            ]
            error = float(np.std(repeats)) if repeats else float("nan")
            results[name].append((float(value), error))
    return results


def print_table(title: str, scores: dict, quantiles: tuple) -> None:
    """Print one table: a row per model, a column per metric and horizon."""
    print(f"\n{title}")
    header = f"{'model':<16}"
    for name in METRICS:
        for quantile in quantiles:
            header += f"{name + '@' + str(int(quantile * 100)):>16}"
    print(header)
    for model_name, results in scores.items():
        line = f"{model_name:<16}"
        for name in METRICS:
            for value, error in results[name]:
                if np.isnan(error):  # no bootstrap was run
                    line += f"{value:>16.4f}"
                else:
                    line += f"{value:>9.4f}±{error:<6.4f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--data", type=Path, help="path to the real support2.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50, help="most epochs")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        root = Path(workdir)
        prepare_data(args.data, root, args.seed)
        dataset = Support2Dataset(
            root=str(root), tables=["support2"], cache_dir=root / "cache", num_workers=1
        )

        # Raw pass: unstandardized features, NaN where a value is missing.
        # These are used to fit the statistics and to find non-White patients.
        raw = load_all(dataset.set_task(SurvivalDCMSupport2(), num_workers=1))
        patient_ids = raw["patient_id"]
        duration = raw["duration"].numpy()
        event = raw["event"].numpy()
        white_column = DCM_SUPPORT2_FEATURES.index("race=white")
        non_white = raw["features"][:, white_column].numpy() == 0
        times = np.quantile(duration[event == 1], QUANTILES)
        print(
            f"{len(patient_ids)} patients, {100 * (1 - event.mean()):.2f}% censored, "
            f"{100 * non_white.mean():.2f}% non-White, horizons {times.tolist()}"
        )

        # Assign every patient to one of the folds at random.
        rng = np.random.default_rng(args.seed)
        fold_of = rng.permutation(len(patient_ids)) % args.folds
        row_of = {pid: row for row, pid in enumerate(patient_ids)}
        predictions = {
            name: np.zeros((len(patient_ids), len(times))) for name in MODELS
        }

        for fold in range(args.folds):
            is_test = fold_of == fold
            mean, scale = fit_standardizer(raw["features"][~is_test])

            # Standardized pass, with this fold's training statistics.
            task = SurvivalDCMSupport2(feature_mean=mean, feature_scale=scale)
            samples = dataset.set_task(task, num_workers=1)
            # The samples can come back in a different order, so match them
            # to the raw pass by patient ID.
            position = {pid: i for i, pid in enumerate(load_all(samples)["patient_id"])}
            order = np.array([position[pid] for pid in patient_ids])
            train_samples = samples.subset(order[~is_test].tolist())
            test = load_all(samples.subset(order[is_test].tolist()))
            test_rows = np.array([row_of[pid] for pid in test["patient_id"]])

            for name, settings in MODELS.items():
                model, best_epoch = train_model(
                    train_samples, settings, args.epochs, args.patience, args.seed
                )
                survival = model.predict_survival(
                    test["features"], torch.as_tensor(times, dtype=torch.float32)
                )
                predictions[name][test_rows] = survival.numpy()
                print(
                    f"fold {fold + 1}/{args.folds}: trained {name}, "
                    f"best epoch {best_epoch}"
                )

        # Pool the test predictions of all folds, then score them.
        for title, rows in (("All patients", slice(None)), ("Non-White", non_white)):
            scores = {
                name: score(
                    duration[rows],
                    event[rows],
                    predictions[name][rows],
                    times,
                    args.bootstrap,
                    args.seed,
                )
                for name in MODELS
            }
            print_table(title, scores, QUANTILES)


if __name__ == "__main__":
    main()
