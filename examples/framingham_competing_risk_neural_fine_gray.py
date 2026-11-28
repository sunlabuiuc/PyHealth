"""Framingham competing-risk pipeline with Neural Fine-Gray and two ablations.

Contributors: Bin Zeng (bzeng48), Birkneh Tilahun Tadesse (btadesse7).
Paper: Jeanselme et al., "Neural Fine-Gray: Monotonic Neural Networks for
Competing Risks", CHIL 2023. https://arxiv.org/abs/2305.06703

Pipeline (every arm uses the same path)::

    FraminghamDataset -> FraminghamCompetingRiskPrediction (set_task)
      -> patient-level split -> CompetingRisksPreprocessor (fit on train)
      -> NeuralFineGray -> competing_risk_concordance / competing_risk_brier

Ablations:

1. Model, ``cdf_mode``: the paper's ``1 - exp(-t * m_r(t, x))`` versus the
   SumoNet-style ``1 - exp(-m_r(t, x))``, which is not forced to zero at t=0.
2. Task, ``horizon_days``: training labels administratively censored at the
   75th percentile of training event times, scored on the uncensored test
   labels. This asks how much the early-horizon predictions depend on
   follow-up beyond the horizon.

Every arm is scored for cause 1 (cardiovascular disease) at the 25th, 50th and
75th percentiles of uncensored training event times, in years, so the arms are
directly comparable.

Runs offline by default on a synthetic cohort written to a temporary
directory (no clinical data needed)::

    python examples/framingham_competing_risk_neural_fine_gray.py

Public Framingham cohort (downloads a commit-pinned CSV if absent)::

    python examples/framingham_competing_risk_neural_fine_gray.py \\
        --data-root ./data/framingham --download --epochs 100

Recorded results and setup: ``examples/neural_fine_gray_results/README.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import FraminghamDataset
from pyhealth.datasets.framingham import FRAMINGHAM_FEATURES
from pyhealth.metrics import (
    CensoringDistribution,
    competing_risk_brier,
    competing_risk_concordance,
)
from pyhealth.models import NeuralFineGray
from pyhealth.tasks import (
    CompetingRisksPreprocessor,
    FraminghamCompetingRiskPrediction,
)

DAYS_PER_YEAR = 365.25
QUANTILES = (0.25, 0.5, 0.75)
Sample = dict[str, Any]


def write_synthetic_cohort(root: Path, patients: int, seed: int) -> None:
    """Write a fabricated prepared ``framingham.csv`` with two competing causes.

    Cause-specific latent times are exponential with covariate-dependent rates,
    so the true model is known and both causes depend on the features.

    Args:
        root: Directory that will hold ``framingham.csv``.
        patients: Number of synthetic participants.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(patients, len(FRAMINGHAM_FEATURES)))
    rate_cvd = np.exp(-2.6 + 0.7 * x[:, 0] - 0.3 * x[:, 1]) / DAYS_PER_YEAR
    rate_death = np.exp(-2.9 - 0.5 * x[:, 0] + 0.4 * x[:, 2]) / DAYS_PER_YEAR
    latent = np.column_stack(
        [
            rng.uniform(5, 24, size=patients) * DAYS_PER_YEAR,  # censoring
            rng.exponential(1 / rate_cvd),
            rng.exponential(1 / rate_death),
        ]
    )
    event = latent.argmin(axis=1)
    frame = pd.DataFrame(x, columns=list(FRAMINGHAM_FEATURES))
    frame.insert(0, "patient_id", [f"s{i:05d}" for i in range(patients)])
    frame["duration_days"] = latent[np.arange(patients), event] + 1.0
    frame["event"] = event
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "framingham.csv", index=False)


def load_samples(
    root: Path,
    cache: Path,
    horizon_days: float | None = None,
    download: bool = False,
) -> list[Sample]:
    """Run the dataset and task; return raw samples sorted by patient.

    Args:
        root: Framingham directory.
        cache: PyHealth cache directory.
        horizon_days: Optional administrative censoring horizon for the task.
        download: Allow downloading the public source.

    Returns:
        One sample per participant with ``features``, ``time`` (years) and
        ``event``.
    """
    dataset = FraminghamDataset(
        root=str(root), cache_dir=str(cache), num_workers=1, download=download
    )
    task = FraminghamCompetingRiskPrediction(horizon_days=horizon_days)
    samples = dataset.set_task(task, num_workers=1)
    return sorted((dict(s) for s in samples), key=lambda s: s["patient_id"])


def split_ids(samples: list[Sample], seed: int) -> dict[str, set[str]]:
    """Stratified 70/15/15 patient split by event cause."""
    rng = np.random.default_rng(seed)
    parts: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    events = np.array([s["event"] for s in samples])
    ids = np.array([s["patient_id"] for s in samples])
    for cause in np.unique(events):
        members = rng.permutation(ids[events == cause])
        first, second = int(0.7 * len(members)), int(0.85 * len(members))
        parts["train"].update(members[:first])
        parts["valid"].update(members[first:second])
        parts["test"].update(members[second:])
    return parts


def _arrays(samples: list[Sample]) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([s["features"] for s in samples], dtype=torch.float64),
        torch.tensor([s["time"] for s in samples], dtype=torch.float64),
        torch.tensor([s["event"] for s in samples], dtype=torch.float64),
    )


def train(
    train_set: list[Sample],
    valid_set: list[Sample],
    cdf_mode: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> NeuralFineGray:
    """Fit Neural Fine-Gray with Adam and keep the best validation NLL.

    Args:
        train_set: Preprocessed training samples.
        valid_set: Preprocessed validation samples.
        cdf_mode: ``"nfg"`` or ``"sumonet"``.
        epochs: Maximum epochs.
        batch_size: Mini-batch size.
        seed: Seed for initialization and shuffling.

    Returns:
        The model restored to its best validation checkpoint, in eval mode.
    """
    torch.manual_seed(seed)
    features, times, events = _arrays(train_set)
    valid = _arrays(valid_set)
    model = NeuralFineGray(
        None,
        input_dim=features.shape[1],
        num_risks=2,
        hidden_dims=(50, 50),
        survival_dims=(50, 50),
        dropout=0.25,
        cdf_mode=cdf_mode,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for index in torch.randperm(len(times)).split(batch_size):
            optimizer.zero_grad()
            model(features[index], times[index], events[index])["loss"].backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(model(*valid)["loss"])
        if loss < best:
            best, best_state, stale = loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate(
    model: NeuralFineGray,
    processor: CompetingRisksPreprocessor,
    train_raw: list[Sample],
    test_raw: list[Sample],
    horizons: np.ndarray,
) -> dict[str, float]:
    """Score cause-1 CIFs at fixed horizons (years) on raw test labels.

    Args:
        model: Trained model.
        processor: Preprocessor fitted on this arm's training split.
        train_raw: Uncensored training labels for the censoring distribution.
        test_raw: Uncensored test samples.
        horizons: Horizons in years.

    Returns:
        Test NLL in this arm's scaled time units plus C-index and Brier score
        at each horizon.
    """
    test = processor.transform(test_raw)
    features, times, events = _arrays(test)
    scale = processor.state["time_scale"]
    with torch.no_grad():
        nll = float(model(features, times, events)["loss"])
        cif = model.predict_cif(features, horizons / scale)[:, :, 0].numpy()
    censoring = CensoringDistribution(
        np.array([s["time"] for s in train_raw]),
        np.array([s["event"] for s in train_raw]),
    )
    test_times = np.array([s["time"] for s in test_raw])
    test_events = np.array([s["event"] for s in test_raw])
    result = {"test_nll_scaled": nll}
    for column, (quantile, horizon) in enumerate(zip(QUANTILES, horizons)):
        label = f"q{int(quantile * 100)}"
        args = (test_times, test_events, cif[:, column], float(horizon), 1)
        result[f"{label}_years"] = float(horizon)
        result[f"{label}_c_index"] = competing_risk_concordance(*args, censoring)[0]
        result[f"{label}_brier"] = competing_risk_brier(*args, censoring)
    return result


def run(
    root: Path,
    cache: Path,
    download: bool,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Run the three arms on one split and return their metrics."""
    full = load_samples(root, cache / "full", download=download)
    parts = split_ids(full, seed)

    def subset(samples: list[Sample], name: str) -> list[Sample]:
        return [s for s in samples if s["patient_id"] in parts[name]]

    train_raw, test_raw = subset(full, "train"), subset(full, "test")
    event_times = [s["time"] for s in train_raw if s["event"] > 0]
    horizons = np.quantile(event_times, QUANTILES)
    cutoff_days = float(horizons[-1] * DAYS_PER_YEAR)
    truncated = load_samples(root, cache / "truncated", horizon_days=cutoff_days)

    arms = {
        "nfg_full_followup": ("nfg", full),
        "sumonet_full_followup": ("sumonet", full),
        "nfg_censored_at_q75": ("nfg", truncated),
    }
    results: dict[str, Any] = {
        "patients": {name: len(parts[name]) for name in parts},
        "events_train": np.bincount(
            [s["event"] for s in train_raw], minlength=3
        ).tolist(),
        "truncation_years": horizons[-1],
    }
    for arm, (cdf_mode, samples) in arms.items():
        arm_train, arm_valid = subset(samples, "train"), subset(samples, "valid")
        processor = CompetingRisksPreprocessor().fit(arm_train)
        model = train(
            processor.transform(arm_train),
            processor.transform(arm_valid),
            cdf_mode,
            epochs,
            batch_size,
            seed,
        )
        results[arm] = evaluate(model, processor, train_raw, test_raw, horizons)
        print(f"{arm}: {json.dumps(results[arm])}", flush=True)
    return results


def main() -> None:
    """Parse arguments, build the cohort, and print JSON results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, help="Framingham directory")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--patients", type=int, default=800, help="synthetic size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    np.random.seed(args.seed)
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        root = args.data_root
        if root is None:
            root = scratch_path / "synthetic"
            write_synthetic_cohort(root, args.patients, args.seed)
        results = run(
            root,
            scratch_path / "cache",
            args.download,
            args.epochs,
            args.batch_size,
            args.seed,
        )
    results["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    text = json.dumps(results, indent=2, default=float)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
