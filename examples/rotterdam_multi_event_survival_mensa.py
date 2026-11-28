# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
"""Rotterdam multi-event survival with MENSA: experiments and baselines.

This example follows the MENSA authors' Rotterdam protocol (thecml/mensa,
commit 6b219cd). It provides:

* :func:`stratified_split`: a reproducible 70/10/20 patient split,
  stratified on both events as in the reference ``make_stratified_split``.
* :func:`make_time_grid`: the evaluation time grid, built from training
  times only as in the reference ``make_time_bins``.
* :func:`baseline_curves`: CoxPH and DeepSurv baselines, one model per event,
  as in the reference ``sota_models.py`` and its Rotterdam configs. Their
  survival curves are put on the same time grid as MENSA's.
* :func:`evaluate_survival`: global and local C-index (paper Eqs. 22 and 23,
  reference ``global_C_index`` / ``local_C_index``), and per-event AUC, IBS,
  MAE (Margin) and D-calibration from SurvivalEVAL, as in the reference
  ``train_models_multi_event.py``. Next to the default IBS it reports
  IBS(tau90), the same score restricted to times up to the 90th percentile
  of the training times (see :func:`_horizon_brier` for why).
* :func:`run_once`: one method on one seed (split, training, test curves,
  metrics), saved as one JSON file; :func:`summarize_runs` turns the run
  files into a CSV of mean ± standard deviation per method and metric.
* :func:`run_learning_curve_experiment` (E2): each method trained on nested
  25% ⊂ 50% ⊂ 100% subsets of every seed's training split, with paired
  MENSA-minus-baseline differences and CI_L, CI_G, IBS and IBS(tau90)
  learning curves.
* :func:`run_variant_experiment` (E3 and E4): variants of E1's
  configurations on the full training split, with paired per-seed
  differences in the same format as E2. E3 (``ablation``): MENSA with 10 or
  8 features (without ``hormon``/``chemo``) and lambda 0.25 or 0, plus CoxPH
  with 8 features; lambda = 0 is already in the paper (verification only).
  E4 (``model_ablation``): MENSA with the reference code's trajectory loss,
  and with the reference Rotterdam config (``hidden_dims=(16,)``, dropout
  0.5, weight decay 1e-4).
* :func:`constant_order_local_ci` and :func:`report_constant_order`: a
  no-model reference for CI_L (recurrence predicted before death for every
  test patient), printed with every summary next to each method's CI_L.
  It is computed from the split alone, so it is not part of the run files.

Results on the real data (2,982 patients, ``EXAMPLE_VERSION`` 2). Values
are x100; E1 uses seeds 0-9, E2-E4 seeds 0-4 (their 100% and unchanged rows
are E1's seeds 0-4, so they differ slightly from E1's means):

* E1 reproduces the paper's MENSA: CI_G 69.3 ± 1.1, CI_L 90.9 ± 1.3, AUC
  73.7 (paper 69.4, 90.1, 73.8). CoxPH reaches CI_G / CI_L 68.5 / 83.1 and
  DeepSurv 68.7 / 85.5 (paper 69.7 / 83.6 and 70.0 / 84.6), so here MENSA's
  CI_G is above both on 9 of 10 seeds (+0.85, +0.66). MAE (Margin) is
  2,130 / 2,251 / 2,194 days (MENSA / CoxPH / DeepSurv); every method is
  D-calibrated for both events on every seed (paper: MENSA 18/20).
* CI_L reference. About 91% of the patients who count for CI_L had
  recurrence as their first event, so predicting that order for everyone,
  with no model, scores 90.97 ± 1.31 (:func:`constant_order_local_ci`).
  MENSA equals it on 8 of 10 seeds and is below it on 2 (-0.09 ± 0.20);
  CoxPH (-7.91) and DeepSurv (-5.47) are below it on all 10. MENSA's CI_L
  lead is +7.8 over CoxPH and +5.4 over DeepSurv (10/10 seeds), and most
  of it is explained by the event order itself. The reference is not an
  upper bound: a model can beat it by ranking the death-first patients
  correctly. The paper's 90.1 is consistent with this reference level;
  its splits cannot be checked.
* IBS. The default IBS, the paper's metric, is 22.5-23.1 ± 5.5-7.0 here
  against 18.8-19.6 ± 1.5-1.6 in the paper, and the same seeds (1, 3, 8,
  9) are high for every method: their test sets have censoring times of
  6,700-7,000 days, where the training censoring survival is 0.001-0.006
  and single patients get IPCW weights of up to about 720. Up to tau90
  (:func:`_horizon_brier`; censoring survival 0.21-0.24, weights at most
  5), IBS is 17.03 ± 0.79 for MENSA, 17.42 ± 0.93 for CoxPH and
  17.40 ± 0.86 for DeepSurv; MENSA is lower by 0.39 (9 of 10 seeds) and
  0.37 (10 of 10). The default IBS shows no consistent difference (MENSA
  lower on 5 and 4 of 10 seeds).
* E2, less training data: mixed. MENSA's CI_L lead over DeepSurv grows
  (+4.4, +6.4, +6.1 at 100%, 50%, 25%) because DeepSurv's CI_L drops
  (86.3, 84.0, 83.2); over CoxPH there is no trend (+6.9, +7.9, +6.0).
  MENSA falls below the no-model CI_L reference on 3 of 5 seeds at 50% and
  at 25%. CI_G: MENSA leads at 100% and 50% (+0.6 to +1.0) and is about
  level at 25% (+0.3). IBS tau90: MENSA is worse at 25% on 5 of 5 seeds
  against both baselines (+0.83, +0.93), level at 50%, and better at 100%
  (-0.33 on 5 of 5 seeds, -0.25 on 4 of 5).
* E3: no effect. Dropping ``hormon``/``chemo`` (MENSA and CoxPH) and
  setting lambda = 0 change every paired metric by at most 0.18 points.
  MENSA's mean CI_L drops slightly in all three comparisons
  (-0.12 to -0.18); CoxPH's mean CI_L rises by 0.18 without
  ``hormon``/``chemo``. With 8 features, MENSA still beats CoxPH on CI_L
  (+6.60, 5 of 5 seeds), CI_G (+0.64, 4 of 5) and IBS tau90 (-0.26, 4 of
  5 seeds).
* E4: the reference code's trajectory loss changes little (CI_G -0.33,
  higher on 1 of 5 seeds; IBS tau90 +0.07). The reference Rotterdam
  config is slightly worse on every seed for CI_G (-0.56) and IBS tau90
  (+0.14), but better on the default IBS (-0.87 ± 1.06, lower on 4 of 5
  seeds), which depends on a few large tail weights (see the IBS bullet).
  All three MENSA variants equal the no-model CI_L reference on 5 of 5
  seeds.

Units: printed C-indexes, AUC and IBS, the plots, and the paired and
constant-order CSVs are x100 (as in the paper); differences are x100
points. Run files and the ``*_summary.csv`` files keep the raw 0-1 values
(they also hold MAE in days and counts).

Extra dependencies, used only by this example (not by PyHealth itself)::

    pip install scikit-multilearn SurvivalEVAL scikit-survival

Run the whole pipeline on ~200 synthetic patients (one seed, two epochs, in
a temporary directory; no data needed)::

    python examples/rotterdam_multi_event_survival_mensa.py --synthetic

E1, the main comparison (MENSA, CoxPH and DeepSurv on seeds 0-9); keep the
output directory outside the repository::

    python examples/rotterdam_multi_event_survival_mensa.py --data-root ../data \
        --experiment main --seeds 0-9 --output-dir ../runs/e1

E2, learning curves on seeds 0-4. Use E1's output directory so the 100% runs
are reused instead of retrained::

    python examples/rotterdam_multi_event_survival_mensa.py --data-root ../data \
        --experiment learning_curve --seeds 0-4 --output-dir ../runs/e1

E3 and E4 on seeds 0-4, also in E1's output directory (the unchanged E1
configurations are reused)::

    python examples/rotterdam_multi_event_survival_mensa.py --data-root ../data \
        --experiment ablation --seeds 0-4 --output-dir ../runs/e1
    python examples/rotterdam_multi_event_survival_mensa.py --data-root ../data \
        --experiment model_ablation --seeds 0-4 --output-dir ../runs/e1

Check the C-index code on a hand-worked example (no data needed)::

    python examples/rotterdam_multi_event_survival_mensa.py --self-check

Check the split and grid on the real data (sizes, outcome mix, and that the
same seed reproduces the same split); add ``--baselines`` to also fit both
baselines per seed and print their C-indexes::

    python examples/rotterdam_multi_event_survival_mensa.py --data-root ../data \
        --experiment split_check --seeds 0 --baselines
"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import RotterdamDataset, SampleDataset, get_dataloader
from pyhealth.datasets.rotterdam import ROTTERDAM_COLUMNS, ROTTERDAM_SIZE_CATEGORIES
from pyhealth.models import MENSA
from pyhealth.tasks import MultiEventSurvivalRotterdam

SPLIT_FRACTIONS: tuple[float, float, float] = (0.7, 0.1, 0.2)
NUM_TIME_BINS_FOR_STRATIFICATION = 20  # equal-width bin edges per event time
EVENT_NAMES: tuple[str, ...] = ("recurrence", "death")
AUC_QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75)
# Bump when a change makes earlier run files incomparable. Version 2 added
# IBS(tau90) (``ibs_tau90_*``); version-1 run files lack it and are rerun.
EXAMPLE_VERSION = "2"
# IBS(tau90) horizon: this quantile of each event's training times.
IBS_HORIZON_QUANTILE = 0.9
METHODS: tuple[str, ...] = ("mensa", "coxph", "deepsurv")
METHOD_LABELS: dict[str, str] = {
    "mensa": "MENSA",
    "coxph": "CoxPH",
    "deepsurv": "DeepSurv",
}
# Rotterdam results as printed in Lillelund et al., "MENSA: A Multi-Event Network
# for Survival Analysis with Trajectory-based Likelihood Estimation", PMLR 297
# (ML4H 2025), pp. 543-571, Tables 2 (CI_G, CI_L, AUC) and 3 (IBS, mMAE, D-Cal):
# mean ± SD over 10 experiments; CI, AUC and IBS x100; D-Cal counts calibrated
# (experiment, event) pairs. The paper does not state mMAE's unit.
PAPER_ROTTERDAM: dict[str, dict[str, str]] = {
    "mensa": {
        "global_ci": "69.4 ± 1.20",
        "local_ci": "90.1 ± 1.55",
        "auc_mean": "73.8 ± 2.31",
        "ibs_mean": "18.8 ± 1.59",
        "mae_margin_mean": "22.6 ± 0.68",
        "dcal": "18/20",
    },
    "coxph": {
        "global_ci": "69.7 ± 1.16",
        "local_ci": "83.6 ± 2.09",
        "auc_mean": "75.1 ± 1.48",
        "ibs_mean": "19.6 ± 1.53",
        "mae_margin_mean": "23.2 ± 1.15",
        "dcal": "20/20",
    },
    "deepsurv": {
        "global_ci": "70.0 ± 1.24",
        "local_ci": "84.6 ± 2.93",
        "auc_mean": "75.4 ± 1.75",
        "ibs_mean": "19.4 ± 1.47",
        "mae_margin_mean": "22.7 ± 1.02",
        "dcal": "18/20",
    },
}
LEARNING_CURVE_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 1.0)
# (metric key, short name, plot title, True if higher is better)
LEARNING_CURVE_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("local_ci", "CI_L", "Local C-index", True),
    ("global_ci", "CI_G", "Global C-index", True),
    ("ibs_mean", "IBS", "Integrated Brier score", False),
    ("ibs_tau90_mean", "IBS_tau90", "Integrated Brier score up to tau90", False),
)
# E3 and E4 variants. Each is a method plus changes to its E1 configuration;
# the unchanged E1 configurations reuse E1's run files. "note" marks variants
# the paper already reports (re-run here only to verify).
IN_PAPER_NOTE = "lambda = 0 was already ablated in the paper; verification only"
ABLATION_VARIANTS: tuple[dict, ...] = (
    {"name": "mensa", "method": "mensa", "label": "MENSA", "overrides": {}},
    {
        "name": "mensa_lambda0",
        "method": "mensa",
        "label": "MENSA, lambda=0",
        "overrides": {"trajectory_weight": 0.0},
        "note": IN_PAPER_NOTE,
    },
    {
        "name": "mensa_8feat",
        "method": "mensa",
        "label": "MENSA, 8 features",
        "overrides": {"use_treatment": False},
    },
    {
        "name": "mensa_8feat_lambda0",
        "method": "mensa",
        "label": "MENSA, 8 features, lambda=0",
        "overrides": {"use_treatment": False, "trajectory_weight": 0.0},
        "note": IN_PAPER_NOTE,
    },
    {"name": "coxph", "method": "coxph", "label": "CoxPH", "overrides": {}},
    {
        "name": "coxph_8feat",
        "method": "coxph",
        "label": "CoxPH, 8 features",
        "overrides": {"use_treatment": False},
    },
)
# Pairs (a, b): the paired difference is a - b on each seed.
ABLATION_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("mensa_lambda0", "mensa"),
    ("mensa_8feat", "mensa"),
    ("mensa_8feat_lambda0", "mensa_8feat"),
    ("coxph_8feat", "coxph"),
    ("mensa", "coxph"),
    ("mensa_8feat", "coxph_8feat"),
)
MODEL_ABLATION_VARIANTS: tuple[dict, ...] = (
    {"name": "mensa", "method": "mensa", "label": "MENSA", "overrides": {}},
    {
        "name": "mensa_reference_loss",
        "method": "mensa",
        "label": "MENSA, reference trajectory loss",
        "overrides": {"trajectory_variant": "reference"},
    },
    {
        # The reference code's configs/mensa/rotterdam.yaml (commit 6b219cd).
        "name": "mensa_reference_config",
        "method": "mensa",
        "label": "MENSA, reference config (16, 0.5, 1e-4)",
        "overrides": {"hidden_dims": [16], "dropout": 0.5, "weight_decay": 1e-4},
    },
)
MODEL_ABLATION_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("mensa_reference_loss", "mensa"),
    ("mensa_reference_config", "mensa"),
)
# Reference Rotterdam baseline configs (configs/coxph and configs/deepsurv).
COXPH_CONFIG: dict[str, float] = {"alpha": 10.0, "n_iter": 100, "tol": 1e-5}
DEEPSURV_CONFIG: dict[str, float] = {
    "hidden_size": 64,
    "dropout": 0.5,
    "lr": 1e-3,
    "c1": 0.1,
    "num_epochs": 2000,
    "patience": 20,
}
# MENSA settings for the experiments: the model's defaults plus the training
# settings used with PyHealth's Trainer.
MENSA_CONFIG: dict[str, object] = {
    "hidden_dims": [32],
    "dropout": 0.25,
    "n_dists": 3,
    "trajectory_weight": 0.25,
    "trajectory_variant": "paper",
    "use_treatment": True,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "max_grad_norm": 1.0,
    "max_epochs": 1000,
    "patience": 20,
}


def dataset_arrays(
    dataset: SampleDataset,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read patient IDs, features, event times and indicators from samples.

    Patients are returned sorted by ID, so the result does not depend on the
    order in which samples were cached. Used by :func:`survival_arrays` and
    :func:`baseline_curves`.

    Args:
        dataset: Samples from :class:`~pyhealth.tasks.MultiEventSurvivalRotterdam`
            (or any task with ``features``, ``event_times`` and
            ``event_indicators``).

    Returns:
        ``(patient_ids, features, times, indicators)``: IDs sorted as strings,
        then ``[N, F]``, ``[N, K]`` and ``[N, K]`` float64 arrays in that order.

    Raises:
        ValueError: If a patient has more than one sample.

    Examples:
        >>> ids, x, times, events = dataset_arrays(samples)  # doctest: +SKIP
        >>> x.shape, times.shape  # doctest: +SKIP
        ((2982, 10), (2982, 2))
    """
    by_patient: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for batch in get_dataloader(dataset, batch_size=1024, shuffle=False):
        for pid, x, t, e in zip(
            batch["patient_id"],
            batch["features"],
            batch["event_times"],
            batch["event_indicators"],
        ):
            if pid in by_patient:
                raise ValueError(f"patient {pid!r} has more than one sample")
            by_patient[pid] = (x.numpy(), t.numpy(), e.numpy())
    ids = sorted(by_patient, key=str)
    features, times, indicators = (
        np.stack([by_patient[p][j] for p in ids]).astype(np.float64) for j in range(3)
    )
    return ids, features, times, indicators


def survival_arrays(
    dataset: SampleDataset,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read patient IDs, event times and indicators from a sample dataset.

    Same order as :func:`dataset_arrays`, without the features. Used by
    :func:`stratified_split` and :func:`make_time_grid`.

    Args:
        dataset: Samples with ``event_times`` and ``event_indicators``.

    Returns:
        ``(patient_ids, times, indicators)``: IDs sorted as strings, and
        ``[N, K]`` float64 arrays in the same order.

    Examples:
        >>> ids, times, events = survival_arrays(samples)  # doctest: +SKIP
        >>> times.shape  # doctest: +SKIP
        (2982, 2)
    """
    ids, _, times, indicators = dataset_arrays(dataset)
    return ids, times, indicators


@contextmanager
def _seeded_numpy_global_rng(seed: int) -> Iterator[None]:
    """Seed NumPy's global RNG for a block, then restore its previous state.

    scikit-multilearn's iterative stratification breaks ties with
    ``np.random.choice`` and ignores its ``random_state`` argument, so the
    global RNG must be seeded for the split to be reproducible. Restoring the
    state keeps this from affecting any other code.

    Args:
        seed: Seed for the duration of the block.

    Yields:
        Nothing; the block runs with the seeded global RNG.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def stratification_labels(times: np.ndarray, indicators: np.ndarray) -> np.ndarray:
    """Build the reference code's multi-event stratification labels.

    For each event, in order: the indicator, then the time's bin among 20
    equal-width edges spanning that event's full time range
    (``np.digitize(..., right=True)``), as in the reference
    ``make_stratified_split(..., stratify_colname="multi")``.

    Note that scikit-multilearn treats every nonzero entry as a present
    label, so any time bin above 0 counts as present. In practice the split
    therefore balances the event indicators and their combinations, and the
    time bins have little effect. This matches the reference behavior.

    Args:
        times: ``[N, K]`` event times from the full dataset.
        indicators: ``[N, K]`` 0/1 event indicators.

    Returns:
        ``[N, 2K]`` integer labels: ``e1, bin(t1), e2, bin(t2), ...``.

    Examples:
        >>> labels = stratification_labels(
        ...     np.array([[1.0, 1.0], [50.0, 100.0], [100.0, 100.0]]),
        ...     np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
        ... )
        >>> labels.tolist()
        [[1, 0, 1, 0], [0, 10, 1, 19], [1, 19, 0, 19]]
    """
    columns = []
    for k in range(times.shape[1]):
        t = times[:, k]
        edges = np.linspace(t.min(), t.max(), NUM_TIME_BINS_FOR_STRATIFICATION)
        columns.append(indicators[:, k].astype(np.int64))
        columns.append(np.digitize(t, edges, right=True).astype(np.int64))
    return np.stack(columns, axis=1)


def stratified_split_ids(
    patient_ids: Sequence[str],
    times: np.ndarray,
    indicators: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> dict[str, list[str]]:
    """Split patient IDs into train/validation/test with iterative stratification.

    Follows the reference ``make_stratified_split``, which calls its
    ``multilabel_train_test_split`` twice: shuffle the rows with the seed and
    take the training fraction with scikit-multilearn's
    ``iterative_train_test_split``; then shuffle the remainder again with the
    same seed and split it into validation and test the same way. The global
    NumPy RNG that scikit-multilearn relies on is seeded with the split seed,
    where the reference seeds it once with 0 at startup, so the same seed
    always gives the same split.

    Args:
        patient_ids: One ID per row, in the same order as ``times``.
        times: ``[N, K]`` event times.
        indicators: ``[N, K]`` 0/1 event indicators.
        seed: Seed for the shuffle and the stratification.
        fractions: Train, validation and test fractions.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``, each sorted, with
        every patient in exactly one split.

    Raises:
        ImportError: If scikit-multilearn is not installed.
        ValueError: If the fractions are invalid or the inputs disagree.

    Examples:
        >>> ids, times, events = survival_arrays(samples)  # doctest: +SKIP
        >>> split = stratified_split_ids(ids, times, events, seed=0)  # doctest: +SKIP
        >>> [len(split[k]) for k in ("train", "val", "test")]  # doctest: +SKIP
        [2087, 299, 596]
    """
    try:
        from skmultilearn.model_selection import iterative_train_test_split
    except ImportError as exc:
        raise ImportError(
            "stratified_split needs scikit-multilearn: pip install scikit-multilearn"
        ) from exc
    from sklearn.utils import shuffle

    train_frac, val_frac, test_frac = fractions
    if min(fractions) <= 0 or not math.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be positive and sum to 1")
    n = len(patient_ids)
    if times.shape != indicators.shape or times.shape[0] != n:
        raise ValueError("patient_ids, times and indicators must have the same rows")
    if len(set(patient_ids)) != n:
        raise ValueError("patient_ids must be unique")

    rows = np.arange(n).reshape(-1, 1)
    labels = stratification_labels(times, indicators)
    with _seeded_numpy_global_rng(seed):
        rows, labels = shuffle(rows, labels, random_state=seed)
        train_rows, _, rest_rows, rest_labels = iterative_train_test_split(
            rows, labels, test_size=1.0 - train_frac
        )
        # The reference reshuffles on every call of multilabel_train_test_split.
        rest_rows, rest_labels = shuffle(rest_rows, rest_labels, random_state=seed)
        val_rows, _, test_rows, _ = iterative_train_test_split(
            rest_rows, rest_labels, test_size=test_frac / (val_frac + test_frac)
        )
    split = {
        name: sorted((patient_ids[i] for i in part[:, 0]), key=str)
        for name, part in (
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        )
    }
    assert sum(len(v) for v in split.values()) == n
    return split


def stratified_split(
    dataset: SampleDataset,
    seed: int,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[SampleDataset, SampleDataset, SampleDataset, dict[str, list[str]]]:
    """Split a sample dataset by patient with multi-event stratification.

    Stratification labels use the whole dataset, as in the reference code.
    Subsets are built with ``dataset.subset`` like
    :func:`~pyhealth.datasets.split_by_patient`, so they share the processed
    samples.

    Args:
        dataset: Samples from :class:`~pyhealth.tasks.MultiEventSurvivalRotterdam`.
        seed: Seed for the split.
        fractions: Train, validation and test fractions.

    Returns:
        ``(train, val, test, patient_ids)``: the three subsets, and the
        patient IDs in each, for saving with results.

    Examples:
        >>> train, val, test, ids = stratified_split(samples, seed=0)  # doctest: +SKIP
        >>> len(train), len(val), len(test)  # doctest: +SKIP
        (2087, 299, 596)
    """
    patient_ids, times, indicators = survival_arrays(dataset)
    split = stratified_split_ids(patient_ids, times, indicators, seed, fractions)
    subsets = tuple(
        dataset.subset(
            [i for pid in split[name] for i in dataset.patient_to_index[pid]]
        )
        for name in ("train", "val", "test")
    )
    return (*subsets, split)


def make_time_grid(train: SampleDataset) -> torch.Tensor:
    """Evaluation time grid from training times only.

    As in the reference ``make_time_bins`` (called with ``event=None``):
    ``ceil(sqrt(N))`` quantiles, from 0 to 1, of all training event times
    pooled across both events, with duplicates removed and 0 prepended. ``N``
    is the number of training patients.

    Args:
        train: The training split only, so held-out times never shape the
            grid.

    Returns:
        Increasing ``float32`` tensor of times, starting at 0, in days.

    Examples:
        >>> grid = make_time_grid(train)  # doctest: +SKIP
        >>> grid[0].item(), len(grid)  # doctest: +SKIP
        (0.0, 47)
    """
    _, times, _ = survival_arrays(train)
    num_bins = math.ceil(math.sqrt(times.shape[0]))
    edges = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    return torch.tensor(np.concatenate([[0.0], edges]), dtype=torch.float32)


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize features with means and scales from the training set only.

    Population standard deviations, as scikit-learn's ``StandardScaler``
    (used by the reference ``preprocess_data``); constant features get scale 1.

    Args:
        train: ``[N, F]`` training features; the statistics come from these.
        *others: Further ``[M, F]`` arrays (validation, test) to transform.

    Returns:
        The standardized training array followed by the others, in order.

    Examples:
        >>> standardize(np.array([[0.0, 5.0], [2.0, 5.0]]), np.array([[1.0, 6.0]]))
        (array([[-1.,  0.],
               [ 1.,  0.]]), array([[0., 1.]]))
    """
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return tuple((x - mean) / scale for x in (train, *others))


def curves_on_grid(
    curves: np.ndarray, curve_times: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Move survival curves from a model's own times onto the shared grid.

    Linear interpolation with linear extrapolation beyond the model's times,
    capped at 1, as in the reference ``train_models_multi_event.py``. Two
    additions keep the curves in [0, 1] and non-increasing like MENSA's:
    values are floored at 0 (extrapolating past the last training time can
    otherwise go negative), and a running minimum removes float rounding
    increases (about 1e-16) where a curve is flat. As in the reference, the
    value at time 0 comes from extrapolation and can be just below 1.

    Args:
        curves: ``[N, U]`` survival probabilities at ``curve_times``.
        curve_times: ``[U]`` increasing times.
        grid: ``[T]`` target times.

    Returns:
        ``[N, T]`` survival probabilities on ``grid``.

    Examples:
        >>> curves_on_grid(np.array([[1.0, 0.5]]), np.array([0.0, 10]),
        ...                np.array([0.0, 5, 30]))
        array([[1.  , 0.75, 0.  ]])
    """
    from scipy.interpolate import interp1d

    spline = interp1d(curve_times, curves, kind="linear", fill_value="extrapolate")
    return np.minimum.accumulate(np.clip(spline(grid), 0.0, 1.0), axis=-1)


def fit_coxph(
    train_x: np.ndarray, train_times: np.ndarray, train_events: np.ndarray
) -> list:
    """Fit one scikit-survival Cox model per event (reference ``make_coxph_model``).

    ``CoxPHSurvivalAnalysis(alpha=10, n_iter=100, tol=1e-5)`` with Breslow
    ties, the reference Rotterdam config.

    Args:
        train_x: ``[N, F]`` standardized training features.
        train_times: ``[N, K]`` training times.
        train_events: ``[N, K]`` training 0/1 indicators.

    Returns:
        ``K`` fitted models, in event order.

    Raises:
        ImportError: If scikit-survival is not installed.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=(40, 3))
        >>> times = rng.uniform(1, 100, size=(40, 2))
        >>> events = rng.integers(0, 2, size=(40, 2))
        >>> models = fit_coxph(x, times, events)
        >>> len(models), type(models[0]).__name__
        (2, 'CoxPHSurvivalAnalysis')
    """
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError(
            "The CoxPH baseline needs scikit-survival: pip install scikit-survival"
        ) from exc
    models = []
    for k in range(train_times.shape[1]):
        y = Surv.from_arrays(train_events[:, k].astype(bool), train_times[:, k])
        model = CoxPHSurvivalAnalysis(
            alpha=COXPH_CONFIG["alpha"],
            n_iter=int(COXPH_CONFIG["n_iter"]),
            tol=COXPH_CONFIG["tol"],
        )
        models.append(model.fit(train_x, y))
    return models


def predict_coxph(models: list, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Survival curves from fitted Cox models, on the shared grid.

    Each model's step functions are evaluated at its own training event
    times (``unique_times_``), then moved to ``grid`` with
    :func:`curves_on_grid`, as in the reference.

    Args:
        models: Output of :func:`fit_coxph`.
        x: ``[N, F]`` standardized features.
        grid: ``[T]`` evaluation times.

    Returns:
        ``[N, K, T]`` survival probabilities.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=(40, 3))
        >>> models = fit_coxph(
        ...     x, rng.uniform(1, 100, size=(40, 2)), rng.integers(0, 2, size=(40, 2))
        ... )
        >>> curves = predict_coxph(models, x[:5], np.array([0.0, 25.0, 50.0, 100.0]))
        >>> curves.shape, bool((curves[..., 0] == 1).all())
        ((5, 2, 4), True)
    """
    curves = []
    for model in models:
        functions = model.predict_survival_function(x)
        own = np.vstack([fn(model.unique_times_) for fn in functions])
        curves.append(curves_on_grid(own, model.unique_times_, grid))
    return np.stack(curves, axis=1)


class DeepSurvNet(nn.Module):
    """DeepSurv network: ``Linear -> ReLU -> Dropout -> Linear(1)``.

    Same layers as the reference ``DeepSurv`` (64 hidden units, dropout
    0.5 for Rotterdam). The output is the log relative hazard.

    Args:
        in_features: Number of input features.
        hidden_size: Hidden layer width.
        dropout: Dropout probability.

    Examples:
        >>> net = DeepSurvNet(in_features=10, hidden_size=64, dropout=0.5)
        >>> net(torch.zeros(4, 10)).shape
        torch.Size([4, 1])
    """

    def __init__(self, in_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, 1]`` log relative hazards for ``[B, F]`` features.

        Args:
            x: ``[B, F]`` standardized features.

        Returns:
            ``[B, 1]`` log relative hazards.

        Examples:
            >>> DeepSurvNet(3, 8, 0.0)(torch.ones(2, 3)).shape
            torch.Size([2, 1])
        """
        return self.fc1(self.hidden(x))


def cox_nll(
    log_hazard: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    model: nn.Module | None = None,
    c1: float = 0.0,
) -> torch.Tensor:
    """Cox partial negative log-likelihood, as in the reference ``cox_nll``.

    ``-(1/D) sum_{i: event} [f_i - log sum_{j: t_j >= t_i} exp(f_j)]``, where
    ``D`` is the number of events. With a model and ``c1 > 0``, adds
    ``c1 / 2 * ||W||`` for every weight matrix ``W``, using the plain
    (not squared) Frobenius norm exactly as the reference does.

    Args:
        log_hazard: ``[N]`` or ``[N, 1]`` predicted log relative hazards.
        times: ``[N]`` times.
        events: ``[N]`` 0/1 indicators; at least one must be 1.
        model: Model whose weights are penalized, when ``c1 > 0``.
        c1: Regularization strength (0.1 for training, 0 for validation).

    Returns:
        Scalar loss.

    Examples:
        >>> cox_nll(torch.zeros(2), torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
        tensor(0.3466)
    """
    f = log_hazard.reshape(-1)
    t = times.reshape(-1)
    e = events.reshape(-1).to(f.dtype)
    at_risk = (t[None, :] >= t[:, None]).to(f.dtype)  # [i, j]: j still at risk at t_i
    peak = f.max()
    log_risk_sum = (
        torch.log((torch.exp(f - peak)[None, :] * at_risk).sum(1) + 1e-20) + peak
    )
    loss = -torch.sum((f - log_risk_sum) * e) / e.sum()
    if model is not None and c1 > 0:
        for name, parameter in model.named_parameters():
            if "weight" in name:
                loss = loss + c1 / 2 * torch.norm(parameter, p=2)
    return loss


def breslow_baseline(
    log_hazard: np.ndarray, times: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow baseline survival from training predictions.

    As in the reference ``calculate_baseline_hazard``: at each unique
    training time ``u``, hazard ``d_u / sum_{j: t_j >= u} exp(f_j)``;
    cumulative hazard ``H``; baseline survival ``exp(-H)`` made
    non-increasing, with time 0 (survival 1) prepended.

    Args:
        log_hazard: ``[N]`` training log relative hazards.
        times: ``[N]`` training times.
        events: ``[N]`` training 0/1 indicators.

    Returns:
        ``(baseline_times, baseline_survival)``, each ``[U]``, starting at 0.

    Examples:
        >>> breslow_baseline(np.zeros(2), np.array([1.0, 2.0]), np.array([1.0, 0.0]))
        (array([0., 1., 2.]), array([1.        , 0.60653066, 0.60653066]))
    """
    order = np.argsort(times, kind="stable")
    t, e, risk = times[order], events[order], np.exp(log_hazard[order])
    unique_times, first = np.unique(t, return_index=True)
    at_risk = np.cumsum(risk[::-1])[::-1][first]
    hazard = np.add.reduceat(e, first) / at_risk
    if unique_times[0] != 0:
        unique_times = np.concatenate([[0.0], unique_times])
        hazard = np.concatenate([[0.0], hazard])
    survival = np.minimum.accumulate(np.exp(-np.cumsum(hazard)))
    return unique_times, survival


def fit_deepsurv(
    train_x: np.ndarray,
    train_times: np.ndarray,
    train_events: np.ndarray,
    val_x: np.ndarray,
    val_times: np.ndarray,
    val_events: np.ndarray,
    seed: int,
    num_epochs: int | None = None,
) -> list[dict]:
    """Train one DeepSurv model per event, as the reference ``train_deepsurv_model``.

    Adam (lr 1e-3), the whole training set as one batch (one step per epoch),
    loss :func:`cox_nll` with ``c1 = 0.1``, up to 2000 epochs. Training stops
    once the validation Cox NLL (``c1 = 0``) has not improved for more than
    20 epochs. Three reference behaviors are kept on purpose so results stay
    comparable with the paper: the weights from the **last** epoch are used
    (not the best), and dropout stays **on** while computing the validation
    loss and the Breslow baseline (the reference never switches to
    evaluation mode there). Dropout is off for prediction.

    Args:
        train_x: ``[N, F]`` standardized training features.
        train_times: ``[N, K]`` training times.
        train_events: ``[N, K]`` training 0/1 indicators.
        val_x: ``[M, F]`` standardized validation features.
        val_times: ``[M, K]`` validation times.
        val_events: ``[M, K]`` validation 0/1 indicators.
        seed: Seeds ``torch`` once before the first event's model, so the
            whole fit is reproducible.
        num_epochs: Optional cap on epochs (default: the config's 2000).

    Returns:
        One dict per event with ``net``, ``baseline_times``,
        ``baseline_survival`` and ``epochs`` (epochs actually run).

    Examples:
        >>> fitted = fit_deepsurv(  # doctest: +SKIP
        ...     train_x, train_times, train_events, val_x, val_times, val_events, seed=0
        ... )
        >>> len(fitted)  # doctest: +SKIP
        2
    """
    cfg = DEEPSURV_CONFIG
    torch.manual_seed(seed)
    tx, vx = (torch.tensor(a, dtype=torch.float32) for a in (train_x, val_x))
    fitted = []
    for k in range(train_times.shape[1]):
        tt, te = (
            torch.tensor(a[:, k], dtype=torch.float32)
            for a in (train_times, train_events)
        )
        vt, ve = (
            torch.tensor(a[:, k], dtype=torch.float32) for a in (val_times, val_events)
        )
        net = DeepSurvNet(tx.shape[1], int(cfg["hidden_size"]), float(cfg["dropout"]))
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        net.train()
        best_loss, best_epoch, epoch = float("inf"), -1, -1
        for epoch in range(int(num_epochs or cfg["num_epochs"])):
            optimizer.zero_grad()
            loss = cox_nll(net(tx), tt, te, net, c1=cfg["c1"])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_loss = cox_nll(net(vx), vt, ve).item()  # dropout on, as reference
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
            if epoch - best_epoch > cfg["patience"]:
                break
        with torch.no_grad():  # dropout still on, as in the reference
            train_log_hazard = net(tx).reshape(-1).double().numpy()
        times, survival = breslow_baseline(
            train_log_hazard, train_times[:, k], train_events[:, k]
        )
        fitted.append(
            {
                "net": net,
                "baseline_times": times,
                "baseline_survival": survival,
                "epochs": epoch + 1,
            }
        )
    return fitted


def predict_deepsurv(fitted: list[dict], x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Survival curves ``S0(t) ** exp(f(x))`` from DeepSurv, on the shared grid.

    Args:
        fitted: Output of :func:`fit_deepsurv`.
        x: ``[N, F]`` standardized features.
        grid: ``[T]`` evaluation times.

    Returns:
        ``[N, K, T]`` survival probabilities.

    Examples:
        >>> curves = predict_deepsurv(fitted, test_x, grid)  # doctest: +SKIP
        >>> curves.shape == (len(test_x), 2, len(grid))  # doctest: +SKIP
        True
    """
    curves = []
    for model in fitted:
        model["net"].eval()
        with torch.no_grad():
            relative = torch.exp(model["net"](torch.tensor(x, dtype=torch.float32)))
        own = model["baseline_survival"][None, :] ** relative.double().numpy()
        curves.append(curves_on_grid(own, model["baseline_times"], grid))
    return np.stack(curves, axis=1)


def baseline_curves(
    method: str,
    train: SampleDataset,
    val: SampleDataset,
    test: SampleDataset,
    grid: np.ndarray | torch.Tensor,
    seed: int,
    num_epochs: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a baseline on the training split and predict test survival curves.

    Features are standardized with training statistics only. One model is
    fitted per event (recurrence, death). The curves are on ``grid``, in the
    same patient order as :func:`survival_arrays` of ``test``, with the same
    ``[N, K, T]`` layout as MENSA's ``predict_survival``.

    Args:
        method: ``"coxph"`` or ``"deepsurv"``.
        train: Training split.
        val: Validation split (DeepSurv early stopping only).
        test: Test split.
        grid: ``[T]`` evaluation times, e.g. from :func:`make_time_grid`.
        seed: Seed for DeepSurv's initialization and dropout.
        num_epochs: Optional cap on DeepSurv's epochs.

    Returns:
        ``(curves, info)``: ``[N, K, T]`` test curves, and run details
        (DeepSurv's epochs per event).

    Raises:
        ValueError: If ``method`` is unknown.

    Examples:
        >>> curves, info = baseline_curves(  # doctest: +SKIP
        ...     "coxph", train, val, test, make_time_grid(train), seed=0
        ... )
        >>> curves.shape  # doctest: +SKIP
        (597, 2, 47)
    """
    _, train_x, train_t, train_e = dataset_arrays(train)
    _, val_x, val_t, val_e = dataset_arrays(val)
    _, test_x, _, _ = dataset_arrays(test)
    train_x, val_x, test_x = standardize(train_x, val_x, test_x)
    grid = np.asarray(grid, dtype=np.float64)
    if method == "coxph":
        return predict_coxph(fit_coxph(train_x, train_t, train_e), test_x, grid), {}
    if method == "deepsurv":
        fitted = fit_deepsurv(
            train_x, train_t, train_e, val_x, val_t, val_e, seed, num_epochs
        )
        info = {
            f"epochs_{name}": float(m["epochs"]) for name, m in zip(EVENT_NAMES, fitted)
        }
        return predict_deepsurv(fitted, test_x, grid), info
    raise ValueError(f"method must be 'coxph' or 'deepsurv', got {method!r}")


def median_survival_times(curves: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Predicted median survival time of each curve, as SurvivalEVAL computes it.

    Calls SurvivalEVAL's ``predict_median_st`` with linear interpolation, the
    default its ``concordance`` uses: the first grid time where the curve
    reaches 0.5, interpolated linearly; a curve that never reaches 0.5 is
    extended by the line from (0, 1) through its last point. Using the same
    function keeps our C-index risk scores identical to SurvivalEVAL's.

    Args:
        curves: ``[N, T]`` survival probabilities on ``grid``.
        grid: ``[T]`` increasing times, starting at 0.

    Returns:
        ``[N]`` predicted median times (``inf`` if a curve stays at 1).

    Examples:
        >>> median_survival_times(np.array([[1.0, 0.5, 0.2]]), np.array([0, 10, 20]))
        array([10.])
    """
    from SurvivalEVAL.evaluations.util import predict_median_st

    return np.atleast_1d(predict_median_st(curves, grid, "Linear")).astype(float)


def concordance_pairs(
    times: np.ndarray,
    events: np.ndarray,
    predicted_times: np.ndarray,
    tied_tol: float = 1e-8,
) -> tuple[float, int]:
    """Count Harrell concordant and comparable pairs, as SurvivalEVAL does.

    A pair (i, j) is comparable when i had an observed event and j's time is
    later, or j was censored at the same time. It is concordant when i is
    predicted to fail first (smaller predicted time). Predicted-time ties
    (within ``tied_tol``) count as half; pairs of events at the same time are
    left out. This matches SurvivalEVAL's ``concordance`` defaults
    (``method="Harrell"``, ``ties="Risk"``).

    Args:
        times: ``[M]`` observed or censoring times.
        events: ``[M]`` 0/1 indicators.
        predicted_times: ``[M]`` predicted times (e.g. median survival).
        tied_tol: Absolute tolerance for predicted-time ties.

    Returns:
        ``(concordant, comparable)``: concordant pairs (with half credit for
        ties) and the number of comparable pairs.

    Examples:
        >>> concordance_pairs(
        ...     np.array([5.0, 12.0, 25.0]),
        ...     np.array([1, 1, 0]),
        ...     np.array([10.0, 30.0, 20.0]),
        ... )
        (2.0, 3)
    """
    anchor = (events == 1)[:, None]
    later = times[None, :] > times[:, None]
    censored_same_time = (times[None, :] == times[:, None]) & (events == 0)[None, :]
    comparable = anchor & (later | censored_same_time)
    np.fill_diagonal(comparable, False)
    with np.errstate(invalid="ignore"):
        difference = predicted_times[None, :] - predicted_times[:, None]
    tied = np.abs(difference) <= tied_tol
    concordant = comparable & (difference > 0) & ~tied
    return (
        float(concordant.sum() + 0.5 * (comparable & tied).sum()),
        int(comparable.sum()),
    )


def global_c_index(
    predicted_times: np.ndarray, times: np.ndarray, events: np.ndarray
) -> tuple[float, float, int]:
    """Global C-index: pairs between patients, pooled over all events.

    Paper Eq. 22 and the reference ``global_C_index`` (``weight=True``):
    count concordant and comparable pairs separately for each event, then
    divide the summed concordant pairs by the summed comparable pairs.

    Args:
        predicted_times: ``[N, K]`` predicted median times per event.
        times: ``[N, K]`` observed or censoring times.
        events: ``[N, K]`` 0/1 indicators.

    Returns:
        ``(c_index, concordant, comparable)``; the C-index is ``nan`` when
        there are no comparable pairs.

    Examples:
        Three patients, (recurrence, death) per row; 3 of the 6 comparable
        pairs are concordant:

        >>> predicted = np.array([[10.0, 30.0], [20.0, 40.0], [30.0, 20.0]])
        >>> times = np.array([[5.0, 15.0], [25.0, 25.0], [12.0, 35.0]])
        >>> events = np.array([[1, 1], [0, 1], [1, 0]])
        >>> global_c_index(predicted, times, events)
        (0.5, 3.0, 6)
    """
    concordant, comparable = 0.0, 0
    for k in range(times.shape[1]):
        c, n = concordance_pairs(times[:, k], events[:, k], predicted_times[:, k])
        concordant, comparable = concordant + c, comparable + n
    c_index = concordant / comparable if comparable else float("nan")
    return c_index, concordant, comparable


def local_c_index(
    predicted_times: np.ndarray, times: np.ndarray, events: np.ndarray
) -> tuple[float, float, int, int]:
    """Local C-index: pairs between one patient's own events, pooled.

    Paper Eq. 23 and the reference ``local_C_index`` (``weight=True``): for
    each patient with at least one observed event, compare the order of
    their events with the order of their predicted times, using the same
    pair rules as :func:`concordance_pairs`. Patients without comparable
    pairs (no events, or tied event times) add nothing.

    Args:
        predicted_times: ``[N, K]`` predicted median times per event.
        times: ``[N, K]`` observed or censoring times.
        events: ``[N, K]`` 0/1 indicators.

    Returns:
        ``(c_index, concordant, comparable, patients)``, where ``patients``
        is the number of patients contributing at least one pair.

    Examples:
        Each of the three patients contributes one comparable pair; only the
        first patient's events are predicted in the right order:

        >>> predicted = np.array([[10.0, 30.0], [20.0, 40.0], [30.0, 20.0]])
        >>> times = np.array([[5.0, 15.0], [25.0, 25.0], [12.0, 35.0]])
        >>> events = np.array([[1, 1], [0, 1], [1, 0]])
        >>> local_c_index(predicted, times, events)
        (0.3333333333333333, 1.0, 3, 3)
    """
    concordant, comparable, patients = 0.0, 0, 0
    for i in np.flatnonzero(events.sum(axis=1) > 0):
        c, n = concordance_pairs(times[i], events[i], predicted_times[i])
        concordant, comparable, patients = (
            concordant + c,
            comparable + n,
            patients + (n > 0),
        )
    c_index = concordant / comparable if comparable else float("nan")
    return c_index, concordant, comparable, patients


def constant_order_local_ci(
    test_times: np.ndarray, test_events: np.ndarray
) -> dict[str, float]:
    """Local C-index of a no-model predictor: events in ``EVENT_NAMES`` order.

    The predictor gives every patient the same predicted times, earliest
    for the first event in ``EVENT_NAMES`` (recurrence before death), so it
    uses no patient information. On Rotterdam about 91% of the patients
    who count for the local C-index had recurrence as their first event,
    so this predictor scores about 0.91 (0.9097 ± 0.0131 on seeds 0-9).
    It is a reference, not an upper bound: a model can beat it by ranking
    the death-first patients correctly. A model with an equal score has the
    same number of concordant pairs, not necessarily the same pairs. It
    depends only on the test split, so it is computed at summary time and
    not stored in the run files.

    Args:
        test_times: ``[N, K]`` test times, events in ``EVENT_NAMES`` order.
        test_events: ``[N, K]`` test 0/1 indicators.

    Returns:
        ``local_ci``, ``local_concordant``, ``local_pairs`` and
        ``local_patients``, as from :func:`local_c_index`.

    Examples:
        >>> times = np.array([[5.0, 15.0], [25.0, 25.0], [12.0, 35.0]])
        >>> events = np.array([[1, 1], [0, 1], [1, 0]])
        >>> round(constant_order_local_ci(times, events)["local_ci"], 4)
        0.6667
    """
    test_times = np.asarray(test_times, dtype=float)
    n, k_events = test_times.shape
    predicted = np.tile(np.arange(1.0, k_events + 1.0), (n, 1))
    c_index, concordant, pairs, patients = local_c_index(
        predicted, test_times, np.asarray(test_events, dtype=float)
    )
    return {
        "local_ci": float(c_index),
        "local_concordant": float(concordant),
        "local_pairs": float(pairs),
        "local_patients": float(patients),
    }


def report_constant_order(
    reference: dict[int, dict[str, float]],
    groups: Sequence[tuple[str, Sequence[dict]]],
    path: Path,
) -> list[dict]:
    """Compare runs' CI_L with the constant-order reference; write two CSVs.

    For each group of runs, and each seed, CI_L minus the reference on the
    same seed (:func:`constant_order_local_ci`), and whether the run's CI_L
    equals, falls below or exceeds it. All CI_L values are x100. Writes:

    * ``path``: one row per group and seed, with ``group``, ``seed``,
      ``local_ci_x100``, ``reference_x100``, ``difference_x100`` (run minus
      reference), ``relation`` (``equal``, ``below`` or ``above``), and the
      reference's ``reference_concordant``, ``reference_pairs`` and
      ``reference_patients``. The reference's own rows come first, as group
      ``"reference (no model)"`` with blank difference and relation.
    * ``<path stem>_summary.csv``: one row per group (reference first),
      with ``group``, ``n``, ``local_ci_mean_x100``, ``local_ci_std_x100``,
      ``difference_mean_x100``, ``difference_std_x100``, ``n_equal``,
      ``n_below`` and ``n_above``.

    Both are also printed.

    Args:
        reference: Per seed, the output of :func:`constant_order_local_ci`
            on that seed's test set.
        groups: ``(label, run records)`` pairs, e.g. one per method.
        path: Per-seed CSV file to write.

    Returns:
        The summary rows.

    Examples:
        >>> reference = {  # doctest: +SKIP
        ...     0: constant_order_local_ci(test_times, test_events)
        ... }
        >>> summary = report_constant_order(  # doctest: +SKIP
        ...     reference, [("MENSA", mensa_records)], Path("main_constant_order.csv")
        ... )
    """
    reference_label = "reference (no model)"
    seeds = sorted(reference)

    def ref_columns(seed: int) -> dict:
        ref = reference[seed]
        return {
            "reference_x100": 100 * ref["local_ci"],
            "reference_concordant": ref["local_concordant"],
            "reference_pairs": ref["local_pairs"],
            "reference_patients": ref["local_patients"],
        }

    def std(values: np.ndarray) -> float:
        return float(values.std(ddof=1)) if values.size > 1 else float("nan")

    per_seed = [
        {
            "group": reference_label,
            "seed": seed,
            "local_ci_x100": 100 * reference[seed]["local_ci"],
            "difference_x100": "",
            "relation": "",
            **ref_columns(seed),
        }
        for seed in seeds
    ]
    values = np.array([100 * reference[seed]["local_ci"] for seed in seeds])
    summary = [
        {
            "group": reference_label,
            "n": int(values.size),
            "local_ci_mean_x100": float(values.mean()),
            "local_ci_std_x100": std(values),
            "difference_mean_x100": "",
            "difference_std_x100": "",
            "n_equal": "",
            "n_below": "",
            "n_above": "",
        }
    ]
    for label, runs in groups:
        rows = []
        for run in sorted(runs, key=lambda r: r["seed"]):
            if run["seed"] not in reference:
                continue
            difference = run["metrics"]["local_ci"] - reference[run["seed"]]["local_ci"]
            if abs(difference) <= 1e-9:
                relation = "equal"
            else:
                relation = "below" if difference < 0 else "above"
            rows.append(
                {
                    "group": label,
                    "seed": run["seed"],
                    "local_ci_x100": 100 * run["metrics"]["local_ci"],
                    "difference_x100": 100 * difference,
                    "relation": relation,
                    **ref_columns(run["seed"]),
                }
            )
        if not rows:
            continue
        per_seed.extend(rows)
        level = np.array([r["local_ci_x100"] for r in rows])
        d = np.array([r["difference_x100"] for r in rows])
        relations = [r["relation"] for r in rows]
        summary.append(
            {
                "group": label,
                "n": len(rows),
                "local_ci_mean_x100": float(level.mean()),
                "local_ci_std_x100": std(level),
                "difference_mean_x100": float(d.mean()),
                "difference_std_x100": std(d),
                "n_equal": relations.count("equal"),
                "n_below": relations.count("below"),
                "n_above": relations.count("above"),
            }
        )

    summary_path = path.with_name(f"{path.stem}_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    for rows, file in ((per_seed, path), (summary, summary_path)):
        with file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"\nCI_L x100 against a reference without a model ("
        f"{' before '.join(EVENT_NAMES)} for every test patient), seeds {seeds} "
        f"(CSV: {path}, {summary_path.name}):"
    )
    print(f"{'':44s}{'CI_L':>16s}{'minus reference':>18s}   seeds equal/below/above")
    for row in summary:
        level = _x100(
            row["local_ci_mean_x100"] / 100, row["local_ci_std_x100"] / 100, 2
        )
        if row["group"] == reference_label:
            print(f"  {row['group']:42s}{level:>16s}")
            continue
        diff = _x100(
            row["difference_mean_x100"] / 100,
            row["difference_std_x100"] / 100,
            2,
            sign=True,
        )
        print(
            f"  {row['group']:42s}{level:>16s}{diff:>18s}   "
            f"{row['n_equal']}/{row['n_below']}/{row['n_above']} of {row['n']}"
        )
    return summary


def _integrated_brier(
    evaluator,
    test_times: np.ndarray,
    test_events: np.ndarray,
    train_times: np.ndarray,
) -> tuple[float, float]:
    """Integrated Brier score with SurvivalEVAL's defaults, shortened only if needed.

    By default SurvivalEVAL integrates over the test set's censoring times,
    weighting by the training censoring survival. When a test patient is
    censored at or after the last training time, that survival is 0 there
    and SurvivalEVAL 0.9.0 raises an error (the reference used an older
    version). Only then are those late times dropped, keeping the horizon
    below the last training time, where the weights are defined.

    Args:
        evaluator: A SurvivalEVAL ``LifelinesEvaluator`` for one event.
        test_times: ``[N]`` test times for that event.
        test_events: ``[N]`` test 0/1 indicators for that event.
        train_times: ``[M]`` training times for that event.

    Returns:
        ``(ibs, truncated)``: the score (``nan`` if no usable times remain),
        and 1.0 if the horizon was shortened, else 0.0.
    """
    try:
        return float(evaluator.integrated_brier_score()), 0.0
    except ValueError:
        censored = np.unique(test_times[test_events == 0])
        targets = censored[censored < train_times.max()]
        if targets.size == 0:
            return float("nan"), 1.0
        return float(evaluator.integrated_brier_score(target_times=targets)), 1.0


def _horizon_brier(
    evaluator,
    test_times: np.ndarray,
    test_events: np.ndarray,
    train_times: np.ndarray,
    quantile: float = IBS_HORIZON_QUANTILE,
) -> tuple[float, float]:
    """Integrated Brier score up to a fixed horizon, IBS(tau90).

    The default IBS (:func:`_integrated_brier`) is kept as the paper's
    metric, but on Rotterdam it is unstable across splits. It integrates
    over every test censoring time, and the Brier score at time t divides
    each patient's term by G(t), the training-set probability of still
    being uncensored at t. Some splits put censored test patients at
    6,700-7,000 days, where only a handful of training patients are still
    followed: G drops to about 0.001-0.003, single patients get weights of
    several hundred, and the Brier score at those times can exceed 1 (2.6
    for CoxPH recurrence on seed 8). Those few late times then dominate the
    average, so the same seeds score high for every method (seeds 1, 3, 8
    and 9 of 0-9; CoxPH IBS 22.8 ± 6.0 x100, against 19.6 ± 1.5 in the
    paper). This score uses the same SurvivalEVAL Brier scores and
    trapezoidal average, but only over the test censoring times at or
    below tau, the ``quantile`` of this event's training times (events and
    censorings), with tau itself added as the last time point so every
    split integrates up to its own tau. tau depends only on training data.
    On seeds 0-9 tau is about 3,900-4,000 days for recurrence and
    4,150-4,280 for death, G(tau) is 0.21-0.24 (weights of at most 5),
    against tail weights of up to about 720. On the same CoxPH curves it gives
    17.4 ± 0.9 (x100), every seed between 16.5 and 19.1. It is not
    comparable to the paper's IBS, which integrates further out; methods
    compared on the same seed share tau, so paired differences are fair.

    Args:
        evaluator: A SurvivalEVAL ``LifelinesEvaluator`` for one event.
        test_times: ``[N]`` test times for that event.
        test_events: ``[N]`` test 0/1 indicators for that event.
        train_times: ``[M]`` training times for that event.
        quantile: Horizon as a quantile of ``train_times``, in (0, 1).

    Returns:
        ``(ibs, tau)``: the score (``nan`` if fewer than two time points
        fall in ``[first test censoring time, tau]``) and tau in days.

    Examples:
        >>> ibs, tau = _horizon_brier(  # doctest: +SKIP
        ...     evaluator, test_t, test_e, train_t
        ... )
    """
    tau = float(np.quantile(train_times, quantile))
    censored = np.unique(test_times[test_events == 0])
    targets = np.unique(np.append(censored[censored <= tau], tau))
    if targets.size < 2:
        return float("nan"), tau
    return float(evaluator.integrated_brier_score(target_times=targets)), tau


def evaluate_survival(
    curves: np.ndarray | torch.Tensor,
    grid: np.ndarray | torch.Tensor,
    test_times: np.ndarray,
    test_events: np.ndarray,
    train_times: np.ndarray,
    train_events: np.ndarray,
) -> dict[str, float]:
    """All evaluation metrics for one method on one test set.

    * Global and local C-index (:func:`global_c_index`,
      :func:`local_c_index`), from predicted median survival times.
    * Per event, with SurvivalEVAL's ``LifelinesEvaluator`` as in the
      reference ``train_models_multi_event.py``: time-dependent AUC averaged
      over the 25th/50th/75th percentiles of that event's observed test
      times (0.5 when SurvivalEVAL cannot compute one, as in the reference;
      the number of such fallbacks is reported), integrated Brier score
      (see :func:`_integrated_brier`), the integrated Brier score up to
      tau90 (see :func:`_horizon_brier`), MAE with the Margin method, and
      the D-calibration p-value. Training times and indicators are needed for
      the censoring-weighted metrics.

    Args:
        curves: ``[N, K, T]`` survival probabilities, events in
            ``EVENT_NAMES`` order.
        grid: ``[T]`` increasing times, starting at 0 (see
            :func:`make_time_grid`).
        test_times: ``[N, K]`` test times.
        test_events: ``[N, K]`` test 0/1 indicators.
        train_times: ``[M, K]`` training times.
        train_events: ``[M, K]`` training 0/1 indicators.

    Returns:
        Flat dict of floats: ``global_ci``, ``local_ci``, their pair counts,
        ``local_patients``; per event ``auc_<event>``, ``ibs_<event>``,
        ``mae_margin_<event>``, ``dcal_p_<event>``, ``auc_fallbacks_<event>``,
        ``ibs_truncated_<event>`` (1 if the IBS horizon had to be shortened),
        ``ibs_tau90_<event>`` and its horizon ``tau90_days_<event>``;
        means over events ``auc_mean``, ``ibs_mean``, ``ibs_tau90_mean``,
        ``mae_margin_mean``;
        and ``dcal_calibrated``, the number of events with p > 0.05.

    Raises:
        ImportError: If SurvivalEVAL is not installed.
        ValueError: If the shapes disagree.

    Examples:
        >>> curves = model.predict_survival(test_features, grid)  # doctest: +SKIP
        >>> scores = evaluate_survival(  # doctest: +SKIP
        ...     curves, grid, test_times, test_events, train_times, train_events
        ... )
        >>> scores["global_ci"], scores["local_ci"]  # doctest: +SKIP
    """
    try:
        import pandas as pd
        from SurvivalEVAL import LifelinesEvaluator
    except ImportError as exc:
        raise ImportError(
            "evaluate_survival needs SurvivalEVAL: pip install SurvivalEVAL"
        ) from exc

    curves = np.asarray(curves, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    test_times, test_events = (
        np.asarray(test_times, float),
        np.asarray(test_events, float),
    )
    train_times, train_events = (
        np.asarray(train_times, float),
        np.asarray(train_events, float),
    )
    n, k_events, n_times = curves.shape
    if grid.shape != (n_times,) or test_times.shape != (n, k_events):
        raise ValueError("curves must be [N, K, T] matching grid [T] and times [N, K]")
    if test_events.shape != test_times.shape or train_times.shape != train_events.shape:
        raise ValueError("times and events must have matching shapes")
    if k_events > len(EVENT_NAMES):
        raise ValueError(f"at most {len(EVENT_NAMES)} events are named")

    predicted = np.stack(
        [median_survival_times(curves[:, k], grid) for k in range(k_events)], axis=1
    )
    global_ci, global_concordant, global_pairs = global_c_index(
        predicted, test_times, test_events
    )
    local_ci, local_concordant, local_pairs, local_patients = local_c_index(
        predicted, test_times, test_events
    )
    scores: dict[str, float] = {
        "global_ci": global_ci,
        "local_ci": local_ci,
        "global_concordant": global_concordant,
        "global_pairs": float(global_pairs),
        "local_concordant": local_concordant,
        "local_pairs": float(local_pairs),
        "local_patients": float(local_patients),
    }
    for k, name in enumerate(EVENT_NAMES[:k_events]):
        evaluator = LifelinesEvaluator(
            pd.DataFrame(curves[:, k, :].T, index=grid),
            test_times[:, k],
            test_events[:, k],
            train_times[:, k],
            train_events[:, k],
        )
        observed = test_times[test_events[:, k] == 1, k]
        aucs, fallbacks = [], 0
        for t in np.quantile(observed, AUC_QUANTILES):
            try:
                aucs.append(float(evaluator.auc(t)))
            except ValueError:
                aucs.append(0.5)
                fallbacks += 1
        scores[f"auc_{name}"] = float(np.mean(aucs))
        scores[f"auc_fallbacks_{name}"] = float(fallbacks)
        scores[f"ibs_{name}"], scores[f"ibs_truncated_{name}"] = _integrated_brier(
            evaluator, test_times[:, k], test_events[:, k], train_times[:, k]
        )
        scores[f"ibs_tau90_{name}"], scores[f"tau90_days_{name}"] = _horizon_brier(
            evaluator, test_times[:, k], test_events[:, k], train_times[:, k]
        )
        scores[f"mae_margin_{name}"] = float(evaluator.mae(method="Margin"))
        scores[f"dcal_p_{name}"] = float(evaluator.d_calibration()[0])
    names = EVENT_NAMES[:k_events]
    for metric in ("auc", "ibs", "ibs_tau90", "mae_margin"):
        scores[f"{metric}_mean"] = float(
            np.mean([scores[f"{metric}_{e}"] for e in names])
        )
    scores["dcal_calibrated"] = float(sum(scores[f"dcal_p_{e}"] > 0.05 for e in names))
    return scores


def self_check() -> None:
    """Check both C-indexes on a hand-worked example; raise if anything differs.

    Grid 0, 10, 20, 30, 40 days. Each curve equals exactly 0.5 at its
    predicted median, so medians are grid times. Patients (recurrence,
    death): A (5, event; 15, event), predicted (10, 30); B (25, censored;
    25, event), predicted (20, 40); C (12, event; 35, censored), predicted
    (30, 20); D (40, censored; 40, censored), predicted (40, 40).

    Global, recurrence: A beats B, C and D; C loses to B, beats D -> 4 of 5.
    Global, death: A beats B and D, loses to C; B loses to C, ties D (half)
    -> 2.5 of 5. Global C-index = 6.5 / 10 = 0.65.

    Local: A recurrence (5) before death (15), predicted 10 < 30 -> 1 of 1.
    B died at 25 with recurrence censored at 25 (comparable), predicted
    death 40 after recurrence 20 -> 0 of 1. C recurred at 12 before death
    censored at 35, predicted 30 > 20 -> 0 of 1. D has no events (skipped).
    Local C-index = 1 / 3.

    The per-event and per-patient pair counts are also compared with
    SurvivalEVAL's own ``concordance``.

    Raises:
        AssertionError: If any value differs from the hand calculation or
            from SurvivalEVAL.

    Examples:
        >>> self_check()  # doctest: +ELLIPSIS
        Self-check passed: global C-index 0.6500 (6.5/10), local C-index 0.3333 ...
    """
    import pandas as pd
    from SurvivalEVAL import LifelinesEvaluator

    grid = np.array([0.0, 10, 20, 30, 40])
    curve_for = {
        10: [1.0, 0.5, 0.25, 0.1, 0.05],
        20: [1.0, 0.8, 0.5, 0.25, 0.1],
        30: [1.0, 0.9, 0.7, 0.5, 0.2],
        40: [1.0, 0.9, 0.8, 0.7, 0.5],
    }
    medians = np.array([[10, 30], [20, 40], [30, 20], [40, 40]])
    curves = np.array([[curve_for[m] for m in row] for row in medians])
    times = np.array([[5.0, 15], [25, 25], [12, 35], [40, 40]])
    events = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])

    predicted = np.stack(
        [median_survival_times(curves[:, k], grid) for k in range(2)], 1
    )
    assert np.array_equal(predicted, medians), predicted
    assert concordance_pairs(times[:, 0], events[:, 0], predicted[:, 0]) == (4.0, 5)
    assert concordance_pairs(times[:, 1], events[:, 1], predicted[:, 1]) == (2.5, 5)
    g = global_c_index(predicted, times, events)
    assert g == (0.65, 6.5, 10), g
    loc = local_c_index(predicted, times, events)
    assert loc[1:] == (1.0, 3, 3) and math.isclose(loc[0], 1 / 3), loc

    def survival_eval_pairs(curve_rows: np.ndarray, t: np.ndarray, e: np.ndarray):
        evaluator = LifelinesEvaluator(pd.DataFrame(curve_rows.T, index=grid), t, e)
        _, concordant, total = evaluator.concordance()
        return float(concordant), int(total)

    for k in range(2):
        ours = concordance_pairs(times[:, k], events[:, k], predicted[:, k])
        assert ours == survival_eval_pairs(curves[:, k], times[:, k], events[:, k])
    for i in range(3):  # patients with events
        ours = concordance_pairs(times[i], events[i], predicted[i])
        assert ours == survival_eval_pairs(curves[i], times[i], events[i])
    print(
        f"Self-check passed: global C-index {g[0]:.4f} (6.5/10), "
        f"local C-index {loc[0]:.4f} (1/3); pair counts match SurvivalEVAL."
    )


def make_synthetic_cohort(root: Path, n_patients: int = 200, seed: int = 0) -> Path:
    """Write a synthetic ``rotterdam.csv`` with the real file's 15 columns.

    Covariates are drawn to resemble the Rotterdam cohort. Recurrence and
    death times are Weibull, both shortened by a shared risk score (more
    nodes, larger tumors, grade 3); death follows recurrence sooner, and
    follow-up ends at a uniform censoring time. Recurrence is censored at
    death or end of follow-up, and for about 10% of patients who died
    without recurrence, recurrence follow-up ended earlier, as for 43 real
    patients. All four outcome combinations occur. The file passes
    :class:`~pyhealth.datasets.RotterdamDataset` validation.

    Args:
        root: Folder to write ``rotterdam.csv`` into (created if needed).
        n_patients: Number of synthetic patients.
        seed: Seed for the generator.

    Returns:
        Path of the written CSV.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = make_synthetic_cohort(Path(tmp), n_patients=20)
        ...     rows = path.read_text().splitlines()
        >>> len(rows), rows[0].split(",")[:3]
        (21, ['pid', 'year', 'age'])
    """
    rng = np.random.default_rng(seed)
    n = n_patients
    age = np.clip(rng.normal(55, 13, n), 24, 90).round()
    size = rng.choice(ROTTERDAM_SIZE_CATEGORIES, n, p=[0.47, 0.44, 0.09])
    grade = rng.choice([2, 3], n, p=[0.27, 0.73])
    nodes = np.where(rng.random(n) < 0.48, 0, rng.poisson(4, n)).clip(0, 34)
    hormon = (rng.random(n) < 0.11).astype(int)
    chemo = (rng.random(n) < 0.19).astype(int)
    risk = (
        0.08 * nodes
        + 0.3 * (size == "20-50")
        + 0.6 * (size == ">50")
        + 0.3 * (grade == 3)
        - 0.3 * hormon
        + 0.01 * (age - 55)
    )
    recurrence = 4000 * np.exp(-risk) * rng.weibull(1.2, n)
    death_without = 9000 * np.exp(-0.03 * (age - 55) - 0.5 * risk) * rng.weibull(1.5, n)
    death_after = recurrence + 2500 * np.exp(-0.3 * risk) * rng.weibull(1.3, n)
    death = np.where(
        recurrence < death_without,
        np.minimum(death_after, death_without),
        death_without,
    )
    censor = rng.uniform(1000, 7000, n)
    dtime = np.maximum(np.ceil(np.minimum(death, censor)), 1)
    died = (death <= censor).astype(int)
    recurred = (recurrence <= np.minimum(death, censor)).astype(int)
    rtime = np.where(recurred == 1, np.maximum(np.ceil(recurrence), 1), dtime)
    rtime = np.minimum(rtime, dtime)
    early_end = (recurred == 0) & (died == 1) & (rng.random(n) < 0.1)
    rtime = np.where(
        early_end, np.maximum(np.floor(dtime * rng.uniform(0.5, 0.95, n)), 1), rtime
    )
    columns = {
        "pid": np.arange(1, n + 1),
        "year": rng.integers(1978, 1994, n),
        "age": age.astype(int),
        "meno": ((age > 50) ^ (rng.random(n) < 0.1)).astype(int),
        "size": size,
        "grade": grade,
        "nodes": nodes,
        "pgr": np.round(rng.lognormal(3.5, 1.5, n)).astype(int),
        "er": np.round(rng.lognormal(4.0, 1.4, n)).astype(int),
        "hormon": hormon,
        "chemo": chemo,
        "rtime": rtime.astype(int),
        "recur": recurred,
        "dtime": dtime.astype(int),
        "death": died,
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "rotterdam.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(ROTTERDAM_COLUMNS)
        writer.writerows(zip(*(columns[c] for c in ROTTERDAM_COLUMNS)))
    return path


def load_samples(
    data_root: Path, cache_dir: Path | None, use_treatment: bool, num_workers: int
) -> SampleDataset:
    """Load the cohort and apply the multi-event task.

    Args:
        data_root: Folder containing ``rotterdam.csv``.
        cache_dir: PyHealth cache folder (``None`` for PyHealth's default).
        use_treatment: Whether the task keeps ``hormon`` and ``chemo``.
        num_workers: Workers for loading and task processing.

    Returns:
        One sample per patient.

    Examples:
        >>> samples = load_samples(Path("../data"), None, True, 1)  # doctest: +SKIP
        >>> samples[0]["features"].shape  # doctest: +SKIP
        torch.Size([10])
    """
    dataset = RotterdamDataset(
        root=str(data_root), cache_dir=cache_dir, num_workers=num_workers
    )
    task = MultiEventSurvivalRotterdam(use_treatment=use_treatment)
    return dataset.set_task(task, num_workers=num_workers)


def prepare_seed(samples: SampleDataset, seed: int) -> dict:
    """Split one seed and collect everything the methods share.

    Args:
        samples: Output of :func:`load_samples`.
        seed: Split seed.

    Returns:
        Dict with the ``train``/``val``/``test`` subsets, ``patient_ids`` per
        split, the time ``grid``, and training/test arrays (``train_times``,
        ``train_events``, ``test_features``, ``test_times``, ``test_events``).

    Examples:
        >>> prepared = prepare_seed(samples, seed=0)  # doctest: +SKIP
        >>> sorted(prepared["patient_ids"])  # doctest: +SKIP
        ['test', 'train', 'val']
    """
    train, val, test, patient_ids = stratified_split(samples, seed)
    _, train_t, train_e = survival_arrays(train)
    _, test_x, test_t, test_e = dataset_arrays(test)
    return {
        "train": train,
        "val": val,
        "test": test,
        "patient_ids": patient_ids,
        "grid": make_time_grid(train),
        "train_times": train_t,
        "train_events": train_e,
        "test_features": test_x,
        "test_times": test_t,
        "test_events": test_e,
    }


def _quiet_trainer_progress_bars() -> None:
    """Turn off the Trainer's per-epoch progress bars for long runs.

    tqdm reads ``TQDM_DISABLE`` only when it is first imported, which has
    already happened by the time this script parses its options, so the
    Trainer module's two progress-bar functions are wrapped instead. Only
    this script's run is affected; ``--progress`` keeps the bars.
    """
    from functools import partial

    import pyhealth.trainer as trainer_module

    trainer_module.tqdm = partial(trainer_module.tqdm, disable=True)
    trainer_module.trange = partial(trainer_module.trange, disable=True)


def train_mensa(
    samples: SampleDataset, prepared: dict, seed: int, config: dict, device: str
) -> tuple[MENSA, dict[str, float]]:
    """Train MENSA with PyHealth's Trainer on one seed's split.

    Preprocessing is fitted on the training split only. Training uses Adam
    with the config's learning rate and weight decay, gradient clipping,
    ``drop_last`` batches (BatchNorm cannot train on a batch of one), and
    early stopping on the validation loss; the best epoch's weights are
    restored. The model seed equals the split seed.

    Args:
        samples: Full sample dataset (defines the model's input/output sizes).
        prepared: Output of :func:`prepare_seed`.
        seed: Seed for initialization, shuffling and dropout.
        config: MENSA settings, as in :data:`MENSA_CONFIG`.
        device: Torch device, e.g. ``"cpu"``.

    Returns:
        ``(model, info)``: the trained model, and ``epochs_run`` and
        ``best_epoch``.

    Examples:
        >>> config = method_config("mensa", MENSA_CONFIG, max_epochs=2)
        >>> model, info = train_mensa(  # doctest: +SKIP
        ...     samples, prepared, 0, config, "cpu"
        ... )
        >>> sorted(info)  # doctest: +SKIP
        ['best_epoch', 'epochs_run']
    """
    from pyhealth.trainer import Trainer

    torch.manual_seed(seed)
    model = MENSA(
        samples,
        hidden_dims=tuple(config["hidden_dims"]),
        n_dists=int(config["n_dists"]),
        dropout=float(config["dropout"]),
        trajectory_weight=float(config["trajectory_weight"]),
        trajectory_variant=str(config["trajectory_variant"]),
    )
    model.fit_preprocessing(prepared["train"])
    train_loader = get_dataloader(
        prepared["train"],
        batch_size=int(config["batch_size"]),
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    val_loader = get_dataloader(prepared["val"], batch_size=1024)
    trainer = Trainer(model=model, device=device, enable_logging=False)
    history: list[float] = []
    evaluate = trainer.evaluate

    def evaluate_and_record(dataloader):
        scores = evaluate(dataloader)
        if dataloader is val_loader:
            history.append(scores["loss"])
        return scores

    trainer.evaluate = evaluate_and_record
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=int(config["max_epochs"]),
        optimizer_params={"lr": float(config["lr"])},
        weight_decay=float(config["weight_decay"]),
        max_grad_norm=float(config["max_grad_norm"]),
        monitor="loss",
        monitor_criterion="min",
        patience=int(config["patience"]),
    )
    trainer.evaluate = evaluate
    model.eval()
    finite = np.isfinite(np.asarray(history, dtype=float))
    # 0 when no epoch had a finite validation loss (training diverged).
    best = (
        int(np.nanargmin(np.where(finite, history, np.nan))) + 1 if finite.any() else 0
    )
    return model, {"epochs_run": float(len(history)), "best_epoch": float(best)}


def method_config(method: str, mensa_config: dict, max_epochs: int | None) -> dict:
    """The settings that define a run, stored in its JSON file.

    Args:
        method: ``"mensa"``, ``"coxph"`` or ``"deepsurv"``.
        mensa_config: MENSA settings.
        max_epochs: Optional epoch cap for MENSA and DeepSurv.

    Returns:
        A JSON-serializable dict; runs are reused only if it matches.

    Examples:
        >>> method_config("coxph", MENSA_CONFIG, None)
        {'alpha': 10.0, 'n_iter': 100, 'tol': 1e-05}
        >>> method_config("mensa", MENSA_CONFIG, max_epochs=5)["max_epochs"]
        5
    """
    if method == "mensa":
        config = dict(mensa_config)
        if max_epochs is not None:
            config["max_epochs"] = int(max_epochs)
        return config
    if method == "coxph":
        return dict(COXPH_CONFIG)
    config = dict(DEEPSURV_CONFIG)
    if max_epochs is not None:
        config["num_epochs"] = int(max_epochs)
    return config


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, to record exactly which data a run used.

    Args:
        path: File to hash.

    Returns:
        Hex digest.

    Examples:
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder) / "data.csv"
        ...     _ = path.write_bytes(b"abc")
        ...     digest = file_sha256(path)
        >>> digest[:16]
        'ba7816bf8f01cfea'
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_once(
    method: str,
    seed: int,
    samples: SampleDataset,
    prepared: dict,
    config: dict,
    data_info: dict,
    run_path: Path,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict:
    """Run one method on one seed, evaluate it, and save one JSON file.

    If ``run_path`` already holds a run with the same method, seed, settings,
    data and example version, it is reused instead of rerun, so an
    interrupted experiment resumes where it stopped.

    Args:
        method: ``"mensa"``, ``"coxph"`` or ``"deepsurv"``.
        seed: Split and model seed.
        samples: Full sample dataset.
        prepared: Output of :func:`prepare_seed` for this seed.
        config: Output of :func:`method_config`.
        data_info: ``{"path", "sha256", "n_patients", "synthetic"}``.
        run_path: JSON file to write.
        device: Torch device for MENSA.
        overwrite: Rerun even if a matching run file exists.

    Returns:
        The run record: version information, method, seed, settings, data,
        split sizes and patient IDs, run details, seconds and metrics. A
        reused record also has ``"reused": True`` (not saved to the file).

    Examples:
        >>> config = method_config("coxph", MENSA_CONFIG, None)
        >>> record = run_once(  # doctest: +SKIP
        ...     "coxph", 0, samples, prepared, config, data_info, Path("run.json")
        ... )
        >>> sorted(record["split_sizes"])  # doctest: +SKIP
        ['test', 'train', 'val']
    """
    identity = {
        "example_version": EXAMPLE_VERSION,
        "method": method,
        "seed": seed,
        "config": config,
        "data_sha256": data_info["sha256"],
    }
    if run_path.is_file() and not overwrite:
        previous = json.loads(run_path.read_text())
        if {k: previous.get(k) for k in identity} == identity:
            return {**previous, "reused": True}
    start = time.perf_counter()
    grid = prepared["grid"]
    if method == "mensa":
        model, info = train_mensa(samples, prepared, seed, config, device)
        curves = model.predict_survival(
            torch.tensor(prepared["test_features"], dtype=torch.float32), grid
        ).cpu()
    else:
        epochs = config.get("num_epochs") if method == "deepsurv" else None
        curves, info = baseline_curves(
            method,
            prepared["train"],
            prepared["val"],
            prepared["test"],
            grid,
            seed,
            num_epochs=int(epochs) if epochs else None,
        )
    curves = np.asarray(curves, dtype=np.float64)
    if not (
        ((curves >= 0) & (curves <= 1)).all() and (np.diff(curves, axis=-1) <= 0).all()
    ):
        n_nan = int(np.isnan(curves).sum())
        n_out = int(((curves < 0) | (curves > 1)).sum())
        n_up = int((np.diff(curves, axis=-1) > 0).sum())
        raise ValueError(
            f"{method} seed {seed}: invalid survival curves ({n_nan} NaN, "
            f"{n_out} outside [0, 1], {n_up} increases; run details {info})"
        )
    metrics = evaluate_survival(
        curves,
        grid,
        prepared["test_times"],
        prepared["test_events"],
        prepared["train_times"],
        prepared["train_events"],
    )
    record = {
        **identity,
        "pyhealth_version": importlib.metadata.version("pyhealth"),
        "data": data_info,
        "split_sizes": {k: len(v) for k, v in prepared["patient_ids"].items()},
        "split_patient_ids": prepared["patient_ids"],
        "grid_points": len(grid),
        "info": info,
        "seconds": round(time.perf_counter() - start, 2),
        "metrics": metrics,
    }
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(record, indent=1, allow_nan=True))
    return record


def summarize_runs(records: Sequence[dict]) -> list[dict]:
    """Mean and standard deviation over seeds for each method and metric.

    Args:
        records: Run records from :func:`run_once`.

    Returns:
        One row per (method, metric): ``method``, ``metric``, ``n`` (seeds),
        ``mean``, ``std`` (sample standard deviation; ``nan`` for one seed)
        and ``mean_std``, a readable ``"mean ± std"``. Rows follow
        :data:`METHODS` order, then the metrics' order in the run files.

    Examples:
        >>> rows = summarize_runs([
        ...     {"method": "coxph", "metrics": {"global_ci": 0.68}},
        ...     {"method": "coxph", "metrics": {"global_ci": 0.70}},
        ... ])
        >>> rows[0]["mean_std"]
        '0.6900 ± 0.0141'
    """
    rows = []
    methods = [m for m in METHODS if any(r["method"] == m for r in records)]
    for method in methods:
        runs = [r for r in records if r["method"] == method]
        for metric in runs[0]["metrics"]:
            values = np.array([r["metrics"][metric] for r in runs], dtype=float)
            values = values[~np.isnan(values)]
            mean = float(values.mean()) if values.size else float("nan")
            std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            digits = 1 if abs(mean) >= 100 else 4
            text = f"{mean:.{digits}f}" + (
                f" ± {std:.{digits}f}" if values.size > 1 else ""
            )
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": mean,
                    "std": std,
                    "mean_std": text,
                }
            )
    return rows


def write_summary_csv(rows: Sequence[dict], path: Path) -> None:
    """Write :func:`summarize_runs` rows to a CSV file.

    Args:
        rows: Summary rows.
        path: Output CSV path (parent folders are created).

    Examples:
        >>> rows = summarize_runs([
        ...     {"method": "coxph", "metrics": {"global_ci": 0.68}},
        ...     {"method": "coxph", "metrics": {"global_ci": 0.70}},
        ... ])
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder) / "summary.csv"
        ...     write_summary_csv(rows, path)
        ...     header = path.read_text().splitlines()[0]
        >>> header
        'method,metric,n,mean,std,mean_std'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["method", "metric", "n", "mean", "std", "mean_std"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict], records: Sequence[dict]) -> None:
    """Print the headline metrics, as the paper reports them (x100).

    Under each method, a ``paper`` row gives the published Rotterdam values
    (:data:`PAPER_ROTTERDAM`). The paper does not state mMAE's unit; our MAE
    is in days, and dividing it by 100 puts it on the paper's scale, so the
    ``MAE (100 d)*`` column shows our MAE / 100 next to the paper's mMAE
    under that assumed unit. ``IBS tau90`` (see :func:`_horizon_brier`)
    sits next to the default IBS; the paper has no counterpart, so its
    paper cell is empty.

    Args:
        rows: Output of :func:`summarize_runs`.
        records: The run records, to count D-calibrated events.

    Examples:
        >>> print_summary(summarize_runs(records), records)  # doctest: +SKIP
    """
    shown = [
        ("global_ci", "CI_G", 100),
        ("local_ci", "CI_L", 100),
        ("auc_mean", "AUC", 100),
        ("ibs_mean", "IBS", 100),
        ("ibs_tau90_mean", "IBS tau90", 100),
        ("mae_margin_mean", "MAE (days)", 1),
        ("mae_margin_mean", "MAE (100 d)*", 0.01),
    ]
    lookup = {(r["method"], r["metric"]): r for r in rows}
    print(
        f"{'method':10s}"
        + "".join(f"{label:>16s}" for _, label, _ in shown)
        + "    D-cal"
    )
    for method in [m for m in METHODS if any(r["method"] == m for r in records)]:
        cells = []
        for metric, _, scale in shown:
            row = lookup[(method, metric)]
            mean, std = row["mean"] * scale, row["std"] * scale
            digits = 0 if scale == 1 else 1
            cell = f"{mean:.{digits}f}" + (
                "" if math.isnan(std) else f" ± {std:.{digits}f}"
            )
            cells.append(f"{cell:>16s}")
        runs = [r for r in records if r["method"] == method]
        calibrated = int(sum(r["metrics"]["dcal_calibrated"] for r in runs))
        print(
            f"{method:10s}"
            + "".join(cells)
            + f"    {calibrated}/{len(runs) * len(EVENT_NAMES)}"
        )
        paper = PAPER_ROTTERDAM.get(method)
        if paper is not None:
            paper_cells = []
            for metric, label, _ in shown:
                value = "" if label == "MAE (days)" else paper.get(metric, "")
                paper_cells.append(f"{value:>16s}")
            print(f"{'  paper':10s}" + "".join(paper_cells) + f"    {paper['dcal']}")
    print(
        "paper = Lillelund et al., PMLR 297 (ML4H 2025), Tables 2-3, Rotterdam, "
        "mean ± SD over 10 experiments.\n"
        "* Assumed unit of 100 days: the paper does not state mMAE's unit; our "
        "MAE (days) / 100 is shown next to its mMAE.\n"
        f"IBS tau90: IBS over times up to the {IBS_HORIZON_QUANTILE:.0%} quantile "
        "of training times, where the censoring weights are stable; not in the "
        "paper."
    )


def run_main_experiment(
    samples: SampleDataset,
    seeds: Sequence[int],
    methods: Sequence[str],
    data_info: dict,
    output_dir: Path,
    max_epochs: int | None,
    device: str,
    overwrite: bool,
) -> list[dict]:
    """E1: every method on every seed, then the summary CSV.

    Writes ``runs/main_<method>_seed<seed>.json`` per run,
    ``main_summary.csv`` and ``main_constant_order.csv`` (see
    :func:`report_constant_order`) in ``output_dir``, and prints progress,
    the headline table and the CI_L reference.

    Args:
        samples: Output of :func:`load_samples` (ten features).
        seeds: Split and model seeds.
        methods: Methods to run, from :data:`METHODS`.
        data_info: Description of the data file, stored with each run.
        output_dir: Where run files and the summary go.
        max_epochs: Optional epoch cap for MENSA and DeepSurv.
        device: Torch device for MENSA.
        overwrite: Rerun runs that already have matching files.

    Returns:
        The run records.

    Examples:
        >>> records = run_main_experiment(  # doctest: +SKIP
        ...     samples, range(10), METHODS, data_info, Path("../runs/e1"),
        ...     max_epochs=None, device="cpu", overwrite=False,
        ... )
    """
    records = []
    reference = {}
    for seed in seeds:
        prepared = prepare_seed(samples, seed)
        reference[seed] = constant_order_local_ci(
            prepared["test_times"], prepared["test_events"]
        )
        for method in methods:
            config = method_config(method, MENSA_CONFIG, max_epochs)
            run_path = output_dir / "runs" / f"main_{method}_seed{seed}.json"
            record = run_once(
                method,
                seed,
                samples,
                prepared,
                config,
                data_info,
                run_path,
                device,
                overwrite,
            )
            records.append(record)
            m = record["metrics"]
            took = "reused" if record.get("reused") else f"{record['seconds']:.0f}s"
            print(
                f"seed {seed} {method:8s} CI_G {100 * m['global_ci']:.1f}  "
                f"CI_L {100 * m['local_ci']:.1f}  IBS {100 * m['ibs_mean']:.1f}  "
                f"({took})",
                flush=True,
            )
    rows = summarize_runs(records)
    summary_path = output_dir / "main_summary.csv"
    write_summary_csv(rows, summary_path)
    print(f"\nE1 summary over seeds {list(seeds)} (CSV: {summary_path}):")
    print_summary(rows, records)
    report_constant_order(
        reference,
        [
            (METHOD_LABELS[m], [r for r in records if r["method"] == m])
            for m in METHODS
            if m in methods
        ],
        output_dir / "main_constant_order.csv",
    )
    return records


def nested_training_subsets(
    train_ids: Sequence[str],
    train_events: np.ndarray,
    seed: int,
    fractions: Sequence[float] = LEARNING_CURVE_FRACTIONS,
) -> dict[float, list[str]]:
    """Nested training subsets, stratified by recurrence/death outcome.

    Within each of the four outcome combinations, the training patients are
    put in one random order (seeded); each fraction takes the first
    ``round(fraction * n)`` of them (at least one if the group is
    nonempty). Every smaller subset is therefore contained in every larger
    one (25% ⊂ 50% ⊂ 100%), and each keeps the training split's outcome mix.

    Args:
        train_ids: Training patient IDs, in the order of ``train_events``.
        train_events: ``[N, K]`` 0/1 indicators of the training patients.
        seed: Seed for the order within each group.
        fractions: Fractions in ``(0, 1]``.

    Returns:
        ``{fraction: sorted patient IDs}``.

    Raises:
        ValueError: If a fraction is outside ``(0, 1]``.

    Examples:
        >>> ids = [f"p{i}" for i in range(8)]
        >>> events = np.array([[1, 1], [0, 0]] * 4)
        >>> subsets = nested_training_subsets(ids, events, seed=0)
        >>> [len(subsets[f]) for f in (0.25, 0.5, 1.0)]
        [2, 4, 8]
        >>> set(subsets[0.25]) <= set(subsets[0.5]) <= set(subsets[1.0])
        True
    """
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("fractions must be in (0, 1]")
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, ...], list[str]] = {}
    for pid, row in zip(train_ids, train_events.astype(int)):
        groups.setdefault(tuple(row), []).append(pid)
    subsets: dict[float, list[str]] = {f: [] for f in fractions}
    for key in sorted(groups):
        ordered = [groups[key][i] for i in rng.permutation(len(groups[key]))]
        for fraction in fractions:
            count = max(1, round(fraction * len(ordered)))
            subsets[fraction].extend(ordered[:count])
    return {f: sorted(ids, key=str) for f, ids in subsets.items()}


def with_training_subset(
    samples: SampleDataset, prepared: dict, train_ids: Sequence[str]
) -> dict:
    """A copy of one seed's prepared data with a smaller training set.

    Only the training subset (and its recorded patient IDs) changes. The
    validation and test sets, the time grid and the full training split's
    times and indicators (used for censoring weights in
    :func:`evaluate_survival`) stay the same, so every fraction is scored
    identically. Methods fit their preprocessing on the subset: MENSA calls
    ``fit_preprocessing`` on it and the baselines standardize with it.

    Args:
        samples: Full sample dataset.
        prepared: Output of :func:`prepare_seed`.
        train_ids: Patient IDs of the training subset.

    Returns:
        The modified copy.

    Examples:
        >>> subsets = nested_training_subsets(  # doctest: +SKIP
        ...     prepared["patient_ids"]["train"], prepared["train_events"], seed=0
        ... )
        >>> part = with_training_subset(  # doctest: +SKIP
        ...     samples, prepared, subsets[0.25]
        ... )
        >>> len(part["patient_ids"]["train"]) == len(subsets[0.25])  # doctest: +SKIP
        True
    """
    indices = [i for pid in train_ids for i in samples.patient_to_index[pid]]
    return {
        **prepared,
        "train": samples.subset(indices),
        "patient_ids": {**prepared["patient_ids"], "train": list(train_ids)},
    }


def _fraction_label(fraction: float) -> str:
    """Return ``"25"`` for 0.25, for file names and tables."""
    return f"{round(fraction * 100):d}"


def _x100(
    mean: float, std: float = float("nan"), digits: int = 1, sign: bool = False
) -> str:
    """Format a 0-1 score or difference on the x100 scale as ``"mean ± std"``.

    Args:
        mean: Mean on the 0-1 scale.
        std: Standard deviation on the 0-1 scale; ``nan`` omits it.
        digits: Decimal places.
        sign: Always show the sign of the mean (for differences).

    Returns:
        The formatted text.

    Examples:
        >>> _x100(0.90972, 0.01311, digits=2)
        '90.97 ± 1.31'
        >>> _x100(-0.00093, digits=2, sign=True)
        '-0.09'
    """
    text = format(100 * mean, f"{'+' if sign else ''}.{digits}f")
    if not math.isnan(std):
        text += f" ± {100 * std:.{digits}f}"
    return text


def paired_differences(
    records: Sequence[dict],
    metrics: Sequence[str],
    fractions: Sequence[float] = LEARNING_CURVE_FRACTIONS,
) -> list[dict]:
    """MENSA minus each baseline, per seed, at each training fraction.

    For each fraction, baseline and metric: the per-seed difference
    ``MENSA - baseline`` (only seeds where both ran), its mean and sample
    standard deviation, and how many seeds have a positive difference. For
    fractions below 1 it also reports how that difference changed from the
    full training set, ``diff(fraction) - diff(100%)``, per seed. For the
    C-indexes a positive difference favors MENSA; for IBS and IBS_tau90
    (lower is better) a negative one does. ``n_favor_mensa`` and
    ``n_change_favor_mensa`` count seeds in MENSA's favor with that direction
    already applied: MENSA better than the baseline, and MENSA's advantage
    larger (or its deficit smaller) than with the full training set.

    Args:
        records: Run records carrying ``train_fraction``, ``method``,
            ``seed`` and ``metrics``.
        metrics: Metric keys to compare.
        fractions: Training fractions.

    Returns:
        One row per (fraction, comparison, metric). Differences are x100
        points: ``mean_x100``, ``std_x100``, ``mean_std_x100`` and
        ``change_vs_full_mean_std_x100``.

    Examples:
        >>> def run(method, seed, local_ci):
        ...     return {"train_fraction": 1.0, "method": method, "seed": seed,
        ...             "metrics": {"local_ci": local_ci}}
        >>> records = [run("mensa", 0, 0.91), run("coxph", 0, 0.84),
        ...            run("mensa", 1, 0.90), run("coxph", 1, 0.83)]
        >>> row = paired_differences(records, ["local_ci"], fractions=(1.0,))[1]
        >>> row["comparison"], row["mean_std_x100"], row["n_favor_mensa"]
        ('mensa-coxph', '+7.00 ± 0.00', 2)
    """

    def text(values: np.ndarray) -> str:
        if values.size == 0:
            return ""
        spread = values.std(ddof=1) if values.size > 1 else float("nan")
        return _x100(values.mean(), spread, 2, sign=True)

    by_key = {
        (r["train_fraction"], r["method"], r["seed"]): r["metrics"] for r in records
    }
    higher_better = {m: better for m, _, _, better in LEARNING_CURVE_METRICS}
    rows = []
    for fraction in fractions:
        for baseline in ("deepsurv", "coxph"):
            for metric in metrics:
                seeds = sorted(
                    seed
                    for (f, m, seed) in by_key
                    if f == fraction
                    and m == "mensa"
                    and (fraction, baseline, seed) in by_key
                )

                def diff(f: float, seed: int, base: str = baseline, key: str = metric):
                    return (
                        by_key[(f, "mensa", seed)][key] - by_key[(f, base, seed)][key]
                    )

                d = np.array([diff(fraction, s) for s in seeds], dtype=float)
                sign = 1.0 if higher_better.get(metric, True) else -1.0
                row = {
                    "fraction": fraction,
                    "comparison": f"mensa-{baseline}",
                    "metric": metric,
                    "n": int(d.size),
                    "mean_x100": 100 * float(d.mean()) if d.size else float("nan"),
                    "std_x100": (
                        100 * float(d.std(ddof=1)) if d.size > 1 else float("nan")
                    ),
                    "mean_std_x100": text(d),
                    "n_positive": int((d > 0).sum()),
                    "n_favor_mensa": int((sign * d > 0).sum()),
                    "change_vs_full_mean_std_x100": "",
                    "n_change_positive": "",
                    "n_change_favor_mensa": "",
                }
                both = [
                    s
                    for s in seeds
                    if (1.0, "mensa", s) in by_key and (1.0, baseline, s) in by_key
                ]
                if fraction < 1 and both:
                    change = np.array(
                        [diff(fraction, s) - diff(1.0, s) for s in both], dtype=float
                    )
                    row["change_vs_full_mean_std_x100"] = text(change)
                    row["n_change_positive"] = int((change > 0).sum())
                    row["n_change_favor_mensa"] = int((sign * change > 0).sum())
                rows.append(row)
    return rows


def plot_learning_curves(summary: Sequence[dict], output_dir: Path) -> list[Path]:
    """Save one learning-curve plot per metric (CI_L, CI_G, IBS, IBS_tau90).

    Each plot shows every method's mean over seeds against the mean training
    size, with error bars of one standard deviation. Methods keep fixed
    colors and marker shapes and are labeled directly as well as in the
    legend.

    Args:
        summary: Rows of ``learning_curve_summary.csv`` (see
            :func:`run_learning_curve_experiment`).
        output_dir: Folder for ``learning_curve_<metric>.png``.

    Returns:
        Paths of the saved images.

    Examples:
        >>> paths = plot_learning_curves(summary, Path("../runs/e1"))  # doctest: +SKIP
        >>> paths[0].name  # doctest: +SKIP
        'learning_curve_ci_l.png'
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mensa": "#2a78d6", "coxph": "#eb6834", "deepsurv": "#1baf7a"}
    markers = {"mensa": "o", "coxph": "s", "deepsurv": "^"}
    ink, muted, grid_color, axis_color = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    lookup = {(r["fraction"], r["method"], r["metric"]): r for r in summary}
    fractions = sorted({r["fraction"] for r in summary})
    methods = [m for m in METHODS if any(r["method"] == m for r in summary)]
    sizes = {
        f: np.mean([r["train_size"] for r in summary if r["fraction"] == f])
        for f in fractions
    }
    paths = []
    for metric, short, title, higher_better in LEARNING_CURVE_METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        span = max(sizes.values()) - min(sizes.values()) or 1.0
        ends: list[list] = []  # [y, x, method] of each line's last point
        for j, method in enumerate(methods):
            offset = (j - (len(methods) - 1) / 2) * 0.012 * span  # keep bars apart
            points = [
                (sizes[f] + offset, lookup[(f, method, metric)])
                for f in fractions
                if (f, method, metric) in lookup
            ]
            x = [p[0] for p in points]
            y = [100 * p[1]["mean"] for p in points]
            err = [
                0.0 if math.isnan(p[1]["std"]) else 100 * p[1]["std"] for p in points
            ]
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=colors[method],
                marker=markers[method],
                markersize=6,
                linewidth=2,
                elinewidth=1,
                capsize=3,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1,
                label=METHOD_LABELS[method],
            )
            ends.append([y[-1], x[-1], method])
        # Direct labels at the line ends, nudged apart so they never overlap.
        low, high = ax.get_ylim()
        gap = 0.05 * (high - low)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        # If nudging pushed the top label past the axes, shift the stack down.
        overflow = ends[-1][0] - (high - 0.5 * gap)
        if overflow > 0:
            for end in ends:
                end[0] -= overflow
        for y_end, x_end, method in ends:
            ax.annotate(
                METHOD_LABELS[method],
                (x_end, y_end),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=muted,
                annotation_clip=False,
            )
        ax.set_xticks([sizes[f] for f in fractions])
        ax.set_xticklabels(
            [f"{_fraction_label(f)}%\n(n={sizes[f]:.0f})" for f in fractions]
        )
        ax.set_xlim(min(sizes.values()) - 0.08 * span, max(sizes.values()) + 0.2 * span)
        better = "higher is better" if higher_better else "lower is better"
        ax.set_ylabel(f"{short} x100 ({better})", color=muted, fontsize=9)
        ax.set_xlabel(
            "Training set (fraction of the split, mean size)", color=muted, fontsize=9
        )
        n_seeds = max(
            lookup[(f, m, metric)]["n"]
            for f in fractions
            for m in methods
            if (f, m, metric) in lookup
        )
        ax.set_title(
            f"{title} vs training-set size\n(mean ± SD over {n_seeds} "
            f"seed{'s' if n_seeds != 1 else ''})",
            color=ink,
            fontsize=10,
            loc="left",
        )
        ax.grid(axis="y", color=grid_color, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis_color)
        ax.tick_params(colors=muted, labelsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=muted, loc="best")
        fig.tight_layout()
        path = output_dir / f"learning_curve_{short.lower()}.png"
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


def run_learning_curve_experiment(
    samples: SampleDataset,
    seeds: Sequence[int],
    methods: Sequence[str],
    data_info: dict,
    output_dir: Path,
    max_epochs: int | None,
    device: str,
    overwrite: bool,
) -> list[dict]:
    """E2: every method on nested 25%/50%/100% training subsets of each seed.

    Uses E1's split for each seed. Validation and test sets are unchanged,
    and evaluation uses the full training split's time grid and censoring
    weights, so only the amount of training data varies. Each subset is
    refitted from scratch: MENSA calls ``fit_preprocessing`` on it, and the
    baselines standardize and fit on it. The 100% runs are E1 runs (same
    file names and settings), so they are reused when ``output_dir`` already
    holds E1's run files.

    Writes, in ``output_dir``: ``runs/lc<25|50>_<method>_seed<seed>.json``
    (100%: ``runs/main_<method>_seed<seed>.json``),
    ``learning_curve_summary.csv`` (fraction, mean training size, method,
    metric, n, mean, std, mean ± std), ``learning_curve_paired.csv`` (see
    :func:`paired_differences`),
    ``learning_curve_<ci_l|ci_g|ibs|ibs_tau90>.png`` and
    ``learning_curve_constant_order.csv`` (see :func:`report_constant_order`;
    the test sets do not change with the training fraction).

    Args:
        samples: Output of :func:`load_samples` (ten features).
        seeds: Split and model seeds.
        methods: Methods to run, from :data:`METHODS`.
        data_info: Description of the data file, stored with each run.
        output_dir: Where run files, summaries and plots go.
        max_epochs: Optional epoch cap for MENSA and DeepSurv.
        device: Torch device for MENSA.
        overwrite: Rerun runs that already have matching files.

    Returns:
        The run records, each with ``train_fraction``.

    Examples:
        >>> records = run_learning_curve_experiment(  # doctest: +SKIP
        ...     samples, range(5), METHODS, data_info, Path("../runs/e1"),
        ...     max_epochs=None, device="cpu", overwrite=False,
        ... )
    """
    records = []
    reference = {}
    for seed in seeds:
        prepared = prepare_seed(samples, seed)
        reference[seed] = constant_order_local_ci(
            prepared["test_times"], prepared["test_events"]
        )
        train_ids = prepared["patient_ids"]["train"]
        subsets = nested_training_subsets(train_ids, prepared["train_events"], seed)
        fractions = sorted(subsets)
        for small, large in itertools.pairwise(fractions):
            assert set(subsets[small]) <= set(subsets[large]), "subsets not nested"
        assert subsets[1.0] == list(train_ids)
        sizes = ", ".join(f"{_fraction_label(f)}%={len(subsets[f])}" for f in fractions)
        print(
            f"seed {seed}: training subsets {sizes} (nested); val "
            f"{len(prepared['patient_ids']['val'])}, test "
            f"{len(prepared['patient_ids']['test'])} unchanged",
            flush=True,
        )
        for fraction in fractions:
            if fraction == 1.0:
                part, prefix = prepared, "main"
            else:
                part = with_training_subset(samples, prepared, subsets[fraction])
                prefix = f"lc{_fraction_label(fraction)}"
            for method in methods:
                config = method_config(method, MENSA_CONFIG, max_epochs)
                if fraction < 1.0:
                    config = {**config, "train_fraction": fraction}
                run_path = output_dir / "runs" / f"{prefix}_{method}_seed{seed}.json"
                record = run_once(
                    method,
                    seed,
                    samples,
                    part,
                    config,
                    data_info,
                    run_path,
                    device,
                    overwrite,
                )
                record = {**record, "train_fraction": fraction}
                records.append(record)
                m = record["metrics"]
                took = "reused" if record.get("reused") else f"{record['seconds']:.0f}s"
                print(
                    f"  {_fraction_label(fraction):>3s}% {method:8s} "
                    f"CI_G {100 * m['global_ci']:.1f}  CI_L {100 * m['local_ci']:.1f}  "
                    f"IBS {100 * m['ibs_mean']:.1f}  ({took})",
                    flush=True,
                )

    summary = []
    for fraction in sorted({r["train_fraction"] for r in records}):
        part = [r for r in records if r["train_fraction"] == fraction]
        size = float(np.mean([r["split_sizes"]["train"] for r in part]))
        for row in summarize_runs(part):
            summary.append({"fraction": fraction, "train_size": size, **row})
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "learning_curve_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metric_keys = [m for m, _, _, _ in LEARNING_CURVE_METRICS]
    paired = paired_differences(records, metric_keys) if "mensa" in methods else []
    paired_path = output_dir / "learning_curve_paired.csv"
    if paired:
        with paired_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    plots = plot_learning_curves(summary, output_dir)

    names = {m: short for m, short, _, _ in LEARNING_CURVE_METRICS}
    lookup = {(r["fraction"], r["method"], r["metric"]): r for r in summary}
    print(
        f"\nE2 learning curves over seeds {list(seeds)}, x100 (CSV, raw 0-1: "
        f"{summary_path}):"
    )
    print(
        f"{'train':>12s} {'method':9s}" + "".join(f"{n:>18s}" for n in names.values())
    )
    for fraction in sorted({r["fraction"] for r in summary}):
        for method in [m for m in METHODS if m in methods]:
            if (fraction, method, metric_keys[0]) not in lookup:
                continue
            first = lookup[(fraction, method, metric_keys[0])]
            label = f"{_fraction_label(fraction)}% n={first['train_size']:.0f}"
            cells = "".join(
                f"{_x100(r['mean'], r['std']):>18s}"
                for r in (lookup[(fraction, method, k)] for k in metric_keys)
            )
            print(f"{label:>12s} {method:9s}{cells}")
    if paired:
        print(
            "\nPaired MENSA - baseline per seed, x100 points (CSV: "
            f"{paired_path}): mean ± SD, and in how many seeds "
            "MENSA is better; then the change vs 100%, and in how many seeds "
            "MENSA's advantage grew. Positive favors MENSA for CI_L/CI_G, "
            "negative for IBS/IBS_tau90; the counts already account for this."
        )
        for row in paired:
            change = (
                f"   change {row['change_vs_full_mean_std_x100']} "
                f"(advantage grew {row['n_change_favor_mensa']}/{row['n']})"
                if row["change_vs_full_mean_std_x100"]
                else ""
            )
            print(
                f"{_fraction_label(row['fraction']):>4s}% {row['comparison']:15s} "
                f"{names[row['metric']]:9s} {row['mean_std_x100']:>14s} "
                f"(MENSA better {row['n_favor_mensa']}/{row['n']}){change}"
            )
    report_constant_order(
        reference,
        [
            (
                f"{_fraction_label(f)}% {METHOD_LABELS[m]}",
                [r for r in records if r["train_fraction"] == f and r["method"] == m],
            )
            for f in sorted({r["train_fraction"] for r in records})
            for m in METHODS
            if m in methods
        ],
        output_dir / "learning_curve_constant_order.csv",
    )
    print("Plots: " + ", ".join(str(p) for p in plots))
    return records


def variant_paired_differences(
    records: Sequence[dict],
    comparisons: Sequence[tuple[str, str]],
    labels: dict[str, str],
    notes: dict[str, str],
) -> list[dict]:
    """Paired per-seed differences between variants, in E2's format.

    For each pair ``(a, b)`` and each of :data:`LEARNING_CURVE_METRICS`, the
    difference ``a - b`` on every seed both have, its mean ± standard
    deviation, and in how many seeds ``a`` is better (higher C-index, lower
    IBS and IBS_tau90; ``n_favor_a`` already applies that direction).

    Args:
        records: Run records, each with a ``variant`` name.
        comparisons: ``(a, b)`` variant-name pairs.
        labels: Display label per variant name.
        notes: Note per variant name (e.g. already in the paper).

    Returns:
        One row per comparison and metric, with ``comparison``, ``metric``,
        ``n``, ``mean_x100``, ``std_x100``, ``mean_std_x100`` (x100
        points), ``n_positive``, ``n_favor_a`` and ``note``.

    Examples:
        >>> rows = variant_paired_differences(  # doctest: +SKIP
        ...     records, ABLATION_COMPARISONS, labels, notes
        ... )
    """
    by_key = {(r["variant"], r["seed"]): r["metrics"] for r in records}
    rows = []
    for a, b in comparisons:
        seeds = sorted(s for (v, s) in by_key if v == a and (b, s) in by_key)
        if not seeds:
            continue
        for metric, _, _, higher_better in LEARNING_CURVE_METRICS:
            d = np.array(
                [by_key[(a, s)][metric] - by_key[(b, s)][metric] for s in seeds],
                dtype=float,
            )
            sign = 1.0 if higher_better else -1.0
            spread = float(d.std(ddof=1)) if d.size > 1 else float("nan")
            text = _x100(float(d.mean()), spread, 2, sign=True)
            rows.append(
                {
                    "comparison": f"{a} - {b}",
                    "a": labels[a],
                    "b": labels[b],
                    "metric": metric,
                    "n": int(d.size),
                    "mean_x100": 100 * float(d.mean()),
                    "std_x100": 100 * spread,
                    "mean_std_x100": text,
                    "n_positive": int((d > 0).sum()),
                    "n_favor_a": int((sign * d > 0).sum()),
                    "note": notes.get(a, ""),
                }
            )
    return rows


def run_variant_experiment(
    name: str,
    variants: Sequence[dict],
    comparisons: Sequence[tuple[str, str]],
    samples_by_treatment: dict[bool, SampleDataset],
    seeds: Sequence[int],
    data_info: dict,
    output_dir: Path,
    max_epochs: int | None,
    device: str,
    overwrite: bool,
) -> list[dict]:
    """Run E3 (``"ablation"``) or E4 (``"model_ablation"``) and summarize.

    Every variant uses the full training set and E1's split for each seed.
    A variant is a method plus changes to its E1 configuration
    (:data:`ABLATION_VARIANTS`, :data:`MODEL_ABLATION_VARIANTS`). Variants
    without changes are E1's own runs and reuse its run files when
    ``output_dir`` is E1's. 8-feature variants (``use_treatment=False``) use
    their own sample set; their split is checked to be identical to the
    10-feature split. MENSA refits its preprocessing and the baselines their
    standardization on every run, as in E1.

    Writes ``<name>_summary.csv`` (mean ± SD per variant and metric),
    ``<name>_paired.csv`` (:func:`variant_paired_differences`) and
    ``<name>_constant_order.csv`` (:func:`report_constant_order`), and prints
    them, in the same format as E2.

    Args:
        name: ``"ablation"`` or ``"model_ablation"``; also the file prefix.
        variants: Variant specifications.
        comparisons: ``(a, b)`` pairs for the paired differences.
        samples_by_treatment: Sample sets keyed by ``use_treatment``.
        seeds: Split seeds.
        data_info: Data path, checksum and size, saved with each run.
        output_dir: Where run files and summaries go.
        max_epochs: Optional cap on MENSA and DeepSurv epochs.
        device: Torch device for MENSA.
        overwrite: Rerun runs that already exist.

    Returns:
        The run records, each with ``variant``.

    Examples:
        >>> run_variant_experiment(  # doctest: +SKIP
        ...     "ablation", ABLATION_VARIANTS, ABLATION_COMPARISONS,
        ...     {True: samples, False: samples_8}, [0, 1, 2, 3, 4], data_info,
        ...     Path("../runs/e1"), None, "cpu", False,
        ... )
    """
    tag = {"ablation": "e3", "model_ablation": "e4"}[name]
    labels = {v["name"]: v["label"] for v in variants}
    notes = {v["name"]: v.get("note", "") for v in variants}
    records = []
    reference = {}
    for seed in seeds:
        prepared = {
            flag: prepare_seed(samples, seed)
            for flag, samples in samples_by_treatment.items()
        }
        if len(prepared) > 1:
            same = prepared[True]["patient_ids"] == prepared[False]["patient_ids"]
            assert same, "8- and 10-feature splits differ"
        any_prepared = next(iter(prepared.values()))
        reference[seed] = constant_order_local_ci(
            any_prepared["test_times"], any_prepared["test_events"]
        )
        print(f"seed {seed}:", flush=True)
        for variant in variants:
            method, overrides = variant["method"], variant["overrides"]
            use_treatment = bool(overrides.get("use_treatment", True))
            config = {**method_config(method, MENSA_CONFIG, max_epochs), **overrides}
            # Unchanged E1 configurations share E1's run files.
            stem = f"main_{method}" if not overrides else f"{tag}_{variant['name']}"
            run_path = output_dir / "runs" / f"{stem}_seed{seed}.json"
            record = run_once(
                method,
                seed,
                samples_by_treatment[use_treatment],
                prepared[use_treatment],
                config,
                data_info,
                run_path,
                device,
                overwrite,
            )
            record = {**record, "variant": variant["name"]}
            records.append(record)
            m = record["metrics"]
            took = "reused" if record.get("reused") else f"{record['seconds']:.0f}s"
            print(
                f"  {variant['label']:42s} CI_G {100 * m['global_ci']:.1f}  "
                f"CI_L {100 * m['local_ci']:.1f}  IBS {100 * m['ibs_mean']:.1f}  "
                f"({took})",
                flush=True,
            )

    summary = []
    for variant in variants:
        part = [r for r in records if r["variant"] == variant["name"]]
        for row in summarize_runs(part):
            summary.append(
                {
                    "variant": variant["name"],
                    "label": variant["label"],
                    "note": notes[variant["name"]],
                    **row,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = variant_paired_differences(records, comparisons, labels, notes)
    paired_path = output_dir / f"{name}_paired.csv"
    if paired:
        with paired_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)

    metric_keys = [m for m, _, _, _ in LEARNING_CURVE_METRICS]
    names = {m: short for m, short, _, _ in LEARNING_CURVE_METRICS}
    lookup = {(r["variant"], r["metric"]): r for r in summary}
    print(
        f"\n{tag.upper()} {name} over seeds {list(seeds)}, x100 (CSV, raw 0-1: "
        f"{summary_path}):"
    )
    print(f"{'variant':44s}" + "".join(f"{n:>18s}" for n in names.values()))
    for variant in variants:
        if (variant["name"], metric_keys[0]) not in lookup:
            continue
        mark = " †" if notes[variant["name"]] else ""
        cells = "".join(
            f"{_x100(r['mean'], r['std']):>18s}"
            for r in (lookup[(variant["name"], k)] for k in metric_keys)
        )
        print(f"{variant['label'] + mark:44s}{cells}")
    if paired:
        print(
            f"\nPaired a - b per seed, x100 points (CSV: {paired_path}): mean ± SD, "
            "and in how "
            "many seeds a is better (higher CI_L/CI_G, lower IBS/IBS_tau90)."
        )
        for row in paired:
            mark = " †" if row["note"] else ""
            print(
                f"{row['comparison'] + mark:44s} {names[row['metric']]:9s} "
                f"{row['mean_std_x100']:>14s}  (a better {row['n_favor_a']}/{row['n']})"
            )
    report_constant_order(
        reference,
        [
            (v["label"], [r for r in records if r["variant"] == v["name"]])
            for v in variants
        ],
        output_dir / f"{name}_constant_order.csv",
    )
    if any(notes.values()):
        print(f"† {IN_PAPER_NOTE}.")
    return records


def _parse_seeds(text: str) -> list[int]:
    """Parse ``"0-9"`` or ``"0,3,5"`` into a list of seeds.

    Args:
        text: A range ``a-b`` (inclusive) or comma-separated integers.

    Returns:
        The seeds, in order.
    """
    if "-" in text:
        start, end = (int(v) for v in text.split("-", 1))
        return list(range(start, end + 1))
    return [int(v) for v in text.split(",")]


def run_split_check(
    samples: SampleDataset, seeds: Sequence[int], baselines: bool
) -> None:
    """Print split sizes, outcome mix and time grid per seed, and check reruns.

    Args:
        samples: Output of :func:`load_samples`.
        seeds: Split seeds to check.
        baselines: Also fit both baselines per seed and print C-indexes.

    Examples:
        >>> run_split_check(samples, seeds=[0], baselines=False)  # doctest: +SKIP
    """
    ids, times, indicators = survival_arrays(samples)
    combos = {
        "both": (1, 1),
        "recur only": (1, 0),
        "death only": (0, 1),
        "neither": (0, 0),
    }

    def mix(rows: np.ndarray) -> str:
        e = indicators[rows]
        return " ".join(
            f"{np.mean((e[:, 0] == a) & (e[:, 1] == b)):.3f}"
            for a, b in combos.values()
        )

    position = {pid: i for i, pid in enumerate(ids)}
    overall = mix(np.arange(len(ids)))
    print(f"{len(ids)} patients; outcome mix ({', '.join(combos)}): {overall}")
    print("seed  train  val  test  identical-rerun  grid(len, first>0, last)  test mix")
    for seed in seeds:
        train, val, test, split = stratified_split(samples, seed)
        rerun = stratified_split_ids(ids, times, indicators, seed)
        grid = make_time_grid(train)
        test_rows = np.array([position[p] for p in split["test"]])
        print(
            f"{seed:4d}  {len(train):5d}  {len(val):3d}  {len(test):4d}  "
            f"{rerun == split!s:>15s}  ({len(grid)}, {grid[1].item():.0f}, "
            f"{grid[-1].item():.0f})  {mix(test_rows)}"
        )
        if baselines:
            _, train_t, train_e = survival_arrays(train)
            _, test_t, test_e = survival_arrays(test)
            for method in ("coxph", "deepsurv"):
                curves, info = baseline_curves(method, train, val, test, grid, seed)
                valid = bool(
                    ((curves >= 0) & (curves <= 1)).all()
                    and (np.diff(curves, axis=-1) <= 0).all()
                )
                scores = evaluate_survival(
                    curves, grid, test_t, test_e, train_t, train_e
                )
                truncated = {
                    f"ibs_truncated_{e}": scores[f"ibs_truncated_{e}"]
                    for e in EVENT_NAMES
                    if scores[f"ibs_truncated_{e}"]
                }
                extra = " ".join(
                    f"{k}={v:.0f}" for k, v in {**info, **truncated}.items()
                )
                print(
                    f"      {method:8s} global C {100 * scores['global_ci']:.1f}  "
                    f"local C {100 * scores['local_ci']:.1f}  curves valid {valid}  "
                    f"{extra}"
                )


def main() -> None:
    """Command line: self-check, synthetic smoke run, E1-E4, or the split check.

    Examples:
        From the shell, e.g.
        ``python examples/rotterdam_multi_event_survival_mensa.py --self-check``:

        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(
        description="MENSA on the Rotterdam cohort: experiments and baselines."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data-root", type=Path, help="folder containing rotterdam.csv"
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="run the whole pipeline once on ~200 synthetic patients "
        "(one seed, two epochs, temporary directory)",
    )
    source.add_argument(
        "--self-check",
        action="store_true",
        help="check the C-index code on a hand-worked example and exit",
    )
    parser.add_argument(
        "--experiment",
        choices=["main", "learning_curve", "ablation", "model_ablation", "split_check"],
        default="main",
        help="main = E1 (all methods, all seeds); learning_curve = E2 "
        "(25/50/100%% training subsets); ablation = E3 (8 vs 10 features x "
        "lambda 0.25 vs 0, CoxPH with 8 features); model_ablation = E4 "
        "(reference trajectory loss, reference config); E2-E4 use seeds 0-4 "
        "by default; split_check = split diagnostics",
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=None, help='e.g. "0-9" or "0,3"'
    )
    parser.add_argument(
        "--methods",
        type=lambda text: text.split(","),
        default=list(METHODS),
        help="comma-separated subset of " + ",".join(METHODS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where run files and summaries go; keep it outside the repository",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="cap on MENSA and DeepSurv epochs (default 1000 and 2000)",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--overwrite", action="store_true", help="rerun runs that already exist"
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="with split_check: also fit both baselines per seed",
    )
    parser.add_argument(
        "--progress", action="store_true", help="show the Trainer's progress bars"
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.synthetic and args.data_root is None:
        parser.error("give --data-root, --synthetic or --self-check")
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    logging.getLogger("pyhealth").setLevel(logging.WARNING)
    if not args.progress:
        _quiet_trainer_progress_bars()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        if args.synthetic:
            data_root = Path(tmp) / "data"
            make_synthetic_cohort(data_root)
            seeds = args.seeds or [0]
            max_epochs = args.max_epochs or 2
            cache_dir = Path(tmp) / "cache"
            output_dir = args.output_dir or Path(tmp) / "runs"
            if args.output_dir is None:
                print(
                    "Synthetic run: outputs go to a temporary folder deleted at exit;"
                )
                print("pass --output-dir to keep the run files and summary CSV.")
        else:
            if args.output_dir is None and args.experiment != "split_check":
                parser.error("--output-dir is required with --data-root")
            data_root, cache_dir = args.data_root, args.cache_dir
            default_seeds = 10 if args.experiment in ("main", "split_check") else 5
            seeds = args.seeds or list(range(default_seeds))
            max_epochs = args.max_epochs
            output_dir = args.output_dir
        csv_path = data_root / "rotterdam.csv"
        if not csv_path.is_file():
            parser.error(f"rotterdam.csv not found in {data_root}")
        start = time.perf_counter()
        samples = load_samples(data_root, cache_dir, True, args.num_workers)
        data_info = {
            "path": "synthetic" if args.synthetic else str(csv_path.resolve()),
            "sha256": file_sha256(csv_path),
            "n_patients": len(samples),
            "synthetic": bool(args.synthetic),
        }
        print(
            f"Loaded {len(samples)} {'synthetic ' if args.synthetic else ''}"
            f"patients in {time.perf_counter() - start:.1f}s"
        )
        if args.experiment == "split_check":
            run_split_check(samples, seeds, args.baselines)
        elif args.experiment in ("ablation", "model_ablation"):
            samples_by_treatment = {True: samples}
            if args.experiment == "ablation":
                samples_by_treatment[False] = load_samples(
                    data_root, cache_dir, False, args.num_workers
                )
            ablation = args.experiment == "ablation"
            run_variant_experiment(
                args.experiment,
                ABLATION_VARIANTS if ablation else MODEL_ABLATION_VARIANTS,
                ABLATION_COMPARISONS if ablation else MODEL_ABLATION_COMPARISONS,
                samples_by_treatment,
                seeds,
                data_info,
                output_dir,
                max_epochs,
                args.device,
                args.overwrite,
            )
        elif args.experiment == "learning_curve":
            run_learning_curve_experiment(
                samples,
                seeds,
                args.methods,
                data_info,
                output_dir,
                max_epochs,
                args.device,
                args.overwrite,
            )
        else:
            run_main_experiment(
                samples,
                seeds,
                args.methods,
                data_info,
                output_dir,
                max_epochs,
                args.device,
                args.overwrite,
            )
        print(f"Finished in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
