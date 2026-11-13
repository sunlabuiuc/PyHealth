"""Synthetic longitudinal liver-lab cohorts for survival models.

Generates small, seeded cohorts in the Mayo PBC ``pbcseq`` layout (one row per
visit; ``id``, ``day``, ``futime``, ``status``, and one column per lab), with
outcomes drawn from a time-varying Cox model whose true coefficients are known.
Use them to test survival models, tasks, and metrics without real patient data,
and to run the MAS example end to end.

Every cohort is **tie-free**: no two patients share an outcome day, and no
outcome day equals any visit day. That keeps comparisons against reference
implementations independent of the tie-handling method (Breslow vs. Efron).

The outcome process: each patient has a latent severity per lab that drifts
over time. Labs are log-normal in that severity, and the hazard on the interval
after visit ``k`` is ``baseline_hazard * exp(sum_f |beta_f| * severity_f(k))``.
Higher severity means a worse lab, so a lab's true effect on its log scale has
the sign in :data:`FEATURE_PROFILES`. Patients are censored at a uniform time,
and a fraction are transplanted first (status 1), as in ``pbcseq``.
"""

import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..processors import TensorProcessor
from ..tasks.liver_survival import (
    DAYS_PER_YEAR,
    STATUS_CENSORED,
    STATUS_DEATH,
    STATUS_TRANSPLANT,
    counting_process_samples_from_frame,
)
from .sample_dataset import SampleDataset, create_sample_dataset

MAS4_FEATURES: tuple[str, ...] = ("bili", "albumin", "ast", "protime")
"""The four MAS labs available in Mayo PBC (prothrombin time stands in for INR)."""

MAS6_FEATURES: tuple[str, ...] = (
    "bili",
    "albumin",
    "ast",
    "inr",
    "creatinine",
    "alt",
)
"""The six labs of the published MAS score."""

FEATURE_PROFILES: dict[str, tuple[float, float, float]] = {
    # name: (median value, log-scale spread, true log-hazard coefficient)
    "bili": (1.4, 0.9, 0.8),
    "albumin": (3.5, 0.12, -0.5),
    "ast": (110.0, 0.5, 0.3),
    "protime": (10.7, 0.08, 0.4),
    "inr": (1.1, 0.15, 0.4),
    "creatinine": (1.0, 0.3, 0.3),
    "alt": (60.0, 0.6, 0.1),
}
"""Lab profiles. Labs not listed here get ``(1.0, 0.3, 0.0)``."""

_DEFAULT_PROFILE = (1.0, 0.3, 0.0)

FEATURE_PROCESSOR_KWARGS = {"normalization": "zscore", "impute_strategy": "zero"}
"""``TensorProcessor`` options for lab features: z-score, then missing -> mean."""


def _visit_schedule(rng: np.random.Generator, max_years: float) -> np.ndarray:
    """Draws visit days: day 0, about 6 months, 1 year, then yearly."""
    targets = [0.0, 182.0, 365.0]
    targets += [365.0 * k for k in range(2, int(max_years) + 1)]
    days = [0]
    for target in targets[1:]:
        jitter = rng.integers(-20, 21)
        days.append(int(target + jitter))
    return np.array(days, dtype=np.int64)


def generate_synthetic_pbcseq(
    n_patients: int = 20,
    feature_names: Sequence[str] = MAS4_FEATURES,
    seed: int = 0,
    max_years: float = 10.0,
    baseline_hazard: float = 0.08,
    transplant_fraction: float = 0.1,
    missing_rate: float = 0.0,
) -> pd.DataFrame:
    """Generates a seeded, tie-free synthetic visit table.

    Args:
        n_patients: Number of patients (at least 1).
        feature_names: Lab columns to generate; any number of features.
        seed: Seed for the random generator. The same arguments give the same
            table.
        max_years: Longest follow-up, in years. Outcome days may be shifted a
            few days later to keep every outcome day unique.
        baseline_hazard: Event rate per year at average severity.
        transplant_fraction: Probability that a patient is scheduled for a
            transplant, which ends follow-up with status 1 if it comes first.
        missing_rate: Probability that each follow-up lab value is missing
            (NaN). Day-0 labs are always observed.

    Returns:
        A DataFrame with columns ``id``, ``day``, ``futime``, ``status``, and
        one column per feature, sorted by ``id`` and ``day``. Every visit is
        before its patient's ``futime``.

    Raises:
        ValueError: If an argument is out of range.

    Examples:
        >>> df = generate_synthetic_pbcseq(n_patients=3, seed=1)
        >>> list(df.columns)
        ['id', 'day', 'futime', 'status', 'bili', 'albumin', 'ast', 'protime']
    """
    if n_patients < 1:
        raise ValueError(f"n_patients must be >= 1, got {n_patients}.")
    if not feature_names:
        raise ValueError("feature_names must not be empty.")
    if max_years < 1:
        raise ValueError(f"max_years must be >= 1, got {max_years}.")
    if baseline_hazard <= 0:
        raise ValueError(f"baseline_hazard must be positive, got {baseline_hazard}.")
    if not 0 <= transplant_fraction <= 1 or not 0 <= missing_rate < 1:
        raise ValueError(
            "transplant_fraction must be in [0, 1], missing_rate in [0, 1)."
        )

    rng = np.random.default_rng(seed)
    profiles = np.array(
        [FEATURE_PROFILES.get(name, _DEFAULT_PROFILE) for name in feature_names]
    )
    medians, spreads, betas = profiles[:, 0], profiles[:, 1], profiles[:, 2]
    max_day = int(max_years * DAYS_PER_YEAR)

    schedules = [_visit_schedule(rng, max_years) for _ in range(n_patients)]
    taken_days = {int(day) for schedule in schedules for day in schedule}
    rows = []
    for index, schedule in enumerate(schedules):
        severity0 = rng.normal(0.0, 1.0, size=len(feature_names))
        drift = rng.normal(0.1, 0.2, size=len(feature_names))
        years = schedule / DAYS_PER_YEAR
        severity = severity0[None, :] + drift[None, :] * years[:, None]

        # Event time from a piecewise-constant hazard set at each visit.
        hazards = baseline_hazard * np.exp(severity @ np.abs(betas))
        ends = np.append(years[1:], np.inf)
        target = rng.exponential(1.0)
        cumulative, event_years = 0.0, np.inf
        for start, end, hazard in zip(years, ends, hazards):
            if cumulative + hazard * (end - start) >= target:
                event_years = start + (target - cumulative) / hazard
                break
            cumulative += hazard * (end - start)
        censor_years = rng.uniform(1.0, max_years)
        transplant_years = (
            rng.uniform(0.5, max_years)
            if rng.random() < transplant_fraction
            else np.inf
        )
        end_years = min(event_years, censor_years, transplant_years)
        if end_years == event_years:
            status = STATUS_DEATH
        elif end_years == transplant_years:
            status = STATUS_TRANSPLANT
        else:
            status = STATUS_CENSORED

        # Integer days, at least 1, and unused by any visit or other outcome.
        futime = min(max(int(np.ceil(end_years * DAYS_PER_YEAR)), 1), max_day)
        while futime in taken_days:
            futime += 1
        taken_days.add(futime)

        keep = schedule < futime
        labs = medians * np.exp(spreads * np.sign(betas + (betas == 0)) * severity)
        labs = labs[keep]
        if missing_rate > 0 and labs.shape[0] > 1:
            mask = rng.random(labs[1:].shape) < missing_rate
            labs[1:][mask] = np.nan
        for day, values in zip(schedule[keep], labs):
            rows.append([index + 1, int(day), futime, status, *values.tolist()])

    columns = ["id", "day", "futime", "status", *feature_names]
    return pd.DataFrame(rows, columns=columns)


def write_synthetic_pbcseq_csv(
    directory: str | pathlib.Path,
    n_patients: int = 200,
    seed: int = 0,
    **generator_kwargs,
) -> pathlib.Path:
    """Writes a synthetic ``pbcseq.csv`` that :class:`PBCSeqDataset` can load.

    The four MAS labs (``bili``, ``albumin``, ``ast``, ``protime``) and the
    outcomes come from :func:`generate_synthetic_pbcseq`. The other ``pbcseq``
    columns are filled with plausible, outcome-independent values: treatment,
    age, sex, clinical signs, cholesterol, alkaline phosphatase, platelets, and
    a histologic stage that is constant per patient. Columns use the R names
    (``alk.phos``), as in a real export. Use it to run the full PBC pipeline
    without the real data.

    Args:
        directory: Directory to write ``pbcseq.csv`` into (created if needed).
        n_patients: Number of patients.
        seed: Seed for the labs, outcomes, and filler columns.
        **generator_kwargs: Extra arguments for
            :func:`generate_synthetic_pbcseq` (not ``feature_names``).

    Returns:
        The path of the written CSV.

    Raises:
        ValueError: If ``feature_names`` is passed.

    Examples:
        >>> import tempfile
        >>> path = write_synthetic_pbcseq_csv(tempfile.mkdtemp(), n_patients=3)
        >>> path.name
        'pbcseq.csv'
    """
    if "feature_names" in generator_kwargs:
        raise ValueError("write_synthetic_pbcseq_csv always writes the MAS-4 labs.")
    visits = generate_synthetic_pbcseq(
        n_patients=n_patients,
        feature_names=MAS4_FEATURES,
        seed=seed,
        **generator_kwargs,
    )
    rng = np.random.default_rng([seed, 1])
    ids = visits["id"].unique()
    per_patient = pd.DataFrame(
        {
            "id": ids,
            "trt": rng.integers(1, 3, len(ids)),
            "age": np.round(rng.normal(50.0, 10.0, len(ids)), 2),
            "sex": np.where(rng.random(len(ids)) < 0.9, "f", "m"),
            "stage": rng.integers(1, 5, len(ids)),
        }
    )
    frame = visits.merge(per_patient, on="id", how="left")
    n = len(frame)
    frame["ascites"] = (rng.random(n) < 0.1).astype(int)
    frame["hepato"] = (rng.random(n) < 0.5).astype(int)
    frame["spiders"] = (rng.random(n) < 0.3).astype(int)
    frame["edema"] = rng.choice([0.0, 0.5, 1.0], n, p=[0.85, 0.1, 0.05])
    frame["chol"] = np.round(rng.lognormal(np.log(330.0), 0.4, n))
    frame["alk.phos"] = np.round(rng.lognormal(np.log(1200.0), 0.6, n))
    frame["platelet"] = np.round(rng.normal(250.0, 80.0, n).clip(30.0))
    columns = [
        "id", "futime", "status", "trt", "age", "sex", "day", "ascites", "hepato",
        "spiders", "edema", "bili", "chol", "albumin", "alk.phos", "ast",
        "platelet", "protime", "stage",
    ]  # fmt: skip
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pbcseq.csv"
    frame[columns].to_csv(path, index=False)
    return path


def create_synthetic_liver_dataset(
    n_patients: int = 20,
    feature_names: Sequence[str] = MAS4_FEATURES,
    seed: int = 0,
    event: str = "death",
    visits: pd.DataFrame | None = None,
    **generator_kwargs,
) -> SampleDataset:
    """Creates a model-ready counting-process dataset from synthetic visits.

    Each sample is one at-risk interval with:

    - ``features``: the visit's labs, z-scored by ``TensorProcessor``
      (``normalization="zscore"``, missing values filled with the mean);
      batches have shape ``(B, F)`` with ``F = len(feature_names)``.
    - ``survival_interval``: ``[start, stop, event]`` in years
      (``"survival_interval"``); batches have shape ``(B, 3)``.
    - ``patient_id``, ``visit_id``, and ``landmark_time`` pass through
      unprocessed.

    The feature processor is fitted on **all** generated samples, because this
    dataset has no held-out split. For train/validation/test experiments, fit
    the processor on the training samples only.

    Args:
        n_patients: Number of patients to generate.
        feature_names: Lab columns to generate and use, in order.
        seed: Seed for :func:`generate_synthetic_pbcseq`.
        event: ``"death"`` (transplant censored) or ``"death_or_transplant"``.
        visits: An existing visit table to use instead of generating one.
            ``n_patients`` and ``seed`` are then unused, ``feature_names`` must
            name its feature columns, and ``generator_kwargs`` must be empty.
        **generator_kwargs: Extra arguments for
            :func:`generate_synthetic_pbcseq`.

    Returns:
        An in-memory ``SampleDataset``. Its models have ``mode = None``.

    Raises:
        ValueError: If ``visits`` is given together with generator arguments,
            or lacks a column named in ``feature_names``.

    Examples:
        >>> from pyhealth.datasets import get_dataloader
        >>> dataset = create_synthetic_liver_dataset(n_patients=10, seed=0)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=len(dataset))))
        >>> batch["features"].shape[1], batch["survival_interval"].shape[1]
        (4, 3)
    """
    if visits is not None and generator_kwargs:
        raise ValueError(
            f"Generator arguments {sorted(generator_kwargs)} are ignored when "
            "`visits` is given."
        )
    if visits is None:
        visits = generate_synthetic_pbcseq(
            n_patients=n_patients,
            feature_names=feature_names,
            seed=seed,
            **generator_kwargs,
        )
    samples = counting_process_samples_from_frame(visits, feature_names, event=event)
    processor = TensorProcessor(**FEATURE_PROCESSOR_KWARGS)
    processor.fit(samples, "features")
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": ("tensor", FEATURE_PROCESSOR_KWARGS)},
        output_schema={"survival_interval": "survival_interval"},
        dataset_name="synthetic_pbcseq",
        task_name="liver_survival_counting_process",
        input_processors={"features": processor},
    )
