"""Framingham competing-risks cohort used by Neural Fine-Gray.

The prepared table contains one baseline row per participant, 18 covariates,
follow-up duration, and a three-valued outcome: censoring, cardiovascular
disease, or death without a prior cardiovascular event.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

__all__ = ["FRAMINGHAM_FEATURES", "FraminghamDataset", "prepare_framingham"]

FRAMINGHAM_FEATURES: tuple[str, ...] = (
    "sex",
    "current_smoker",
    "diabetes",
    "bp_meds",
    "education",
    "previous_chd",
    "previous_angina",
    "previous_mi",
    "previous_stroke",
    "previous_hypertension",
    "total_cholesterol",
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cigarettes_per_day",
    "bmi",
    "heart_rate",
    "glucose",
)

_SOURCE_COLUMNS: tuple[str, ...] = (
    "SEX",
    "CURSMOKE",
    "DIABETES",
    "BPMEDS",
    "educ",
    "PREVCHD",
    "PREVAP",
    "PREVMI",
    "PREVSTRK",
    "PREVHYP",
    "TOTCHOL",
    "AGE",
    "SYSBP",
    "DIABP",
    "CIGPDAY",
    "BMI",
    "HEARTRTE",
    "GLUCOSE",
)

# Pinned public copy used by the Neural Fine-Gray authors through their
# DeepSurvivalMachines dependency. Pinning prevents silent data drift.
_SOURCE_URL = (
    "https://raw.githubusercontent.com/Jeanselme/DeepSurvivalMachines/"
    "b29ab6dda51066e5c0b1b1f02e8bd7baca875094/dsm/datasets/framingham.csv"
)
_RAW_NAME = "framingham_longitudinal.csv"
_CSV_NAME = "framingham.csv"


def _load_raw(root: Path, download: bool) -> pd.DataFrame:
    raw_path = root / _RAW_NAME
    if raw_path.exists():
        return pd.read_csv(raw_path)
    if not download:
        raise FileNotFoundError(
            f"{raw_path} not found. Put the longitudinal Framingham CSV there, "
            "provide an already prepared framingham.csv, or pass download=True."
        )
    logger.info("Downloading pinned Framingham source from %s", _SOURCE_URL)
    try:
        with urllib.request.urlopen(_SOURCE_URL, timeout=60) as response:
            payload = response.read()
    except OSError as error:
        raise OSError(
            "Could not download the Framingham source. Download it manually "
            f"from {_SOURCE_URL} and save it as {raw_path}."
        ) from error
    root.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)
    return pd.read_csv(io.BytesIO(payload))


def prepare_framingham(root: str | Path, download: bool = False) -> Path:
    """Create the one-row-per-patient Neural Fine-Gray cohort.

    The transformation matches the public reference preprocessing: retain the
    first observation of each participant, use the 18 published covariates,
    and mean-impute missing covariates. Cardiovascular disease is cause 1;
    death without a prior cardiovascular event is cause 2; zero is censoring.

    Args:
        root: Dataset directory.
        download: Download the pinned public longitudinal CSV when absent.

    Returns:
        Path to the prepared ``framingham.csv``.
    """
    root = Path(root)
    output = root / _CSV_NAME
    if output.exists():
        return output
    raw = _load_raw(root, download)
    required = {
        "RANDID",
        "TIME",
        "TIMECVD",
        "TIMEDTH",
        "CVD",
        "DEATH",
        *_SOURCE_COLUMNS,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Framingham source is missing columns: {sorted(missing)}")
    baseline = raw.groupby("RANDID", sort=True, as_index=False).first()
    features = baseline.loc[:, _SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.mean())
    if features.isna().any().any():
        raise ValueError(
            "A Framingham feature is entirely missing and cannot be imputed"
        )

    cvd = pd.to_numeric(baseline["CVD"], errors="coerce").eq(1)
    death = pd.to_numeric(baseline["DEATH"], errors="coerce").eq(1)
    start = pd.to_numeric(baseline["TIME"], errors="coerce")
    cvd_time = pd.to_numeric(baseline["TIMECVD"], errors="coerce") - start
    death_time = pd.to_numeric(baseline["TIMEDTH"], errors="coerce") - start
    duration = np.where(cvd, cvd_time, death_time).astype(float) + 1.0
    event = np.select([cvd, death], [1, 2], default=0).astype(int)
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("Framingham follow-up durations must be finite and positive")

    prepared = pd.DataFrame(
        {
            "patient_id": baseline["RANDID"].astype(str),
            **{
                target: features[source].astype(float)
                for source, target in zip(_SOURCE_COLUMNS, FRAMINGHAM_FEATURES)
            },
            "duration_days": duration,
            "event": event,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    return output


class FraminghamDataset(BaseDataset):
    """Framingham baseline cohort for competing-risk survival prediction.

    The dataset follows the preprocessing used by Neural Fine-Gray and Deep
    Survival Machines. It contains 4,434 participants when built from the full
    source, with cardiovascular disease as cause 1 and death as cause 2.

    Args:
        root: Directory containing prepared ``framingham.csv`` or the raw
            ``framingham_longitudinal.csv``.
        tables: Tables to load. Defaults to ``["framingham"]``.
        dataset_name: Optional display name.
        config_path: Optional custom dataset configuration.
        download: Download the pinned public source when local data are absent.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import FraminghamDataset
        >>> dataset = FraminghamDataset(  # doctest: +SKIP
        ...     root="/data/framingham", download=True
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        if not (Path(root) / _CSV_NAME).exists():
            prepare_framingham(root, download=download)
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "framingham.yaml")
        super().__init__(
            root=root,
            tables=tables or ["framingham"],
            dataset_name=dataset_name or "framingham",
            config_path=config_path,
            **kwargs,
        )
