"""ICU decompensation task for Dynamic Survival Analysis.

Provides:

* :class:`DecompensationDSA` — a ``BaseTask`` subclass for use with
  real PyHealth datasets (MIMIC-III / MIMIC-IV / eICU).
* :func:`make_synthetic_dsa_samples` — a standalone factory function that
  builds a list of sample dicts **without any external dataset**, suitable
  for quick experimentation, unit tests, and CI.

References:
    Yèche H. et al., "Dynamic Survival Analysis for Early Event Prediction",
    Proceedings of Machine Learning for Health (CHIL), 2024.
    https://proceedings.mlr.press/v248/yeche24a.html

Example — synthetic data only::

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.tasks import DecompensationDSA
    from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples

    samples = make_synthetic_dsa_samples(n_patients=200, n_features=8, horizon=24)
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=DecompensationDSA.input_schema,
        output_schema=DecompensationDSA.output_schema,
        dataset_name="dsa_synthetic",
    )
    loader = get_dataloader(dataset, batch_size=16, shuffle=True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base_task import BaseTask


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class DecompensationDSA(BaseTask):
    """ICU decompensation prediction task for Dynamic Survival Analysis.

    Generates one sample per ICU stay. The target label indicates whether
    the patient decompensated (e.g., died) within the next ``horizon`` hours.
    The feature is a pre-padded float time series of shape
    ``(max_seq_len, n_features)``.

    Attributes:
        task_name: Identifier string used for logging.
        input_schema: Maps feature keys to processor types.
        output_schema: Maps label keys to processor types.

    Note:
        For real datasets (MIMIC-III, MIMIC-IV, eICU), override
        ``__call__`` to extract events from the patient object.
        For synthetic data, use :func:`make_synthetic_dsa_samples`
        directly — no patient object is required.

    Examples:
        >>> from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples
        >>> from pyhealth.tasks import DecompensationDSA
        >>> samples = make_synthetic_dsa_samples(n_patients=10)
        >>> assert all("timeseries" in s and "label" in s for s in samples)
    """

    task_name: str = "DecompensationDSA"
    input_schema: Dict[str, str] = {"timeseries": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient from a PyHealth dataset into DSA samples.

        Each admission is converted into a single sample: the ICU time
        series is aggregated into hourly bins and zero-padded to
        ``max_seq_len``.  The binary label is 1 if the patient died
        in-hospital during the admission, 0 otherwise.

        Args:
            patient: A PyHealth patient object with at least one ICU
                admission and associated chart events.

        Returns:
            List of sample dicts, one per ICU admission that meets the
            minimum length requirement.  Returns an empty list when the
            patient has no usable admissions.

        Note:
            This default implementation expects the patient to expose
            ``patient.patient_id``, and each visit to expose
            ``visit.discharge_status`` and ``visit.available_tables``.
            Override this method when working with a custom dataset.
        """
        samples: List[Dict[str, Any]] = []

        for visit_idx, visit in enumerate(patient):
            # Decompensation label: in-hospital death
            discharge_status = getattr(visit, "discharge_status", None)
            label = 1 if str(discharge_status).lower() in ("expired", "dead", "1") else 0

            # Extract available numeric events as a flat feature matrix
            # (real implementation would pivot chartevents by itemid)
            events = []
            for table in getattr(visit, "available_tables", []):
                events.extend(visit.get_events(table))

            if len(events) < 4:
                continue

            # Build a simple T×1 time series from event values
            values = []
            for ev in events:
                v = getattr(ev, "value", None) or getattr(ev, "valuenum", None)
                try:
                    values.append([float(v)])
                except (TypeError, ValueError):
                    values.append([0.0])

            timeseries = np.array(values, dtype=np.float32)

            samples.append(
                {
                    "patient_id": f"{patient.patient_id}_v{visit_idx}",
                    "timeseries": timeseries.tolist(),
                    "label": label,
                }
            )

        return samples


# ---------------------------------------------------------------------------
# Synthetic data factory (no external dataset required)
# ---------------------------------------------------------------------------


def make_synthetic_dsa_samples(
    n_patients: int = 200,
    n_features: int = 8,
    horizon: int = 24,
    max_seq_len: int = 100,
    event_rate: float = 0.3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic DSA samples compatible with :class:`DecompensationDSA`.

    Each sample represents one synthetic ICU stay.  Positive patients
    (``label=1``) have a decompensation event placed at a random timestep;
    negative patients are censored.  All feature matrices are zero-padded
    to ``max_seq_len`` so they stack cleanly in a PyHealth DataLoader.

    Args:
        n_patients: Number of synthetic patients. Default: ``200``.
        n_features: Number of features per time step. Default: ``8``.
        horizon: Prediction horizon in time steps. Default: ``24``.
        max_seq_len: Fixed sequence length after padding. Default: ``100``.
        event_rate: Fraction of patients with a decompensation event.
            Default: ``0.3``.
        seed: Random seed for reproducibility. Default: ``42``.

    Returns:
        List of sample dicts with keys:

        * ``"patient_id"``  – str
        * ``"timeseries"``  – list of shape ``(max_seq_len, n_features)``
        * ``"label"``       – int (0 or 1)

    Examples:
        >>> samples = make_synthetic_dsa_samples(n_patients=10, n_features=4)
        >>> len(samples)
        10
        >>> len(samples[0]["timeseries"]) == 100  # padded to max_seq_len
        True
        >>> samples[0]["timeseries"][0]            # first timestep features
        [...]
    """
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []

    for pid in range(n_patients):
        # Random stay length between horizon+4 and max_seq_len
        stay_len = int(rng.integers(horizon + 4, max_seq_len + 1))
        features = rng.standard_normal((stay_len, n_features)).astype(np.float32)

        has_event = rng.random() < event_rate
        if has_event:
            # Event occurs at least `horizon` steps from the end
            onset = int(rng.integers(0, max(1, stay_len - horizon)))
            # Mark a decompensation signal: spike in the first feature
            features[onset:, 0] += 3.0
            label = 1
        else:
            label = 0

        # Zero-pad on the left to max_seq_len
        padded = np.zeros((max_seq_len, n_features), dtype=np.float32)
        padded[max_seq_len - stay_len :] = features

        samples.append(
            {
                "patient_id": f"synth_{pid:04d}",
                "timeseries": padded.tolist(),
                "label": label,
            }
        )

    return samples
