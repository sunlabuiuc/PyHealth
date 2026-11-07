"""SUPPORT2 time-to-death samples retaining right-censored subjects.

Contributor: Taejun Oh (NetID: taejuno2).
Paper: GRAFT: Decoupling Ranking and Calibration for Survival Analysis.
Paper link: https://arxiv.org/abs/2602.07884v2.
"""

import math
from typing import Any

from .base_task import BaseTask


class SurvivalTimeSupport2(BaseTask):
    """Extract dense predictors, positive follow-up time and death indicator.

    Missing numeric predictors or invalid follow-up times exclude a record.
    Missing race remains a category. Invalid event codes and unknown nominal
    categories raise. Features are fixed encodings with no learned vocabulary,
    imputation or scaling; GRAFT fits numeric preprocessing on training only.
    Some SUPPORT measurements are collected through day three. These are not
    uniformly admission-time predictors. No supplied prognosis, outcome-derived
    field, post-event field or patient identifier is used as a predictor.

    Args:
        encoding: ``"onehot22"`` (default) or ``"benchmark14"``. The latter
            uses the verified DeepSurv/pycox ordering and numeric category codes.
            The two encodings are distinct experimental conditions.

    Examples:
        >>> from pyhealth.tasks import SurvivalTimeSupport2
        >>> task = SurvivalTimeSupport2(encoding="benchmark14")
        >>> len(task.feature_names)
        14
        >>> task.output_schema
        {'duration': 'tensor', 'event': 'tensor'}
    """

    task_name = "SurvivalTimeSupport2"
    input_schema = {"features": "tensor"}
    output_schema = {"duration": "tensor", "event": "tensor"}
    numeric = (
        "age",
        "num.co",
        "diabetes",
        "dementia",
        "meanbp",
        "hrt",
        "resp",
        "temp",
        "wblc",
        "sod",
        "crea",
    )
    categories = {
        "sex": ("female", "male"),
        "race": ("asian", "black", "hispanic", "other", "white", "<missing>"),
        "ca": ("metastatic", "no", "yes"),
    }
    benchmark_names = (
        "age",
        "sex",
        "num.co",
        "race",
        "diabetes",
        "dementia",
        "ca",
        "meanbp",
        "hrt",
        "resp",
        "temp",
        "sod",
        "wblc",
        "crea",
    )
    benchmark_codes = {
        "sex": {"male": 0, "female": 1},
        "race": {
            "other": 0,
            "white": 1,
            "black": 2,
            "hispanic": 3,
            "asian": 4,
            "<missing>": 5,
        },
        "ca": {"metastatic": 0, "no": 1, "yes": 2},
    }

    def __init__(self, encoding: str = "onehot22") -> None:
        super().__init__()
        if encoding not in {"onehot22", "benchmark14"}:
            raise ValueError("encoding must be 'onehot22' or 'benchmark14'.")
        self.encoding = encoding
        # Distinct names prevent set_task cache collisions between encodings.
        self.task_name = f"SurvivalTimeSupport2_{encoding}"
        self.feature_names = (
            list(self.benchmark_names)
            if encoding == "benchmark14"
            else (
                list(self.numeric)
                + [
                    f"{key}={level}"
                    for key, levels in self.categories.items()
                    for level in levels
                ]
            )
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Return zero or one labeled sample from the patient's SUPPORT2 row.

        Args:
            patient: PyHealth patient exposing one ``support2`` event with
                clinical attributes, ``d.time`` and ``death``.

        Returns:
            One dictionary containing patient_id, features, duration and event,
            or an empty list for invalid follow-up/missing numeric predictors.

        Raises:
            ValueError: For multiple/missing rows, invalid event coding or
                nominal categories outside the documented fixed encoding.
        """
        rows = patient.get_events(event_type="support2")
        if len(rows) != 1:
            raise ValueError("Expected exactly one SUPPORT2 record per patient.")
        row = rows[0]
        try:
            duration = float(row["d.time"])
        except (TypeError, ValueError):
            return []
        if not math.isfinite(duration) or duration <= 0:
            return []
        try:
            event = float(row["death"])
        except (TypeError, ValueError) as exc:
            raise ValueError("death must be 0=censored or 1=observed event.") from exc
        if event not in (0, 1):
            raise ValueError("death must be 0=censored or 1=observed event.")
        numeric = {}
        for name in self.numeric:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                return []
            if not math.isfinite(value):
                return []
            numeric[name] = value
        categorical = {}
        for name, levels in self.categories.items():
            value = row[name]
            missing = value is None or (isinstance(value, float) and math.isnan(value))
            value = "<missing>" if missing else str(value)
            if value not in levels:
                raise ValueError(f"Unrecognized {name} category: {value}")
            categorical[name] = value
        if self.encoding == "benchmark14":
            values = dict(numeric)
            values.update(
                {k: float(self.benchmark_codes[k][v]) for k, v in categorical.items()}
            )
            features = [values[name] for name in self.benchmark_names]
        else:
            features = [numeric[name] for name in self.numeric]
            for name, levels in self.categories.items():
                features.extend(float(categorical[name] == level) for level in levels)
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "duration": duration,
                "event": event,
            }
        ]
