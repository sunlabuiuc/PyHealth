"""Training-only impute-then-standardize preprocessing for competing risks.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

from copy import deepcopy
from typing import Any

import numpy as np


class CompetingRisksPreprocessor:
    """Preserve clinical width and learn statistics only from training patients.

    First mean-impute, then compute population standard deviations on the
    imputed matrix. All-missing columns use mean zero and scale one. Time is
    divided by the maximum training follow-up (at least 1e-8). Masks always
    describe missingness before imputation.

    Attributes:
        state: JSON-serializable fitted mean, scale, time_scale, and
            all_missing_indices, or None before fitting.

    Examples:
        >>> train = [{"features": [1., float("nan")], "time": 2., "event": 1}]
        >>> processor = CompetingRisksPreprocessor().fit(train)
        >>> processor.transform(train)[0]["features"]
        [0.0, 0.0]
    """

    def __init__(self) -> None:
        """Start unfitted; learn statistics from one training split with fit."""
        self.state: dict[str, Any] | None = None

    def fit(self, samples: list[dict[str, Any]]) -> "CompetingRisksPreprocessor":
        """Fit imputation, scaling, and time units before transforming any split.

        Args:
            samples: Nonempty training samples with equal-width numeric features
                and finite, nonnegative times. NaN means missing.

        Returns:
            This fitted instance.

        Raises:
            ValueError: Training data is empty, malformed, or contains infinity.
        """
        if not samples:
            raise ValueError("Cannot fit empty training samples")
        features = np.asarray([sample["features"] for sample in samples], dtype=float)
        if features.ndim != 2 or features.shape[1] == 0 or np.isinf(features).any():
            raise ValueError("Features must be a nonempty matrix without infinity")
        times = np.asarray([sample["time"] for sample in samples], dtype=float)
        if times.shape != (len(samples),) or not np.isfinite(times).all():
            raise ValueError("Times must be one finite value per training sample")
        if (times < 0).any():
            raise ValueError("Times cannot be negative")
        count = np.isfinite(features).sum(axis=0)
        mean = np.divide(
            np.nansum(features, axis=0),
            count,
            out=np.zeros(features.shape[1]),
            where=count > 0,
        )
        imputed = np.where(np.isnan(features), mean, features)
        scale = imputed.std(axis=0)
        scale[scale < 1e-12] = 1.0
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Feature statistics overflowed")
        self.state = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "time_scale": max(float(times.max()), 1e-8),
            "all_missing_indices": np.flatnonzero(count == 0).tolist(),
        }
        return self

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CompetingRisksPreprocessor":
        """Restore validated preprocessing saved alongside a checkpoint.

        Args:
            state: Fitted JSON-serializable state from fit.

        Returns:
            An independent preprocessor with a copy of the supplied state.

        Raises:
            ValueError: State shape, statistics, or time scale are invalid.
        """
        mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
        time_scale = float(state["time_scale"])
        indices = state["all_missing_indices"]
        if (
            mean.ndim != 1
            or not mean.size
            or mean.shape != scale.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
            or not np.isfinite(time_scale)
            or time_scale <= 0
            or any(not isinstance(i, int) or not 0 <= i < len(mean) for i in indices)
        ):
            raise ValueError("Invalid preprocessing state")
        result = cls()
        result.state = deepcopy(state)
        return result

    def transform(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply fitted statistics without altering them or the source samples.

        Args:
            samples: Samples with matching feature width and original time units.

        Returns:
            Copies containing standardized imputed features, masks, and times.

        Raises:
            RuntimeError: fit has not been called.
            ValueError: Feature shape, values, or times are invalid.
        """
        if self.state is None:
            raise RuntimeError("Call fit before transform")
        mean, scale = np.asarray(self.state["mean"]), np.asarray(self.state["scale"])
        result = []
        for sample in samples:
            features = np.asarray(sample["features"], dtype=float)
            time = float(sample["time"])
            if (
                features.shape != mean.shape
                or np.isinf(features).any()
                or not np.isfinite(time)
                or time < 0
            ):
                raise ValueError("Invalid feature schema, values, or time")
            missing = np.isnan(features)
            values = (np.where(missing, mean, features) - mean) / scale
            if not np.isfinite(values).all():
                raise ValueError("Transformed features overflowed")
            result.append(
                {
                    **sample,
                    "features": values.tolist(),
                    "missing_mask": missing.astype(float).tolist(),
                    "time": time / self.state["time_scale"],
                }
            )
        return result
