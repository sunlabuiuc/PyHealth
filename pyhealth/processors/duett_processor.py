from typing import Any, Dict, Iterable, Tuple, Optional

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("duett_ts")
class DuETTTimeSeriesProcessor(FeatureProcessor):
    """Feature processor for DuETT's time-series format.

    Calculates global mean and std for valid inputs. Applies Z-score normalization
    and concatenates observation frequencies along the feature dimension.
    """

    def __init__(self):
        self.means = None
        self.stds = None
        self.n_features = None

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Calculates global mean and std deviation from valid sequence matrices.

        Args:
            samples (Iterable[Dict[str, Any]]): The samples.
            field (str): The specific dictionary key holding features.
        """
        for sample in samples:
            if field in sample and sample[field] is not None:
                x_ts_raw, _ = sample[field]
                self.n_features = x_ts_raw.shape[1]
                break

        if self.n_features is None:
            raise ValueError(f"Could not infer n_features for field {field}")

        sums = np.zeros(self.n_features, dtype=np.float64)
        sq_sums = np.zeros(self.n_features, dtype=np.float64)
        counts = np.zeros(self.n_features, dtype=np.float64)

        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            x_ts_raw, x_ts_counts = sample[field]
            mask = x_ts_counts > 0
            for i in range(self.n_features):
                valid_vals = x_ts_raw[:, i][mask[:, i]]
                sums[i] += valid_vals.sum()
                sq_sums[i] += (valid_vals ** 2).sum()
                counts[i] += len(valid_vals)

        safe_counts = np.maximum(counts, 1.0)
        self.means = sums / safe_counts
        variance = (sq_sums / safe_counts) - (self.means ** 2)
        self.stds = np.sqrt(np.maximum(variance, 0.0))

    def process(self, value: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
        """Applies normalization and concatenates arrays.

        Args:
            value (Tuple[np.ndarray, np.ndarray]): Tuple of (values, counts).

        Returns:
            torch.Tensor: Evaluated tensor matrix shaped (S, F * 2).
        """
        x_ts_raw, x_ts_counts = value
        x_ts_raw = torch.tensor(x_ts_raw, dtype=torch.float32)
        x_ts_counts = torch.tensor(x_ts_counts, dtype=torch.float32)
        means = torch.tensor(self.means, dtype=torch.float32)
        stds = torch.tensor(self.stds, dtype=torch.float32)
        
        mask = x_ts_counts > 0
        x_ts_norm = torch.zeros_like(x_ts_raw)
        
        exp_means = means.expand_as(x_ts_raw)
        exp_stds = stds.expand_as(x_ts_raw)
        
        x_ts_norm[mask] = (x_ts_raw[mask] - exp_means[mask]) / (exp_stds[mask] + 1e-7)
        return torch.cat([x_ts_norm, x_ts_counts], dim=-1)

    def size(self) -> Optional[int]:
        return self.n_features * 2 if self.n_features else None

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple:
        return ("value",)

    def dim(self) -> tuple:
        return (2,)

    def spatial(self) -> tuple:
        return (True, False)

    def __repr__(self) -> str:
        return f"DuETTTimeSeriesProcessor(n_features={self.n_features})"