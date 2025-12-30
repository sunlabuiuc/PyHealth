from datetime import datetime, timedelta
from typing import Any, List, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("timeseries")
class TimeseriesProcessor(FeatureProcessor):
    """
    Feature processor for irregular time series with missing values.

    Input:
        - timestamps: List[datetime]
        - values: np.ndarray of shape (T, F)

    Processing:
        1. Uniform sampling at fixed intervals.
        2. Imputation for missing values.

    Output:
        - torch.Tensor of shape (S, F), where S is the number of sampled time steps.
    """

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        impute_strategy: str = "forward_fill",
    ):
        # Configurable sampling rate and imputation method
        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.n_features = None

    def fit(self, samples: Any, field: str) -> None:
        """Fit the processor by determining n_features from the first valid sample.

        Args:
            samples: Iterable of sample dictionaries.
            field: The field name to extract from samples.
        """
        # Extract n_features from the first valid sample without full processing
        for sample in samples:
            if field in sample and sample[field] is not None:
                _, values = sample[field]
                values = np.asarray(values)
                if values.ndim == 2:
                    self.n_features = values.shape[1]
                    break
                elif values.ndim == 1:
                    self.n_features = 1
                    break

    def process(self, value: Tuple[List[datetime], np.ndarray]) -> torch.Tensor:
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        values = np.asarray(values)
        num_features = values.shape[1]

        # Step 1: Uniform sampling
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_steps = int((end_time - start_time) / self.sampling_rate) + 1

        sampled_times = [
            start_time + i * self.sampling_rate for i in range(total_steps)
        ]
        sampled_values = np.full((total_steps, num_features), np.nan)

        # Map original timestamps to indices in the sampled grid
        for t, v in zip(timestamps, values):
            idx = int((t - start_time) / self.sampling_rate)
            if 0 <= idx < total_steps:
                sampled_values[idx] = v

        # Step 2: Imputation
        if self.impute_strategy == "forward_fill":
            for f in range(num_features):
                last_value = 0.0
                for t in range(total_steps):
                    if not np.isnan(sampled_values[t, f]):
                        last_value = sampled_values[t, f]
                    else:
                        sampled_values[t, f] = last_value
        elif self.impute_strategy == "zero":
            sampled_values = np.nan_to_num(sampled_values, nan=0.0)
        else:
            raise ValueError(f"Unsupported imputation strategy: {self.impute_strategy}")

        if self.n_features is None:
            self.n_features = sampled_values.shape[1]

        return torch.tensor(sampled_values, dtype=torch.float)

    def size(self):
        # Size equals number of features, unknown until first process
        return self.n_features

    def __repr__(self):
        return (
            f"TimeSeriesProcessor(sampling_rate={self.sampling_rate}, "
            f"impute_strategy='{self.impute_strategy}')"
        )
