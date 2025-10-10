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
            self, sampling_rate: timedelta = timedelta(hours=1), 
            impute_strategy: str = "forward_fill",
            normalize: bool = False,
            norm_method: str = "z_score", 
            norm_axis: str = "global"
            ):
        # Configurable sampling rate and imputation method
        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.size = None
        self.normalize_method = norm_method
        self.normalize_axis = norm_axis
        self.normalize_flag = normalize
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

        sampled_times = [start_time + i * self.sampling_rate for i in range(total_steps)]
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
        
        if self.size is None:
            self.size = sampled_values.shape[1]
        if self.normalize_flag and hasattr(self, 'mean_'):  
          sampled_values = self._apply_normalization(sampled_values)

        return torch.tensor(sampled_values, dtype=torch.float)
    def size(self):
        # Size equals number of features, unknown until first process
        return self.size

    def __repr__(self):
        return (
            f"TimeSeriesProcessor(sampling_rate={self.sampling_rate}, "
            f"impute_strategy='{self.impute_strategy}')"
        )
    def _compute_global_stats(self, data: np.ndarray) -> Any:
        """
        Compute global statistics for normalization across the entire dataset.

        Depending on `self.normalize_method`, calculates:
            - "z_score": mean and standard deviation over all values.
        - "min_max": minimum and maximum over all values.
        - "robust": median and median absolute deviation (MAD) over all values.

        Parameters
        ----------
        data : np.ndarray
            The input array containing all values to compute statistics on.

        Raises
        ------
        ValueError
            If `self.normalize_method` is unsupported.
            """
        if self.normalize_method == "z_score":
            self.mean = np.mean(data)
            self.std = np.std(data)
        elif self.normalize_method == "min_max":
            self.min = np.min(data)
            self.max = np.max(data)
        elif self.normalize_method == "robust" :
            self.median = np.median(data)
            self.mad_ = np.median(np.abs(data - self.median))
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalize_method}")
    def _compute_per_feature_stats(self, data: np.ndarray) -> Any:
        """
        Compute per-feature statistics for normalization.

        Calculates statistics independently for each column (feature) based on 
        `self.normalize_method`:
        - "z_score": mean and standard deviation per feature.
        - "min_max": minimum and maximum per feature.
        - "robust": median and median absolute deviation (MAD) per feature.

        Parameters
        ----------
        data : np.ndarray
            The input 2D array where each column represents a feature.

        Raises
        ------
        ValueError
            If `self.normalize_method` is unsupported.
        """
        if self.normalize_method == "z_score":
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        elif self.normalize_method == "min_max":
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
        elif self.normalize_method == "robust" :
            self.median = np.median(data, axis=0)
            self.mad_ = np.median(np.abs(data - self.median), axis=0)
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalize_method}")
    def fit(self, samples: List[dict[str, Any]], field: str) -> None:
        """
            Fit normalization statistics to a dataset.

            Extracts values from the given `samples`, processes them, and computes 
            normalization statistics either globally or per feature, depending on 
            `self.normalize_axis`.

            Parameters
            ----------
            samples : list of dict[str, Any]
                A list of sample dictionaries. Each dictionary must contain `field` 
                mapping to a tuple of (timestamps, values).
            field : str
                The key in each sample dictionary from which to extract the data.

            Notes
            -----
            - Uses `self.process()` to preprocess each sample's values before computing statistics.
            - Does nothing if `self.normalize_flag` is False.

            Raises
            ------
            ValueError
                If `self.normalize_axis` is unsupported.
        """
        if not self.normalize_flag:
            return
        all_values = []
        for sample in samples:
            timestamps, values = sample[field]
            processed_values = self.process((timestamps, values))
            all_values.append(processed_values.numpy())
        combined_values = np.vstack(all_values)
        if self.normalize_axis == "global":
            self._compute_global_stats(combined_values)
        elif self.normalize_axis == "per_feature":
            self._compute_per_feature_stats(combined_values)
        else:
            raise ValueError(f"Unsupported normalization axis: {self.normalize_axis}")
    def _apply_normalization(self, value: np.ndarray) -> np.ndarray:
        """
        Apply normalization to an array using precomputed statistics.

        Normalization method is determined by `self.normalize_method`:
        - "z_score": (value - mean) / std
        - "min_max": (value - min) / (max - min)
        - "robust": (value - median) / MAD

        Parameters
        ----------
        value : np.ndarray
            The array of values to normalize.

        Returns
        -------
        np.ndarray
            The normalized array.

        Raises
        ------
        ValueError
            If `self.normalize_method` is unsupported.
        """
        if self.normalize_method == "z_score":
            return (value - self.mean) / (self.std + 1e-8)
        elif self.normalize_method == "min_max":
            return (value - self.min) / (self.max - self.min + 1e-8)
        elif self.normalize_method == "robust":
            return (value - self.median) / (self.mad_ + 1e-8)
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalize_method}")
        
        