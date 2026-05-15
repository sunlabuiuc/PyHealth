"""
Provides utilities for PyHealth models
"""
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


def batch_to_multihot(label: List[List[int]], num_labels: int) -> torch.tensor:
    """Converts label to multihot format.

    Args:
        label: [batch size, *]
        num_labels: total number of labels

    Returns:
        multihot: [batch size, num_labels]
    """
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot


def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1
        # Clamp to 0 so that samples with an all-zero mask (no valid
        # visits) fall back to the first timestep instead of producing
        # a negative index that would crash torch.gather.
        last_visit = last_visit.clamp(min=0)
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state


class DataLoaderToNumpy:
    """
    Converts a DataLoader to numpy arrays for sklearn models.

    Args:
        feature_keys: list of feature keys
        label_key: label key

    Examples:
        >>> converter = DataLoaderToNumpy(
        ...     feature_keys=["conditions", "procedures"],
        ...     label_key="los"
        ... )
        >>> X, y = converter.transform(dataloader)
    """

    def __init__(self,
                 feature_keys: List[str],
                 label_key: str,
                 pad_batches: bool = True):
        """
        Initializes the dataloader to numpy converter.

        Args:
            feature_keys: Keys for input features to include.
            label_key: Key for the target label.
            pad_batches: Whether to pad features to a consistent size
                across batches. Defaults to True.

        Examples:
            >>> converter = DataLoaderToNumpy(
            ...     feature_keys=["conditions", "procedures"],
            ...     label_key="los"
            ... )
        """

        self.feature_keys = feature_keys
        self.label_key = label_key

        self.pad_batches = pad_batches
        self._key_dims: Dict[str, int] = {}
        self._fitted = False

    def transform(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Converts a DataLoader to numpy arrays for sklearn models.

        Args:
            dataloader: PyHealth DataLoader

        Returns:
            X: numpy array of shape (n_samples, n_features).
            y: numpy array of shape (n_samples,).

        Examples:
            >>> X, y = converter.transform(dataloader)
            >>> X.shape
            (num_samples, num_features)
            >>> y.shape
            (num_samples,)
        """
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        for batch in dataloader:
            x_parts.append(self._process_features(batch))
            y_parts.append(self._process_labels(batch))

            self._fitted = True

        return np.vstack(x_parts), np.concatenate(y_parts)

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        """Converts a Tensor or list to a numpy array

        Args:
            value: torch.Tensor, numpy array, or Python list from a DataLoader batch.

        Returns:
            numpy array.

        Examples:
            >>> import torch
            >>> arr = DataLoaderToNumpy._to_numpy(torch.tensor([1, 2, 3]))
        """
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.array(value)

        return arr.astype(np.float32)

    def _flatten_feature(self, arr: np.ndarray, key: str) -> np.ndarray:
        """Converts the feature to a two-dimensional array and pads if padding is
        enabled and needed.

        Args:
            arr: numpy array
            key: feature key

        Returns:
            numpy array of shape (batch_size, expected_width).

        Raises:
            ValueError: if dimensions are not consistent and pad_batches is False.

        Examples:
            >>> arr = np.array([[1, 2], [3, 4]])
            >>> converter._flatten_feature(arr, "conditions")
        """
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        width = arr.shape[1]

        if not self._fitted:
            self._key_dims[key] = width
        elif width != self._key_dims[key]:
            expected = self._key_dims[key]

            if not self.pad_batches:
                raise ValueError(
                    f"Inconsistent batch sizes across features. Set pad_batches=True t"
                    f"o allow padding."
                )
            arr = (
                np.pad(arr, ((0, 0), (0, expected - width)), mode = "constant")
                if width < expected
                else arr[:, :expected]
            )

        return arr

    def _process_features(self, batch: dict) -> np.ndarray:
        """Concatenate all features from one batch.

        Args:
            batch: dictionary from PyHealth DataLoader containing feature keys

        Returns:
            numpy array of shape (batch_size, total_feature_dim).

        Examples:
            >>> X_batch = converter._process_features(batch)
        """
        return np.concatenate(
            [self._flatten_feature(self._to_numpy(batch[k]), k) for k in
             self.feature_keys],
            axis = 1,
        )

    def _process_labels(self, batch: dict) -> np.ndarray:
        """Extract and flatten the label array from one batch.

        Args:
            batch: dictionary from a PyHealth DataLoader containing label_key.

        Returns:
            numpy array of shape (batch_size,).

        Examples:
            >>> y_batch = converter._process_labels(batch)
            >>> y_batch.shape
            (batch_size,)
        """
        return self._to_numpy(batch[self.label_key]).reshape(-1)
