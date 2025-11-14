from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .sample_dataset import SampleDataset

# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


def split_by_patient_stream(
    patient_ids: List[str],
    ratios: List[float],
    seed: int = 42,
) -> Tuple[List[str], ...]:
    """Split patient IDs into train/val/test or other proportions.

    This function provides deterministic patient-level splitting by operating
    on patient ID lists rather than SampleDataset objects. This is ideal for:
    - Streaming mode datasets where you filter patients before task application
    - Pre-computing splits to save for reproducibility
    - Creating custom patient-level cross-validation folds

    Unlike `split_by_patient` which operates on `SampleDataset` objects and
    returns `Subset` objects, this function operates on raw patient ID lists.

    Args:
        patient_ids: List of all patient IDs to split
        ratios: List of floats that sum to 1.0 specifying split proportions.
            Common patterns:
            - [0.8, 0.2] for train/test
            - [0.8, 0.1, 0.1] for train/val/test
            - [0.7, 0.15, 0.15] for larger validation/test sets
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of patient ID lists, one per ratio specified.
        Length matches len(ratios).

    Raises:
        AssertionError: If ratios don't sum to 1.0 (within 1e-6 tolerance)

    Examples:
        >>> # Standard train/val/test split
        >>> from pyhealth.datasets import split_by_patient_stream
        >>> patient_ids = ["patient-1", "patient-2", ..., "patient-100"]
        >>> train, val, test = split_by_patient_stream(
        ...     patient_ids, [0.8, 0.1, 0.1]
        ... )
        >>> len(train), len(val), len(test)
        (80, 10, 10)

        >>> # Use with streaming datasets
        >>> base_dataset = MIMIC4Dataset(..., stream=True)
        >>> all_ids = base_dataset.patient_ids
        >>> train_ids, val_ids, test_ids = split_by_patient_stream(
        ...     all_ids, [0.8, 0.1, 0.1]
        ... )
        >>> # Then filter when creating sample datasets
        >>> train_samples = base_dataset.set_task(
        ...     task, patient_ids=train_ids  # Filter to train patients
        ... )

    Note:
        Patient-level splitting is essential in medical ML to prevent:
        - Data leakage from multiple visits of same patient
        - Optimistically biased performance estimates
        - Models that memorize patient-specific patterns

    See Also:
        - `split_by_patient`: Splits SampleDataset objects into Subset objects
        - `split_by_visit`: Splits by samples/visits
    """
    import random

    # Validation
    assert isinstance(patient_ids, list), "patient_ids must be a list"
    assert isinstance(ratios, list), "ratios must be a list"
    assert len(ratios) >= 2, "Must provide at least 2 ratios for splitting"
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"
    assert all(r > 0 for r in ratios), "All ratios must be positive"

    # Shuffle patient IDs deterministically
    random.seed(seed)
    shuffled_ids = patient_ids.copy()
    random.shuffle(shuffled_ids)

    # Calculate split indices
    n_total = len(shuffled_ids)
    splits = []
    start_idx = 0

    for i, ratio in enumerate(ratios[:-1]):
        # Calculate size for this split
        split_size = int(n_total * ratio)
        end_idx = start_idx + split_size

        splits.append(shuffled_ids[start_idx:end_idx])
        start_idx = end_idx

    # Last split gets all remaining patients (handles rounding)
    splits.append(shuffled_ids[start_idx:])

    # Print summary
    split_names = (
        ["train", "val", "test"]
        if len(splits) == 3
        else (
            ["train", "test"]
            if len(splits) == 2
            else [f"split_{i+1}" for i in range(len(splits))]
        )
    )
    print(f"Split {n_total} patients by patient ID (seed={seed}):")
    for name, split in zip(split_names, splits):
        pct = len(split) / n_total * 100
        print(f"  {name.capitalize():8s}: " f"{len(split):6d} patients ({pct:5.1f}%)")

    return tuple(splits)


def split_by_visit(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
        int(len(dataset) * ratios[0]) : int(len(dataset) * (ratios[0] + ratios[1]))
    ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])) :]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_patient(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys())
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    val_patient_indx = patient_indx[
        int(num_patients * ratios[0]) : int(num_patients * (ratios[0] + ratios[1]))
    ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])) :]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_sample(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
        int(len(dataset) * ratios[0]) : int(len(dataset) * (ratios[0] + ratios[1]))
    ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])) :]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(test_index),
        )
    else:
        return train_dataset, val_dataset, test_dataset


def split_by_visit_conformal(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples) for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / cal / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset: four subsets
            of the dataset of type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, `cal_dataset.dataset`, and
            `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert len(ratios) == 4, "ratios must have 4 elements for train/val/cal/test"
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"

    index = np.arange(len(dataset))
    np.random.shuffle(index)

    # Calculate split points
    train_end = int(len(dataset) * ratios[0])
    val_end = int(len(dataset) * (ratios[0] + ratios[1]))
    cal_end = int(len(dataset) * (ratios[0] + ratios[1] + ratios[2]))

    train_index = index[:train_end]
    val_index = index[train_end:val_end]
    cal_index = index[val_end:cal_end]
    test_index = index[cal_end:]

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    cal_dataset = torch.utils.data.Subset(dataset, cal_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    return train_dataset, val_dataset, cal_dataset, test_dataset


def split_by_patient_conformal(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / cal / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset: four subsets
            of the dataset of type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, `cal_dataset.dataset`, and
            `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert len(ratios) == 4, "ratios must have 4 elements for train/val/cal/test"
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"

    patient_indx = list(dataset.patient_to_index.keys())
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)

    # Calculate split points
    train_end = int(num_patients * ratios[0])
    val_end = int(num_patients * (ratios[0] + ratios[1]))
    cal_end = int(num_patients * (ratios[0] + ratios[1] + ratios[2]))

    train_patient_indx = patient_indx[:train_end]
    val_patient_indx = patient_indx[train_end:val_end]
    cal_patient_indx = patient_indx[val_end:cal_end]
    test_patient_indx = patient_indx[cal_end:]

    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    cal_index = list(chain(*[dataset.patient_to_index[i] for i in cal_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    cal_dataset = torch.utils.data.Subset(dataset, cal_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    return train_dataset, val_dataset, cal_dataset, test_dataset


def split_by_sample_conformal(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / cal / test
        seed: random seed for shuffling the dataset
        get_index: if True, return indices instead of Subset objects

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset: four subsets
            of the dataset of type `torch.utils.data.Subset`, or four tensors
            of indices if get_index=True.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, `cal_dataset.dataset`, and
            `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert len(ratios) == 4, "ratios must have 4 elements for train/val/cal/test"
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"

    index = np.arange(len(dataset))
    np.random.shuffle(index)

    # Calculate split points
    train_end = int(len(dataset) * ratios[0])
    val_end = int(len(dataset) * (ratios[0] + ratios[1]))
    cal_end = int(len(dataset) * (ratios[0] + ratios[1] + ratios[2]))

    train_index = index[:train_end]
    val_index = index[train_end:val_end]
    cal_index = index[val_end:cal_end]
    test_index = index[cal_end:]

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(cal_index),
            torch.tensor(test_index),
        )
    else:
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        cal_dataset = torch.utils.data.Subset(dataset, cal_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        return train_dataset, val_dataset, cal_dataset, test_dataset
