from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .sample_dataset import SampleDataset

# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


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
    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore
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
    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore
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
    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore

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

    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    cal_dataset = dataset.subset(cal_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore

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

    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    cal_dataset = dataset.subset(cal_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore

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
        train_dataset = dataset.subset(train_index) # type: ignore
        val_dataset = dataset.subset(val_index) # type: ignore
        cal_dataset = dataset.subset(cal_index) # type: ignore
        test_dataset = dataset.subset(test_index) # type: ignore
        return train_dataset, val_dataset, cal_dataset, test_dataset
