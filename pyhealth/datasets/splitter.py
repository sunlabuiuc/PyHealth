from itertools import chain
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from pyhealth.datasets import SampleBaseDataset


# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


def split_by_visit(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleBaseDataset` object
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
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleBaseDataset` object
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
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample

    Args:
        dataset: a `SampleBaseDataset` object
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
                int(len(dataset) * ratios[0]): int(
                    len(dataset) * (ratios[0] + ratios[1]))
                ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])):]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    
    if get_index:
        return torch.tensor(train_index), torch.tensor(val_index), torch.tensor(test_index)
    else:
        return train_dataset, val_dataset, test_dataset