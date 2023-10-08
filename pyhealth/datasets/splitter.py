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
    balanced: bool = False,
    pos_ids: Optional[List[str]] = None,
    neg_ids: Optional[List[str]] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset
        balanced: whether to balance the postive/negative samples in the val/test set
        pos_ids: a list of patient ids that are positive
        neg_ids: a list of patient ids that are negative

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
    np.random.shuffle(patient_indx)
    
    if balanced and (pos_ids is None or neg_ids is None):
            raise ValueError("pos_ids and neg_ids must be provided when balanced is True")
    
    if hasattr(dataset, "pos_neg_labels"):
        balanced = True
        pos_ids = dataset.pos_neg_labels[0]
        neg_ids = dataset.pos_neg_labels[1]
    else:
        balanced = False
        pos_ids = None
        neg_ids = None
        
    if balanced:
        # Separate positive and negative patient indices
        pos_patient_indx = [i for i in patient_indx if i in pos_ids]
        neg_patient_indx = [i for i in patient_indx if i in neg_ids]

        # Allocate positive and negative samples to test and validation sets
        test_pos_indx = np.random.choice(pos_patient_indx, int(len(pos_patient_indx) * ratios[2]), replace=False)
        val_pos_indx = np.random.choice(list(set(pos_patient_indx) - set(test_pos_indx)), int(len(pos_patient_indx) * ratios[1]), replace=False)
        
        test_neg_indx = np.random.choice(neg_patient_indx, int(len(neg_patient_indx) * ratios[2]), replace=False)
        val_neg_indx = np.random.choice(list(set(neg_patient_indx) - set(test_neg_indx)), int(len(neg_patient_indx) * ratios[1]), replace=False)

        # Combine positive and negative samples for test and validation sets
        test_patient_indx = list(test_pos_indx) + list(test_neg_indx)
        val_patient_indx = list(val_pos_indx) + list(val_neg_indx)

        # Remaining samples are used for training
        train_patient_indx = list(set(patient_indx) - set(test_patient_indx) - set(val_patient_indx))

    else:
        # If not balancing, simply split the data according to the ratios
        num_patients = len(patient_indx)
        train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
        val_patient_indx = patient_indx[
            int(num_patients * ratios[0]) : int(num_patients * (ratios[0] + ratios[1]))
        ]
        test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])) :]

    # Get sample indices for each split
    train_index = list(chain(*[dataset.patient_to_index[i] for i in train_patient_indx]))
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    # Create data subsets
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    return train_dataset, val_dataset, test_dataset