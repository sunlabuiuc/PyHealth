from itertools import chain
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from pyhealth.datasets import SampleBaseDataset


def split(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by its outermost indexed items

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
    train_dataset = [{**train_dataset[i], **{'train': True, 'hyperparameters': dataset.task_spec_param}} for i in range(len(train_dataset))]

    val_dataset = torch.utils.data.Subset(dataset, val_index)
    val_dataset = [{**val_dataset[i], 'train': False} for i in range(len(val_dataset))]
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    test_dataset = [{**test_dataset[i], 'train': False} for i in range(len(test_dataset))]
    return train_dataset, val_dataset, test_dataset