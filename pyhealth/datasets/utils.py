import hashlib
import os
import pickle
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from dateutil.parser import parse as dateutil_parse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory

MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "datasets")
create_directory(MODULE_CACHE_PATH)


# basic tables which are a part of the defined datasets
DATASET_BASIC_TABLES = {
    "MIMIC3Dataset": {"PATIENTS", "ADMISSIONS"},
    "MIMIC4Dataset": {"patients", "admission"},
}


def hash_str(s):
    return hashlib.md5(s.encode()).hexdigest()


def strptime(s: str) -> Optional[datetime]:
    """Helper function which parses a string to datetime object.

    Args:
        s: str, string to be parsed.

    Returns:
        Optional[datetime], parsed datetime object. If s is nan, return None.
    """
    # return None if s is nan
    if s != s:
        return None
    return dateutil_parse(s)

def padyear(year: str, month='1', day='1') -> str:
    """Pad a date time year of format 'YYYY' to format 'YYYY-MM-DD'
    
    Args: 
        year: str, year to be padded. Must be non-zero value.
        month: str, month string to be used as padding. Must be in [1, 12]
        day: str, day string to be used as padding. Must be in [1, 31]
        
    Returns:
        padded_date: str, padded year.
    
    """
    return f"{year}-{month}-{day}"

def flatten_list(l: List) -> List:
    """Flattens a list of list.

    Args:
        l: List, the list of list to be flattened.

    Returns:
        List, the flattened list.

    Examples:
        >>> flatten_list([[1], [2, 3], [4]])
        [1, 2, 3, 4]R
        >>> flatten_list([[1], [[2], 3], [4]])
        [1, [2], 3, 4]
    """
    assert isinstance(l, list), "l must be a list."
    return sum(l, [])


def list_nested_levels(l: List) -> Tuple[int]:
    """Gets all the different nested levels of a list.

    Args:
        l: the list to be checked.

    Returns:
        All the different nested levels of the list.

    Examples:
        >>> list_nested_levels([])
        (1,)
        >>> list_nested_levels([1, 2, 3])
        (1,)
        >>> list_nested_levels([[]])
        (2,)
        >>> list_nested_levels([[1, 2, 3], [4, 5, 6]])
        (2,)
        >>> list_nested_levels([1, [2, 3], 4])
        (1, 2)
        >>> list_nested_levels([[1, [2, 3], 4]])
        (2, 3)
    """
    if not isinstance(l, list):
        return tuple([0])
    if not l:
        return tuple([1])
    levels = []
    for i in l:
        levels.extend(list_nested_levels(i))
    levels = [i + 1 for i in levels]
    return tuple(set(levels))


def is_homo_list(l: List) -> bool:
    """Checks if a list is homogeneous.

    Args:
        l: the list to be checked.

    Returns:
        bool, True if the list is homogeneous, False otherwise.

    Examples:
        >>> is_homo_list([1, 2, 3])
        True
        >>> is_homo_list([])
        True
        >>> is_homo_list([1, 2, "3"])
        False
        >>> is_homo_list([1, 2, 3, [4, 5, 6]])
        False
    """
    if not l:
        return True

    # if the value vector is a mix of float and int, convert all to float
    l = [float(i) if type(i) == int else i for i in l]
    return all(isinstance(i, type(l[0])) for i in l)


def collate_fn_dict(batch: List[dict]) -> dict:
    """Collates a batch of data into a dictionary of lists.

    Args:
        batch: List of dictionaries, where each dictionary represents a data sample.

    Returns:
        A dictionary where each key corresponds to a list of values from the batch.
    """
    return {key: [d[key] for d in batch] for key in batch[0]}


def collate_fn_dict_with_padding(batch: List[dict]) -> dict:
    """Collates a batch of data into a dictionary with padding for tensor values.

    Args:
        batch: List of dictionaries, where each dictionary represents a data sample.

    Returns:
        A dictionary where each key corresponds to a list of values from the batch.
        Tensor values are padded to the same shape.
    """
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            # Check if shapes are the same
            shapes = [v.shape for v in values]
            if all(shape == shapes[0] for shape in shapes):
                # Same shape, just stack
                collated[key] = torch.stack(values)
            else:
                # Variable shapes, pad
                if values[0].dim() == 0:
                    # Scalars, treat as stackable
                    collated[key] = torch.stack(values)
                elif values[0].dim() >= 1:
                    collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
                else:
                    raise ValueError(f"Unsupported tensor shape: {values[0].shape}")
        else:
            # Non-tensor data: keep as list
            collated[key] = values

    return collated


def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Creates a DataLoader for a given dataset.

    Args:
        dataset: The dataset to load data from.
        batch_size: The number of samples per batch.
        shuffle: Whether to shuffle the data at every epoch.

    Returns:
        A DataLoader instance for the dataset.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict_with_padding,
    )

    return dataloader


if __name__ == "__main__":
    print(list_nested_levels([1, 2, 3]))
    print(list_nested_levels([1, [2], 3]))
    print(list_nested_levels([[1, [2], [[3]]]]))
    print(is_homo_list([1, 2, 3]))
    print(is_homo_list([1, 2, [3]]))
    print(is_homo_list([1, 2.0]))
