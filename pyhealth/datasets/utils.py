import hashlib
import os
from datetime import datetime
from typing import List, Tuple, Optional
import torch
from itertools import cycle

from dateutil.parser import parse as dateutil_parse
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


def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}


def get_dataloader(dataset, batch_size, shuffle=False):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict,
    )

    return dataloader


def collate_fn_kg_train(batch):
    positive_sample = torch.stack([d[0] for d in batch], dim=0)
    negative_sample = torch.stack([d[1] for d in batch], dim=0)
    subsample_weight = torch.cat([d[2] for d in batch], dim=0)
    mode = batch[0][3]
    
    return {
        "positive_sample": positive_sample,
        "negative_sample": negative_sample, 
        "subsample_weight": subsample_weight, 
        "mode": mode,
        "train": True
    }


def collate_fn_kg_test(batch):
    positive_sample = torch.stack([d[0] for d in batch], dim=0)
    negative_sample = torch.stack([d[1] for d in batch], dim=0)
    filter_bias = torch.stack([d[2] for d in batch], dim=0)
    mode = batch[0][3]
    
    return {
        "positive_sample": positive_sample,
        "negative_sample": negative_sample, 
        "filter_bias": filter_bias, 
        "mode": mode,
        "train": False
    }


def get_dataloader_kg(dataset, batch_size, train, shuffle=False):

    if train:

        dataloader_head = DataLoader(
            dataset['head_train'],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_kg_train,
        )

        dataloader_tail = DataLoader(
            dataset['tail_train'],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_kg_train,
        )
    
    # valid/test dataloader
    else:
        dataloader_head = DataLoader(
            dataset['head_test'],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_kg_test,
        )

        dataloader_tail = DataLoader(
            dataset['tail_test'],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_kg_test,
        )

    num_batch = len(dataloader_head) + len(dataloader_tail)

    return interleaved_dataloader(dataloader_head, dataloader_tail), num_batch


def interleaved_dataloader(dataloader1, dataloader2):
    dataloader1_iter = cycle(iter(dataloader1))
    dataloader2_iter = cycle(iter(dataloader2))

    while True:
        try:
            yield next(dataloader1_iter)
        except StopIteration:
            dataloader1_iter = cycle(iter(dataloader1))
            yield next(dataloader1_iter)

        try:
            yield next(dataloader2_iter)
        except StopIteration:
            dataloader2_iter = cycle(iter(dataloader2))
            yield next(dataloader2_iter)


if __name__ == "__main__":
    print(list_nested_levels([1, 2, 3]))
    print(list_nested_levels([1, [2], 3]))
    print(list_nested_levels([[1, [2], [[3]]]]))
    print(is_homo_list([1, 2, 3]))
    print(is_homo_list([1, 2, [3]]))
    print(is_homo_list([1, 2.0]))
