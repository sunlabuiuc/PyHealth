import hashlib
import os
from datetime import datetime
from typing import List
from typing import Optional

from dateutil.parser import parse as dateutil_parse
from torch.utils.data import DataLoader

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory

MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "datasets")
create_directory(MODULE_CACHE_PATH)


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


def list_nested_level(l: List) -> int:
    """Gets the nested level of a list.

    Args:
        l: List, the list to be checked.

    Returns:
        int, the nested level of the list.

    Examples:
        >>> list_nested_level([1, 2, 3])
        1
        >>> list_nested_level([[1, 2, 3], [4, 5, 6]])
        2
        >>> list_nested_level([1, [2, 3], 4])
        2
    """
    if not isinstance(l, list):
        return 0
    if not l:
        return 1
    return 1 + max(list_nested_level(i) for i in l)


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
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_dict
    )
    return dataloader


if __name__ == "__main__":
    print(list_nested_level([1, 2, 3]))
    print(list_nested_level([1, [2], 3]))
    print(list_nested_level([1, [2], [[3]]]))
    print(is_homo_list([1, 2, 3]))
    print(is_homo_list([1, 2, [3]]))
    print(is_homo_list([1, 2.0]))
