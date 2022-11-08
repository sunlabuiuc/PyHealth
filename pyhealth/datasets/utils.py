import os
from datetime import datetime
from typing import Optional
import hashlib
from dateutil.parser import parse as dateutil_parse

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
