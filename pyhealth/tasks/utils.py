from typing import List

from pyhealth.data import Event
from datetime import datetime


def get_code_from_list_of_event(
    list_of_event: List[Event],
    remove_duplicate: bool = True,
) -> List[str]:
    """
    Args:
        list_of_event: List[Event], a list of Event objects (e.g., conditions, procedures, drugs)
        remove_duplicate: whether to remove duplicate codes (but keep the order)
    Returns
        list_of_code: List[str], a list of codes (e.g., ICD9, ICD10, etc.)
    """
    list_of_code = [
        event.code
        for event in sorted(
            list_of_event,
            key=lambda event: event.timestamp if event.timestamp else 0,
            reverse=False,
        )
    ]
    if remove_duplicate is True:
        # remove duplicate codes but keep the order
        list_of_code = list(dict.fromkeys(list_of_code))
    return list_of_code


def datetime_string_to_datetime(datetime_string: str) -> datetime:
    """Converts a datetime string to a datetime object.

    Args:
        datetime_string: str, datetime string.

    Returns:
        datetime, datetime object.
    """

    if ":" in datetime_string:
        return datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
    else:
        return datetime.strptime(datetime_string, "%Y-%m-%d")
