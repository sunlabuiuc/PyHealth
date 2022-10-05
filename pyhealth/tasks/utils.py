from typing import List

from pyhealth.data import Event


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
    list_of_code = [event.code for event in list_of_event]
    if remove_duplicate is True:
        # remove duplicate codes but keep the order
        list_of_code = list(dict.fromkeys(list_of_code))
    return list_of_code
