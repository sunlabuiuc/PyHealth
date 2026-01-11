"""Utilities for SDOH ICD-9 V-code detection tasks."""

import logging
from typing import Iterable, Sequence, Set

import pandas as pd
import torch

logger = logging.getLogger(__name__)


# Standard SDOH ICD-9 V-codes
TARGET_CODES = [
    "V600",
    "V602",
    "V604",
    "V620",
    "V625",
    "V1541",
    "V1542",
    "V1584",
    "V6141",
]


def parse_codes(codes_str: object, target_codes: Sequence[str]) -> Set[str]:
    """Parse ICD-9 codes from various string formats.

    Args:
        codes_str: String representation of codes (comma/semicolon separated)
        target_codes: Valid target codes to filter for

    Returns:
        Set of valid codes found in the string
    """
    if pd.isna(codes_str) or str(codes_str).strip() == "":
        return set()

    # Clean string
    codes = (
        str(codes_str)
        .replace("[", "")
        .replace("]", "")
        .replace('"', "")
        .replace("'", "")
    )

    # Split by delimiter
    if "," in codes:
        values = [c.strip() for c in codes.split(",")]
    elif ";" in codes:
        values = [c.strip() for c in codes.split(";")]
    else:
        values = [codes.strip()]

    # Filter to valid target codes
    target_set = {code.upper() for code in target_codes}
    parsed = {value.upper() for value in values if value.strip()}
    return {code for code in parsed if code in target_set}


def codes_to_multihot(codes: Iterable[str], target_codes: Sequence[str]) -> torch.Tensor:
    """Convert code set to multi-hot encoding.

    Args:
        codes: Iterable of code strings
        target_codes: Ordered list of target codes

    Returns:
        Multi-hot tensor aligned with target_codes
    """
    code_set = {code.upper() for code in codes}
    return torch.tensor(
        [1.0 if code in code_set else 0.0 for code in target_codes],
        dtype=torch.float32,
    )
