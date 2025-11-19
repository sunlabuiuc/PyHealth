"""Dataset processing module.

This module provides different processing modes for PyHealth datasets:
- normal: Traditional in-memory processing for smaller datasets
- streaming: Disk-backed streaming processing for large datasets

These are internal implementation details and should not be imported directly
by users. The public API is through BaseDataset.set_task().
"""

from .normal import set_task_normal
from .streaming import (
    build_patient_cache,
    iter_patients_streaming,
    set_task_streaming,
    setup_streaming_cache,
    _create_patients_from_dataframe,
)

__all__ = [
    "set_task_normal",
    "set_task_streaming",
    "setup_streaming_cache",
    "build_patient_cache",
    "iter_patients_streaming",
    "_create_patients_from_dataframe",
]
