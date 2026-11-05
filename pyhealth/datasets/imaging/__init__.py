"""Dataset-level helpers for medical imaging.

This package holds imaging utilities that are not tied to one dataset loader:
file download/archive helpers and foreground intensity statistics.
"""

from .stats import (
    compute_foreground_intensity_stats as compute_foreground_intensity_stats,
)

__all__ = ["compute_foreground_intensity_stats"]
