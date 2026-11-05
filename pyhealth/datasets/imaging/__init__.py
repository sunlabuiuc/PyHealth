"""Dataset-level helpers for medical imaging.

This package holds imaging utilities that are not tied to one dataset loader:
file download/archive helpers, the patch-based dataset view, and foreground
intensity statistics.
"""

from .patches import PatchDataset as PatchDataset
from .stats import (
    compute_foreground_intensity_stats as compute_foreground_intensity_stats,
)

__all__ = ["PatchDataset", "compute_foreground_intensity_stats"]
