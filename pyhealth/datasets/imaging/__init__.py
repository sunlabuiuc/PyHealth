"""Dataset-level helpers for medical imaging.

This package holds imaging utilities that are not tied to one dataset loader:
the patch-based dataset view.
"""

from .patches import PatchDataset as PatchDataset

__all__ = ["PatchDataset"]
