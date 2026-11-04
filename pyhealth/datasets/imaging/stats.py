"""Utilities for segmentation datasets.

The most important helper here is
:func:`compute_foreground_intensity_stats`, which mirrors nnU-Net's dataset
fingerprinting: it collects image intensities inside the segmentation
foreground and reports the percentiles and mean/std that CT normalisation
uses.

Examples:
    >>> import numpy as np
    >>> from pyhealth.datasets.imaging.stats import (
    ...     compute_foreground_intensity_stats,
    ... )
    >>> samples = [
    ...     {
    ...         "image": np.array([[-1000.0, 10.0], [20.0, 30.0]]),
    ...         "mask": np.array([[0, 1], [1, 1]]),
    ...     }
    ... ]
    >>> stats = compute_foreground_intensity_stats(samples)
    >>> sorted(stats)
    ['mean', 'percentile_00_5', 'percentile_99_5', 'std']
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like value to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_value(value: Any) -> np.ndarray:
    """Load an ``ImageRef`` or convert an already-materialised value."""
    if hasattr(value, "load") and hasattr(value, "reader"):
        value = value.load()
    return _to_numpy(value)


def compute_foreground_intensity_stats(
    dataset: Iterable[dict[str, Any]],
    input_key: str = "image",
    label_key: str = "mask",
    modality: str | None = None,
    max_cases: int | None = None,
    max_voxels: int = 1_000_000,
    seed: int = 0,
    raw: bool = True,
) -> dict[str, float]:
    """Collect foreground intensity statistics for CT normalisation.

    This follows nnU-Net's ``foreground_intensity_properties_per_channel``:
    only voxels where ``mask > 0`` are used. Compute the statistics on the
    training split, not on validation or test data.

    Args:
        dataset: Iterable of sample dictionaries, typically a ``SampleDataset``.
        input_key: Key of the image tensor or array.
        label_key: Key of the integer segmentation mask.
        modality: Optional modality filter. Samples whose ``modality`` field is
            present and different are skipped. Use this to compute statistics
            for CT only in a mixed CT/MR dataset such as CHAOS.
        max_cases: Optional limit on the number of cases to read.
        max_voxels: Maximum number of foreground voxels kept in memory.
        seed: Seed for the final subsampling step.
        raw: Read raw image values through lazy ``ImageRef`` objects instead of
            using the processed samples. This is important because CT
            normalisation must be estimated from raw Hounsfield units, not
            from already-normalised tensors. Defaults to True.

    Returns:
        A dictionary with ``percentile_00_5``, ``percentile_99_5``, ``mean``
        and ``std``.

    Raises:
        ValueError: If no sample contains a foreground voxel.

    Examples:
        >>> import numpy as np
        >>> samples = [
        ...     {"image": np.array([0.0, 10.0, 20.0]),
        ...      "mask": np.array([0, 1, 1])}
        ... ]
        >>> stats = compute_foreground_intensity_stats(samples)
        >>> round(stats["mean"], 1)
        15.0
    """
    values: list[np.ndarray] = []
    seen = 0
    restore_refs = False
    if raw and hasattr(dataset, "_resolve_refs") and dataset._resolve_refs:
        # Lazy datasets expose raw ImageRef objects when resolve_refs is off.
        # CT statistics must come from raw Hounsfield units, not from tensors
        # that the task has already normalised.
        dataset._resolve_refs = False
        restore_refs = True
    try:
        for sample in dataset:
            if max_cases is not None and seen >= max_cases:
                break
            sample_modality = sample.get("modality")
            if (
                modality is not None
                and sample_modality is not None
                and str(sample_modality).upper() != str(modality).upper()
            ):
                continue
            image = _load_value(sample[input_key])
            mask = _load_value(sample[label_key])
            if mask.ndim == image.ndim:
                # Both have a channel axis: keep the mask channel dimension
                # for broadcasting against the image channels.
                mask = mask.astype(bool, copy=False)
            elif mask.ndim + 1 == image.ndim:
                mask = mask[None].astype(bool, copy=False)
            elif mask.ndim == image.ndim + 1 and mask.shape[0] == 1:
                mask = mask[0].astype(bool, copy=False)
            else:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match image shape "
                    f"{image.shape}."
                )
            foreground = np.broadcast_to(mask, image.shape)
            sample_values = image[foreground]
            if sample_values.size:
                values.append(sample_values.astype(np.float32, copy=False))
            seen += 1
    finally:
        if restore_refs:
            dataset._resolve_refs = True

    if not values:
        raise ValueError(
            "No foreground voxels found. Check the mask values and label_key."
        )

    combined = np.concatenate(values)
    if combined.size > max_voxels:
        rng = np.random.default_rng(seed)
        indices = rng.choice(combined.size, size=max_voxels, replace=False)
        combined = combined[indices]

    lower, upper = np.percentile(combined, [0.5, 99.5])
    return {
        "percentile_00_5": float(lower),
        "percentile_99_5": float(upper),
        "mean": float(combined.mean()),
        "std": float(combined.std()),
    }
