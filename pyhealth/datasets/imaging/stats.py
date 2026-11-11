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
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like value to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _is_lazy_value(value: Any) -> bool:
    """Whether a processed field holds lazy references instead of pixels."""
    if isinstance(value, (str, Path, dict)):
        return True
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, Path)) for item in value
    )


def _load_value(value: Any, processor: Any = None) -> np.ndarray:
    """Materialise a lazy value with its processor, or convert an array."""
    if _is_lazy_value(value):
        if processor is None or not hasattr(processor, "materialize"):
            raise TypeError(
                "Lazy image values need a dataset processor with "
                "materialize(); pass a SampleDataset or materialise the "
                "value first."
            )
        value = processor.materialize(value)
    return _to_numpy(value)


def _update_reservoir(
    reservoir: np.ndarray,
    values: np.ndarray,
    seen: int,
    max_voxels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Add foreground values to a fixed-size reservoir."""
    if values.size == 0:
        return reservoir, seen
    remaining = max_voxels - reservoir.size
    if remaining > 0:
        take = min(int(values.size), remaining)
        reservoir = np.concatenate((reservoir, values[:take]))
        values = values[take:]
        seen += take
    if values.size:
        # Standard reservoir sampling: item ``t`` (1-based) enters the
        # reservoir with probability ``max_voxels / t`` and replaces a random
        # slot. This keeps memory bounded while preserving a uniform sample.
        totals = np.arange(seen + 1, seen + values.size + 1, dtype=np.float64)
        selected = rng.random(values.size) < (max_voxels / totals)
        if selected.any():
            slots = rng.integers(0, max_voxels, size=int(selected.sum()))
            reservoir[slots] = values[selected]
        seen += int(values.size)
    return reservoir, seen


def compute_foreground_intensity_stats(
    dataset: Iterable[dict[str, Any]],
    input_key: str = "image",
    label_key: str = "mask",
    modality: str | None = None,
    max_cases: int | None = None,
    max_voxels: int = 1_000_000,
    seed: int = 0,
) -> dict[str, float]:
    """Collect foreground intensity statistics for CT normalisation.

    This follows nnU-Net's ``foreground_intensity_properties_per_channel``:
    only voxels where ``mask > 0`` are used. Compute the statistics on the
    training split, not on validation or test data.

    Lazy values are loaded through the dataset's own processors with their
    public ``materialize()`` hook, the same hook ``PatchDataset`` uses, so the
    statistics follow each processor's reader and normalization settings.

    Args:
        dataset: Iterable of sample dictionaries, typically a ``SampleDataset``.
        input_key: Key of the image tensor or array.
        label_key: Key of the integer segmentation mask.
        modality: Optional modality filter. Samples whose ``modality`` field is
            present and different are skipped. Use this to compute statistics
            for CT only in a mixed CT/MR dataset such as CHAOS.
        max_cases: Optional limit on the number of cases to read.
        max_voxels: Maximum number of foreground voxels kept in memory.
            Larger inputs are reservoir-sampled while iterating.
        seed: Seed for reservoir sampling.

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
    if max_voxels < 1:
        raise ValueError("max_voxels must be at least 1.")
    reservoir = np.empty(0, dtype=np.float32)
    seen_voxels = 0
    rng = np.random.default_rng(seed)
    seen = 0
    processors = {
        **getattr(dataset, "input_processors", {}),
        **getattr(dataset, "output_processors", {}),
    }
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
        image = _load_value(sample[input_key], processors.get(input_key))
        mask = _load_value(sample[label_key], processors.get(label_key))
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
        reservoir, seen_voxels = _update_reservoir(
            reservoir,
            sample_values.astype(np.float32, copy=False),
            seen_voxels,
            max_voxels,
            rng,
        )
        seen += 1

    if reservoir.size == 0:
        raise ValueError(
            "No foreground voxels found. Check the mask values and label_key."
        )

    combined = reservoir
    lower, upper = np.percentile(combined, [0.5, 99.5])
    return {
        "percentile_00_5": float(lower),
        "percentile_99_5": float(upper),
        "mean": float(combined.mean()),
        "std": float(combined.std()),
    }
