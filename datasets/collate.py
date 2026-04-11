"""Universal collation utilities for TemporalFeatureProcessor outputs.

Because every ``TemporalFeatureProcessor.process_temporal()`` returns a
``dict[str, Tensor]``, batching is trivial: stack/pad each key independently.
No per-processor custom collation logic is needed.

Usage::

    from pyhealth.datasets.collate import collate_temporal
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_temporal)

"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def _stack_or_pad(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack if all shapes match; pad along dim-0 otherwise."""
    if all(t.shape == tensors[0].shape for t in tensors):
        return torch.stack(tensors)
    return pad_sequence(tensors, batch_first=True)


def collate_temporal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Universal collator for datasets that contain ``TemporalFeatureProcessor``
    dict outputs alongside ordinary tensors and labels.

    Handling rules (applied recursively for nested dicts):

    * ``Tensor``         — stack if same shape, pad to longest otherwise
    * ``dict[str, ...]`` — recursively collate each sub-key  *(temporal feature)*
    * ``int / float``    — ``torch.tensor(...)``
    * anything else      — kept as a plain Python list

    Args:
        batch: List of sample dicts as returned by the DataLoader's dataset.

    Returns:
        A single collated dict ready for model ``forward(**batch)``.
    """
    if not batch:
        return {}

    result: dict[str, Any] = {}

    for key in batch[0]:
        vals = [s[key] for s in batch]
        first = vals[0]

        if isinstance(first, dict):
            # ── Temporal feature dict — collate each sub-key ─────────────
            sub_result: dict[str, Any] = {}
            for sub_key in first:
                sub_vals = [v[sub_key] for v in vals]
                if sub_vals[0] is None:
                    sub_result[sub_key] = [None] * len(sub_vals)
                elif isinstance(sub_vals[0], torch.Tensor):
                    sub_result[sub_key] = _stack_or_pad(sub_vals)
                else:
                    sub_result[sub_key] = sub_vals
            result[key] = sub_result

        elif isinstance(first, torch.Tensor):
            result[key] = _stack_or_pad(vals)

        elif isinstance(first, (int, float)):
            result[key] = torch.tensor(vals)

        else:
            result[key] = vals

    return result
