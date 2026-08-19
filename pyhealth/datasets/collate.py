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
import torch.nn.functional as F


def _pad_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Right-pad same-rank tensors to the per-dimension max, then stack.

    ``pad_sequence`` only pads dimension 0 and requires every trailing dimension
    to already match. Tokenized notes are ``(n_notes, seq_len)`` and, once the
    text processor pads to the longest note in a sample rather than to a fixed
    ``max_length``, BOTH dimensions vary across samples.
    """
    if len({t.dim() for t in tensors}) != 1:
        raise ValueError("cannot pad tensors of differing rank")
    target = [max(t.shape[d] for t in tensors) for d in range(tensors[0].dim())]
    padded = []
    for t in tensors:
        spec: list[int] = []
        for d in range(t.dim() - 1, -1, -1):
            spec.extend([0, target[d] - t.shape[d]])
        padded.append(F.pad(t, spec) if any(spec) else t)
    return torch.stack(padded)


def _stack_or_pad(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack if all shapes match; pad every ragged dimension otherwise."""
    if all(t.shape == tensors[0].shape for t in tensors):
        return torch.stack(tensors)
    return _pad_stack(tensors)


def _pad_mask(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Event-level validity for the tensor :func:`_stack_or_pad` just built.

    Batch padding is created here and nowhere else, so it has to be recorded
    here. A padded slot carries value 0.0 and time 0.0, which is
    indistinguishable from a real measurement taken at admission time, so a
    model given no mask treats padding as data.

    This is deliberately NOT called ``mask``. A field may carry its own
    ``{field}_mask`` meaning "was this value observed", which is a different
    question from "is this slot real".
    """
    lengths = torch.tensor([t.shape[0] for t in tensors])
    width = int(lengths.max())
    return torch.arange(width)[None, :] < lengths[:, None]


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
                    if sub_key == "time":
                        sub_result["pad_mask"] = _pad_mask(sub_vals)
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
