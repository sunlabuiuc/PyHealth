"""Small tensor conversion helpers for medical image processors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional


def _pad_volumes(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pad equal-rank tensors to their maximum shape and stack them."""
    if not values:
        raise ValueError("Cannot collate an empty batch.")
    max_shape = [
        max(int(value.shape[axis]) for value in values)
        for axis in range(values[0].dim())
    ]
    padded = []
    for value in values:
        pad = []
        for axis in range(value.dim() - 1, -1, -1):
            pad.extend([0, max_shape[axis] - int(value.shape[axis])])
        padded.append(functional.pad(value, pad))
    return torch.stack(padded)


def _collate_volumes(values: Sequence[Any]) -> Any:
    """Stack equal-shape tensors or pad volumes to their maximum shape."""
    if not values or not isinstance(values[0], torch.Tensor):
        return values
    if all(value.shape == values[0].shape for value in values):
        return torch.stack(values)
    return _pad_volumes(values)


def _materialize_and_collate(processor: Any, values: Sequence[Any]) -> Any:
    """Materialise lazy values, then stack or pad them."""
    if values and getattr(processor, "lazy", False):
        values = [processor.materialize(value) for value in values]
    return _collate_volumes(values)


def _to_torch(array: np.ndarray, dtype: str = "float32") -> torch.Tensor:
    """Convert a NumPy array to a contiguous torch tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(getattr(torch, dtype))


def _channels_first(array: np.ndarray) -> torch.Tensor:
    """Convert ``(H, W)`` or ``(H, W, C)`` to a ``(C, H, W)`` tensor."""
    if array.ndim == 2:
        return torch.from_numpy(np.ascontiguousarray(array))[None]
    if array.ndim == 3:
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    raise ValueError(f"Expected a 2D or 3D array, found shape {array.shape}.")


def _resize_tensor(
    tensor: torch.Tensor,
    size: int | None,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a ``(C, H, W)`` tensor to ``(size, size)``."""
    if size is None or tensor.shape[-2:] == (size, size):
        return tensor
    align_corners = False if mode == "bilinear" else None
    resized = functional.interpolate(
        tensor.unsqueeze(0),
        size=(size, size),
        mode=mode,
        align_corners=align_corners,
    )
    return resized.squeeze(0)


def _normalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Normalise a ``(C, H, W)`` tensor with per-channel mean and std."""
    if tensor.shape[0] != len(mean) or tensor.shape[0] != len(std):
        raise ValueError(
            "Normalization mean/std length must match the image channel count."
        )
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``(1, H, W)`` or ``(3, H, W)`` tensor to an 8-bit image."""
    array = tensor.detach().cpu().float()
    low = float(array.min())
    high = float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    else:
        array = torch.zeros_like(array)
    array = (array.clamp(0.0, 1.0) * 255.0).byte().numpy()
    if array.ndim == 3:
        if array.shape[0] == 1:
            return Image.fromarray(array[0])
        if array.shape[0] in (3, 4):
            return Image.fromarray(np.transpose(array, (1, 2, 0)))
    raise ValueError(
        f"Expected 1, 3 or 4 channels, found shape {tuple(tensor.shape)}."
    )
