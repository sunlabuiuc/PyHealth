from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor
from .image_io.readers.nifti import NiftiReader
from .image_io.tensor_utils import _materialize_and_collate, _to_torch


@register_processor("nifti")
class NiftiProcessor(FeatureProcessor):
    """Processor for NIfTI images and volumes.

    Args:
        lazy: Store the file path in the sample cache instead of the volume.
            Defaults to True; set False to cache tensors.
        channels: Expected channel count. If None, the reader infers it.
        canonical: Reorient to canonical RAS. Defaults to True.
        target_spacing: Target spacing in ``(D, H, W)`` order.
        target_shape: Target shape in ``(D, H, W)`` order.
        normalize: Intensity normalisation mode. ``"ct"`` and ``"mri"`` follow
            nnU-Net's CT and z-score schemes.
        clip: Optional ``(low, high)`` intensity clipping range.
        percentile_range: Percentiles for ``normalize="percentile"``.
        intensity_stats: Foreground statistics for ``normalize="ct"``.
        use_nonzero_mask: For ``normalize="mri"``, normalise only non-zero
            voxels. Defaults to True.
        mri_background_threshold: For ``normalize="mri"``, optional fraction
            of the 99th percentile below which voxels are treated as air/noise
            and set to zero. Defaults to None, which keeps all non-zero voxels.
        dtype: Output dtype. Defaults to ``"float32"``.
        is_mask: Treat the input as a segmentation mask.
        channel_axis: Channel axis for 4D NIfTI inputs. Only the last axis is
            supported, because the NIfTI affine describes the first three
            axes.
        order: Interpolation order. Defaults to 0 for masks and 1 otherwise.
        num_classes: Optional number of segmentation classes, used by models
            to configure the output head.

    Examples:
        >>> from pyhealth.processors import NiftiProcessor
        >>> processor = NiftiProcessor(lazy=True, target_shape=(32, 32, 32))
        >>> processor.stores_tensor
        False
        >>> processor.is_token()
        False
    """

    def __init__(
        self,
        lazy: bool = True,
        channels: int | None = None,
        canonical: bool = True,
        target_spacing: Sequence[float] | None = None,
        target_shape: Sequence[int] | None = None,
        normalize: str | None = None,
        clip: tuple[float, float] | None = None,
        percentile_range: Sequence[float] = (1.0, 99.0),
        intensity_stats: dict[str, float] | None = None,
        use_nonzero_mask: bool = True,
        mri_background_threshold: float | None = None,
        dtype: str = "float32",
        is_mask: bool = False,
        channel_axis: int = -1,
        order: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        """Initialize the NIfTI processor.

        Args:
            lazy: Store references instead of pixel tensors.
            channels: Expected channel count.
            canonical: Reorient to canonical RAS.
            target_spacing: Target spacing in ``(D, H, W)`` order.
            target_shape: Target shape in ``(D, H, W)`` order.
            normalize: Intensity normalisation mode.
            clip: Optional intensity clipping range.
            percentile_range: Percentiles for percentile normalisation.
            dtype: Output dtype.
            is_mask: Treat the input as a mask.
            channel_axis: Channel axis for 4D inputs. Only the last axis is
                supported.
            order: Interpolation order.
        """
        if normalize == "auto":
            raise ValueError(
                "NiftiProcessor does not infer a modality for "
                "normalize='auto'; use 'ct' or 'mri' explicitly."
            )
        self.lazy = bool(lazy)
        self.channels = channels
        self.canonical = bool(canonical)
        self.target_spacing = (
            None if target_spacing is None else tuple(target_spacing)
        )
        self.target_shape = None if target_shape is None else tuple(target_shape)
        self.normalize = normalize
        self.clip = clip
        self.percentile_range = tuple(percentile_range)
        self.intensity_stats = (
            None if intensity_stats is None else dict(intensity_stats)
        )
        self.use_nonzero_mask = bool(use_nonzero_mask)
        self.mri_background_threshold = mri_background_threshold
        self.dtype = dtype
        self.is_mask = bool(is_mask)
        self.channel_axis = channel_axis
        self.order = order
        self.num_classes = num_classes

    @property
    def stores_tensor(self) -> bool:
        """Whether the processed cache stores a tensor (not a reference)."""
        return not self.lazy

    def process(self, value: str | Path) -> Any:
        """Process a NIfTI path.

        Args:
            value: File path.

        Returns:
            An absolute path string in lazy mode, otherwise a tensor.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {path}")
        reader = NiftiReader()
        if self.lazy:
            # Validate the header at task-build time, not mid-training.
            reader.read_metadata(
                str(path),
                canonical=self.canonical,
                channel_axis=self.channel_axis,
            )
            return str(path.resolve())
        array = reader.load(str(path), **self._reader_kwargs())
        return _to_torch(array, dtype=self._output_dtype())

    def materialize(self, value: str | Path) -> Any:
        """Load a lazy path and apply this processor's settings.

        Args:
            value: Path returned by :meth:`process`.

        Returns:
            A tensor with shape ``(C, H, W)`` or ``(C, D, H, W)``.
        """
        array = NiftiReader().load(str(value), **self._reader_kwargs())
        return _to_torch(array, dtype=self._output_dtype())

    def _output_dtype(self) -> str:
        """Return the tensor dtype, preserving integer masks."""
        return "int64" if self.is_mask else self.dtype

    def _reader_kwargs(self) -> dict[str, Any]:
        """Return options forwarded to the NIfTI reader."""
        return {
            "channels": self.channels,
            "canonical": self.canonical,
            "target_spacing": self.target_spacing,
            "target_shape": self.target_shape,
            "normalize": self.normalize,
            "clip": self.clip,
            "percentile_range": self.percentile_range,
            "intensity_stats": getattr(self, "intensity_stats", None),
            "use_nonzero_mask": getattr(self, "use_nonzero_mask", True),
            "mri_background_threshold": getattr(
                self, "mri_background_threshold", None
            ),
            "dtype": self.dtype,
            "is_mask": self.is_mask,
            "channel_axis": self.channel_axis,
            "order": self.order,
        }

    def collate(self, values: list[Any]) -> Any:
        """Materialise lazy refs, then pad and stack volumes."""
        return _materialize_and_collate(self, values)

    def is_token(self) -> bool:
        """Return False because image intensities are continuous."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Return the single-tensor output schema."""
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Return the output tensor rank."""
        return (4,)

    def spatial(self) -> tuple[bool, ...]:
        """Return which axes are spatial."""
        return (False, True, True, True)

    def __repr__(self) -> str:
        return (
            f"NiftiProcessor(lazy={self.lazy}, channels={self.channels}, "
            f"target_spacing={self.target_spacing}, "
            f"target_shape={self.target_shape}, normalize={self.normalize}, "
            f"is_mask={self.is_mask})"
        )
