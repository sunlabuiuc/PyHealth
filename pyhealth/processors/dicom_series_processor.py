from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor
from .image_io.readers.dicom import DicomSeriesReader
from .image_io.tensor_utils import _materialize_and_collate, _to_torch


@register_processor("dicom_series")
class DicomSeriesProcessor(FeatureProcessor):
    """Processor for a directory or list of DICOM slices.

    Args:
        lazy: Store the ordered slice paths instead of the volume. Defaults
            to True; set False to cache tensors.
        channels: Expected channel count. DICOM series currently produce one
            channel.
        canonical: Reorient to canonical RAS. Defaults to True.
        target_spacing: Target spacing in ``(D, H, W)`` order.
        target_shape: Target shape in ``(D, H, W)`` order.
        normalize: Intensity normalisation mode.
        clip: Optional ``(low, high)`` intensity clipping range.
        percentile_range: Percentiles for ``normalize="percentile"``.
        dtype: Output dtype. Defaults to ``"float32"``.
        is_mask: Treat the series as a mask.
        order: Interpolation order. Defaults to 0 for masks and 1 otherwise.
        series_uid: Series to select when a directory contains several.
        apply_rescale: Apply per-slice rescale values. Defaults to True.
        invert_monochrome1: Invert ``MONOCHROME1`` slices. Defaults to True.
        window: Apply each slice's DICOM window. Defaults to False.

    Examples:
        >>> from pyhealth.processors import DicomSeriesProcessor
        >>> processor = DicomSeriesProcessor(lazy=True, target_shape=(64, 64, 64))
        >>> processor.stores_tensor
        False
        >>> processor.is_token()
        False
    """

    def __init__(
        self,
        lazy: bool = True,
        channels: int | None = 1,
        canonical: bool = True,
        target_spacing: Sequence[float] | None = None,
        target_shape: Sequence[int] | None = None,
        normalize: str | None = None,
        clip: tuple[float, float] | None = None,
        percentile_range: Sequence[float] = (1.0, 99.0),
        intensity_stats: dict[str, float] | None = None,
        use_nonzero_mask: bool = True,
        mri_background_threshold: float | None = 0.05,
        dtype: str = "float32",
        is_mask: bool = False,
        order: int | None = None,
        series_uid: str | None = None,
        apply_rescale: bool = True,
        invert_monochrome1: bool = True,
        window: bool = False,
        num_classes: int | None = None,
    ) -> None:
        """Initialize the DICOM series processor.

        Args:
            lazy: Store references instead of volume tensors.
            channels: Expected channel count.
            canonical: Reorient to canonical RAS.
            target_spacing: Target spacing in ``(D, H, W)`` order.
            target_shape: Target shape in ``(D, H, W)`` order.
            normalize: Intensity normalisation mode. ``"auto"`` selects
                ``"ct"`` or ``"mri"`` from the DICOM modality.
            clip: Optional intensity clipping range.
            percentile_range: Percentiles for percentile normalisation.
            intensity_stats: Foreground statistics for ``normalize="ct"``.
            use_nonzero_mask: For ``normalize="mri"``, normalise only non-zero
                voxels. Defaults to True.
            mri_background_threshold: For ``normalize="mri"``, fraction of the
                99th percentile below which voxels are treated as air/noise
                and set to zero. Defaults to 0.05 for DICOM MRI because raw
                DICOM series usually include an air background.
                ``NiftiProcessor`` defaults to None because NIfTI volumes are
                often already skull-stripped.
            dtype: Output dtype.
            is_mask: Treat the input as a mask.
            order: Interpolation order.
            series_uid: Series UID to select.
            apply_rescale: Apply per-slice rescale values.
            invert_monochrome1: Invert MONOCHROME1 slices.
            window: Apply each slice's DICOM window.
            num_classes: Optional number of segmentation classes, used by
                models to configure the output head.
        """
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
        self.order = order
        self.series_uid = series_uid
        self.apply_rescale = bool(apply_rescale)
        self.invert_monochrome1 = bool(invert_monochrome1)
        self.window = bool(window)
        self.num_classes = num_classes

    @property
    def stores_tensor(self) -> bool:
        """Whether the processed cache stores a tensor (not a reference)."""
        return not self.lazy

    def process(self, value: str | Path | Sequence[str]) -> Any:
        """Process a DICOM series path.

        Args:
            value: Directory or file list.

        Returns:
            An ordered list of absolute slice paths in lazy mode, otherwise a
            tensor.

        Raises:
            FileNotFoundError: If a path does not exist.
        """
        if isinstance(value, (str, Path)):
            source: str | Path = value
            if not Path(value).exists():
                raise FileNotFoundError(f"DICOM series not found: {value}")
        else:
            source = tuple(str(item) for item in value)
            for item in source:
                if not Path(item).exists():
                    raise FileNotFoundError(f"DICOM slice not found: {item}")
        reader = DicomSeriesReader()
        if self.lazy:
            metadata = reader.read_metadata(
                source,
                series_uid=self.series_uid,
                canonical=self.canonical,
            )
            # Validate and record slice order once; materialize trusts it.
            return [
                str(Path(item).resolve()) for item in metadata["source"]
            ]
        array = reader.load(source, **self._reader_kwargs())
        return self._to_tensor(array)

    def materialize(self, value: Sequence[str]) -> Any:
        """Load lazy slice paths and apply this processor's settings.

        Args:
            value: Ordered slice paths returned by :meth:`process`.

        Returns:
            A tensor with shape ``(1, D, H, W)``.
        """
        array = DicomSeriesReader().load(
            list(value),
            sorted_paths=True,
            **self._reader_kwargs(),
        )
        return self._to_tensor(array)

    def _reader_kwargs(self) -> dict[str, Any]:
        """Return options forwarded to the DICOM series reader."""
        return {
            "series_uid": self.series_uid,
            "canonical": self.canonical,
            "target_spacing": self.target_spacing,
            "target_shape": self.target_shape,
            "normalize": self.normalize,
            "clip": self.clip,
            "percentile_range": self.percentile_range,
            "intensity_stats": getattr(self, "intensity_stats", None),
            "use_nonzero_mask": getattr(self, "use_nonzero_mask", True),
            "mri_background_threshold": getattr(
                self, "mri_background_threshold", 0.05
            ),
            "dtype": self.dtype,
            "is_mask": self.is_mask,
            "order": self.order,
            "apply_rescale": self.apply_rescale,
            "invert_monochrome1": self.invert_monochrome1,
            "window": self.window,
        }

    def _to_tensor(self, array: Any) -> Any:
        """Convert a reader array to a tensor and validate channels."""
        if self.channels is not None and array.shape[0] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, found {array.shape[0]}."
            )
        dtype = "int64" if self.is_mask else self.dtype
        return _to_torch(array, dtype=dtype)

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
            f"DicomSeriesProcessor(lazy={self.lazy}, "
            f"target_spacing={self.target_spacing}, "
            f"target_shape={self.target_shape}, "
            f"normalize={self.normalize}, is_mask={self.is_mask}, "
            f"series_uid={self.series_uid})"
        )
