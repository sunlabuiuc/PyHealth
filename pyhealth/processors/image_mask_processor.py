"""Processor for reading a directory of image segmentation masks."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import register_processor
from .base_processor import FeatureProcessor
from .image_io.readers.image_mask import (
    ImageMaskReader,
    _mask_files,
    _prepare_mask_files,
)
from .image_io.tensor_utils import _materialize_and_collate, _to_torch


@register_processor("image_mask_dir")
class ImageMaskDirectoryProcessor(FeatureProcessor):
    """Read a directory of image slices as a 3D integer mask volume.

    The processor opens each mask image, stacks the slices into
    ``(1, D, H, W)``, and preserves integer label values. The format is not
    fixed: ``pattern`` defaults to ``"*.png"`` but can be any PIL-readable
    image glob such as ``"*.tif"``. Use a lossless format: JPEG compression
    is lossy and corrupts integer mask labels.

    Args:
        mode: Optional PIL mode conversion. Usually left as None.
        pattern: Glob pattern for mask files. Defaults to ``"*.png"``.
        canonical: Reorient the mask like the reference DICOM volume.
        lazy: Store the mask directory and reference series path instead of
            the stacked volume. Defaults to True; set False to cache tensors.
        label_map: Optional mapping from raw pixel values to class indices.
            Use this for datasets whose masks store scaled or sparse label
            values, such as CHAOS MR masks. In lazy mode, unmapped labels are
            reported when the sample is materialized, not when it is indexed.
        num_classes: Optional number of classes, used by models to configure
            the output head.
        target_spacing: Target spacing in ``(D, H, W)`` order. Requires a
            reference DICOM series for the affine.
        target_shape: Target shape in ``(D, H, W)`` order.
        strict_mask_names: Require mask filenames to match the DICOM stems
            when a reference series is given. Defaults to True. Set it to
            False to allow natural-order positional pairing with a warning.

    Examples:
        >>> from pyhealth.processors import ImageMaskDirectoryProcessor
        >>> processor = ImageMaskDirectoryProcessor(lazy=True)
        >>> processor.stores_tensor
        False
        >>> processor.dim()
        (4,)
    """

    def __init__(
        self,
        mode: str | None = None,
        pattern: str = "*.png",
        canonical: bool = True,
        lazy: bool = True,
        label_map: dict[int, int] | None = None,
        num_classes: int | None = None,
        target_spacing: Sequence[float] | None = None,
        target_shape: Sequence[int] | None = None,
        strict_mask_names: bool = True,
    ) -> None:
        """Initialize the image mask directory processor.

        Args:
            mode: Optional PIL mode conversion.
            pattern: Glob pattern for mask files.
            canonical: Reorient the mask like the reference DICOM volume.
            lazy: Store references instead of the stacked mask volume.
            label_map: Optional raw-value to class-index mapping.
            num_classes: Optional number of classes.
            target_spacing: Target spacing in ``(D, H, W)`` order. Requires a
                reference DICOM series for the affine.
            target_shape: Target shape in ``(D, H, W)`` order.
            strict_mask_names: Require mask names to match the DICOM stems.
                Defaults to True; set False for positional pairing.
        """
        self.mode = mode
        self.pattern = pattern
        self.canonical = bool(canonical)
        self.lazy = bool(lazy)
        self.label_map = (
            None if label_map is None else {int(k): int(v) for k, v in label_map.items()}
        )
        self.num_classes = num_classes
        self.target_spacing = (
            None if target_spacing is None else tuple(target_spacing)
        )
        self.target_shape = None if target_shape is None else tuple(target_shape)
        self.strict_mask_names = bool(strict_mask_names)

    @property
    def stores_tensor(self) -> bool:
        """Whether the processed cache stores a tensor (not a reference)."""
        return not self.lazy

    def process(
        self,
        value: str | Path | tuple[str, str],
    ) -> Any:
        """Read image masks and return a tensor or lazy value.

        Args:
            value: Directory containing image mask slices or a
                ``(mask_dir, series_path)`` tuple.

        Returns:
            A ``torch.int64`` tensor in eager mode, otherwise a dictionary
            with the absolute mask directory and optional series path.

        Raises:
            FileNotFoundError: If the directory or any mask file is missing.
            ValueError: If the optional reference DICOM series cannot be read,
                has a different number of slices, or has mismatched in-plane
                rows/columns.
        """
        mask_dir, series_path = self._split_value(value)
        files, metadata = self._prepare(mask_dir, series_path)
        if self.lazy:
            return {
                "mask_dir": str(mask_dir.resolve()),
                "series_path": (
                    None
                    if series_path is None
                    else str(Path(series_path).resolve())
                ),
            }
        return self._load_prepared(files, metadata)

    def materialize(self, value: Any) -> Any:
        """Load a lazy mask value and return an integer mask tensor.

        Args:
            value: Dictionary or path value returned by :meth:`process`.

        Returns:
            A ``torch.int64`` tensor with shape ``(1, D, H, W)``.
        """
        if isinstance(value, dict):
            mask_dir = Path(value["mask_dir"])
            series_path = value.get("series_path")
        else:
            mask_dir, series_path = self._split_value(value)
        files, metadata = self._prepare(mask_dir, series_path)
        return self._load_prepared(files, metadata)

    def _split_value(self, value: Any) -> tuple[Path, str | Path | None]:
        """Return ``(mask_dir, series_path)`` from a mask value."""
        if isinstance(value, tuple) and len(value) == 2:
            return Path(value[0]), value[1]
        if isinstance(value, list):
            raise TypeError(
                "Pass masks as a directory or a (mask_dir, series_path) "
                "tuple, not as a list of files."
            )
        return Path(value), None

    def _prepare(
        self, mask_dir: Path, series_path: str | Path | None
    ) -> tuple[list[Path], dict[str, Any] | None]:
        """Validate mask files and return their order and reference metadata."""
        files = _mask_files(mask_dir, self.pattern)
        files, metadata = _prepare_mask_files(
            files,
            series_path,
            require_name_match=getattr(self, "strict_mask_names", False),
            canonical=self.canonical,
        )
        reader = ImageMaskReader()
        mask_metadata = reader.read_metadata(tuple(files))
        if metadata is None:
            metadata = mask_metadata
        else:
            source_shape = metadata.get("source_shape")
            mask_shape = tuple(int(v) for v in mask_metadata["shape"][1:])
            if source_shape is not None and mask_shape != tuple(source_shape):
                raise ValueError(
                    f"Mask volume shape {mask_shape} does not match the "
                    f"reference image shape {tuple(source_shape)}; check the "
                    "mask slice count and in-plane rows/columns."
                )
        if (
            self.target_shape is not None or self.target_spacing is not None
        ) and (metadata is None or metadata.get("affine") is None):
            raise ValueError(
                "Mask resampling requires a reference affine; pass the "
                "DICOM series path as the second mask value."
            )
        return files, metadata

    def _load_prepared(
        self, files: list[Path], metadata: dict[str, Any] | None
    ) -> Any:
        """Load prepared masks and apply this processor's settings."""
        array = ImageMaskReader().load(
            tuple(files),
            metadata=metadata,
            canonical=self.canonical,
            mode=self.mode,
            target_spacing=getattr(self, "target_spacing", None),
            target_shape=getattr(self, "target_shape", None),
        )
        array = self._apply_label_map(array)
        return _to_torch(array, dtype="int64")

    def collate(self, values: list[Any]) -> Any:
        """Materialise lazy refs, then pad and stack mask volumes."""
        return _materialize_and_collate(self, values)

    def _apply_label_map(self, array: "np.ndarray") -> "np.ndarray":
        """Map raw mask values to class indices, rejecting unknown values."""
        label_map = getattr(self, "label_map", None)
        if label_map is None:
            return array
        mapped = array.copy()
        unknown = sorted(
            int(value)
            for value in np.unique(array)
            if int(value) not in label_map
        )
        if unknown:
            raise ValueError(
                "Found mask labels without a mapping: "
                f"{unknown}. Add them to label_map."
            )
        for source, target in label_map.items():
            mapped[array == source] = target
        return mapped

    def is_token(self) -> bool:
        """Mask values are integer labels, not continuous tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Return the single output field name."""
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Return the output tensor rank."""
        return (4,)

    def spatial(self) -> tuple[bool, ...]:
        """Return which axes are spatial."""
        return (False, True, True, True)

    def __repr__(self) -> str:
        return (
            f"ImageMaskDirectoryProcessor(mode={self.mode}, "
            f"pattern={self.pattern!r}, canonical={self.canonical}, "
            f"lazy={self.lazy})"
        )
