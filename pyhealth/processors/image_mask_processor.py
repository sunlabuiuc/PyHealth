"""Processor for reading a directory of image segmentation masks."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import register_processor
from .image_io.readers.image_mask import (
    ImageMaskReader,
    _mask_files,
    _prepare_mask_files,
)
from .image_io.tensor_utils import _to_torch
from .image_io.types import ImageRef
from .image_processor import ImageProcessor


@register_processor("image_mask_dir")
class ImageMaskDirectoryProcessor(ImageProcessor):
    """Read a directory of image slices as a 3D integer mask volume.

    The processor reuses :class:`ImageProcessor` to open each mask image, then
    stacks the slices into ``(1, D, H, W)``. Masks are not resized or
    normalized, so integer label values are preserved. The format is not
    fixed: ``pattern`` defaults to ``"*.png"`` but can be any PIL-readable
    image glob such as ``"*.jpg"`` or ``"*.tif"``.

    Args:
        mode: Optional PIL mode conversion. Usually left as None.
        pattern: Glob pattern for mask files. Defaults to ``"*.png"``.
        canonical: Reorient the mask like the reference DICOM volume.
        lazy: Store an :class:`ImageRef` instead of the stacked mask volume.
            Defaults to True, matching the other 3D image processors.
        label_map: Optional mapping from raw pixel values to class indices.
            Use this for datasets whose masks store scaled or sparse label
            values, such as CHAOS MR masks.
        num_classes: Optional number of classes, used by models to configure
            the output head.
        target_spacing: Target spacing in ``(D, H, W)`` order. Requires a
            reference DICOM series for the affine.
        target_shape: Target shape in ``(D, H, W)`` order.

    Examples:
        >>> from pyhealth.processors import ImageMaskDirectoryProcessor
        >>> processor = ImageMaskDirectoryProcessor()
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
        """
        super().__init__(
            image_size=None,
            to_tensor=False,
            normalize=False,
            mode=mode,
        )
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

    @property
    def stores_tensor(self) -> bool:
        """Whether the processed cache stores a tensor (not a reference)."""
        return not self.lazy

    def process(
        self,
        value: str | Path | tuple[str, str] | ImageRef,
    ) -> Any:
        """Read image masks and return a tensor or an :class:`ImageRef`.

        Args:
            value: Directory containing image mask slices, a
                ``(mask_dir, series_path)`` tuple, or an existing
                :class:`ImageRef`. The series path is used to match mask
                slices to the DICOM slice order.

        Returns:
            An :class:`ImageRef` in lazy mode, otherwise a
            ``torch.int64`` tensor with shape ``(1, D, H, W)``.

        Raises:
            FileNotFoundError: If the directory or any mask file is missing.
            ValueError: If the optional reference DICOM series cannot be read
                or has a different number of slices than the mask directory.
        """
        if isinstance(value, ImageRef):
            return self.materialize(value)

        series_path: str | Path | None = None
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mask_value, series_path = value
            mask_dir = Path(mask_value)
        else:
            mask_dir = Path(value)
        files = _mask_files(mask_dir, self.pattern)
        files, metadata = _prepare_mask_files(files, series_path)
        reader = ImageMaskReader()
        if metadata is None:
            metadata = reader.read_metadata(tuple(files))

        if self.lazy:
            return ImageRef(
                path=tuple(str(path) for path in files),
                reader=reader.name,
                metadata=metadata,
                read_kwargs={
                    "canonical": self.canonical,
                    "mode": self.mode,
                },
            )

        array = reader.load(
            tuple(files),
            metadata=metadata,
            canonical=self.canonical,
            mode=self.mode,
            target_spacing=getattr(self, "target_spacing", None),
            target_shape=getattr(self, "target_shape", None),
        )
        array = self._apply_label_map(array)
        return _to_torch(array, dtype="int64")

    def materialize(self, ref: ImageRef) -> Any:
        """Load an :class:`ImageRef` and return an integer mask tensor.

        Args:
            ref: Reference created by :meth:`process`.

        Returns:
            A ``torch.int64`` tensor with shape ``(1, D, H, W)``.
        """
        array = ImageMaskReader().load(
            ref.path,
            metadata=ref.metadata,
            canonical=self.canonical,
            mode=self.mode,
            target_spacing=getattr(self, "target_spacing", None),
            target_shape=getattr(self, "target_shape", None),
        )
        array = self._apply_label_map(array)
        return _to_torch(array, dtype="int64")

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
