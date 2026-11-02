"""Processor for reading a directory of image segmentation masks."""

from pathlib import Path
from typing import Any

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
    ) -> None:
        """Initialize the image mask directory processor.

        Args:
            mode: Optional PIL mode conversion.
            pattern: Glob pattern for mask files.
            canonical: Reorient the mask like the reference DICOM volume.
            lazy: Store references instead of the stacked mask volume.
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
        )
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
        )
        return _to_torch(array, dtype="int64")

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
