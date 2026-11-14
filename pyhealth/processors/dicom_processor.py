from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from . import register_processor
from .image_io.readers.dicom import DicomReader
from .image_io.tensor_utils import (
    _channels_first,
    _materialize_and_collate,
    _normalize_tensor,
    _resize_tensor,
    _tensor_to_pil,
)
from .image_processor import ImageProcessor


@register_processor("dicom")
class DicomProcessor(ImageProcessor):
    """Processor for single-frame DICOM images.

    The processor keeps the resizing and normalisation options of
    :class:`ImageProcessor`. DICOM-specific options control rescale, windowing
    and lazy loading.

    Args:
        image_size: Output size for a square resize. Defaults to 224.
        to_tensor: Return a tensor instead of a PIL image. Defaults to True.
        normalize: Normalise with the supplied mean and std. Defaults to False.
        mean: Per-channel means for normalisation.
        std: Per-channel standard deviations for normalisation.
        mode: Output channel mode such as ``"L"`` or ``"RGB"``.
        lazy: Store the file path instead of decoded pixels. Defaults to
            False. Lazy 2D values are path strings; existing 2D vision models
            inspect raw sample values for channel inference, so use eager
            loading or materialize paths in a wrapper.
        apply_rescale: Apply DICOM rescale slope/intercept. Defaults to True.
        invert_monochrome1: Invert ``MONOCHROME1`` images. Defaults to True.
        window: Apply the DICOM window. Defaults to False.
        window_center: Override the DICOM window centre.
        window_width: Override the DICOM window width.
        intensity_mode: Normalisation applied by the reader before tensor
            conversion. Defaults to ``"minmax"``.
        clip: Optional ``(low, high)`` intensity range.
        dtype: Output dtype for tensor values. Defaults to ``"float32"``.
        is_mask: Treat the image as a mask.

    Examples:
        >>> from pyhealth.processors import DicomProcessor
        >>> processor = DicomProcessor(image_size=128, mode="L")
        >>> processor.stores_tensor
        True
        >>> processor.is_token()
        False
    """

    def __init__(
        self,
        image_size: int = 224,
        to_tensor: bool = True,
        normalize: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        mode: str | None = None,
        lazy: bool = False,
        apply_rescale: bool = True,
        invert_monochrome1: bool = True,
        window: bool = False,
        window_center: float | None = None,
        window_width: float | None = None,
        intensity_mode: str | None = "minmax",
        clip: tuple[float, float] | None = None,
        dtype: str = "float32",
        is_mask: bool = False,
    ) -> None:
        """Initialize the DICOM processor.

        Args:
            image_size: Output size for a square resize.
            to_tensor: Return a tensor instead of a PIL image.
            normalize: Normalise with the supplied mean and std.
            mean: Per-channel means.
            std: Per-channel standard deviations.
            mode: Output channel mode.
            lazy: Store references instead of decoded pixels.
            apply_rescale: Apply DICOM rescale slope/intercept.
            invert_monochrome1: Invert MONOCHROME1 images.
            window: Apply the DICOM window.
            window_center: Override the window centre.
            window_width: Override the window width.
            intensity_mode: Reader-side intensity normalisation.
            clip: Optional intensity clipping range.
            dtype: Output dtype.
            is_mask: Treat the image as a mask.
        """
        super().__init__(
            image_size=image_size,
            to_tensor=to_tensor,
            normalize=normalize,
            mean=mean,
            std=std,
            mode=mode,
        )
        self.lazy = bool(lazy)
        self.apply_rescale = bool(apply_rescale)
        self.invert_monochrome1 = bool(invert_monochrome1)
        self.window = bool(window)
        self.window_center = window_center
        self.window_width = window_width
        self.intensity_mode = intensity_mode
        self.clip = clip
        self.dtype = dtype
        self.is_mask = bool(is_mask)
        if self.is_mask and not self.to_tensor:
            raise ValueError(
                "DicomProcessor(is_mask=True) requires to_tensor=True; "
                "returning a mask as a PIL image would rescale its labels."
            )
        if self.is_mask and self.normalize:
            raise ValueError(
                "DicomProcessor(is_mask=True) cannot use normalize=True; "
                "mean/std normalization would corrupt integer labels."
            )

    @property
    def stores_tensor(self) -> bool:
        """Whether the processed cache stores a tensor (not a reference)."""
        return bool(self.to_tensor and not self.lazy)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Records the widest output shape ``(C, H, W)`` of the images.

        This replaces :meth:`ImageProcessor.fit`, which reads headers with
        PIL and cannot open DICOM files. A DICOM image has one channel, which
        ``mode="RGB"`` or ``"RGBA"`` repeats, so ``C`` is 3, 4 or otherwise
        1. ``H`` and ``W`` are ``image_size``, or, without one, the largest
        height and width of any image, read from the DICOM headers (not the
        pixel data).

        Args:
            samples: Sample dictionaries.
            field: The field holding DICOM paths.
        """
        channels = {"RGB": 3, "RGBA": 4}.get(self.mode or "", 1)
        if self.image_size is not None:
            size = self.image_size
            self._shape = (channels, size, size)
            return
        height = width = 0
        reader = DicomReader()
        for sample in samples:
            path = sample.get(field)
            if path is None:
                continue
            _, h, w = reader.read_metadata(str(path))["shape"]
            height, width = max(height, h), max(width, w)
        self._shape = (channels, height, width) if height else None

    def process(self, value: str | Path) -> Any:
        """Process a DICOM path.

        Args:
            value: File path.

        Returns:
            An absolute path string in lazy mode, otherwise a tensor or PIL
            image.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path}")
        if self.lazy:
            # Validate the header at task-build time, not mid-training.
            DicomReader().read_metadata(str(path))
            return str(path.resolve())
        return self._array_to_output(self._load_array(str(path)))

    def materialize(self, value: str | Path) -> Any:
        """Load a lazy path and apply this processor's settings.

        Args:
            value: Path returned by :meth:`process`.

        Returns:
            A tensor or PIL image, matching ``to_tensor``.
        """
        array = DicomReader().load(str(value), **self._reader_kwargs())
        return self._array_to_output(array)

    def _load_array(self, path: str) -> Any:
        """Read and intensity-normalise one DICOM image."""
        return DicomReader().load(
            path,
            apply_rescale=self.apply_rescale,
            invert_monochrome1=self.invert_monochrome1,
            window=self.window,
            window_center=self.window_center,
            window_width=self.window_width,
            normalize=self.intensity_mode,
            clip=self.clip,
            dtype=self.dtype,
            is_mask=self.is_mask,
        )

    def _reader_kwargs(self) -> dict[str, Any]:
        """Return options forwarded to the DICOM reader."""
        return {
            "apply_rescale": self.apply_rescale,
            "invert_monochrome1": self.invert_monochrome1,
            "window": self.window,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "normalize": self.intensity_mode,
            "clip": self.clip,
            "dtype": self.dtype,
            "is_mask": self.is_mask,
        }

    def _array_to_output(self, array: Any) -> Any:
        """Convert a reader array to a tensor or PIL image.

        This array path is deliberate: :class:`ImageProcessor` runs its PIL
        pipeline, which converts images to 8-bit and resizes with
        interpolation. That would quantize masks (changing integer labels)
        and would quantize the float32 DICOM intensities of the non-mask path
        before resize/normalize. Both paths therefore stay on this array
        helper instead of splitting into two diverging pipelines.
        """
        tensor = _channels_first(array).float()
        if self.mode == "L" and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        elif self.mode in {"RGB", "RGBA"}:
            target_channels = 4 if self.mode == "RGBA" else 3
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(target_channels, 1, 1)
            elif self.mode == "RGB" and tensor.shape[0] == 4:
                tensor = tensor[:3]
            elif self.mode == "RGBA" and tensor.shape[0] == 3:
                alpha = torch.ones_like(tensor[:1])
                tensor = torch.cat([tensor, alpha], dim=0)
        # Masks must use nearest-neighbour so labels stay integers.
        interpolation = "nearest" if self.is_mask else "bilinear"
        tensor = _resize_tensor(
            tensor,
            self.image_size,
            mode=interpolation,
        )
        if self.normalize:
            tensor = _normalize_tensor(tensor, self.mean, self.std)
        if self.to_tensor:
            dtype = "int64" if self.is_mask else self.dtype
            return tensor.to(getattr(torch, dtype))
        return _tensor_to_pil(tensor)

    def collate(self, values: list[Any]) -> Any:
        """Materialise lazy refs, then pad and stack images."""
        return _materialize_and_collate(self, values)

    def __repr__(self) -> str:
        return (
            f"DicomProcessor(image_size={self.image_size}, "
            f"to_tensor={self.to_tensor}, lazy={self.lazy}, "
            f"normalize={self.normalize}, mode={self.mode}, "
            f"apply_rescale={self.apply_rescale}, window={self.window})"
        )
