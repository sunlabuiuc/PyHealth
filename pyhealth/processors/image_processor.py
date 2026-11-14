from functools import partial
from pathlib import Path
from collections.abc import Iterable
from typing import Any, List, Optional, Union

import torchvision.transforms as transforms
from PIL import Image

from . import register_processor
from .base_processor import LegacyFeatureProcessor


@register_processor("image")
class ImageProcessor(LegacyFeatureProcessor):
    """Feature processor for loading images from disk and converting them to tensors.

    Args:
        image_size: Desired output image size (will resize to square image).
            Defaults to 224.
        to_tensor: Whether to convert image to tensor. Defaults to True.
        normalize: Whether to normalize image values to [0, 1]. Defaults to False.
        mean: Precomputed mean for normalization.
        std: Precomputed std for normalization.
        mode: PIL image mode to convert the image to before processing.
            Common modes: 'RGB', 'RGBA', 'L' (grayscale), 'P' (palette).
            If None, keeps the original mode. Defaults to None.

    Raises:
        ValueError: If normalization parameters are inconsistent.

    Examples:
        >>> from pyhealth.processors import ImageProcessor
        >>> p = ImageProcessor(image_size=224)
        >>> p.stores_tensor  # to_tensor=False returns PIL images instead
        True
        >>> p.process("chest_xray.png").shape  # doctest: +SKIP
        torch.Size([3, 224, 224])
    """

    _shape: tuple[int, int, int] | None = None
    """The widest ``(C, H, W)`` that ``fit()`` saw, or None if unknown (not
    fitted, or fitted before this was recorded)."""

    @property
    def stores_tensor(self) -> bool:
        """True when ``process()`` returns a tensor (``to_tensor=True``)."""
        return bool(getattr(self, "to_tensor", True))

    def __init__(
        self,
        image_size: int = 224,
        to_tensor: bool = True,
        normalize: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.image_size = image_size
        self.to_tensor = to_tensor
        self.normalize = normalize

        self.mean = mean
        self.std = std
        self.mode = mode

        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError(
                "Normalization requires both mean and std to be provided."
            )
        if not self.normalize and (self.mean is not None or self.std is not None):
            raise ValueError(
                "Mean and std are provided but normalize is set to False. "
                "Either provide normalize=True, or remove mean and std."
            )

        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        transform_list = []
        if self.mode is not None:
            transform_list.append(
                transforms.Lambda(partial(_convert_mode, mode=self.mode))
            )
        if self.image_size is not None:
            transform_list.append(
                transforms.Resize((self.image_size, self.image_size))
            )
        if self.to_tensor:
            transform_list.append(transforms.ToTensor())
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        return transforms.Compose(transform_list)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Records the widest output shape ``(C, H, W)`` of the images.

        ``C`` is the channel count of ``mode`` (or, without one, the most
        bands of any image's own mode), and ``H``, ``W`` are ``image_size``
        (or, without one, the largest height and width of any image). With
        both ``mode`` and ``image_size`` the shape is known without reading
        any image; otherwise every image's header is read (not its pixels).

        Args:
            samples: Sample dictionaries.
            field: The field holding image paths.
        """
        if self.mode is not None and self.image_size is not None:
            size = self.image_size
            self._shape = (Image.getmodebands(self.mode), size, size)
            return
        channels = height = width = 0
        for sample in samples:
            path = sample.get(field)
            if path is None:
                continue
            with Image.open(path) as img:  # lazy: reads the header only
                channels = max(channels, Image.getmodebands(self.mode or img.mode))
                w, h = (
                    (self.image_size, self.image_size)
                    if self.image_size is not None
                    else img.size
                )
            height, width = max(height, h), max(width, w)
        self._shape = (channels, height, width) if channels else None

    def process(self, value: Union[str, Path]) -> Any:
        """Process a single image path into a transformed tensor image.

        Args:
            value: Path to image file as string or Path object.

        Returns:
            Transformed image tensor.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = Path(value)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            img.load()  # Avoid "too many open files" errors
            return self.transform(img)

    def is_token(self) -> bool:
        """Image data is continuous (float-valued pixel intensities), not discrete tokens.

        Returns:
            False.
        """
        return False

    def schema(self) -> tuple[str, ...]:
        """Single tensor output.

        Returns:
            ("value",)
        """
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output tensor has 3 dimensions: (C, H, W).

        Returns:
            (3,)
        """
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        """Spatial axes for the output tensor (C, H, W).

        Channels are not spatial; height and width are.

        Returns:
            (False, True, True)
        """
        return (False, True, True)

    def __repr__(self) -> str:
        return (
            f"ImageLoadingProcessor(image_size={self.image_size}, "
            f"to_tensor={self.to_tensor}, normalize={self.normalize}, "
            f"mean={self.mean}, std={self.std}, mode={self.mode})"
        )


def _convert_mode(img: Image.Image, mode: str) -> Image.Image:
    """Convert a PIL image to the requested mode."""
    return img.convert(mode)
