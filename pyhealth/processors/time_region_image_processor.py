"""Crops anatomical regions from time-stamped images.

Region cropping is adapted from DiPro (commit 64b12ec,
``dataset_module/dipro_dataset.py``, https://github.com/Chenliu-svg/DiPro),
MIT License, Copyright (c) 2025 LIU Chen @ The Hong Kong Polytechnic University.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from . import register_processor
from .base_processor import FeatureProcessor, ModalityType


@register_processor("time_region_image")
class TimeRegionImageProcessor(FeatureProcessor):
    """Loads images, crops the same regions from each, and keeps their times.

    The output is a 3-tuple like that of
    :class:`~pyhealth.processors.TimeImageProcessor`, with one more dimension
    for the regions. Crops are resized with nearest-neighbor interpolation and
    scaled to ``[0, 1]``, as in DiPro. A box without area (DiPro stores a
    region that is not on the image as ``[0, 0, 0, 0]``) gives a crop of zeros.

    Args:
        num_regions: Number of boxes per image.
        image_size: Side of each square crop.
        mode: PIL image mode.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image
        >>> from pyhealth.processors import TimeRegionImageProcessor
        >>> folder = tempfile.TemporaryDirectory()
        >>> path = str(Path(folder.name) / "cxr.png")
        >>> Image.new("L", (32, 32), 255).save(path)
        >>> processor = TimeRegionImageProcessor(num_regions=2, image_size=8)
        >>> boxes = [[0, 0, 16, 16], [0, 0, 0, 0]]
        >>> regions, hours, tag = processor.process(([path], [1.5], [boxes]))
        >>> regions.shape
        torch.Size([1, 2, 3, 8, 8])
        >>> regions[0, 0].max().item(), regions[0, 1].max().item()
        (1.0, 0.0)
        >>> hours, tag
        (tensor([1.5000]), 'image')
        >>> folder.cleanup()
    """

    def __init__(
        self, num_regions: int, image_size: int = 224, mode: str = "RGB"
    ) -> None:
        self.num_regions = num_regions
        self.image_size = image_size
        self.mode = mode
        self.n_channels = len(Image.new(mode, (1, 1)).getbands())
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def _crop(self, image: Image.Image, box: Sequence[float]) -> torch.Tensor:
        """Returns one region ``[C, H, W]``, zeros if its box has no area."""
        left, top, right, bottom = (round(float(side)) for side in box)
        if right <= left or bottom <= top:
            return torch.zeros(self.n_channels, self.image_size, self.image_size)
        return self.transform(image.crop((left, top, right, bottom)))

    def process(
        self,
        value: tuple[
            Sequence[str | Path], Sequence[float], Sequence[Sequence[Sequence[float]]]
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Loads the images in time order and crops their regions.

        Args:
            value: ``(paths, times, boxes)``, with ``boxes[i]`` holding
                ``num_regions`` boxes ``[x1, y1, x2, y2]`` in the pixels of
                image ``i``.

        Returns:
            ``(regions, times, "image")``: ``regions`` is ``[N, R, C, H, W]``
            and ``times`` is ``[N]``.

        Raises:
            ValueError: If the inputs differ in length, are empty, or an image
                does not have ``num_regions`` boxes.
            FileNotFoundError: If an image file does not exist.
        """
        paths, times, boxes = value
        if not len(paths) == len(times) == len(boxes):
            raise ValueError(
                f"Got {len(paths)} paths, {len(times)} times and {len(boxes)} "
                "lists of boxes; they must match."
            )
        if len(paths) == 0:
            raise ValueError("paths must be non-empty.")
        ordered = sorted(zip(times, paths, boxes), key=lambda item: item[0])

        images = []
        for _, path, image_boxes in ordered:
            if len(image_boxes) != self.num_regions:
                raise ValueError(
                    f"Expected {self.num_regions} boxes per image, got "
                    f"{len(image_boxes)}."
                )
            if not Path(path).exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            with Image.open(path) as image:
                image = image.convert(self.mode)
                images.append(
                    torch.stack([self._crop(image, box) for box in image_boxes])
                )
        sorted_times = torch.tensor([item[0] for item in ordered], dtype=torch.float32)
        return torch.stack(images), sorted_times, "image"

    def size(self) -> int:
        """Returns the number of image channels."""
        return self.n_channels

    def is_token(self) -> bool:
        """Pixels are continuous values."""
        return False

    def modality(self) -> ModalityType:
        """Medical images: the IMAGE modality, as for ``TimeImageProcessor``."""
        return ModalityType.IMAGE

    def process_temporal(self, value) -> dict[str, torch.Tensor]:
        """Returns the output of :meth:`process` as a dict.

        Args:
            value: ``(paths, times, boxes)``, as for :meth:`process`.

        Returns:
            ``{"value": regions [N, R, C, H, W], "time": times [N]}``.
        """
        regions, times, _tag = self.process(value)
        return {"value": regions, "time": times}

    def __repr__(self) -> str:
        return (
            f"TimeRegionImageProcessor(num_regions={self.num_regions}, "
            f"image_size={self.image_size}, mode={self.mode!r})"
        )
