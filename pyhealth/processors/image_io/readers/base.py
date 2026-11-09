"""Reader interface for medical image formats.

Examples:
    >>> from pyhealth.processors.image_io.readers.base import BaseImageReader
    >>> BaseImageReader.__name__
    'BaseImageReader'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import PathLike


class BaseImageReader(ABC):
    """Abstract base class for image and volume readers.

    Subclasses set ``name`` to the registry key used by the reader factory
    and implement :meth:`read_metadata` and :meth:`load`.

    Attributes:
        name: Registry name, for example ``"nifti"`` or ``"dicom"``.

    Examples:
        >>> from pyhealth.processors.image_io.readers.base import BaseImageReader
        >>> issubclass(BaseImageReader, object)
        True
    """

    name: str = "base"

    @abstractmethod
    def read_metadata(self, path: PathLike) -> dict[str, Any]:
        """Read metadata without loading pixel data.

        Args:
            path: File path, directory, or ordered tuple of paths.

        Returns:
            A dictionary describing the image.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: PathLike, **kwargs: Any) -> np.ndarray:
        """Load pixel data as a NumPy array.

        Args:
            path: File path, directory, or ordered tuple of paths.
            **kwargs: Reader-specific loading options.

        Returns:
            A NumPy array in the reader's documented layout.
        """
        raise NotImplementedError
