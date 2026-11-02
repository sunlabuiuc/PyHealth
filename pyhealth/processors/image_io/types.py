"""Shared lightweight types for medical imaging I/O.

Examples:
    >>> from pyhealth.processors.image_io.types import ImageMetadata, ImageRef
    >>> metadata = ImageMetadata(format="nifti", shape=(1, 4, 5, 6))
    >>> ref = ImageRef(path="/tmp/scan.nii.gz", reader="nifti", metadata=metadata)
    >>> ref.reader
    'nifti'
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

PathLike = str | tuple[str, ...]


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata describing one image or volume without its pixel data.

    Shapes follow the PyHealth medical-imaging convention: ``(C, H, W)`` for
    2D images and ``(C, D, H, W)`` for 3D volumes. Affines map source voxel
    indices to RAS world coordinates; spacing is in millimetres and follows
    the same spatial order as ``shape`` (``H, W`` or ``D, H, W``).

    Args:
        format: Source format, for example ``"nifti"`` or ``"dicom_series"``.
        shape: Output shape, including the channel axis.
        spacing: Voxel spacing in millimetres for the spatial axes.
        affine: Source voxel-to-RAS affine, or None for 2D images.
        orientation: Orientation code such as ``"RAS"`` when known.
        dtype: Source data type as a string.
        modality: DICOM modality or dataset modality, such as ``"CT"``.
        photometric: DICOM photometric interpretation.
        intensity: Intensity-handling values such as slope, intercept and
            window centre/width.
        patient_id: Patient identifier when present in the source.
        study_id: Study identifier when present in the source.
        series_id: Series identifier when present in the source.
        source: Source file path(s), ordered for series loading.
        n_frames: Number of frames for multi-frame sources.

    Examples:
        >>> from pyhealth.processors.image_io.types import ImageMetadata
        >>> metadata = ImageMetadata(format="nifti", shape=(1, 4, 5, 6))
        >>> metadata.shape
        (1, 4, 5, 6)
    """

    format: str
    shape: tuple[int, ...] = ()
    spacing: tuple[float, ...] = ()
    affine: tuple[tuple[float, ...], ...] | None = None
    orientation: str | None = None
    dtype: str | None = None
    modality: str | None = None
    photometric: str | None = None
    intensity: Mapping[str, float] = field(default_factory=dict)
    patient_id: str | None = None
    study_id: str | None = None
    series_id: str | None = None
    source: tuple[str, ...] = ()
    n_frames: int | None = None


@dataclass(frozen=True)
class ImageRef:
    """A lightweight reference to an image on disk.

    The processed-sample cache stores ``ImageRef`` objects instead of pixel
    arrays when a processor runs in lazy mode. The processor materialises the
    reference when the sample is read.

    Args:
        path: File path, directory, or ordered tuple of file paths.
        reader: Reader name registered in ``pyhealth.processors.image_io.readers``.
        metadata: Metadata collected when the sample was built.
        read_kwargs: Reader-specific options, such as a series UID.

    Examples:
        >>> from pyhealth.processors.image_io.types import ImageMetadata, ImageRef
        >>> ref = ImageRef(
        ...     path="/tmp/scan.nii.gz",
        ...     reader="nifti",
        ...     metadata=ImageMetadata(format="nifti", shape=(1, 4, 5, 6)),
        ... )
        >>> ref.reader
        'nifti'
    """

    path: PathLike
    reader: str
    metadata: ImageMetadata
    read_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def load(self) -> Any:
        """Read the source and return its raw pixel array.

        This method bypasses processor-specific transforms such as resizing or
        intensity normalisation. Prefer the owning processor's
        ``materialize`` method when a fully processed tensor is needed.

        Returns:
            The array returned by the registered reader.

        Examples:
            >>> from pyhealth.processors.image_io.types import ImageRef
            >>> ref = ImageRef(path="scan.nii.gz", reader="nifti")  # doctest: +SKIP
            >>> ref.load()  # doctest: +SKIP
        """
        # Import here so importing the dataclass does not require imaging
        # dependencies (pydicom/nibabel).
        from .readers.factory import get_reader_by_name

        reader = get_reader_by_name(self.reader)
        return reader.load(
            self.path,
            metadata=self.metadata,
            **dict(self.read_kwargs),
        )
