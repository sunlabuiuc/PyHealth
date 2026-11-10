"""Reader for directories of image segmentation-mask slices.

Examples:
    >>> from pyhealth.processors.image_io.readers.image_mask import ImageMaskReader
    >>> ImageMaskReader().name
    'image_mask'
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..transforms import canonicalize_volume, resample_volume
from ..types import PathLike
from .base import BaseImageReader
from .dicom import DicomSeriesReader

logger = logging.getLogger(__name__)
_WARNED_POSITIONAL: set[str] = set()


def _natural_key(path: Path) -> list[Any]:
    """Sort filenames by embedded integer parts, so ``2`` precedes ``10``."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _mask_files(path: PathLike, pattern: str = "*.png") -> list[Path]:
    """Return mask files in the order supplied, or naturally sorted."""
    if isinstance(path, (tuple, list)):
        return [Path(item) for item in path]
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {source}")
    files = sorted(source.glob(pattern), key=_natural_key)
    if not files:
        raise FileNotFoundError(
            f"No mask files matching {pattern!r} in {source}."
        )
    return files


def _prepare_mask_files(
    mask_files: list[Path],
    series_path: str | Path | None,
    require_name_match: bool = False,
    canonical: bool = True,
) -> tuple[list[Path], dict[str, Any] | None]:
    """Order mask slices like the DICOM series and return its metadata.

    Args:
        mask_files: Mask slice paths in filename order.
        series_path: Optional reference DICOM series.
        require_name_match: Reject positional pairing when mask and DICOM
            filenames differ. Defaults to False because datasets such as
            CHAOS name masks ``mask_*.png`` and slices ``slice_*.dcm``.
        canonical: Report reference metadata in canonical RAS orientation.

    Raises:
        ValueError: If the reference series cannot be read, has a different
            number of slices, or positional pairing is used while
            ``require_name_match`` is set.
    """
    if series_path is None:
        return mask_files, None
    try:
        metadata = DicomSeriesReader().read_metadata(
            series_path, canonical=canonical
        )
    except ImportError:
        raise
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read the reference DICOM series {str(series_path)!r} "
            "needed to align mask slices. Pass the mask directory alone to "
            "use filename order."
        ) from exc
    ordered_dicom = [Path(path) for path in metadata["source"]]
    if len(ordered_dicom) != len(mask_files):
        raise ValueError(
            f"Mask directory has {len(mask_files)} slices but the reference "
            f"DICOM series {str(series_path)!r} has {len(ordered_dicom)}; "
            "cannot align masks with images."
        )

    name_to_index = {path.name: index for index, path in enumerate(ordered_dicom)}
    stem_to_index = {path.stem: index for index, path in enumerate(ordered_dicom)}
    ordered: list[Path | None] = [None] * len(mask_files)
    for mask_file in mask_files:
        index = stem_to_index.get(mask_file.stem)
        if index is not None:
            ordered[index] = mask_file
    if all(path is not None for path in ordered):
        return [path for path in ordered if path is not None], metadata

    if require_name_match:
        raise ValueError(
            f"Mask filenames do not match DICOM filenames for "
            f"{str(series_path)!r}. Rename the masks to match the DICOM "
            "stems, or disable strict_mask_names."
        )

    # Mask filenames often differ from the DICOM names (for example
    # ``mask_0.png`` next to ``slice_0.dcm``). With equal slice counts, pair
    # both lists in natural filename order and map that position to the
    # DICOM acquisition order. Warn once per series so a wrong pairing is
    # visible without flooding the log for every sample.
    key = str(series_path)
    if key not in _WARNED_POSITIONAL:
        logger.warning(
            "Mask filenames do not match DICOM filenames for %s; pairing %d "
            "masks with %d slices in natural filename order. Set "
            "strict_mask_names=True to reject this.",
            key,
            len(mask_files),
            len(ordered_dicom),
        )
        _WARNED_POSITIONAL.add(key)
    ordered = [None] * len(mask_files)
    dicom_by_name = sorted(ordered_dicom, key=_natural_key)
    for mask_file, dicom_file in zip(mask_files, dicom_by_name):
        ordered[name_to_index[dicom_file.name]] = mask_file
    return [path for path in ordered if path is not None], metadata


class ImageMaskReader(BaseImageReader):
    """Read ordered image slices as one integer mask volume.

    The reader accepts any image format that :class:`PIL.Image` can open.
    The caller is responsible for slice order when it differs from filename
    order, for example when matching DICOM positions.

    Examples:
        >>> from pyhealth.processors.image_io.readers.image_mask import ImageMaskReader
        >>> ImageMaskReader().name
        'image_mask'
    """

    name = "image_mask"

    def read_metadata(self, path: PathLike) -> dict[str, Any]:
        """Read mask shape and source files without loading every slice.

        Args:
            path: Directory or ordered tuple of image files.

        Returns:
            Metadata with shape ``(1, D, H, W)``.

        Raises:
            FileNotFoundError: If no image files are found.
        """
        files = _mask_files(path)
        with Image.open(files[0]) as image:
            width, height = image.size
        return {
            "format": "image_mask",
            "shape": (1, len(files), height, width),
            "dtype": "int64",
            "source": tuple(str(file) for file in files),
        }

    def load(
        self,
        path: PathLike,
        metadata: dict[str, Any] | None = None,
        canonical: bool = True,
        mode: str | None = None,
        target_spacing: Sequence[float] | None = None,
        target_shape: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Load image slices and stack them into ``(1, D, H, W)``.

        Args:
            path: Directory or ordered tuple of image files.
            metadata: Optional reference metadata. When it has an affine, the
                mask is reoriented like the reference volume.
            canonical: Reorient to canonical RAS when an affine is available.
            mode: Optional PIL mode conversion.
            target_spacing: Target spacing in ``(D, H, W)`` order. Masks use
                nearest-neighbour interpolation.
            target_shape: Target shape in ``(D, H, W)`` order.

        Returns:
            A contiguous ``int64`` array with shape ``(1, D, H, W)``.

        Raises:
            FileNotFoundError: If no image files are found.
            ValueError: If the masks have more than one colour channel.
        """
        files = _mask_files(path)

        slices = []
        for file in files:
            with Image.open(file) as image:
                if mode is not None:
                    image = image.convert(mode)
                slices.append(np.asarray(image))
        if slices and slices[0].ndim == 3:
            raise ValueError(
                "Image masks must be single-channel; convert the colour mask "
                f"{str(files[0])!r} to grayscale first."
            )
        # Nibabel does not accept bool or int64 data when reorienting, so use
        # int32 for the intermediate volume and widen after canonicalisation.
        volume = np.stack(slices, axis=0).astype(np.int32, copy=False)
        affine = None
        if metadata is not None and metadata.get("affine") is not None:
            affine = np.asarray(
                metadata.get("source_affine", metadata["affine"]), dtype=float
            )
        if canonical and affine is not None:
            volume_xyz = np.transpose(volume, (2, 1, 0))
            volume_xyz, affine = canonicalize_volume(volume_xyz, affine)
            volume = np.transpose(volume_xyz, (2, 1, 0))
        if metadata is not None and len(metadata.get("shape", ())) == 4:
            expected_shape = tuple(int(v) for v in metadata["shape"][1:])
            if volume.shape != expected_shape:
                raise ValueError(
                    f"Mask volume shape {tuple(volume.shape)} does not match "
                    f"the reference image shape {expected_shape}; check the "
                    "mask slice count and in-plane rows/columns."
                )
        if target_spacing is not None or target_shape is not None:
            if affine is None:
                raise ValueError(
                    "Mask resampling requires a reference affine; pass the "
                    "DICOM series path as the second mask value."
                )
            volume_xyz = np.transpose(volume, (2, 1, 0))
            spacing_xyz = (
                None
                if target_spacing is None
                else tuple(reversed(tuple(target_spacing)))
            )
            shape_xyz = (
                None
                if target_shape is None
                else tuple(reversed(tuple(target_shape)))
            )
            volume_xyz, _ = resample_volume(
                volume_xyz,
                affine,
                target_spacing=spacing_xyz,
                target_shape=shape_xyz,
                order=0,
            )
            volume = np.transpose(volume_xyz, (2, 1, 0))
        return np.ascontiguousarray(volume, dtype=np.int64)[None, ...]
