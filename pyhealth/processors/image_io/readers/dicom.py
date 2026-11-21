"""DICOM single-image and series readers built on pydicom.

Examples:
    >>> from pyhealth.processors.image_io.readers.dicom import DicomReader
    >>> DicomReader().name
    'dicom'
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..transforms import (
    _require_nibabel,
    apply_window,
    canonicalize_volume,
    normalize_intensity,
    resample_volume,
)
from ..types import PathLike
from .base import BaseImageReader

logger = logging.getLogger(__name__)

_AXIAL_IOP = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _require_pydicom():
    """Import pydicom or raise an actionable error."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "Medical imaging support requires pydicom. "
            "Install it with `pip install pyhealth[imaging]`."
        ) from exc
    return pydicom


def _iter_files(path: PathLike) -> list[Path]:
    """Return candidate files for a path, directory or explicit sequence."""
    if isinstance(path, (tuple, list)):
        return [Path(item) for item in path]
    source = Path(path)
    if source.is_dir():
        return sorted(
            item
            for item in source.rglob("*")
            if item.is_file()
            and not item.name.startswith("._")
            and item.name != ".DS_Store"
        )
    return [source]


def _float_list(dataset: Any, name: str) -> list[float] | None:
    """Read a DICOM numeric attribute as a list of floats."""
    value = getattr(dataset, name, None)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _read_headers(
    path: PathLike, stop_before_pixels: bool = True
) -> list[tuple[Path, Any]]:
    """Read DICOM headers, skipping files that are not DICOM."""
    pydicom = _require_pydicom()
    entries = []
    for file in _iter_files(path):
        try:
            dataset = pydicom.dcmread(
                str(file), stop_before_pixels=stop_before_pixels
            )
        except pydicom.errors.InvalidDicomError:
            logger.debug("Skipping non-DICOM file %s", file)
            continue
        except (EOFError, OSError, ValueError) as exc:
            # The file looks like DICOM but could not be decoded. Fail
            # instead of silently dropping a slice from the series.
            raise ValueError(
                f"Failed to read DICOM file {str(file)!r}: {exc}"
            ) from exc
        if getattr(dataset, "Rows", None) is None:
            continue
        entries.append((file, dataset))
    if not entries:
        raise ValueError(f"No DICOM files found in {path!r}.")
    return entries


def _sort_entries(entries: list[tuple[Path, Any]]) -> list[tuple[Path, Any]]:
    """Sort slices by position along the slice normal, then InstanceNumber."""
    iop = _float_list(entries[0][1], "ImageOrientationPatient")
    if iop is not None and len(iop) == 6:
        normal = np.cross(np.asarray(iop[:3]), np.asarray(iop[3:]))
        positions = []
        for _, dataset in entries:
            ipp = _float_list(dataset, "ImagePositionPatient")
            if ipp is None or len(ipp) < 3:
                positions = []
                break
            positions.append(float(np.dot(np.asarray(ipp, dtype=float), normal)))
        if positions:
            order = np.argsort(positions)
            ordered_positions = np.asarray(positions, dtype=float)[order]
            diffs = np.diff(ordered_positions)
            if len(diffs) and np.any(np.abs(diffs) < 1e-6):
                raise ValueError(
                    "DICOM series contains duplicate or overlapping slice "
                    "positions; refusing to build a volume."
                )
            return [entries[int(index)] for index in order]

    instance_numbers = []
    for _, dataset in entries:
        instance = getattr(dataset, "InstanceNumber", None)
        try:
            instance_numbers.append(int(instance))
        except (TypeError, ValueError):
            instance_numbers = []
            break
    if instance_numbers:
        order = np.argsort(instance_numbers)
        return [entries[int(index)] for index in order]
    return sorted(entries, key=lambda item: str(item[0]))


def _series_entries(
    path: PathLike,
    series_uid: str | None = None,
    stop_before_pixels: bool = True,
) -> list[tuple[Path, Any]]:
    """Read and group DICOM files, returning one sorted series."""
    entries = _read_headers(path, stop_before_pixels=stop_before_pixels)
    groups: dict[str, list[tuple[Path, Any]]] = {}
    for entry in entries:
        uid = str(getattr(entry[1], "SeriesInstanceUID", "") or "")
        groups.setdefault(uid, []).append(entry)

    if series_uid is not None:
        if series_uid not in groups:
            raise ValueError(
                f"Series {series_uid!r} not found. Available series: "
                f"{sorted(groups)}"
            )
        selected = groups[series_uid]
    elif len(groups) == 1:
        selected = next(iter(groups.values()))
    else:
        summary = {uid: len(items) for uid, items in groups.items()}
        raise ValueError(
            "Multiple DICOM series found. Pass series_uid to choose one: "
            f"{summary}"
        )
    return _sort_entries(selected)


def _pixel_spacing(dataset: Any) -> tuple[float, float]:
    """Return DICOM ``(row_spacing, column_spacing)`` in millimetres."""
    spacing = _float_list(dataset, "PixelSpacing")
    if spacing is None or len(spacing) < 2:
        return (1.0, 1.0)
    return (float(spacing[0]), float(spacing[1]))


def _slice_spacing(entries: list[tuple[Path, Any]]) -> float:
    """Infer spacing from slice positions, falling back to header tags.

    Position gaps describe the reconstructed geometry, which is what matters
    for a volume. ``SliceThickness`` is the acquisition thickness and can
    differ for overlapping or reconstructed series, so it is only a fallback.
    """
    if len(entries) > 1:
        iop = _float_list(entries[0][1], "ImageOrientationPatient")
        if iop is not None and len(iop) == 6:
            normal = np.cross(np.asarray(iop[:3]), np.asarray(iop[3:]))
            positions = []
            for _, dataset in entries:
                ipp = _float_list(dataset, "ImagePositionPatient")
                if ipp is None or len(ipp) < 3:
                    break
                positions.append(float(np.dot(np.asarray(ipp), normal)))
            if len(positions) == len(entries):
                ordered = np.sort(np.asarray(positions, dtype=float))
                raw_diffs = np.diff(ordered)
                if len(raw_diffs) and np.any(np.abs(raw_diffs) < 1e-6):
                    raise ValueError(
                        "DICOM series contains duplicate or overlapping "
                        "slice positions; cannot infer slice spacing."
                    )
                diffs = np.abs(raw_diffs[raw_diffs != 0])
                if len(diffs) > 0:
                    median = float(np.median(diffs))
                    tolerance = max(1e-3, abs(median) * 1e-3)
                    if np.any(np.abs(diffs - median) > tolerance):
                        logger.warning(
                            "DICOM series has non-uniform slice spacing "
                            "(gaps=%s); using the median %.3f mm.",
                            [float(value) for value in diffs],
                            median,
                        )
                    return median
    for name in ("SpacingBetweenSlices", "SliceThickness"):
        value = _float_list(entries[0][1], name)
        if value is not None and value[0] > 0:
            return float(value[0])
    return 1.0


def _affine_from_geometry(
    entries: list[tuple[Path, Any]],
) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    """Build a RAS affine and raw ``(x, y, z)`` shape/spacing."""
    first = entries[0][1]
    iop = _float_list(first, "ImageOrientationPatient")
    if iop is None or len(iop) != 6:
        iop = list(_AXIAL_IOP)
    ipp = _float_list(first, "ImagePositionPatient")
    if ipp is None or len(ipp) < 3:
        ipp = [0.0, 0.0, 0.0]

    row_cosines = np.asarray(iop[:3], dtype=float)
    column_cosines = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row_cosines, column_cosines)
    row_spacing, column_spacing = _pixel_spacing(first)
    z_spacing = _slice_spacing(entries)

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_cosines * column_spacing
    affine_lps[:3, 1] = column_cosines * row_spacing
    affine_lps[:3, 2] = normal * z_spacing
    affine_lps[:3, 3] = np.asarray(ipp[:3], dtype=float)
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps

    rows = int(first.Rows)
    columns = int(first.Columns)
    shape_xyz = (columns, rows, len(entries))
    spacing_xyz = (column_spacing, row_spacing, z_spacing)
    return affine_ras, shape_xyz, spacing_xyz


def _canonical_layout(
    affine: np.ndarray,
    shape_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> tuple[
    tuple[int, int, int],
    tuple[float, float, float],
    str,
    np.ndarray,
]:
    """Return output ``(D, H, W)`` shape/spacing, orientation and affine."""
    nib = _require_nibabel()
    orientation = nib.orientations.io_orientation(affine)
    permutation = [
        int(np.where(orientation[:, 0] == index)[0][0]) for index in range(3)
    ]
    shape_ras = tuple(shape_xyz[index] for index in permutation)
    spacing_ras = tuple(spacing_xyz[index] for index in permutation)
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(
        orientation, shape_xyz
    )
    return (
        tuple(int(value) for value in reversed(shape_ras)),
        tuple(float(value) for value in reversed(spacing_ras)),
        "".join(nib.aff2axcodes(canonical_affine)),
        canonical_affine,
    )


def _require_single_frame(dataset: Any) -> None:
    """Reject multi-frame files, which the readers do not support yet."""
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if frames > 1:
        raise NotImplementedError(
            "Multi-frame DICOM is not supported yet; use one file per slice."
        )


def _require_grayscale(dataset: Any) -> None:
    """Reject colour DICOM images, which are not supported yet."""
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "")
    if samples > 1 or photometric.upper().startswith(
        ("RGB", "YBR", "PALETTE")
    ):
        raise NotImplementedError(
            "Colour DICOM images are not supported yet; expected a "
            "single-channel image, found "
            f"SamplesPerPixel={samples}, "
            f"PhotometricInterpretation={photometric!r}."
        )


def _dicom_dtype(dataset: Any) -> str:
    """Return a NumPy dtype name for a DICOM pixel representation."""
    bits = int(getattr(dataset, "BitsAllocated", 16) or 16)
    signed = int(getattr(dataset, "PixelRepresentation", 0) or 0) == 1
    return f"int{bits}" if signed else f"uint{bits}"


def _intensity(dataset: Any) -> dict[str, float]:
    """Collect rescale and window values from a DICOM header."""
    values: dict[str, float] = {}
    for name, key in (
        ("RescaleSlope", "slope"),
        ("RescaleIntercept", "intercept"),
    ):
        item = _float_list(dataset, name)
        if item is not None:
            values[key] = float(item[0])
    for name, key in (("WindowCenter", "window_center"), ("WindowWidth", "window_width")):
        item = _float_list(dataset, name)
        if item is not None:
            values[key] = float(item[0])
    return values


def _to_lps_affine(dataset: Any) -> np.ndarray:
    """Build a single-slice RAS affine, using identity when tags are absent."""
    iop = _float_list(dataset, "ImageOrientationPatient")
    if iop is None or len(iop) != 6:
        iop = list(_AXIAL_IOP)
    ipp = _float_list(dataset, "ImagePositionPatient")
    if ipp is None or len(ipp) < 3:
        ipp = [0.0, 0.0, 0.0]
    row_cosines = np.asarray(iop[:3], dtype=float)
    column_cosines = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row_cosines, column_cosines)
    row_spacing, column_spacing = _pixel_spacing(dataset)
    spacing = _float_list(dataset, "SliceThickness")
    z_spacing = float(spacing[0]) if spacing and spacing[0] > 0 else 1.0

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_cosines * column_spacing
    affine_lps[:3, 1] = column_cosines * row_spacing
    affine_lps[:3, 2] = normal * z_spacing
    affine_lps[:3, 3] = np.asarray(ipp[:3], dtype=float)
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps


class DicomReader(BaseImageReader):
    """Reader for a single-frame, single-channel DICOM image.

    The reader returns a 2D float array. Colour DICOM images are not supported
    yet and raise :class:`NotImplementedError`. Resizing and model-specific
    transforms belong to the processor.

    Examples:
        >>> from pyhealth.processors.image_io.readers.dicom import DicomReader
        >>> DicomReader().name
        'dicom'
    """

    name = "dicom"

    def read_metadata(self, path: PathLike) -> dict[str, Any]:
        """Read a DICOM header without loading pixel data.

        Args:
            path: Path to one DICOM file.

        Returns:
            Metadata with ``(1, H, W)`` shape.

        Raises:
            NotImplementedError: If the image is colour.
        """
        entries = _read_headers(path, stop_before_pixels=True)
        if len(entries) != 1:
            raise ValueError(
                "DicomReader expects a single file; use DicomSeriesReader "
                "for a series."
            )
        file, dataset = entries[0]
        _require_single_frame(dataset)
        _require_grayscale(dataset)
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        row_spacing, column_spacing = _pixel_spacing(dataset)
        affine = _to_lps_affine(dataset)
        nib = _require_nibabel()
        photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "")
        return {
            "format": "dicom",
            "shape": (1, rows, columns),
            "spacing": (row_spacing, column_spacing),
            "affine": tuple(
                tuple(float(value) for value in row) for row in affine
            ),
            "source_affine": tuple(
                tuple(float(value) for value in row) for row in affine
            ),
            "orientation": "".join(nib.aff2axcodes(affine)),
            "dtype": _dicom_dtype(dataset),
            "modality": str(getattr(dataset, "Modality", "") or "") or None,
            "photometric": photometric or None,
            "intensity": _intensity(dataset),
            "patient_id": str(getattr(dataset, "PatientID", "") or "") or None,
            "study_id": str(getattr(dataset, "StudyInstanceUID", "") or "")
            or None,
            "series_id": str(getattr(dataset, "SeriesInstanceUID", "") or "")
            or None,
            "source": (str(file),),
            "n_frames": int(getattr(dataset, "NumberOfFrames", 1) or 1),
        }

    def load(
        self,
        path: PathLike,
        *,
        metadata: dict[str, Any] | None = None,
        apply_rescale: bool = True,
        invert_monochrome1: bool = True,
        window: bool = False,
        window_center: float | None = None,
        window_width: float | None = None,
        normalize: str | None = None,
        clip: tuple[float, float] | None = None,
        percentile_range: Sequence[float] = (1.0, 99.0),
        dtype: str = "float32",
        is_mask: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Load one DICOM image as a NumPy array.

        Args:
            path: Path to one DICOM file.
            metadata: Optional pre-read metadata.
            apply_rescale: Apply ``RescaleSlope`` and ``RescaleIntercept``.
            invert_monochrome1: Invert ``MONOCHROME1`` images.
            window: Apply the DICOM or supplied window.
            window_center: Override the DICOM window centre.
            window_width: Override the DICOM window width.
            normalize: Intensity normalisation mode.
            clip: Optional ``(low, high)`` clipping range.
            percentile_range: Percentiles for ``normalize="percentile"``.
            dtype: Output dtype for non-mask images.
            is_mask: Treat the input as a mask. Rescale, inversion,
                windowing and normalization are skipped so labels keep their
                stored integer values.
            **kwargs: Ignored extra options for interface compatibility.

        Returns:
            A 2D array.

        Raises:
            ValueError: If the path is not a single-file DICOM image.
            NotImplementedError: If the image is multi-frame or colour.
        """
        del kwargs
        pydicom = _require_pydicom()
        dataset = pydicom.dcmread(str(path))
        if int(getattr(dataset, "NumberOfFrames", 1) or 1) > 1:
            raise NotImplementedError(
                "Multi-frame DICOM is not supported yet; use one file per "
                "slice or extend DicomSeriesReader."
            )
        _require_grayscale(dataset)
        array = np.asarray(dataset.pixel_array)
        if array.ndim != 2:
            raise NotImplementedError(
                "Only single-channel, single-frame DICOM images are supported."
            )
        if is_mask:
            # Mask labels are stored values; rescale, windowing and
            # normalization would change the integer class indices.
            return np.ascontiguousarray(array.astype(np.int64, copy=False))

        array = array.astype(np.float32)
        if apply_rescale:
            slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
            array = array * slope + intercept
        applied_window = False
        if window:
            intensity = _intensity(dataset)
            center = (
                float(window_center)
                if window_center is not None
                else intensity.get("window_center")
            )
            width = (
                float(window_width)
                if window_width is not None
                else intensity.get("window_width")
            )
            if (center is None) != (width is None):
                raise ValueError(
                    "DICOM window requires both WindowCenter and WindowWidth."
                )
            if center is not None and width is not None:
                # Window tags describe the stored intensities, so apply the
                # window before any MONOCHROME1 inversion.
                array = apply_window(array, center, width)
                applied_window = True
        photometric = str(
            getattr(dataset, "PhotometricInterpretation", "") or ""
        )
        if invert_monochrome1 and photometric == "MONOCHROME1":
            array = (
                1.0 - array
                if applied_window
                else float(np.max(array)) - array
            )
        array = normalize_intensity(
            array,
            mode=normalize,
            clip=clip,
            percentile_range=percentile_range,
        )
        return np.ascontiguousarray(array.astype(dtype, copy=False))


class DicomSeriesReader(BaseImageReader):
    """Reader that reconstructs a 3D volume from a DICOM series.

    Slices are grouped by ``SeriesInstanceUID`` and sorted by their position
    along the slice normal. If position tags are missing, ``InstanceNumber``
    is used instead. The output volume is ``(1, D, H, W)`` in canonical RAS
    orientation, with ``D`` superior-inferior. Colour series are not supported
    yet and raise :class:`NotImplementedError`.

    Examples:
        >>> from pyhealth.processors.image_io.readers.dicom import DicomSeriesReader
        >>> DicomSeriesReader().name
        'dicom_series'
    """

    name = "dicom_series"

    def read_metadata(
        self,
        path: PathLike,
        *,
        series_uid: str | None = None,
        canonical: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Read series headers and geometry without loading pixels.

        Args:
            path: Directory, file list, or one DICOM file.
            series_uid: Series to select when several are present.
            canonical: Report the metadata of the canonical RAS volume.
            **kwargs: Ignored extra options for interface compatibility.

        Returns:
            Metadata for the reconstructed volume.

        Raises:
            NotImplementedError: If the series contains colour images.
        """
        del kwargs
        entries = _series_entries(
            path, series_uid=series_uid, stop_before_pixels=True
        )
        _, first = entries[0]
        for _, dataset in entries:
            _require_single_frame(dataset)
        _require_grayscale(first)
        affine, shape_xyz, spacing_xyz = _affine_from_geometry(entries)
        if canonical:
            shape_dhw, spacing_dhw, orientation, metadata_affine = (
                _canonical_layout(affine, shape_xyz, spacing_xyz)
            )
        else:
            nib = _require_nibabel()
            shape_dhw = tuple(int(value) for value in reversed(shape_xyz))
            spacing_dhw = tuple(
                float(value) for value in reversed(spacing_xyz)
            )
            orientation = "".join(nib.aff2axcodes(affine))
            metadata_affine = affine
        return {
            "format": "dicom_series",
            "shape": (1,) + shape_dhw,
            "spacing": spacing_dhw,
            "affine": tuple(
                tuple(float(value) for value in row)
                for row in metadata_affine
            ),
            "source_affine": tuple(
                tuple(float(value) for value in row) for row in affine
            ),
            "orientation": orientation,
            "dtype": _dicom_dtype(first),
            "modality": str(getattr(first, "Modality", "") or "") or None,
            "photometric": str(
                getattr(first, "PhotometricInterpretation", "") or ""
            )
            or None,
            "intensity": _intensity(first),
            "patient_id": str(getattr(first, "PatientID", "") or "") or None,
            "study_id": str(getattr(first, "StudyInstanceUID", "") or "")
            or None,
            "series_id": str(getattr(first, "SeriesInstanceUID", "") or "")
            or None,
            "source": tuple(str(file) for file, _ in entries),
            "source_shape": tuple(int(value) for value in reversed(shape_xyz)),
            "n_frames": len(entries),
        }

    def load(
        self,
        path: PathLike,
        *,
        metadata: dict[str, Any] | None = None,
        series_uid: str | None = None,
        canonical: bool = True,
        sorted_paths: bool = False,
        target_spacing: Sequence[float] | None = None,
        target_shape: Sequence[int] | None = None,
        normalize: str | None = None,
        clip: tuple[float, float] | None = None,
        percentile_range: Sequence[float] = (1.0, 99.0),
        intensity_stats: dict[str, float] | None = None,
        use_nonzero_mask: bool = True,
        mri_background_threshold: float | None = None,
        dtype: str = "float32",
        is_mask: bool = False,
        order: int | None = None,
        apply_rescale: bool = True,
        invert_monochrome1: bool = True,
        window: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Load and reconstruct a DICOM series.

        Args:
            path: Directory, file list, or one DICOM file.
            metadata: Optional pre-read metadata. Its ``source`` order is
                reused so the series is not sorted again.
            series_uid: Series to select when several are present.
            canonical: Reorient the volume to canonical RAS.
            sorted_paths: Treat a path list as already sorted and skip series
                grouping and sorting. Used for lazy DICOM series values.
            target_spacing: Target spacing in ``(D, H, W)`` order.
            target_shape: Target shape in ``(D, H, W)`` order.
            normalize: Intensity normalisation mode.
            clip: Optional ``(low, high)`` clipping range.
            percentile_range: Percentiles for ``normalize="percentile"``.
            intensity_stats: Foreground statistics for ``normalize="ct"``.
            use_nonzero_mask: For ``normalize="mri"``, normalise only non-zero
                voxels. Defaults to True.
            mri_background_threshold: Optional fraction of the 99th percentile
                below which MRI voxels are treated as air/noise and zeroed.
            dtype: Output dtype for non-mask volumes.
            is_mask: Treat the input as a mask. Rescale, inversion,
                windowing and normalization are skipped so labels keep their
                stored integer values.
            order: Interpolation order. Defaults to 0 for masks and 1
                otherwise.
            apply_rescale: Apply per-slice rescale values.
            invert_monochrome1: Invert ``MONOCHROME1`` slices.
            window: Apply each slice's DICOM window.
            **kwargs: Ignored extra options for interface compatibility.

        Returns:
            A ``(1, D, H, W)`` array.

        Raises:
            ValueError: If slice dimensions are inconsistent.
            NotImplementedError: If a multi-frame or colour file is present.
        """
        del kwargs
        pydicom = _require_pydicom()
        if metadata is not None and metadata.get("source"):
            files = [Path(item) for item in metadata["source"]]
            entries = [
                (file, pydicom.dcmread(str(file), stop_before_pixels=False))
                for file in files
            ]
        elif sorted_paths and isinstance(path, (tuple, list)):
            entries = [
                (Path(item), pydicom.dcmread(str(item), stop_before_pixels=False))
                for item in path
            ]
        else:
            entries = _series_entries(
                path, series_uid=series_uid, stop_before_pixels=False
            )

        slices = []
        windows = []
        first_rows = None
        first_columns = None
        monochrome1 = False
        for _, dataset in entries:
            _require_grayscale(dataset)
            rows = int(dataset.Rows)
            columns = int(dataset.Columns)
            if first_rows is None:
                first_rows, first_columns = rows, columns
                monochrome1 = (
                    not is_mask
                    and invert_monochrome1
                    and str(
                        getattr(dataset, "PhotometricInterpretation", "") or ""
                    )
                    == "MONOCHROME1"
                )
            elif rows != first_rows or columns != first_columns:
                raise ValueError(
                    "Inconsistent slice dimensions in DICOM series."
                )
            array = np.asarray(dataset.pixel_array)
            if array.ndim != 2:
                raise NotImplementedError(
                    "Multi-frame DICOM is not supported yet."
                )
            if is_mask:
                # Keep stored integer labels; rescale, windowing and
                # MONOCHROME1 inversion would change class indices.
                slices.append(array)
                windows.append(None)
                continue
            array = array.astype(np.float32)
            if apply_rescale:
                slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
                intercept = float(
                    getattr(dataset, "RescaleIntercept", 0.0) or 0.0
                )
                array = array * slope + intercept
            window_params = None
            if window:
                intensity = _intensity(dataset)
                center = intensity.get("window_center")
                width = intensity.get("window_width")
                if (center is None) != (width is None):
                    raise ValueError(
                        "DICOM window requires both WindowCenter and "
                        "WindowWidth."
                    )
                if center is not None and width is not None:
                    window_params = (center, width)
            slices.append(array)
            windows.append(window_params)

        applied_window = False
        if window:
            tagged = [params is not None for params in windows]
            if any(tagged) and not all(tagged):
                raise ValueError(
                    "DICOM series has window tags on only some slices; "
                    "cannot apply a consistent window to the volume."
                )
            if all(tagged):
                first_window = windows[0]
                if any(params != first_window for params in windows[1:]):
                    raise ValueError(
                        "DICOM series has different window center/width "
                        "values per slice; cannot apply a consistent window."
                    )
                slices = [
                    apply_window(item, *first_window) for item in slices
                ]
                applied_window = True
        if monochrome1:
            if applied_window:
                # Window tags describe stored intensities, so invert the
                # already-windowed [0, 1] values.
                slices = [1.0 - item for item in slices]
            else:
                # Use one offset for the whole volume so slices keep their
                # relative intensities; a per-slice max would shift each
                # slice by a different amount.
                series_max = max(float(np.max(item)) for item in slices)
                slices = [series_max - item for item in slices]

        volume_hwd = np.stack(slices, axis=0)
        volume_xyz = np.transpose(volume_hwd, (2, 1, 0))
        affine = None
        if metadata is not None and metadata.get("affine") is not None:
            affine = np.asarray(
                metadata.get("source_affine", metadata["affine"]), dtype=float
            )
        if affine is None:
            affine, _, _ = _affine_from_geometry(entries)
        if canonical:
            volume_xyz, affine = canonicalize_volume(volume_xyz, affine)

        interpolation = order if order is not None else (0 if is_mask else 1)
        spacing_xyz = (
            None if target_spacing is None else tuple(reversed(target_spacing))
        )
        shape_xyz = None if target_shape is None else tuple(reversed(target_shape))
        if spacing_xyz is not None or shape_xyz is not None:
            volume_xyz, affine = resample_volume(
                volume_xyz,
                affine,
                target_spacing=spacing_xyz,
                target_shape=shape_xyz,
                order=interpolation,
            )

        volume_dhw = np.transpose(volume_xyz, (2, 1, 0))
        volume = volume_dhw[np.newaxis, ...]
        if is_mask:
            return np.ascontiguousarray(volume.astype(np.int64, copy=False))
        if normalize == "auto":
            modality = metadata.get("modality") if metadata is not None else None
            if not modality:
                modality = str(getattr(entries[0][1], "Modality", "") or "")
            if not modality:
                raise ValueError(
                    "normalize='auto' requires a DICOM Modality; pass "
                    "'ct' or 'mri' explicitly."
                )
            normalize = "ct" if modality.upper() == "CT" else "mri"
        volume = normalize_intensity(
            volume,
            mode=normalize,
            clip=clip,
            percentile_range=percentile_range,
            intensity_stats=intensity_stats,
            use_nonzero_mask=use_nonzero_mask,
            mri_background_threshold=mri_background_threshold,
        )
        return np.ascontiguousarray(volume.astype(dtype, copy=False))
