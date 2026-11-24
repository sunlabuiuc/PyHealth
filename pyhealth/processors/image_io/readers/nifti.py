"""NIfTI reader built on nibabel.

Examples:
    >>> from pyhealth.processors.image_io.readers.nifti import NiftiReader
    >>> NiftiReader().name
    'nifti'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..transforms import _require_nibabel, normalize_intensity, resample_volume
from ..types import PathLike
from .base import BaseImageReader


class NiftiReader(BaseImageReader):
    """Reader for NIfTI-1/2 files.

    The reader accepts 2D, 3D and 4D images. The last axis of a 4D image is
    treated as channels. Output arrays are channels-first; 3D volumes use
    ``(C, D, H, W)`` with ``D`` superior-inferior after RAS canonicalisation.

    Examples:
        >>> from pyhealth.processors.image_io.readers.nifti import NiftiReader
        >>> reader = NiftiReader()
        >>> reader.name
        'nifti'
    """

    name = "nifti"

    def read_metadata(
        self, path: PathLike, *, canonical: bool = True, channel_axis: int = -1
    ) -> dict[str, Any]:
        """Read NIfTI header information without loading pixel data.

        Args:
            path: Path to a ``.nii`` or ``.nii.gz`` file.
            canonical: Report the metadata of the canonical RAS image.
            channel_axis: Channel axis for 4D inputs. Only the last axis is
                supported, because the NIfTI affine describes the first
                three axes.

        Returns:
            Metadata for the image.

        Raises:
            ValueError: If the image does not have 2, 3 or 4 dimensions, or
                uses a non-last channel axis for 4D input.
        """
        nib = _require_nibabel()
        source = nib.load(str(path))
        shape = tuple(int(x) for x in source.shape)
        if len(shape) < 2 or len(shape) > 4:
            raise ValueError(
                "NIfTI image must have 2, 3 or 4 dimensions: "
                f"{shape}"
            )

        if len(shape) == 4 and channel_axis % 4 != 3:
            raise ValueError(
                "Channel-first 4D NIfTI is not supported: the NIfTI affine "
                "describes the first three axes. Store channels on the last "
                "axis instead."
            )
        image = nib.as_closest_canonical(source) if canonical else source
        shape = tuple(int(x) for x in image.shape)
        if len(shape) == 4:
            axis = channel_axis % 4
            spatial_shape = tuple(
                int(size)
                for index, size in enumerate(shape)
                if index != axis
            )
            zooms = image.header.get_zooms()[:4]
            spatial_spacing = tuple(
                float(size)
                for index, size in enumerate(zooms)
                if index != axis
            )
            channels = shape[axis]
            n_frames = shape[axis]
        else:
            spatial_shape = shape
            spatial_spacing = tuple(
                float(x) for x in image.header.get_zooms()[: len(shape)]
            )
            channels = 1
            n_frames = None
        output_shape = (channels,) + tuple(reversed(spatial_shape))
        output_spacing = tuple(reversed(spatial_spacing))
        orientation = "".join(nib.aff2axcodes(image.affine))
        metadata_affine = image.affine
        data_dtype = str(image.get_data_dtype())

        slope, intercept = source.header.get_slope_inter()
        intensity = {}
        if slope is not None:
            intensity["slope"] = float(slope)
        if intercept is not None:
            intensity["intercept"] = float(intercept)

        return {
            "format": "nifti",
            "shape": output_shape,
            "spacing": output_spacing,
            "affine": tuple(
                tuple(float(value) for value in row)
                for row in metadata_affine
            ),
            "orientation": orientation,
            "dtype": data_dtype,
            "intensity": intensity,
            "source": (str(path),),
            "n_frames": n_frames,
        }

    def load(
        self,
        path: PathLike,
        *,
        metadata: dict[str, Any] | None = None,
        channels: int | None = None,
        canonical: bool = True,
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
        channel_axis: int = -1,
        order: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Load a NIfTI image as a channels-first array.

        Args:
            path: Path to a ``.nii`` or ``.nii.gz`` file.
            metadata: Optional pre-read metadata (accepted for the shared
                reader interface; the image header is authoritative).
            channels: Expected channel count. Raises if it does not match.
            canonical: Reorient the image to canonical RAS.
            target_spacing: Target spacing in ``(D, H, W)`` order.
            target_shape: Target shape in ``(D, H, W)`` order.
            normalize: Intensity normalisation mode, see
                :func:`~pyhealth.processors.image_io.transforms.normalize_intensity`.
            clip: Optional ``(low, high)`` intensity clipping range.
            percentile_range: Percentiles for ``normalize="percentile"``.
            intensity_stats: Foreground statistics for ``normalize="ct"``.
            use_nonzero_mask: For ``normalize="mri"``, normalise only non-zero
                voxels. Defaults to True.
            mri_background_threshold: Optional fraction of the 99th percentile
                below which MRI voxels are treated as air/noise and zeroed.
            dtype: Output dtype. Masks default to ``int64`` when possible.
            is_mask: Treat the input as a segmentation mask. This disables
                intensity normalisation and uses nearest-neighbour resampling.
            channel_axis: Channel axis for 4D inputs. Only the last axis is
                supported, because the NIfTI affine describes the first
                three axes.
            order: Interpolation order. Defaults to 0 for masks and 1
                otherwise.
            **kwargs: Ignored extra options for interface compatibility.

        Returns:
            An array with shape ``(C, H, W)`` or ``(C, D, H, W)``.

        Raises:
            ValueError: If the image has an unsupported dimensionality or an
                unexpected channel count.
        """
        del metadata, kwargs
        nib = _require_nibabel()
        image = nib.load(str(path))
        if image.ndim == 4 and channel_axis % 4 != 3:
            raise ValueError(
                "Channel-first 4D NIfTI is not supported: the NIfTI affine "
                "describes the first three axes. Store channels on the last "
                "axis instead."
            )
        if canonical:
            image = nib.as_closest_canonical(image)

        data = np.asanyarray(image.dataobj)
        source_affine = np.asarray(image.affine, dtype=float)
        if data.ndim == 4:
            data = np.moveaxis(data, channel_axis, 0)
            if channels is not None and data.shape[0] != channels:
                raise ValueError(
                    f"Expected {channels} channels, found {data.shape[0]}."
                )
        elif data.ndim == 3 or data.ndim == 2:
            data = data[np.newaxis, ...]
        else:
            raise ValueError(
                f"NIfTI image must be 2D, 3D or 4D, found {data.ndim}D."
            )

        if channels is not None and data.shape[0] != channels:
            raise ValueError(
                f"Expected {channels} channels, found {data.shape[0]}."
            )

        spatial_ndim = data.ndim - 1
        if spatial_ndim != 3 and (
            target_shape is not None or target_spacing is not None
        ):
            raise ValueError(
                "target_shape and target_spacing require a 3D NIfTI volume."
            )
        interpolation = order if order is not None else (0 if is_mask else 1)
        if spatial_ndim == 3:
            spacing_xyz = (
                None if target_spacing is None else tuple(reversed(target_spacing))
            )
            shape_xyz = (
                None if target_shape is None else tuple(reversed(target_shape))
            )
            if spacing_xyz is not None or shape_xyz is not None:
                resampled = []
                for channel in data:
                    channel_data, _ = resample_volume(
                        channel,
                        source_affine,
                        target_spacing=spacing_xyz,
                        target_shape=shape_xyz,
                        order=interpolation,
                    )
                    resampled.append(channel_data)
                data = np.stack(resampled, axis=0)
            data = np.transpose(data, (0, 3, 2, 1))
        elif spatial_ndim == 2:
            data = np.transpose(data, (0, 2, 1))
        else:
            raise ValueError(
                "Only 2D and 3D spatial NIfTI images are supported."
            )

        if is_mask:
            out_dtype = np.int64 if np.issubdtype(data.dtype, np.integer) else dtype
        else:
            out_dtype = dtype
        if not is_mask:
            data = normalize_intensity(
                data,
                mode=normalize,
                clip=clip,
                percentile_range=percentile_range,
                intensity_stats=intensity_stats,
                use_nonzero_mask=use_nonzero_mask,
                mri_background_threshold=mri_background_threshold,
            )
        return np.ascontiguousarray(data.astype(out_dtype, copy=False))
