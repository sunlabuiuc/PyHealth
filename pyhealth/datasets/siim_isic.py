import logging
import os
import pydicom
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class SIIMISICDataset(BaseDataset):
    """
    A dataset class for handling the SIIM-ISIC Melanoma Classification dataset.

    The SIIM-ISIC dataset contains dermoscopic images and metadata for melanoma
    classification. A typical metadata CSV row looks like:

        image_name,patient_id,sex,age_approx,anatom_site_general_challenge,diagnosis,benign_malignant,target
        ISIC_2637011,IP_7279968,male,45,head/neck,unknown,benign,0

    Common columns include:
        - image_name: image ID (e.g., "ISIC_2637011")
        - patient_id: de-identified patient identifier (e.g., "IP_7279968")
        - sex: patient sex ("male", "female", etc.)
        - age_approx: approximate age in years (numeric)
        - anatom_site_general_challenge: anatomical site of the lesion
        - diagnosis: diagnosis label (e.g., "nevus", "melanoma", "unknown")
        - benign_malignant: coarse label ("benign" or "malignant")
        - target: binary target (0/1) for melanoma classification

    This class is a thin wrapper around BaseDataset. The actual schema and file
    locations are defined in the YAML config (e.g. configs/siim_isic.yaml),
    which specifies:
        - which CSV file to read (file_path)
        - which column is patient_id
        - which columns are attributes

    Args:
        root (str):
            The root directory where the dataset files are stored. This directory
            is usually expected to contain the CSV file(s) referenced in the
            siim_isic.yaml config (e.g., "siim-isic-metadata-pyhealth.csv" or
            the original "train.csv").
        tables (List[str]):
            A list of tables to be included (typically ["siim_isic"]).
            These must match the table names defined in the YAML config.
        dataset_name (Optional[str]):
            The name of the dataset. Defaults to "siim_isic".
        config_path (Optional[str]):
            The path to the YAML configuration file. If not provided, uses
            the default config at:
                pyhealth/datasets/configs/siim_isic.yaml
        image_dir (Option[str]):
            The path to the directory inside root where all dcm image files are stored

    Examples:
        >>> from pyhealth.datasets import SIIMISICDataset
        >>> dataset = SIIMISICDataset(
        ...     root="/path/to/siim_isic",
        ...     tables=["siim_isic"],
        ...     image_dir="train",
        ... )
        >>> # Get all patient ids
        >>> unique_patients = dataset.unique_patient_ids
        >>> print(f"There are {len(unique_patients)} patients")
        >>>
        >>> # Get all images associated with a patient
        >>> patient_id = "IP_2669371"
        >>> dcm_images = dataset.get_dcm_images(patient_id)
        >>>
        >>> # Display all dmc images assocaited with the patient
        >>> dcm_ids = dataset.get_image_file_ids(patient_id)
        >>> dataset.show_dicom_list(dcm_ids)
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        
    ) -> None:
        # If no config is provided, use the default siim_isic.yaml
        if config_path is None:
            logger.info("No config path provided, using default SIIM-ISIC config")
            config_path = Path(__file__).parent / "configs" / "siim_isic.yaml"

        self.image_dir = image_dir

        super().__init__(
            root=root,
            tables=["siim_isic"],
            dataset_name=dataset_name or "siim_isic",
            config_path=str(config_path),
        )
        return
    
    def get_image_file_ids(self, patient_id: str):
        """
        Retrieve all SIIM-ISIC image identifiers associated with a given patient.

        This function looks up all records belonging to the specified patient in the
        dataset's aggregated event DataFrame and returns the list of DICOM image
        identifiers (without file extensions). These identifiers correspond to the
        base filenames used to load `.dcm` files.

        Parameters
        ----------
        patient_id : str
            The unique patient identifier for which image IDs should be retrieved.
            Must exist in `self.unique_patient_ids`.

        Returns
        -------
        list[str]
            A list of image name strings (e.g., "ISIC_01234567"), suitable for use
            with `process_dcm()`.
        """
        assert (
            patient_id in self.unique_patient_ids
        ), f"Patient {patient_id} not found in dataset"
        df = self.collected_global_event_df.filter(pl.col("patient_id") == patient_id)
        return df["siim_isic/image_name"]
    
    def get_dcm_images(self, patient_id: str):
        """
        Load and process all DICOM images belonging to a specified patient.

        This function retrieves the list of DICOM image identifiers for the patient,
        then uses `process_dcm()` to load and preprocess each image into a normalized
        RGB NumPy array. The resulting list contains one processed image per DICOM
        file associated with the patient.

        Parameters
        ----------
        patient_id : str
            The unique patient identifier whose DICOM images should be loaded.

        Returns
        -------
        list[numpy.ndarray]
            A list of processed image arrays, each of shape (H, W, 3) and dtype
            `uint8`, suitable for visualization or model input.
        """
        image_ids = self.get_image_file_ids(patient_id)
        dcm_images = []
        for image_id in image_ids:
            dcm_images.append(self.process_dcm(image_name=image_id))
        return dcm_images
    
    def process_dcm(self, image_name: str):
        """
        Loads and processes a DICOM image into a normalized RGB numpy array.

        This function reads a DICOM file from the dataset's root directory, extracts
        the pixel data, normalizes the intensity values, corrects dimensionality
        issues, and returns a clean (H, W, 3) uint8 RGB image suitable for display
        or for input into machine learning models.

        Parameters
        ----------
        image_name : str
            The base filename (without ".dcm") of the DICOM image to load.

        Processing Steps
        ----------------
        1. **DICOM Loading**
        - Reads the DICOM using `pydicom.dcmread`.
        - Extracts pixel data via `ds.pixel_array`.

        2. **Intensity Normalization**
        - Converts pixels to float.
        - Divides by the maximum pixel value (if > 0) to produce a [0,1] range.

        3. **Dimensionality Handling**
        Handles multiple possible DICOM formats:
        - (H, W)  → converted to 3-channel RGB by stacking.
        - (H, W, 1) → expanded to (H, W, 3).
        - (H, W, 3) → used as-is.
        - (N, H, W) → assumes multiple frames and selects the first frame.
        - (H, W, C>3) → truncated to the first 3 channels.

        Additional safety checks ensure the final shape is always (H, W, 3).

        4. **Final Conversion**
        - Scales back to [0,255] and converts to `uint8`.

        Returns
        -------
        numpy.ndarray
            A processed image array of shape (H, W, 3) in uint8 format, ready for
            visualization or model input.

        Notes
        -----
        - This function intentionally abstracts away DICOM complexities
        (windowing modes, MONOCHROME1 inversion, multi-frame stacks, etc.)
        to produce a consistent RGB array.

        """

        ds = pydicom.dcmread(Path(self.root) / self.image_dir / f"{image_name}.dcm")
        im = ds.pixel_array.astype(float)

        # normalize safely
        max_val = im.max()
        if max_val > 0:
            im = im / max_val

        # Handle dimensions:
        # - 2D: (H, W) → stack to (H, W, 3)
        # - 3D: assume last dim might be channels or frames
        if im.ndim == 2:
            # grayscale → make 3-channel RGB
            im = np.stack([im] * 3, axis=-1)  # (H, W, 3)

        elif im.ndim == 3:
            # cases:
            # (H, W, C) where C == 1 or 3 or more
            # or (N, H, W) where N is frames; take first
            if im.shape[-1] in (1, 3, 4):
                # already (H,W,channels)-ish
                if im.shape[-1] == 1:
                    # single channel → repeat to 3
                    im = np.repeat(im, 3, axis=-1)
                elif im.shape[-1] > 3:
                    # trim extra channels
                    im = im[..., :3]
            else:
                # likely (N, H, W) → take first frame
                im = im[0]
                if im.ndim == 2:
                    im = np.stack([im] * 3, axis=-1)
                elif im.ndim == 3 and im.shape[-1] == 1:
                    im = np.repeat(im, 3, axis=-1)

        # final safety: if still not (H, W, 3), squeeze leading dims
        while im.ndim > 3:
            im = im[0]
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        if im.ndim == 3 and im.shape[-1] > 3:
            im = im[..., :3]

        im = (255 * im).clip(0, 255).astype(np.uint8)
        return im

    def show_dicom_list(self, dcm_image_ids, cols: int = 3, figsize=(12, 8)):
        """
        Display multiple DICOM images in a grid layout.

        Parameters
        ----------
        dcm_image_ids : list[str]
            A list of DICOM image identifiers (base names without ".dcm") to display.
        cols : int, optional
            Number of columns in the display grid. Default is 3.
        figsize : tuple, optional
            Size of the matplotlib figure. Default is (12, 8).

        Notes
        -----
        - Each image is processed using `process_dcm()`, ensuring consistent shape.
        - The layout automatically adjusts row count based on number of images.
        """
        n = len(dcm_image_ids)
        if n == 0:
            print("No DICOM images to display.")
            return

        rows = (n + cols - 1) // cols  # ceiling division
        plt.figure(figsize=figsize)

        for i, image_id in enumerate(dcm_image_ids):
            im = self.process_dcm(image_id)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(im)
            plt.title(image_id)
            plt.axis("off")

        plt.tight_layout()
        plt.show()
