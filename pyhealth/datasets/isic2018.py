"""
Unified PyHealth dataset for ISIC 2018 Tasks.

This module provides :class:`ISIC2018Dataset`, a single dataset class that
covers both:

* ``task="task3"`` — 7-class skin lesion **classification** (HAM10000 / Task 3).
  Downloads images and ``ISIC2018_Task3_Training_GroundTruth.csv``.

* ``task="task1_2"`` — Lesion **segmentation** & attribute detection (Task 1/2).
  Downloads images and binary segmentation masks.

Both modes support ``download=True`` for automatic data acquisition from the
official ISIC 2018 challenge S3 archive.

The module also exports the URL / directory-name constants and helper functions
(``_download_file``, ``_extract_zip``) that are re-used by
:class:`~pyhealth.datasets.ISIC2018ArtifactsDataset`.

Dataset link:
    https://challenge.isic-archive.com/data/#2018

Licenses:
    Task 1/2 (segmentation & attribute detection):
        CC-0 (Public Domain) — https://creativecommons.org/public-domain/cc0/

    Task 3 (classification):
        CC-BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/
        Attribution required — see references below.

References:
    [1] Noel Codella et al. "Skin Lesion Analysis Toward Melanoma Detection
    2018: A Challenge Hosted by the International Skin Imaging Collaboration
    (ISIC)", 2018; https://arxiv.org/abs/1902.03368

    [2] Tschandl et al. "The HAM10000 dataset, a large collection of
    multi-source dermatoscopic images of common pigmented skin lesions."
    Sci. Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
"""

import hashlib
import logging
import os
import shutil
import zipfile
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants (also imported by isic2018_artifacts.py)
# ---------------------------------------------------------------------------

TASK12_IMAGES_URL: str = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task1-2_Training_Input.zip"
)
TASK12_MASKS_URL: str = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task1_Training_GroundTruth.zip"
)
TASK12_IMAGES_DIR: str = "ISIC2018_Task1-2_Training_Input"
TASK12_MASKS_DIR: str = "ISIC2018_Task1_Training_GroundTruth"
_T12_IMAGES_ZIP = "ISIC2018_Task1-2_Training_Input.zip"
_T12_MASKS_ZIP = "ISIC2018_Task1_Training_GroundTruth.zip"

_T3_IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task3_Training_Input.zip"
)
_T3_LABELS_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task3_Training_GroundTruth.zip"
)
_T3_IMAGES_DIR = "ISIC2018_Task3_Training_Input"
_T3_GROUNDTRUTH_CSV = "ISIC2018_Task3_Training_GroundTruth.csv"
_T3_IMAGES_ZIP = "ISIC2018_Task3_Training_Input.zip"
_T3_LABELS_ZIP = "ISIC2018_Task3_Training_GroundTruth.zip"

VALID_TASKS = ("task3", "task1_2")

# MD5 checksums for ISIC 2018 files. Update with values from archive.
# To compute: python -c "import hashlib; h=hashlib.md5();"
#             "f=open('file.zip','rb'); h.update(f.read()); print(h.hexdigest())"
#
# Verified checksums (downloaded and computed):
#   - ISIC2018_Task3_Training_GroundTruth.zip: verified ✓
#   - ISIC2018_Task1_Training_GroundTruth.zip: verified ✓
#
# Large files with multipart uploads (ETag has -N suffix, not usable):
#   - ISIC2018_Task3_Training_Input.zip: requires download (~2.6 GB)
#   - ISIC2018_Task1-2_Training_Input.zip: requires download (~10.4 GB)
_CHECKSUMS: Dict[str, Optional[str]] = {
    "ISIC2018_Task3_Training_GroundTruth.zip": "8302427e4ce0c107559531b9f444abe9",
    "ISIC2018_Task3_Training_Input.zip": None,  # 2.6 GB - multipart, TODO
    "ISIC2018_Task1-2_Training_Input.zip": None,  # 10.4 GB - multipart, TODO
    "ISIC2018_Task1_Training_GroundTruth.zip": "ee5e5db7771d48fa2613abc7cb5c24e2",
}


# ---------------------------------------------------------------------------
# Public download helpers (also imported by isic2018_artifacts.py)
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str, expected_md5: Optional[str] = None) -> None:
    """Stream *url* to *dest* with 1 MB chunks, logging % progress.

    Args:
        url: Source URL to download from.
        dest: Destination file path.
        expected_md5: Expected MD5 checksum (optional). If provided, verifies
                      downloaded file integrity and raises ValueError if mismatch.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        ValueError: If MD5 checksum verification fails.
    """
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        md5_hash = hashlib.md5()

        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                md5_hash.update(chunk)
                downloaded += len(chunk)
                if total:
                    logger.info(
                        "  %.1f%% (%d / %d bytes)",
                        downloaded / total * 100,
                        downloaded,
                        total,
                    )

    if expected_md5 is not None:
        actual_md5 = md5_hash.hexdigest()
        if actual_md5 != expected_md5:
            os.remove(dest)
            raise ValueError(
                f"MD5 checksum mismatch for {os.path.basename(dest)}\n"
                f"  Expected: {expected_md5}\n"
                f"  Got: {actual_md5}\n"
                f"Download was corrupted or incomplete. File removed."
            )


def _extract_zip(zip_path: str, dest_dir: str, flatten: bool = False) -> None:
    """Safely extract zip, guarding against path-traversal attacks.

    Args:
        zip_path: Path to zip file to extract.
        dest_dir: Destination directory.
        flatten: If True and zip has a single top-level directory, extract
                 its contents to dest_dir (flattening structure). If False,
                 extract normally (preserving directory structure).
    """
    abs_dest = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Security check: prevent path traversal
        for member in zf.infolist():
            member_path = os.path.abspath(os.path.join(abs_dest, member.filename))
            if not member_path.startswith(abs_dest + os.sep):
                raise ValueError(f"Unsafe path in zip archive: '{member.filename}'")

        if flatten:
            # Check if all files are in a single top-level directory
            names = zf.namelist()
            if names:
                # Get top-level entries
                top_level = set()
                for name in names:
                    parts = name.split('/')
                    if parts[0]:  # Skip empty parts from trailing slashes
                        top_level.add(parts[0])

                # If only one top-level item and it's a directory, flatten it
                if len(top_level) == 1:
                    top_dir = list(top_level)[0]
                    # Check if it's a directory (has trailing slash or contains files)
                    is_dir = any(name.startswith(top_dir + '/') for name in names)
                    if is_dir:
                        # Extract to temp location and move contents up
                        temp_dir = os.path.join(abs_dest, '.extract_temp')
                        os.makedirs(temp_dir, exist_ok=True)
                        zf.extractall(temp_dir)

                        # Move contents of top_dir to dest_dir
                        source_dir = os.path.join(temp_dir, top_dir)
                        for item in os.listdir(source_dir):
                            src = os.path.join(source_dir, item)
                            dst = os.path.join(abs_dest, item)
                            if os.path.isdir(src):
                                os.makedirs(dst, exist_ok=True)
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                            else:
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                shutil.copy2(src, dst)

                        # Clean up temp dir
                        shutil.rmtree(temp_dir)
                        return

        # Otherwise, extract normally
        zf.extractall(dest_dir)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ISIC2018Dataset(BaseDataset):
    """Unified ISIC 2018 dataset for Task 1/2 (segmentation) or Task 3 (classification).

    Args:
        root (str): Root directory. Defaults to ".".
        task (str): Which ISIC 2018 task to load. One of:

            - ``"task3"`` (default) — 7-class skin lesion classification.
              Downloads images + ``ISIC2018_Task3_Training_GroundTruth.csv``.
            - ``"task1_2"`` — Lesion segmentation & attribute detection.
              Downloads images + binary segmentation masks.

        download (bool): Download missing data automatically. Defaults to False.
        **kwargs: Forwarded to BaseDataset.

    .. note::
        **Licenses differ by task:**

        * ``task="task1_2"`` — **CC-0** (public domain).
          No attribution required.
        * ``task="task3"`` — **CC-BY-NC 4.0**.
          Attribution is required; commercial use is not permitted.
          See https://challenge.isic-archive.com/data/#2018 for citation details.

    Raises:
        ValueError: If task is not one of VALID_TASKS.
        FileNotFoundError: If required paths are missing and download=False.
        requests.HTTPError: If download fails.

    task="task3" directory layout::

        <root>/
            ISIC2018_Task3_Training_GroundTruth.csv
            ISIC2018_Task3_Training_Input/
                ISIC_0024306.jpg ...

    task="task1_2" directory layout::

        <root>/
            ISIC2018_Task1-2_Training_Input/
                ISIC_0024306.jpg ...
            ISIC2018_Task1_Training_GroundTruth/
                ISIC_0024306_segmentation.png ...

    Event attributes for task="task3" (table "isic2018"):
        image_id, path, mel, nv, bcc, akiec, bkl, df, vasc

    Event attributes for task="task1_2" (table "isic2018_task12"):
        image_id, path, mask_path (empty string if mask absent)

    Example::
        >>> dataset = ISIC2018Dataset(root="/data/isic", task="task3", download=True)
        >>> dataset = ISIC2018Dataset(root="/data/isic", task="task1_2", download=True)
    """

    classes: List[str] = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

    def __init__(self, root=".", task="task3", download=False, **kwargs):
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        self.task = task

        if task == "task3":
            self._image_dir = os.path.join(root, _T3_IMAGES_DIR)
            self._label_path = os.path.join(root, _T3_GROUNDTRUTH_CSV)
        else:  # task1_2
            self._image_dir = os.path.join(root, TASK12_IMAGES_DIR)
            self._mask_dir = os.path.join(root, TASK12_MASKS_DIR)

        if download:
            self._download(root)

        self._verify_data(root)
        config_path = self._index_data(root)

        table = "isic2018" if task == "task3" else "isic2018_task12"
        super().__init__(
            root=root,
            tables=[table],
            dataset_name="ISIC2018",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self):
        if self.task == "task3":
            from pyhealth.tasks import ISIC2018Classification
            return ISIC2018Classification()
        return None  # No native segmentation task yet

    @wraps(BaseDataset.set_task)
    def set_task(self, *args, **kwargs):
        return super().set_task(*args, **kwargs)

    def _download(self, root):
        os.makedirs(root, exist_ok=True)
        if self.task == "task3":
            if not os.path.isfile(self._label_path):
                zip_path = os.path.join(root, _T3_LABELS_ZIP)
                # Skip download if ZIP already exists (may be partial/incomplete)
                if not os.path.isfile(zip_path):
                    logger.info("Downloading ISIC 2018 Task 3 labels...")
                    _download_file(
                        _T3_LABELS_URL, zip_path, _CHECKSUMS.get(_T3_LABELS_ZIP)
                    )
                if os.path.isfile(zip_path):
                    _extract_zip(zip_path, root, flatten=True)
                    os.remove(zip_path)
            if not os.path.isdir(self._image_dir):
                zip_path = os.path.join(root, _T3_IMAGES_ZIP)
                # Skip download if ZIP already exists (may be partial/incomplete)
                if not os.path.isfile(zip_path):
                    logger.info("Downloading ISIC 2018 Task 3 images (~8 GB)...")
                    _download_file(
                        _T3_IMAGES_URL, zip_path, _CHECKSUMS.get(_T3_IMAGES_ZIP)
                    )
                if os.path.isfile(zip_path):
                    _extract_zip(zip_path, root, flatten=False)
                    os.remove(zip_path)
        else:  # task1_2
            if not os.path.isdir(self._image_dir):
                zip_path = os.path.join(root, _T12_IMAGES_ZIP)
                # Skip download if ZIP already exists (may be partial/incomplete)
                if not os.path.isfile(zip_path):
                    logger.info("Downloading ISIC 2018 Task 1/2 images (~8 GB)...")
                    _download_file(
                        TASK12_IMAGES_URL, zip_path, _CHECKSUMS.get(_T12_IMAGES_ZIP)
                    )
                if os.path.isfile(zip_path):
                    _extract_zip(zip_path, root, flatten=False)
                    os.remove(zip_path)
            if not os.path.isdir(self._mask_dir):
                zip_path = os.path.join(root, _T12_MASKS_ZIP)
                # Skip download if ZIP already exists (may be partial/incomplete)
                if not os.path.isfile(zip_path):
                    logger.info("Downloading ISIC 2018 Task 1 masks...")
                    _download_file(
                        TASK12_MASKS_URL, zip_path, _CHECKSUMS.get(_T12_MASKS_ZIP)
                    )
                if os.path.isfile(zip_path):
                    _extract_zip(zip_path, root, flatten=False)
                    os.remove(zip_path)

    def _verify_data(self, root):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        if not os.path.isdir(self._image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self._image_dir}\n"
                "Use download=True or obtain manually from "
                "https://challenge.isic-archive.com/data/#2018"
            )
        if self.task == "task3":
            if not os.path.isfile(self._label_path):
                raise FileNotFoundError(
                    f"Ground-truth CSV not found: {self._label_path}\n"
                    "Use download=True or obtain manually from "
                    "https://challenge.isic-archive.com/data/#2018"
                )
            if not list(Path(self._image_dir).glob("*.jpg")):
                raise ValueError(f"No JPEG images found in '{self._image_dir}'")
        else:  # task1_2
            if not os.path.isdir(self._mask_dir):
                raise FileNotFoundError(
                    f"Mask directory not found: {self._mask_dir}\n"
                    "Use download=True or obtain manually from "
                    "https://challenge.isic-archive.com/data/#2018"
                )

    def _index_data(self, root):
        if self.task == "task3":
            return self._index_task3(root)
        return self._index_task12(root)

    def _index_task3(self, root):
        df = pd.read_csv(self._label_path)
        image_names = {f.stem for f in Path(self._image_dir).glob("*.jpg")}
        df = df[df["image"].isin(image_names)].copy()
        df.rename(columns={c.upper(): c for c in self.classes}, inplace=True)
        df.rename(columns={"image": "image_id"}, inplace=True)
        df["patient_id"] = df["image_id"]
        df["path"] = df["image_id"].apply(
            lambda img_id: os.path.join(self._image_dir, f"{img_id}.jpg")
        )
        metadata_path = os.path.join(root, "isic2018-metadata-pyhealth.csv")
        df.to_csv(metadata_path, index=False)

        config = {
            "version": "1.0",
            "tables": {
                "isic2018": {
                    "file_path": "isic2018-metadata-pyhealth.csv",
                    "patient_id": "patient_id",
                    "timestamp": None,
                    "attributes": ["path", "image_id"] + list(self.classes),
                }
            },
        }
        config_path = os.path.join(root, "isic2018-config-pyhealth.yaml")
        with open(config_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
        logger.info(
            "ISIC2018Dataset (task3): indexed %d images → %s", len(df), metadata_path
        )
        return config_path

    def _index_task12(self, root):
        image_dir = Path(self._image_dir)
        mask_dir = Path(self._mask_dir)
        images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.JPG"))
        if not images:
            raise ValueError(f"No images found in '{self._image_dir}'")
        records = []
        for img_path in images:
            image_id = img_path.stem
            mask_path = mask_dir / f"{image_id}_segmentation.png"
            records.append({
                "image_id": image_id,
                "patient_id": image_id,
                "path": str(img_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
            })
        df = pd.DataFrame(records)
        metadata_path = os.path.join(root, "isic2018-task12-metadata-pyhealth.csv")
        df.to_csv(metadata_path, index=False)
        config = {
            "version": "1.0",
            "tables": {
                "isic2018_task12": {
                    "file_path": "isic2018-task12-metadata-pyhealth.csv",
                    "patient_id": "patient_id",
                    "timestamp": None,
                    "attributes": ["path", "mask_path", "image_id"],
                }
            },
        }
        config_path = os.path.join(root, "isic2018-task12-config-pyhealth.yaml")
        with open(config_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
        logger.info(
            "ISIC2018Dataset (task1_2): indexed %d images (%d with masks) → %s",
            len(df),
            (df["mask_path"] != "").sum(),
            metadata_path,
        )
        return config_path
