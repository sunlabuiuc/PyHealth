"""BCSS breast-cancer semantic segmentation dataset, tiled for classification."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir
from .splitter import SubjectSplit

if TYPE_CHECKING:
    from ..tasks.bcss_tile_classification import BCSSTileClassification

logger = logging.getLogger(__name__)

BCSS_CLASS_CODES: dict[int, str] = {
    1: "tumor",
    2: "stroma",
    3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris",
    5: "glandular_secretions",
    6: "blood",
    7: "exclude",
    8: "metaplasia_NOS",
    9: "fat",
    10: "plasma_cells",
    11: "other_immune_infiltrate",
    12: "mucoid_material",
    13: "normal_acinus_or_duct",
    14: "lymphatics",
    15: "undetermined",
    16: "nerve",
    17: "skin_adnexa",
    18: "blood_vessel",
    19: "angioinvasion",
    20: "dcis",
    21: "other",
}
"""Mask values from ``meta/gtruth_codes.tsv``. Value 0 (``outside_roi``) is
"don't care" and is never a class."""

BCSS_MAIN_CLASSES: tuple[int, ...] = (1, 2, 3, 4)
"""The four main classes used by the BCSS paper's benchmark (and by default)."""

BCSS_TEST_SITES: frozenset[str] = frozenset({"OL", "LL", "E2", "EW", "GM", "S3"})
"""TCGA tissue-source sites of the official held-out test set (45 ROIs)."""

_ROI_NAME = re.compile(
    r"^(?P<patient>TCGA-(?P<site>[A-Z0-9]{2})-[A-Z0-9]{4})-[^_]+_"
    r"xmin(?P<xmin>\d+)_ymin(?P<ymin>\d+)_MPP-[0-9.]+\.png$"
)
_INDEX_CSV = "bcss-tiles-pyhealth.csv"


def parse_bcss_roi_name(name: str) -> dict[str, str] | None:
    """Parse a BCSS ROI file name into patient, site and ROI id.

    Args:
        name: File name such as
            ``"TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"``.

    Returns:
        ``{"patient_id", "site", "roi_id"}``, or ``None`` if the name does
        not follow the BCSS pattern.

    Examples:
        >>> parse_bcss_roi_name("TCGA-E2-A14N-DX1_xmin21383_ymin66838_MPP-0.2500.png")
        {'patient_id': 'TCGA-E2-A14N', 'site': 'E2', 'roi_id': 'TCGA-E2-A14N-DX1_xmin21383_ymin66838_MPP-0.2500'}
        >>> parse_bcss_roi_name("README.md") is None
        True
    """
    match = _ROI_NAME.match(name)
    if match is None:
        return None
    return {
        "patient_id": match["patient"],
        "site": match["site"],
        "roi_id": name[: -len(".png")],
    }


def tile_label(
    mask_tile: np.ndarray, class_codes: Sequence[int], min_purity: float
) -> tuple[int, float] | None:
    """Return the label of one mask tile, or ``None`` if it is not pure enough.

    Purity is the share of *all* pixels of the tile that carry the most
    frequent code among ``class_codes``. Don't-care pixels (value 0) and
    pixels of other codes count against purity, so a tile that is half
    outside the ROI never passes ``min_purity > 0.5``.

    Args:
        mask_tile: 2-D integer array of mask codes.
        class_codes: Codes that may label a tile. ``0`` is ignored if given.
        min_purity: Minimum purity in ``(0, 1]``.

    Returns:
        ``(code, purity)`` or ``None``.

    Examples:
        >>> import numpy as np
        >>> tile = np.array([[1, 1], [1, 0]])
        >>> tile_label(tile, (1, 2), 0.75)
        (1, 0.75)
        >>> tile_label(tile, (1, 2), 0.9) is None
        True
    """
    counts = np.bincount(mask_tile.ravel(), minlength=max(class_codes) + 1)
    best_code, best_count = 0, 0
    for code in class_codes:
        if code != 0 and counts[code] > best_count:
            best_code, best_count = int(code), int(counts[code])
    if best_count == 0:
        return None
    purity = best_count / mask_tile.size
    if purity < min_purity:
        return None
    return best_code, float(purity)


class BCSSDataset(BaseDataset):
    """BCSS breast-cancer pathology ROIs, tiled into labelled patches.

    The Breast Cancer Semantic Segmentation (BCSS) dataset has 151
    hand-annotated regions of interest (ROIs), one per TCGA-BRCA slide and
    patient, at 0.25 microns per pixel (about 1.8k to 6.7k pixels per side).
    Each ROI has an RGB image and a mask whose pixel values follow
    ``meta/gtruth_codes.tsv`` (see :data:`BCSS_CLASS_CODES`). Mask value
    ``0`` means "outside ROI / don't care" and never labels a tile.

    BCSS is a segmentation dataset. To use it for patch classification this
    loader cuts each ROI into square tiles once and writes them, plus an
    index CSV, to a tile directory in the PyHealth cache (never into
    ``root``). One event is created per kept tile. The recipe is fully set
    by the constructor arguments:

    - ``tile_size`` (default 224): side of each square tile in pixels, at
      the native 0.25 MPP (no resizing before tiling).
    - ``stride`` (default ``tile_size``, non-overlapping): step between
      tile origins. Partial tiles at the right and bottom edges are dropped.
    - ``min_purity`` (default 0.9): a tile is kept only if one code in
      ``class_codes`` covers at least this share of *all* its pixels
      (:func:`tile_label`); don't-care pixels count against purity.
    - ``class_codes`` (default :data:`BCSS_MAIN_CLASSES`, i.e. tumor,
      stroma, lymphocytic infiltrate, necrosis/debris): the codes that may
      label a tile. Tiles dominated by other codes are dropped.
    - ``image_dir`` (default ``"rgbs_colorNormalized"``): which RGB folder
      to tile, so the stain-normalised or raw images can be chosen.

    These are this loader's own choices, not CLIMB's, and they will not
    reproduce CLIMB's numbers. Changing any of them builds a new tile
    directory (the directory name holds a hash of the recipe and the
    source file list).

    ``patient_id`` is the TCGA patient barcode (``TCGA-XX-YYYY``), and the
    events carry ``site`` (the TCGA tissue-source site, ``XX``), ``roi_id``,
    the tile position and ``split``: ``"test"`` for the official held-out
    sites :data:`BCSS_TEST_SITES` and ``"train"`` otherwise. Tiles from one
    ROI are strongly correlated, so always split by patient or site; use
    :meth:`official_split` to get the official site-held-out split.

    Differences from CLIMB:
        - CLIMB crops OpenCV contour bounding boxes per class, from 224 up
          to 1024 pixels, so a crop can contain other tissue and crop sizes
          vary before resizing. This loader uses the fixed-grid, purity
          thresholded recipe above.
        - CLIMB carves its validation set out of the training images at
          random (90/10 by image, seed 42). :meth:`official_split` carves
          validation patients instead, so no ROI is in two parts.
        - CLIMB expects an ``images/`` folder; the public release names it
          ``rgbs_colorNormalized/``. Both are accepted.
        - CLIMB needs OpenCV; this loader needs only Pillow and NumPy.

    Getting the data:
        - Source: https://github.com/PathologyDataScience/BCSS
        - Access: open, no account needed.
        - Download: the public Google Drive folder linked from the README
          (``masks/``, ``rgbs_colorNormalized/`` and ``meta/``, about
          3.5 GB), or the repository's ``download_crowdsource_dataset.py``
          script, which pulls the ROIs from the Digital Slide Archive.
        - ``root``: the folder holding ``masks/`` and
          ``rgbs_colorNormalized/`` (or ``images/``), with matching PNG file
          names.

    Licence:
        Data: CC0 1.0 (public domain). Code in the BCSS repository: MIT.

    Citation:
        Amgad M, Elfandy H, Hussein H, et al. Structured crowdsourcing
        enables convolutional segmentation of histology images.
        Bioinformatics 35(18):3461-3467 (2019).
        https://doi.org/10.1093/bioinformatics/btz083

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Args:
        root: Folder holding ``masks/`` and the RGB folder.
        tile_size: Tile side in pixels.
        stride: Step between tiles. Defaults to ``tile_size``.
        min_purity: Minimum share of a tile's pixels carrying its label.
        class_codes: Mask codes that may label a tile.
        image_dir: RGB folder name under ``root``. Defaults to
            ``"rgbs_colorNormalized"``, falling back to ``"images"``.
        dataset_name: Optional dataset name. Defaults to ``"bcss"``.
        config_path: Optional YAML config. Defaults to the bundled
            ``configs/bcss.yaml``.
        cache_dir: Directory for PyHealth caches. Also the default parent of
            the tile directory.
        metadata_dir: Explicit parent directory for the tile directories.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Re-tile even if a finished tile directory exists.

    Raises:
        ValueError: If a tiling parameter is out of range.
        FileNotFoundError: If the mask or RGB folder, or any ROI pair, is
            missing.

    Examples:
        >>> from pyhealth.datasets import BCSSDataset
        >>> from pyhealth.tasks import BCSSTileClassification
        >>> dataset = BCSSDataset(root="/data/bcss", tile_size=224, min_purity=0.9)  # doctest: +SKIP
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     BCSSTileClassification(), split=dataset.official_split(val_ratio=0.1)
        ... )
    """

    default_dataset_name = "bcss"
    default_table = "bcss"

    def __init__(
        self,
        root: str,
        tile_size: int = 224,
        stride: int | None = None,
        min_purity: float = 0.9,
        class_codes: Sequence[int] = BCSS_MAIN_CLASSES,
        image_dir: str | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.tile_size = int(tile_size)
        self.stride = int(stride) if stride is not None else self.tile_size
        self.min_purity = float(min_purity)
        self.class_codes = tuple(sorted({int(c) for c in class_codes}))
        if self.tile_size <= 0 or self.stride <= 0:
            raise ValueError("tile_size and stride must be positive.")
        if not 0.0 < self.min_purity <= 1.0:
            raise ValueError("min_purity must be in (0, 1].")
        if not self.class_codes or any(c not in BCSS_CLASS_CODES for c in self.class_codes):
            raise ValueError(
                f"class_codes must be a non-empty subset of {sorted(BCSS_CLASS_CODES)}."
            )

        self.source_root = Path(root).expanduser().resolve()
        self.mask_root = self.source_root / "masks"
        if image_dir is None:
            image_dir = (
                "rgbs_colorNormalized"
                if (self.source_root / "rgbs_colorNormalized").is_dir()
                else "images"
            )
        self.image_root = self.source_root / image_dir
        for folder in (self.mask_root, self.image_root):
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"BCSS folder not found: {folder}. See the 'Getting the "
                    "data' section of BCSSDataset."
                )
        self._rois = self._find_rois()

        name = dataset_name or self.default_dataset_name
        recipe = {
            "root": str(self.source_root),
            "image_dir": image_dir,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "min_purity": self.min_purity,
            "class_codes": list(self.class_codes),
            "rois": [
                [roi, os.path.getsize(self.mask_root / f"{roi}.png"),
                 os.path.getsize(self.image_root / f"{roi}.png")]
                for roi in sorted(self._rois)
            ],
        }
        key = hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]
        parent = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.tile_dir = parent / (
            f"tiles_t{self.tile_size}_s{self.stride}_"
            f"p{round(self.min_purity * 100)}_{key}"
        )
        self.tile_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.tile_dir / _INDEX_CSV
        with FileLock(str(self.tile_dir) + ".lock"):
            if refresh_metadata or not self.metadata_path.exists():
                self.prepare_metadata()

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "bcss.yaml")
        super().__init__(
            root=str(self.tile_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _find_rois(self) -> dict[str, dict[str, str]]:
        """Return parsed ROI names present in both the mask and RGB folders."""
        rois = {}
        unpaired = []
        for name in sorted(os.listdir(self.mask_root)):
            parsed = parse_bcss_roi_name(name)
            if parsed is None:
                continue
            if not (self.image_root / name).is_file():
                unpaired.append(name)
                continue
            rois[parsed["roi_id"]] = parsed
        if unpaired:
            logger.warning(
                "BCSS: %d masks have no RGB image in %s and are skipped: %s",
                len(unpaired),
                self.image_root,
                unpaired[:5],
            )
        if not rois:
            raise FileNotFoundError(
                f"No BCSS mask/RGB pairs found under {self.source_root}."
            )
        return rois

    def prepare_metadata(self) -> None:
        """Tile every ROI and write the tile PNGs and the index CSV.

        The index CSV is written last, so an interrupted run is redone on
        the next construction.

        Examples:
            >>> ds = BCSSDataset(root="/data/bcss")  # doctest: +SKIP
            >>> ds.prepare_metadata()  # doctest: +SKIP
        """
        from PIL import Image

        rows = []
        old_limit = Image.MAX_IMAGE_PIXELS
        # ROIs are large but trusted dataset files; lift Pillow's
        # decompression-bomb guard while reading them.
        Image.MAX_IMAGE_PIXELS = None
        try:
            for roi_id, info in sorted(self._rois.items()):
                rows.extend(self._tile_roi(Image, roi_id, info))
        finally:
            Image.MAX_IMAGE_PIXELS = old_limit
        columns = [
            "patient_id", "roi_id", "site", "split", "tile_id", "path", "x", "y",
            "label_code", "label", "purity",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        tmp_path = self.metadata_path.with_suffix(".csv.tmp")
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.metadata_path)
        logger.info(
            "BCSS: wrote %d tiles from %d ROIs to %s",
            len(frame),
            len(self._rois),
            self.tile_dir,
        )

    def _tile_roi(self, image_module, roi_id: str, info: dict[str, str]) -> list[dict]:
        """Cut one ROI into tiles and save the kept ones as PNG files."""
        mask = np.asarray(image_module.open(self.mask_root / f"{roi_id}.png"))
        if mask.ndim == 3:
            mask = mask[..., 0]
        rgb = image_module.open(self.image_root / f"{roi_id}.png").convert("RGB")
        if rgb.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(
                f"BCSS ROI {roi_id}: RGB size {rgb.size} does not match mask "
                f"size {(mask.shape[1], mask.shape[0])}."
            )
        rgb = np.asarray(rgb)
        out_dir = self.tile_dir / roi_id
        out_dir.mkdir(parents=True, exist_ok=True)
        split = "test" if info["site"] in BCSS_TEST_SITES else "train"
        height, width = mask.shape
        size = self.tile_size
        rows = []
        for y in range(0, height - size + 1, self.stride):
            for x in range(0, width - size + 1, self.stride):
                result = tile_label(
                    mask[y : y + size, x : x + size], self.class_codes, self.min_purity
                )
                if result is None:
                    continue
                code, purity = result
                tile_id = f"{roi_id}_x{x}_y{y}"
                path = out_dir / f"x{x}_y{y}.png"
                image_module.fromarray(rgb[y : y + size, x : x + size]).save(path)
                rows.append(
                    {
                        "patient_id": info["patient_id"],
                        "roi_id": roi_id,
                        "site": info["site"],
                        "split": split,
                        "tile_id": tile_id,
                        "path": str(path),
                        "x": x,
                        "y": y,
                        "label_code": code,
                        "label": BCSS_CLASS_CODES[code],
                        "purity": round(purity, 4),
                    }
                )
        return rows

    def official_split(self, val_ratio: float = 0.0, seed: int = 0) -> SubjectSplit:
        """Return the official site-held-out split as a ``SubjectSplit``.

        Patients from :data:`BCSS_TEST_SITES` form the test part. With
        ``val_ratio > 0`` a seeded random share of the remaining patients
        (whole ROIs, never single tiles) becomes a validation part.

        Args:
            val_ratio: Share of training patients moved to validation.
            seed: Seed for choosing the validation patients.

        Returns:
            ``SubjectSplit((train, test))`` or
            ``SubjectSplit((train, val, test))``.

        Raises:
            ValueError: If a part would be empty.

        Examples:
            >>> ds = BCSSDataset(root="/data/bcss")  # doctest: +SKIP
            >>> train, val, test = ds.set_task(split=ds.official_split(0.1))  # doctest: +SKIP
        """
        train = sorted(
            {i["patient_id"] for i in self._rois.values() if i["site"] not in BCSS_TEST_SITES}
        )
        test = sorted(
            {i["patient_id"] for i in self._rois.values() if i["site"] in BCSS_TEST_SITES}
        )
        if not train or not test:
            raise ValueError(
                "The official split needs ROIs from both training and test sites."
            )
        if val_ratio <= 0:
            return SubjectSplit((train, test))
        order = np.random.default_rng(seed).permutation(len(train))
        n_val = max(1, round(len(train) * val_ratio))
        if n_val >= len(train):
            raise ValueError("val_ratio leaves no training patients.")
        val = [train[i] for i in order[:n_val]]
        train = [train[i] for i in order[n_val:]]
        return SubjectSplit((train, val, test))

    @property
    def default_task(self) -> BCSSTileClassification:
        """Return the default tile-classification task for this recipe.

        Returns:
            A :class:`~pyhealth.tasks.BCSSTileClassification` over this
            dataset's ``class_codes``.

        Examples:
            >>> ds = BCSSDataset(root="/data/bcss")  # doctest: +SKIP
            >>> ds.default_task.classes  # doctest: +SKIP
            ['tumor', 'stroma', 'lymphocytic_infiltrate', 'necrosis_or_debris']
        """
        from ..tasks.bcss_tile_classification import BCSSTileClassification

        return BCSSTileClassification(
            classes=[BCSS_CLASS_CODES[c] for c in self.class_codes]
        )
