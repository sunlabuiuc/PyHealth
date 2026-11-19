"""Jichi Medical University diabetic retinopathy fundus dataset."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

if TYPE_CHECKING:
    from ..tasks.jichi_davis_grading import JichiDavisGrading

logger = logging.getLogger(__name__)

JICHI_LIST_CSV = "list.csv"
_DERIVED_CSV = "jichi-metadata-pyhealth.csv"
_FINGERPRINT = "jichi-metadata-pyhealth.sha1"
_FILENAME = re.compile(
    r"^(?P<patient>\d+)_(?P<index>\d+)_(?P<eye>[LR])\.jpg$", re.IGNORECASE
)
_LABEL_COLUMNS = {
    "Davis_grading_of_one_figure": "davis_one",
    "Davis_grading_of_concatenated_figures": "davis_concatenated",
}


def parse_jichi_filename(name: str) -> tuple[str, int, str] | None:
    """Split a Jichi image file name into patient id, image index and eye.

    The release does not ship a patient column. The file names have the form
    ``<patient>_<index>_<eye>.jpg``, and the 2,740 distinct ``<patient>``
    prefixes match the 2,740 patients reported by Takahashi et al. (2017),
    so the prefix is used as the patient id.

    Args:
        name: Image file name, for example ``"91_3_L.jpg"``.

    Returns:
        ``(patient_id, image_index, eye)`` or ``None`` if the name does not
        follow the pattern.

    Examples:
        >>> parse_jichi_filename("91_3_L.jpg")
        ('91', 3, 'L')
        >>> parse_jichi_filename("notes.txt") is None
        True
    """
    match = _FILENAME.match(name)
    if match is None:
        return None
    return match["patient"], int(match["index"]), match["eye"].upper()


class JichiDRDataset(BaseDataset):
    """Jichi diabetic retinopathy fundus photographs with Davis grades.

    9,939 posterior-pole fundus photographs (848 x 848 JPEG) of 2,740
    patients from Jichi Medical University, Japan, each graded on the
    modified Davis scale: no DR (``ndr``), simple DR (``sdr``),
    pre-proliferative DR (``ppdr``) and proliferative DR (``pdr``). Two grade
    columns are released: the grade of the single image
    (``davis_one``) and the grade given when the image was read together
    with the patient's other images (``davis_concatenated``). They disagree
    on about 9% of images; the task picks one with ``label_column``.

    One event is created per image. ``patient_id`` is the numeric prefix of
    the file name (see :func:`parse_jichi_filename`); this is inferred from
    the file names, not documented by the authors. Patients have about 3.6
    images each, so split by patient, for example with
    :class:`~pyhealth.datasets.PatientSplit`. ``eye`` (``L``/``R``) and
    ``image_index`` are kept as attributes. The derived metadata CSV is
    written to a metadata directory (by default under the PyHealth cache,
    keyed by a hash of the image folder, ``list.csv`` and the image
    file names), never into
    ``root``. Rows whose image is missing are dropped with a warning.

    Differences from CLIMB:
        - CLIMB's loader (named ``Jinchi``) uses four classes including
          NDR, while its dataset table says "3 classes (SDR, PPDR, PDR)".
          This loader keeps all four grades; the task maps them to ordered
          integers ``0..3``.
        - CLIMB uses a seeded stratified 80/20 image split that ignores
          patients, so a patient's images land in both train and test.
          Here every image carries its patient id for patient-level splits.
        - CLIMB uses the one-figure grade only; both columns are kept here.

    Getting the data:
        - Source: https://figshare.com/articles/dataset/4879853
          ("Davis Grading of One and Concatenated Figures",
          DOI 10.6084/m9.figshare.4879853.v1). Paper: Takahashi et al.,
          PLoS ONE 2017 (see Citation below).
        - Access: open download, no account, registration or data-use
          agreement. On the figshare page click "Download", or use the
          direct link below.
        - Download: the single file ``dmr.tar`` (886,820,858 bytes, about
          887 MB)::

              wget -O dmr.tar https://ndownloader.figshare.com/files/8169134

        - Unpack: despite the ``.tar`` name the archive is gzip-compressed,
          so use ``tar -xzf dmr.tar``. This creates a ``dmr/`` folder.
        - ``root``: the extracted ``dmr/`` folder, or the folder that
          contains it::

              <root>/                 # or <root>/dmr/
                  list.csv            # Image, Davis_grading_of_one_figure,
                                      # Davis_grading_of_concatenated_figures
                  1_1_R.jpg           # <patient>_<index>_<eye>.jpg
                  1_2_L.jpg
                  ...                 # 9,939 JPEGs listed in list.csv

          Nothing is written into ``root``.

    Licence:
        CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/): reuse,
        including commercial use and derivatives, is allowed with
        attribution; there are no NC or ND terms.

    Citation:
        Takahashi H, Tampo H, Arai Y, Inoue Y, Kawashima H. Applying
        artificial intelligence to disease staging: Deep learning for
        improved staging of diabetic retinopathy. PLoS ONE 12(6): e0179790
        (2017). https://doi.org/10.1371/journal.pone.0179790

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Args:
        root: The ``dmr/`` folder, or a folder containing it.
        dataset_name: Optional dataset name. Defaults to ``"jichi_dr"``.
        config_path: Optional YAML config. Defaults to the bundled
            ``configs/jichi.yaml``.
        cache_dir: Directory for PyHealth caches. Also the default parent of
            the metadata directory.
        metadata_dir: Explicit directory for the derived metadata CSV. The
            CSV is written there directly (no hash subfolder), next to a
            ``jichi-metadata-pyhealth.sha1`` fingerprint of the image folder,
            ``list.csv`` and the image names; it is rebuilt whenever that
            fingerprint changes, so pointing the same folder at different
            data does not reuse stale rows.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Regenerate the derived metadata CSV.

    Examples:
        >>> from pyhealth.datasets import JichiDRDataset
        >>> from pyhealth.tasks import JichiDavisGrading
        >>> dataset = JichiDRDataset(root="/data/jichi/dmr")  # doctest: +SKIP
        >>> samples = dataset.set_task(JichiDavisGrading())  # doctest: +SKIP
    """

    default_dataset_name = "jichi_dr"
    default_table = "jichi"

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        source_root = Path(root).expanduser().resolve()
        if (
            not (source_root / JICHI_LIST_CSV).is_file()
            and (source_root / "dmr" / JICHI_LIST_CSV).is_file()
        ):
            source_root = source_root / "dmr"
        self.image_root = source_root
        self.list_csv = source_root / JICHI_LIST_CSV
        if not self.list_csv.is_file():
            raise FileNotFoundError(
                f"{JICHI_LIST_CSV} not found under {source_root} or "
                f"{source_root / 'dmr'}. Download dmr.tar from "
                "https://figshare.com/articles/dataset/4879853"
            )

        name = dataset_name or self.default_dataset_name
        digest = hashlib.sha1(str(self.image_root).encode("utf-8"))
        digest.update(self.list_csv.read_bytes())
        self._present = sorted(
            n for n in os.listdir(self.image_root) if n.lower().endswith(".jpg")
        )
        digest.update("\n".join(self._present).encode("utf-8"))
        fingerprint = digest.hexdigest()
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name) / fingerprint[:10]
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _DERIVED_CSV
        fingerprint_path = self.metadata_dir / _FINGERPRINT
        with FileLock(str(self.metadata_path) + ".lock"):
            stale = (
                not fingerprint_path.is_file()
                or fingerprint_path.read_text().strip() != fingerprint
            )
            if refresh_metadata or stale or not self.metadata_path.exists():
                self.prepare_metadata()
                fingerprint_path.write_text(fingerprint + "\n")

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "jichi.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self) -> None:
        """Write the derived metadata CSV with one row per available image.

        Raises:
            ValueError: If ``list.csv`` lacks a required column.
            FileNotFoundError: If none of the listed images is found.

        Examples:
            >>> ds = JichiDRDataset(root="/data/jichi/dmr")  # doctest: +SKIP
            >>> ds.prepare_metadata()  # doctest: +SKIP
        """
        table = pd.read_csv(self.list_csv, dtype=str)
        required = ["Image", *_LABEL_COLUMNS]
        missing_cols = [c for c in required if c not in table.columns]
        if missing_cols:
            raise ValueError(f"{self.list_csv} is missing columns {missing_cols}.")
        present = set(self._present)
        rows = []
        n_missing = n_bad_name = 0
        for record in table.to_dict(orient="records"):
            image = str(record["Image"]).strip()
            parsed = parse_jichi_filename(image)
            if parsed is None:
                n_bad_name += 1
                continue
            if image not in present:
                n_missing += 1
                continue
            patient_id, index, eye = parsed
            row = {
                "patient_id": patient_id,
                "image_id": image,
                "path": str(self.image_root / image),
                "image_index": index,
                "eye": eye,
            }
            for column, short in _LABEL_COLUMNS.items():
                row[short] = str(record[column]).strip().lower()
            rows.append(row)
        if n_bad_name:
            logger.warning(
                "Jichi: skipped %d rows whose file name does not match "
                "<patient>_<index>_<eye>.jpg.",
                n_bad_name,
            )
        if not rows:
            raise FileNotFoundError(
                f"None of the images listed in {self.list_csv} were found "
                f"under {self.image_root}."
            )
        if n_missing:
            logger.warning(
                "Jichi: dropping %d of %d rows whose image is missing.",
                n_missing,
                len(table),
            )
        tmp_path = self.metadata_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.metadata_path)
        logger.info("Wrote %d Jichi images to %s", len(rows), self.metadata_path)

    @property
    def default_task(self) -> JichiDavisGrading:
        """Return the default four-grade Davis task on the one-figure grade.

        Returns:
            A :class:`~pyhealth.tasks.JichiDavisGrading` instance.

        Examples:
            >>> ds = JichiDRDataset(root="/data/jichi/dmr")  # doctest: +SKIP
            >>> ds.default_task.label_column  # doctest: +SKIP
            'davis_one'
        """
        from ..tasks.jichi_davis_grading import JichiDavisGrading

        return JichiDavisGrading()
