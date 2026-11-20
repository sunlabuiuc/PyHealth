"""Lobachevsky University Electrocardiography Database (LUDB).

Authors: Shray Srivastava (ssriv5@illinois.edu), Sid Wanjara (swanj2@illinois.edu)

Paper: J. Park et al. "Benchmarking ECG Delineation using Deep Neural
Network-based Semantic Segmentation Models." CHIL 2025, PMLR 287:63-88.
https://proceedings.mlr.press/v287/park25a.html

Description: LUDB has 200 ten-second, 12-lead ECGs sampled at 500 Hz, one
per subject. Cardiologists marked where every P wave, QRS complex and T wave
starts and ends on each lead. This module indexes the records and the
diagnoses in ``ludb.csv`` so a task can read them.

Dataset: https://physionet.org/content/ludb/1.0.1/
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pyhealth.tasks import ECGDelineationLUDB

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

CSV_NAME = "ludb-pyhealth.csv"


def _clean_cell(value: str) -> str:
    """Clean one cell of ludb.csv.

    The cells end with a newline and some hold several lines (one diagnosis
    per line), so this strips each line and joins them with "; ".

    Notes:
        How to write it:

        1. Turn the value into a string and split it into lines
           (``str.splitlines``).
        2. Strip each line and drop the empty ones.
        3. Join what's left with ``"; "``.

        Real cells look like ``"F\\n"`` or ``"Left atrial hypertrophy\\nLeft
        ventricular hypertrophy\\n"``, so ``"F\\n"`` becomes ``"F"`` and the
        second becomes ``"Left atrial hypertrophy; Left ventricular
        hypertrophy"``.
    """
    lines = [line.strip() for line in str(value).splitlines()]
    return "; ".join(line for line in lines if line)


class LUDBDataset(BaseDataset):
    """Lobachevsky University Electrocardiography Database (LUDB).

    200 ECG records, 10 s each, 12 leads at 500 Hz. Each record is a
    different subject, so one record is one patient and ``patient_id`` is
    the record number (``"1"`` to ``"200"``). The P, QRS and T annotations
    of each lead are in a file named after the lead, for example ``1.ii``.

    On first use the dataset writes ``ludb-pyhealth.csv`` with one row per
    record: ``signal_file`` (the WFDB record path, without extension) and
    ``sex``, ``age``, ``rhythm`` and ``electric_axis`` from ``ludb.csv``.
    The signals and annotations are read later by
    :class:`~pyhealth.tasks.ECGDelineationLUDB`.

    Getting the data: download the ZIP of LUDB 1.0.1 (about 15 MB, open
    access) from https://physionet.org/content/ludb/1.0.1/ and unzip it.
    ``root`` is the unzipped folder, which holds ``ludb.csv`` and ``data/``.

    Args:
        root: The LUDB folder (or its ``data/`` folder).
        dataset_name: Name of the dataset. Defaults to ``"ludb"``.
        config_path: YAML config. Defaults to ``configs/ludb.yaml``.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`, e.g.
            ``cache_dir`` or ``dev``.

    Raises:
        FileNotFoundError: If ``root`` is not a folder or has no records.

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> dataset = LUDBDataset(root="/path/to/ludb/1.0.1")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> event = dataset.get_patient("1").get_events("records")[0]  # doctest: +SKIP
        >>> event.rhythm  # doctest: +SKIP
        'Sinus bradycardia'
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the metadata CSV if needed, then load it as a BaseDataset.

        Notes:
            How to write it (copy the shape of ``HelsinkiNeonatalDataset``
            in helsinki_neonatal.py):

            1. If no ``config_path`` was given, use ``configs/ludb.yaml``
               next to this file (``Path(__file__).parent``).
            2. Save ``root`` on ``self`` and call ``self.prepare_metadata()``.
               This has to happen before step 3, because BaseDataset reads
               ludb-pyhealth.csv straight away.
            3. Call ``super().__init__`` with
               ``root=resolve_metadata_root(root, "ludb", CSV_NAME)`` (this
               points at the PyHealth cache if the CSV had to be written
               there), ``tables=["records"]`` (the table name in the YAML),
               the dataset name and the config path, and pass ``**kwargs``
               through.

            BaseDataset turns empty CSV cells into None, so later on events
            a missing electric axis is ``None``, not ``""``.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "ludb.yaml")
        self.root = root
        self.prepare_metadata()
        super().__init__(
            root=resolve_metadata_root(root, "ludb", CSV_NAME),
            tables=["records"],
            dataset_name=dataset_name or "ludb",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ludb-pyhealth.csv with one row per record.

        Finds every ``<N>.hea`` record, looks up its row in ``ludb.csv`` and
        writes the result next to the data (or to the PyHealth cache if the
        folder is read only). Does nothing if the CSV already exists.

        Raises:
            FileNotFoundError: If ``root`` is not a folder or has no records.

        Notes:
            How to write it:

            1. If ``metadata_csv_exists(root, "ludb", CSV_NAME)`` is true,
               return. (So an old CSV is never rewritten; delete it by hand
               if you change the columns.)
            2. If ``root`` is not a folder, raise FileNotFoundError.
            3. The records are in ``root / "data"`` if that folder exists,
               otherwise in ``root`` itself (the user may pass ``data/``).
            4. Loop over ``record_dir.glob("*.hea")`` and keep the file
               names that are numbers (``header.stem.isdigit()``). If there
               are none, raise FileNotFoundError with the download link.
            5. Read the metadata with ``_read_ludb_csv``.
            6. For each record id, sorted as numbers (so 2 comes before 10),
               build a dict with ``patient_id``, ``signal_file`` and the four
               metadata fields (use ``""`` if the record isn't in ludb.csv).
               ``signal_file`` is the record path *without* ".hea", because
               that's what wfdb.rdsamp expects.
            7. Save the rows with ``write_metadata_csv(pd.DataFrame(rows),
               root, "ludb", CSV_NAME)``; it falls back to the cache folder
               if ``root`` is read only.

            The column names must match the ``attributes`` in ludb.yaml.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, "ludb", CSV_NAME):
            return
        if not root.is_dir():
            raise FileNotFoundError(f"LUDB root {root} is not a folder.")

        # PhysioNet puts the records in a data/ folder next to ludb.csv.
        record_dir = root / "data" if (root / "data").is_dir() else root
        record_ids = []
        for header in record_dir.glob("*.hea"):
            if header.stem.isdigit():
                record_ids.append(int(header.stem))
        if not record_ids:
            raise FileNotFoundError(
                f"No LUDB records (<N>.hea) found in {record_dir}. Download "
                "LUDB from https://physionet.org/content/ludb/1.0.1/."
            )

        metadata = self._read_ludb_csv(root, record_dir)
        rows = []
        for record_id in sorted(record_ids):
            info = metadata.get(str(record_id), {})
            rows.append(
                {
                    "patient_id": str(record_id),
                    "signal_file": str((record_dir / str(record_id)).absolute()),
                    "sex": info.get("sex", ""),
                    "age": info.get("age", ""),
                    "rhythm": info.get("rhythm", ""),
                    "electric_axis": info.get("electric_axis", ""),
                }
            )
        write_metadata_csv(pd.DataFrame(rows), root, "ludb", CSV_NAME)

    @staticmethod
    def _read_ludb_csv(root: Path, record_dir: Path) -> dict[str, dict[str, str]]:
        """Read sex, age, rhythm and electric axis from ludb.csv.

        Returns a dict from record id to its values, or an empty dict (with
        a warning) if ludb.csv is missing. Ages over 89 stay as ">89".

        Raises:
            ValueError: If ludb.csv has no ID column.

        Notes:
            How to write it:

            1. Look for ``ludb.csv`` in ``root``, then in the folder above
               ``record_dir``. If it's in neither, log a warning and return
               ``{}`` so the dataset still loads without metadata.
            2. Read it with ``pd.read_csv(path, dtype=str,
               keep_default_na=False)``. ``dtype=str`` keeps ids like "1" as
               text; ``keep_default_na=False`` makes empty cells ``""``
               instead of NaN.
            3. If there is no "ID" column, raise ValueError.
            4. For each row (``table.iterrows()``), clean these columns with
               ``_clean_cell``: "Sex", "Age", "Rhythms" and "Electric axis
               of the heart". Use ``row.get(column, "")`` so a missing
               column doesn't crash.
            5. The axis cells read "Electric axis of the heart: normal", so
               remove that prefix to keep just "normal".
            6. Store the four values under the cleaned ID.

            Things you'll see in the real file: 10 paced records have an
            empty axis, one age is ">89", and some rhythms have two lines.
        """
        path = root / "ludb.csv"
        if not path.is_file():
            path = record_dir.parent / "ludb.csv"
        if not path.is_file():
            logger.warning("No ludb.csv found in %s; metadata left empty.", root)
            return {}

        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "ID" not in table.columns:
            raise ValueError(f"{path} has no 'ID' column; is it LUDB's ludb.csv?")

        metadata = {}
        for _, row in table.iterrows():
            # "Electric axis of the heart: normal" -> "normal"
            axis = _clean_cell(row.get("Electric axis of the heart", ""))
            axis = axis.replace("Electric axis of the heart:", "").strip()
            metadata[_clean_cell(row["ID"])] = {
                "sex": _clean_cell(row.get("Sex", "")),
                "age": _clean_cell(row.get("Age", "")),
                "rhythm": _clean_cell(row.get("Rhythms", "")),
                "electric_axis": axis,
            }
        return metadata

    @property
    def default_task(self) -> ECGDelineationLUDB:
        """Return the default task: delineation on all 12 leads, filtered.

        Notes:
            How to write it: return ``ECGDelineationLUDB()``. Import the task
            at the top of this file. The task file must never import from
            pyhealth.datasets, or the two imports go in a circle; that's why
            LUDB_LEADS lives in the task file.
        """
        return ECGDelineationLUDB()
