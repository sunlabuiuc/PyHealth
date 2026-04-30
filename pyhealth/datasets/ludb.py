# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Paper link: https://arxiv.org/abs/2408.07773
# Description: Lobachevsky University Database (LUDB) — 200 subjects
#     with 12-lead ECG at 500 Hz, manually annotated with P wave,
#     T wave, QRS complex, and background classes.
# Source: https://physionet.org/content/ludb/1.0.1/

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.datasets._medtsllm_cache import (
    compute_fingerprint,
    load_or_build,
)

logger = logging.getLogger(__name__)

# Label mapping: wfdb annotation symbol -> class index
# 0 = background, 1 = P wave, 2 = QRS complex, 3 = T wave
WAVE_CLASSES = ["background", "P wave", "QRS complex", "T wave"]
_WAVE_LABELS = {"p": 1, "N": 2, "t": 3}

# Paper's LUDB split: 80/20 by patient, seeded with NumPy legacy RNG
# to match the cs598 reference implementation exactly.
_PAPER_SPLIT_RATIO = 0.8
_PAPER_SPLIT_SEED = 0

# Subdirectory under ``root`` where preprocessed ``.npz`` files live.
_PROCESSED_SUBDIR = "processed"


def _parse_ludb_header(record) -> tuple[str, str, str]:
    """Parse age, sex, and diagnoses from LUDB wfdb header comments.

    LUDB headers look like::

        #<age>: 51
        #<sex>: F
        #<diagnoses>:
        #Rhythm: Sinus bradycardia.
        #Left ventricular hypertrophy.

    Args:
        record: A ``wfdb.Record`` with a ``comments`` attribute.

    Returns:
        Tuple of (age, sex, diagnoses). Diagnoses are joined with
        ``"; "`` separators. Missing fields return empty strings.
    """
    age = ""
    sex = ""
    diagnoses_lines: list[str] = []
    if not record.comments:
        return age, sex, ""

    in_diagnoses = False
    for line in record.comments:
        line = line.strip()
        if line.startswith("<age>:"):
            age = line.split(":", 1)[1].strip()
        elif line.startswith("<sex>:"):
            sex = line.split(":", 1)[1].strip()
        elif line.startswith("<diagnoses>:"):
            in_diagnoses = True
        elif in_diagnoses and line:
            diagnoses_lines.append(line.rstrip(". "))

    return age, sex, "; ".join(diagnoses_lines)


class LUDBDataset(BaseDataset):
    """Lobachevsky University Database (LUDB) for ECG delineation.

    Dataset of 200 subjects with 12-lead ECG recordings at 500 Hz
    (10 seconds each). Each lead is manually annotated by cardiologists
    with P wave, QRS complex, and T wave boundaries.

    Dataset is available at https://physionet.org/content/ludb/1.0.1/

    Paper: Kalyakulina, A. et al. "LUDB: A New Open-Access Validation
    Database for Electrocardiogram Delineation Algorithms."

    Args:
        root: Root directory of the raw LUDB data. Should contain a
            ``data/`` subdirectory with wfdb record files (.dat, .hea).
        dataset_name: Name of the dataset. Default is ``"ludb"``.
        config_path: Path to the YAML config file. Default uses the
            built-in config.
        dev: Whether to enable dev mode (only use first 5 patients).
            Default is False.
        paper_split: If True, populate a ``split`` column with an
            80/20 train/test assignment per patient, using the
            legacy NumPy seed=0 RNG from the paper. Otherwise the
            ``split`` column is left blank. Default False.
        preprocess: If True, decode each ``(patient, lead)`` wfdb
            record once and cache the resulting ``(signal, labels)``
            arrays to ``{root}/processed/{record}_{lead}.npz``.
            Subsequent runs skip wfdb entirely. Default False.
        trim: Only consulted when ``preprocess=True``. Crop each
            lead to the region between the first and last wave
            annotation before caching. Matches the paper's
            preprocessing. Default True.

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> dataset = LUDBDataset(root="/path/to/ludb/")
        >>> dataset.stat()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        paper_split: bool = False,
        preprocess: bool = False,
        trim: bool = True,
    ) -> None:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "ludb.yaml"
            )

        metadata_path = os.path.join(root, "ludb-pyhealth.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(
                root,
                dev=dev,
                paper_split=paper_split,
                preprocess=preprocess,
                trim=trim,
            )

        super().__init__(
            root=root,
            tables=["ecg"],
            dataset_name=dataset_name or "ludb",
            config_path=config_path,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(
        root: str,
        dev: bool = False,
        paper_split: bool = False,
        preprocess: bool = False,
        trim: bool = True,
    ) -> None:
        """Prepare metadata CSV from raw wfdb files.

        Scans the ``data/`` subdirectory for wfdb records and creates
        a CSV with one row per (patient, lead) pair, pointing to the
        signal and annotation files. When ``preprocess=True`` also
        writes per-lead ``.npz`` caches into ``{root}/processed/``.

        Args:
            root: Root directory containing ``data/`` with wfdb files.
            dev: If True, only process first 5 patients.
            paper_split: If True, assign patients to train/test via the
                paper's 80/20 seed=0 split and write values to the
                ``split`` column. Otherwise the column is blank.
            preprocess: If True, build per-lead ``.npz`` signal+label
                caches and point each row's ``processed_file`` at
                its cache.
            trim: When ``preprocess=True``, crop each lead to the
                region between the first and last wave annotation.
        """
        import wfdb

        data_dir = os.path.join(root, "data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"LUDB data directory not found at {data_dir}. "
                "Download from https://physionet.org/content/ludb/1.0.1/"
            )

        records = sorted(
            f.replace(".dat", "")
            for f in os.listdir(data_dir)
            if f.endswith(".dat")
        )
        if dev:
            records = records[:5]

        split_by_record = _assign_paper_split(records) if paper_split else {}
        processed_dir = os.path.join(root, _PROCESSED_SUBDIR)

        rows = []
        for rec_name in records:
            rec_path = os.path.join(data_dir, rec_name)
            record = wfdb.rdrecord(rec_path)
            patient_id = int(rec_name)

            age, sex, diagnoses = _parse_ludb_header(record)

            for lead_idx in range(record.n_sig):
                lead_name = record.sig_name[lead_idx]
                # Check if annotation file exists for this lead
                ann_ext = lead_name.lower()
                ann_path = os.path.join(data_dir, f"{rec_name}.{ann_ext}")
                if not os.path.exists(ann_path):
                    continue

                clip_id = patient_id * 100 + lead_idx
                processed_file = ""
                if preprocess:
                    processed_file = _build_lead_cache(
                        processed_dir=processed_dir,
                        rec_path=rec_path,
                        rec_name=rec_name,
                        lead_name=lead_name,
                        lead_idx=lead_idx,
                        ann_path=ann_path,
                        ann_ext=ann_ext,
                        trim=trim,
                    )
                rows.append({
                    "patient_id": str(patient_id),
                    "lead": lead_name,
                    "clip_id": clip_id,
                    "signal_file": os.path.join(data_dir, rec_name),
                    "label_file": ann_ext,
                    "age": age,
                    "sex": sex,
                    "diagnoses": diagnoses,
                    "split": split_by_record.get(rec_name, ""),
                    "processed_file": processed_file,
                })

        df = pd.DataFrame(rows)
        out_path = os.path.join(root, "ludb-pyhealth.csv")
        df.to_csv(out_path, index=False)
        logger.info(
            "LUDB metadata: %d records from %d patients -> %s",
            len(df),
            df["patient_id"].nunique(),
            out_path,
        )

    @property
    def default_task(self):
        """Returns the default task for this dataset."""
        from pyhealth.tasks.ecg_wave_segmentation import ECGWaveSegmentation

        return ECGWaveSegmentation()


def _build_lead_cache(
    processed_dir: str,
    rec_path: str,
    rec_name: str,
    lead_name: str,
    lead_idx: int,
    ann_path: str,
    ann_ext: str,
    trim: bool,
) -> str:
    """Cache (signal, labels) for one (record, lead) pair.

    Returns the absolute cache path, which is written into the
    metadata CSV's ``processed_file`` column.
    """
    cache_path = os.path.join(
        processed_dir, f"{rec_name}_{lead_name}.npz"
    )
    raw_paths = [rec_path + ".dat", rec_path + ".hea", ann_path]
    params = {"trim": bool(trim)}
    fingerprint = compute_fingerprint(raw_paths, params)

    def _build() -> dict[str, np.ndarray]:
        import wfdb

        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal[:, lead_idx].astype(np.float32)
        ann = wfdb.rdann(rec_path, extension=ann_ext)

        labels = np.zeros(len(signal), dtype=np.int64)
        i = 0
        while i < len(ann.symbol):
            sym = ann.symbol[i]
            if sym == "(" and i + 2 < len(ann.symbol):
                wave_type = ann.symbol[i + 1]
                onset = ann.sample[i]
                offset = (
                    ann.sample[i + 2]
                    if ann.symbol[i + 2] == ")"
                    else ann.sample[i + 1]
                )
                if wave_type in _WAVE_LABELS:
                    labels[onset : offset + 1] = _WAVE_LABELS[wave_type]
                i += 3
            else:
                i += 1

        if trim:
            wave_mask = labels > 0
            if wave_mask.any():
                first = int(np.argmax(wave_mask))
                last = len(wave_mask) - 1 - int(np.argmax(wave_mask[::-1]))
                signal = signal[first : last + 1]
                labels = labels[first : last + 1]

        return {"signal": signal, "labels": labels}

    load_or_build(cache_path, fingerprint, _build)
    return cache_path


def _assign_paper_split(records: list[str]) -> dict[str, str]:
    """Assign each record to train/test per the paper's 80/20 seed=0 split.

    Mirrors the cs598 preprocess_ludb.py recipe: shuffle record names
    with ``np.random.RandomState(0)``, take the first 80% as train and
    the remainder as test.
    """
    rng = np.random.RandomState(_PAPER_SPLIT_SEED)
    order = rng.permutation(len(records))
    cutoff = int(len(records) * _PAPER_SPLIT_RATIO)
    assignment: dict[str, str] = {}
    for rank, idx in enumerate(order):
        assignment[records[idx]] = "train" if rank < cutoff else "test"
    return assignment
