# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Paper link: https://arxiv.org/abs/2408.07773
# Description: MIT-BIH Arrhythmia Database — 48 half-hour excerpts
#     of two-channel ambulatory ECG, 360 Hz. Used for boundary
#     detection (R-peaks) and anomaly detection (arrhythmia).
# Source: https://physionet.org/content/mitdb/1.0.0/

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

# Normal beat types (all others are anomalies)
_NORMAL_BEATS = {"N", "L", "R", "e", "j"}

# Paced rhythm records — excluded per paper
_PACED_RECORDS = {"102", "104", "107", "217"}

# Paper's MIT-BIH split: 80/20 by patient, seeded with NumPy legacy RNG.
_PAPER_SPLIT_RATIO = 0.8
_PAPER_SPLIT_SEED = 0
_VALID_SPLIT_MODES = {None, "random", "abnormal_sorted"}

# Subdirectory under ``root`` for preprocessed ``.npz`` caches.
_PROCESSED_SUBDIR = "processed"


class MITBIHDataset(BaseDataset):
    """MIT-BIH Arrhythmia Database for ECG analysis.

    48 half-hour excerpts of two-channel ambulatory ECG from a mixed
    population of inpatients and outpatients, digitized at 360 Hz.
    4 paced-rhythm records are excluded.

    Supports two tasks:
        - Boundary detection (R-peak localization)
        - Anomaly detection (arrhythmia via reconstruction error)

    Dataset is available at https://physionet.org/content/mitdb/1.0.0/

    Paper: Moody, G.B. & Mark, R.G. "The impact of the MIT-BIH
    Arrhythmia Database." IEEE EMB Magazine, 2001.

    Args:
        root: Root directory of the raw MIT-BIH data. Should contain
            wfdb record files (100.dat, 100.hea, etc.).
        dataset_name: Name of the dataset. Default is ``"mitbih"``.
        config_path: Path to the YAML config file.
        dev: Whether to enable dev mode (first 5 patients).
        paper_split: Split assignment strategy:

            - ``None`` (default): leave the ``split`` column blank.
            - ``"random"``: 80/20 patient split via ``RandomState(0)``,
              matching the paper's segmentation/boundary task setup.
            - ``"abnormal_sorted"``: patients sorted by ``n_abnormal``
              ascending, with all-abnormal patients excluded. The
              least-abnormal 80% become train and the most-abnormal
              20% become test. Matches the paper's anomaly setup.
        preprocess: If True, decode each record once, downsample,
            trim, and cache ``(signal, ann_sample, ann_symbol)`` to
            ``{root}/processed/{record}.npz``. Subsequent runs skip
            wfdb. Default False.
        downsample_factor: Decimation factor applied to the 360 Hz
            raw signal when ``preprocess=True``. Default 3 (120 Hz).
        trim: When ``preprocess=True``, crop each record to the
            region between its first and last beat annotation.
            Matches the paper's preprocessing. Default True.

    Examples:
        >>> from pyhealth.datasets import MITBIHDataset
        >>> dataset = MITBIHDataset(root="/path/to/mitdb/")
        >>> dataset.stat()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        paper_split: Optional[str] = None,
        preprocess: bool = False,
        downsample_factor: int = 3,
        trim: bool = True,
    ) -> None:
        if paper_split not in _VALID_SPLIT_MODES:
            raise ValueError(
                f"paper_split must be one of {_VALID_SPLIT_MODES}, "
                f"got {paper_split!r}"
            )
        if downsample_factor < 1:
            raise ValueError(
                f"downsample_factor must be >= 1, got {downsample_factor}"
            )

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mitbih.yaml"
            )

        metadata_path = os.path.join(root, "mitbih-pyhealth.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(
                root,
                dev=dev,
                paper_split=paper_split,
                preprocess=preprocess,
                downsample_factor=downsample_factor,
                trim=trim,
            )

        super().__init__(
            root=root,
            tables=["ecg"],
            dataset_name=dataset_name or "mitbih",
            config_path=config_path,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(
        root: str,
        dev: bool = False,
        paper_split: Optional[str] = None,
        preprocess: bool = False,
        downsample_factor: int = 3,
        trim: bool = True,
    ) -> None:
        """Prepare metadata CSV from raw wfdb files.

        Args:
            root: Root directory containing wfdb files.
            dev: If True, only process first 5 patients.
            paper_split: ``None``, ``"random"``, or ``"abnormal_sorted"``.
                See class docstring for semantics.
            preprocess: If True, build per-record ``.npz`` caches and
                populate the ``processed_file`` column.
            downsample_factor: Decimation factor when ``preprocess=True``.
            trim: When ``preprocess=True``, crop to first/last beat
                annotation.
        """
        import wfdb

        records = sorted(
            f.replace(".dat", "")
            for f in os.listdir(root)
            if f.endswith(".dat") and f.replace(".dat", "") not in _PACED_RECORDS
        )
        if dev:
            records = records[:5]

        processed_dir = os.path.join(root, _PROCESSED_SUBDIR)

        rows = []
        for rec_name in records:
            rec_path = os.path.join(root, rec_name)
            try:
                record = wfdb.rdrecord(rec_path)
                ann = wfdb.rdann(rec_path, extension="atr")
            except Exception:
                continue

            patient_id = rec_name

            # Parse header for demographics
            age, sex, medications = "", "", ""
            if record.comments:
                first = record.comments[0].strip()
                tokens = first.split()
                if len(tokens) >= 2:
                    age = tokens[0]
                    sex = tokens[1]
                if len(record.comments) > 1:
                    medications = record.comments[1].strip()

            # Count beats excluding rhythm-change markers ("+")
            beat_symbols = [s for s in ann.symbol if s != "+"]
            n_beats = len(beat_symbols)
            n_abnormal = sum(
                1 for s in beat_symbols if s not in _NORMAL_BEATS
            )

            processed_file = ""
            if preprocess:
                processed_file = _build_record_cache(
                    processed_dir=processed_dir,
                    rec_path=rec_path,
                    rec_name=rec_name,
                    downsample_factor=downsample_factor,
                    trim=trim,
                )

            rows.append({
                "patient_id": patient_id,
                "signal_file": os.path.join(root, rec_name),
                "annotation_file": "atr",
                "age": age,
                "sex": sex,
                "medications": medications,
                "n_abnormal": n_abnormal,
                "n_beats": n_beats,
                "processed_file": processed_file,
            })

        rows = _apply_paper_split(rows, paper_split)

        df = pd.DataFrame(rows)
        out_path = os.path.join(root, "mitbih-pyhealth.csv")
        df.to_csv(out_path, index=False)
        logger.info(
            "MIT-BIH metadata: %d records -> %s", len(df), out_path
        )

    @property
    def default_task(self):
        """Returns the default task (boundary detection)."""
        from pyhealth.tasks.ecg_boundary_detection import (
            ECGBoundaryDetection,
        )

        return ECGBoundaryDetection()


def _build_record_cache(
    processed_dir: str,
    rec_path: str,
    rec_name: str,
    downsample_factor: int,
    trim: bool,
) -> str:
    """Cache downsampled + trimmed signal and annotations for one record.

    Returns the absolute cache path, which is written into the
    metadata CSV's ``processed_file`` column. The cache stores:

    - ``signal``: ``(n_timesteps, n_channels)`` post-downsample + trim
    - ``ann_sample``: beat sample indices **relative to the trimmed
      signal** (drops annotations outside the trim range)
    - ``ann_symbol``: beat symbols aligned with ``ann_sample``
    """
    cache_path = os.path.join(processed_dir, f"{rec_name}.npz")
    raw_paths = [rec_path + ".dat", rec_path + ".hea", rec_path + ".atr"]
    params = {
        "downsample_factor": int(downsample_factor),
        "trim": bool(trim),
    }
    fingerprint = compute_fingerprint(raw_paths, params)

    def _build() -> dict[str, np.ndarray]:
        import wfdb

        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, extension="atr")

        signal = record.p_signal.astype(np.float32)
        if downsample_factor > 1:
            signal = signal[::downsample_factor]

        ds_samples = np.asarray(ann.sample) // downsample_factor
        ann_symbols = np.asarray(ann.symbol)

        # Drop annotations outside the signal bounds.
        in_bounds = (ds_samples >= 0) & (ds_samples < len(signal))
        ds_samples = ds_samples[in_bounds]
        ann_symbols = ann_symbols[in_bounds]

        if trim and len(ds_samples) > 0:
            first = int(ds_samples[0])
            last = int(ds_samples[-1])
            if first <= last:
                signal = signal[first : last + 1]
                ds_samples = ds_samples - first
                # After shifting, last kept index is last-first.

        return {
            "signal": signal,
            "ann_sample": ds_samples.astype(np.int64),
            "ann_symbol": ann_symbols.astype("U4"),
        }

    load_or_build(cache_path, fingerprint, _build)
    return cache_path


def _apply_paper_split(
    rows: list[dict], paper_split: Optional[str]
) -> list[dict]:
    """Assign each row to train/test per the paper's split strategy.

    Mutates rows in place by adding a ``split`` key. For
    ``"abnormal_sorted"``, patients with every beat marked abnormal
    are dropped entirely (the paper excludes them from anomaly
    training). Returns the possibly filtered list.
    """
    if not rows:
        return rows

    if paper_split is None:
        for row in rows:
            row["split"] = ""
        return rows

    if paper_split == "random":
        rng = np.random.RandomState(_PAPER_SPLIT_SEED)
        order = rng.permutation(len(rows))
        cutoff = int(len(rows) * _PAPER_SPLIT_RATIO)
        split_by_rank = [
            "train" if rank < cutoff else "test"
            for rank in range(len(rows))
        ]
        for rank, idx in enumerate(order):
            rows[idx]["split"] = split_by_rank[rank]
        return rows

    if paper_split == "abnormal_sorted":
        # Drop all-abnormal patients before splitting.
        kept = [
            row for row in rows
            if row.get("n_beats", 0) == 0
            or row["n_abnormal"] < row["n_beats"]
        ]
        kept.sort(key=lambda r: r["n_abnormal"])
        cutoff = int(len(kept) * _PAPER_SPLIT_RATIO)
        for rank, row in enumerate(kept):
            row["split"] = "train" if rank < cutoff else "test"
        return kept

    raise ValueError(f"Unknown paper_split mode: {paper_split!r}")
