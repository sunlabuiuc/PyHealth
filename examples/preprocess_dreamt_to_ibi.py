"""preprocess_dreamt_to_ibi.py — Convert raw DREAMT PPG recordings to NPZ files.

This is a standalone CLI script in ``examples/``, **not** part of the PyHealth API.
After running this script, the ``dst_dir`` can be passed directly as the ``root``
argument to :class:`pyhealth.datasets.IBISleepDataset`.

Required extra install (not in PyHealth core)::

    pip install neurokit2

Usage::

    python examples/preprocess_dreamt_to_ibi.py \\
        --src_dir /path/to/DREAMT/raw \\
        --dst_dir /path/to/output/npz \\
        --participant_info /path/to/DREAMT/participant_info.csv

DREAMT raw directory layout expected::

    <src_dir>/
        <SID>_PSG_df_updated.csv   # 100 Hz BVP + stage columns
        ...

The ``participant_info.csv`` must contain at minimum:
    - a subject-ID column (``Participant_ID`` or first column)
    - an ``AHI`` column

Output NPZ schema (one file per subject, saved as ``<SID>.npz``)::

    data   : float32 (N,)   IBI time series at 25 Hz
    stages : int32   (N,)   0=W, 1=N1, 2=N2, 3=N3, 4=REM  (sample-level)
    fs     : int64   ()     Always 25
    ahi    : float32 ()     Apnea-Hypopnea Index (NaN if unavailable)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
_DREAMT_FS = 100  # Hz of raw BVP signal
_TARGET_FS = 25
_STRIDE = _DREAMT_FS // _TARGET_FS  # 4
_IBI_OUTLIER_S = 2.0  # zero out intervals >= this


def _extract_ibi_dreamt(bvp: np.ndarray, fs: int = _DREAMT_FS) -> np.ndarray:
    """Return per-sample IBI array at *fs* Hz using neurokit2 PPG processing."""
    try:
        import neurokit2 as nk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "neurokit2 is required for DREAMT preprocessing.  "
            "Install it with:  pip install neurokit2"
        ) from exc

    signals, info = nk.ppg_process(bvp, sampling_rate=fs)
    peaks = info["PPG_Peaks"]  # sample indices of systolic peaks

    ibi = np.zeros(len(bvp), dtype=np.float32)
    for i in range(1, len(peaks)):
        interval_s = (peaks[i] - peaks[i - 1]) / fs
        if interval_s >= _IBI_OUTLIER_S:
            interval_s = 0.0
        ibi[peaks[i - 1] : peaks[i]] = interval_s
    return ibi


def _encode_stages(stage_series: pd.Series) -> np.ndarray:
    """Map DREAMT string labels → int32 (unknown → -1)."""
    return stage_series.map(_STAGE_MAP).fillna(-1).astype(np.int32).to_numpy()


def _process_subject(
    src_dir: str,
    dst_dir: str,
    sid: str,
    ahi: float,
) -> bool:
    """Process one subject.  Returns True on success, False on skip/error."""
    out_path = Path(dst_dir) / f"{sid}.npz"
    if out_path.exists():
        logger.info("Skipping %s — NPZ already exists", sid)
        return True

    csv_candidates = list(Path(src_dir).glob(f"{sid}_PSG_df_updated.csv"))
    if not csv_candidates:
        logger.warning("No PSG CSV found for subject %s — skipping", sid)
        return False
    csv_path = csv_candidates[0]

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot read %s: %s — skipping", csv_path, exc)
        return False

    if "BVP" not in df.columns:
        logger.warning("No BVP column in %s — skipping", csv_path)
        return False

    stage_col = next(
        (c for c in ("stage", "Stage", "sleep_stage", "Sleep_Stage") if c in df.columns),
        None,
    )
    if stage_col is None:
        logger.warning("No stage column in %s — skipping", csv_path)
        return False

    try:
        ibi_100hz = _extract_ibi_dreamt(df["BVP"].to_numpy(dtype=np.float64))
    except Exception as exc:  # noqa: BLE001
        logger.warning("IBI extraction failed for %s: %s — skipping", sid, exc)
        return False

    # Stride-4 downsample: 100 Hz → 25 Hz
    data_25hz = ibi_100hz[::_STRIDE].astype(np.float32)
    stages_25hz = _encode_stages(df[stage_col])[::_STRIDE].astype(np.int32)

    # Align lengths
    n = min(len(data_25hz), len(stages_25hz))
    data_25hz = data_25hz[:n]
    stages_25hz = stages_25hz[:n]

    np.savez(
        out_path,
        data=data_25hz,
        stages=stages_25hz,
        fs=np.int64(_TARGET_FS),
        ahi=np.float32(ahi),
    )
    logger.info("Saved %s  (%d samples, AHI=%.1f)", out_path, n, ahi)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw DREAMT PPG recordings to NPZ files for IBISleepDataset."
    )
    parser.add_argument("--src_dir", required=True, help="Directory with raw DREAMT CSV files")
    parser.add_argument("--dst_dir", required=True, help="Output directory for NPZ files (= IBISleepDataset root)")
    parser.add_argument("--participant_info", required=True, help="Path to participant_info.csv with AHI column")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N subjects")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    os.makedirs(args.dst_dir, exist_ok=True)

    info_df = pd.read_csv(args.participant_info)
    # Accept 'Participant_ID' or first column as subject ID
    id_col = "Participant_ID" if "Participant_ID" in info_df.columns else info_df.columns[0]
    ahi_col = next((c for c in info_df.columns if c.upper() == "AHI"), None)

    if args.limit is not None:
        info_df = info_df.head(args.limit)

    success = fail = skip = 0
    for _, row in info_df.iterrows():
        sid = str(row[id_col])
        ahi = float(row[ahi_col]) if ahi_col is not None else float("nan")
        result = _process_subject(args.src_dir, args.dst_dir, sid, ahi)
        if result:
            success += 1
        else:
            fail += 1

    logger.info("Done. success=%d  failed/skipped=%d", success, fail)


if __name__ == "__main__":
    main()
