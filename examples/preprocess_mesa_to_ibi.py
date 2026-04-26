"""preprocess_mesa_to_ibi.py — Convert raw MESA PPG recordings to NPZ files.

This is a standalone CLI script in ``examples/``, **not** part of the PyHealth API.
After running this script, the ``dst_dir`` can be passed directly as the ``root``
argument to :class:`pyhealth.datasets.IBISleepDataset` with ``source="mesa"``.

Required extra installs (not in PyHealth core)::

    pip install neurokit2 mne

``mne`` is already installed in most PyHealth environments; ``neurokit2`` is the
only additional dependency.

Usage::

    python examples/preprocess_mesa_to_ibi.py \\
        --src_dir /path/to/MESA/polysomnography \\
        --dst_dir /path/to/output/npz \\
        --harmonized_csv /path/to/mesa-sleep-harmonized-dataset.csv

MESA raw directory layout expected (standard NSRR download)::

    <src_dir>/
        edfs/
            mesa-sleep-<id>.edf
        annotations-events-profusion/
            mesa-sleep-<id>-profusion.xml

The harmonized CSV must contain at minimum:
    - ``mesaid`` column (integer)
    - an AHI column (script tries several common names)

Output NPZ schema (one file per subject, e.g. ``mesa-sleep-00001.npz``)::

    data   : float32 (N,)   IBI time series at 25 Hz
    stages : int32   (N,)   0=W, 1=N1, 2=N2, 3=N3, 4=REM  (sample-level)
    fs     : int64   ()     Always 25
    ahi    : float32 ()     Apnea-Hypopnea Index (NaN if unavailable)
"""

from __future__ import annotations

import argparse
import logging
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

_TARGET_FS = 25

# Profusion XML stage codes → unified 5-class scheme (4=N4→N3, 5=REM→4)
_STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}

_AHI_CANDIDATES = [
    "nsrr_ahi_hp3r_aasm15", "nsrr_ahi_hp3u", "nsrr_ahi_hp4u_aasm15",
    "ahi_a0h3a", "ahi_a0h4a", "ahi_a0h3", "ahi_a0h4", "AHI", "ahi",
]
_PLETH_CANDIDATES = ["Pleth", "PLETH", "SpO2", "PPG"]


def _parse_profusion_xml(xml_path: Path) -> np.ndarray:
    """Parse MESA profusion XML → per-epoch stage array (30 s epochs)."""
    try:
        tree = ET.parse(xml_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot parse {xml_path}: {exc}") from exc

    root = tree.getroot()
    stages = []
    for elem in root.iter("SleepStage"):
        try:
            raw = int(elem.text)
        except (TypeError, ValueError):
            raw = -1
        stages.append(_STAGE_MAP.get(raw, -1))
    return np.array(stages, dtype=np.int32)


def _extract_ibi_ppg(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Return per-sample IBI array at *fs* Hz using neurokit2 PPG processing."""
    try:
        import neurokit2 as nk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "neurokit2 is required for MESA preprocessing.  "
            "Install it with:  pip install neurokit2"
        ) from exc

    signals, info = nk.ppg_process(ppg, sampling_rate=fs)
    peaks = info["PPG_Peaks"]

    ibi = np.zeros(len(ppg), dtype=np.float32)
    for i in range(1, len(peaks)):
        interval_s = (peaks[i] - peaks[i - 1]) / fs
        ibi[peaks[i - 1] : peaks[i]] = interval_s
    return ibi


def _process_recording(
    edf_path: Path,
    xml_path: Optional[Path],
    dst_dir: str,
    ahi: float,
) -> bool:
    """Process one MESA recording.  Returns True on success."""
    import mne  # noqa: PLC0415

    sid = edf_path.stem  # e.g. mesa-sleep-00001
    out_path = Path(dst_dir) / f"{sid}.npz"
    if out_path.exists():
        logger.info("Skipping %s — NPZ already exists", sid)
        return True

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot read EDF %s: %s — skipping", edf_path, exc)
        return False

    pleth_ch = next((c for c in raw.ch_names if any(p in c for p in _PLETH_CANDIDATES)), None)
    if pleth_ch is None:
        logger.warning("No Pleth/PPG channel in %s — skipping", edf_path)
        return False

    ppg_data, _ = raw[pleth_ch]
    ppg = ppg_data[0]
    fs_orig = raw.info["sfreq"]

    try:
        ibi_orig = _extract_ibi_ppg(ppg, fs_orig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("IBI extraction failed for %s: %s — skipping", sid, exc)
        return False

    # Resample to TARGET_FS
    gcd = int(np.gcd(int(_TARGET_FS), int(fs_orig)))
    up = _TARGET_FS // gcd
    down = int(fs_orig) // gcd
    data_25hz = resample_poly(ibi_orig, up, down).astype(np.float32)

    n = len(data_25hz)

    # Parse stage annotations from profusion XML
    if xml_path is not None and xml_path.exists():
        try:
            epoch_stages = _parse_profusion_xml(xml_path)
        except ValueError as exc:
            logger.warning("%s — stage parse failed: %s; stages set to -1", sid, exc)
            epoch_stages = np.full(n // (_TARGET_FS * 30) + 1, -1, dtype=np.int32)
    else:
        logger.warning("No profusion XML for %s — stages set to -1", sid)
        epoch_stages = np.full(n // (_TARGET_FS * 30) + 1, -1, dtype=np.int32)

    # Expand epoch-level stages to sample-level at 25 Hz (30 s × 25 = 750 samples/epoch)
    samples_per_epoch = _TARGET_FS * 30
    stages_25hz = np.full(n, -1, dtype=np.int32)
    for ep_idx, stage in enumerate(epoch_stages):
        start = ep_idx * samples_per_epoch
        end = start + samples_per_epoch
        if start >= n:
            break
        stages_25hz[start : min(end, n)] = stage

    np.savez(
        out_path,
        data=data_25hz,
        stages=stages_25hz,
        fs=np.int64(_TARGET_FS),
        ahi=np.float32(ahi),
    )
    logger.info("Saved %s  (%d samples, AHI=%.1f)", out_path, n, ahi)
    return True


def _find_ahi(info_df: pd.DataFrame, mesaid: int) -> float:
    id_col = "mesaid" if "mesaid" in info_df.columns else "nsrrid"
    row = info_df[info_df[id_col] == mesaid]
    if row.empty:
        return float("nan")
    for col in _AHI_CANDIDATES:
        if col in row.columns:
            val = row.iloc[0][col]
            if pd.notna(val):
                return float(val)
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw MESA PPG recordings to NPZ files for IBISleepDataset."
    )
    parser.add_argument("--src_dir", required=True, help="MESA polysomnography root (contains edfs/ and annotations/)")
    parser.add_argument("--dst_dir", required=True, help="Output directory for NPZ files (= IBISleepDataset root)")
    parser.add_argument("--harmonized_csv", required=True, help="MESA harmonized dataset CSV with mesaid and AHI columns")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N recordings")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    os.makedirs(args.dst_dir, exist_ok=True)

    info_df = pd.read_csv(args.harmonized_csv)

    src = Path(args.src_dir)
    edf_paths = sorted((src / "edfs").glob("mesa-sleep-*.edf"))
    if args.limit is not None:
        edf_paths = list(edf_paths)[: args.limit]

    def _args_for(edf_path: Path):
        sid = edf_path.stem  # mesa-sleep-00001
        try:
            mesaid = int(sid.split("-")[-1])
        except ValueError:
            mesaid = -1
        xml_path = src / "annotations-events-profusion" / f"{sid}-profusion.xml"
        ahi = _find_ahi(info_df, mesaid)
        return edf_path, xml_path if xml_path.exists() else None, args.dst_dir, ahi

    success = fail = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_recording, *_args_for(p)): p for p in edf_paths}
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    fail += 1
    else:
        for edf_path in edf_paths:
            if _process_recording(*_args_for(edf_path)):
                success += 1
            else:
                fail += 1

    logger.info("Done. success=%d  failed/skipped=%d", success, fail)


if __name__ == "__main__":
    main()
