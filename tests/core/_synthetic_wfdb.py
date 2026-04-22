"""Synthetic wfdb record generators for ECG/respiratory dataset tests.

Generates minimal in-repo wfdb records for LUDB, MIT-BIH, and BIDMC so
tests never ship real patient data. Each generator writes the same
files the dataset loader expects (``.dat``, ``.hea``, per-lead/``.atr``
/``.breath`` annotations), with seeded RNG for reproducibility.

All signals are sine/noise mixes — not realistic ECGs, but valid wfdb
records that satisfy the dataset parsers and task annotations. Keep
these generators in sync with the real record schemas when the
dataset parsers change.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------- #
# LUDB: 12-lead ECG at 500 Hz, 10 s per record, wave annotations per lead
# --------------------------------------------------------------------- #

_LUDB_LEADS = ("i", "ii", "iii", "avr", "avl", "avf",
               "v1", "v2", "v3", "v4", "v5", "v6")
_LUDB_FS = 500
_LUDB_LEN = 5000  # 10 s
_LUDB_DIAGNOSES = (
    "Rhythm: Sinus rhythm.",
    "Left ventricular hypertrophy.",
    "Non-specific repolarization abnormalities.",
)


def synthesize_ludb(
    dest_root: str,
    n_records: int = 2,
    seed: int = 0,
) -> None:
    """Write ``n_records`` synthetic LUDB records into ``{dest_root}/data/``.

    Each record has:
    * 12-lead signal with a weak sinusoidal "ECG" shape + noise
    * Header comments with ``<age>``, ``<sex>``, ``<diagnoses>``
    * Per-lead wave annotation file (``.i``, ``.ii``, ...) with evenly
      spaced P / N (QRS) / T wave triplets via the LUDB ``( sym )``
      encoding.
    """
    import wfdb

    data_dir = os.path.join(dest_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for rec_idx in range(n_records):
        rec_name = str(rec_idx + 1)
        signal = _synthesize_12lead_ecg(rng)
        comments = [
            f"<age>: {40 + 5 * rec_idx}",
            f"<sex>: {'M' if rec_idx % 2 == 0 else 'F'}",
            "<diagnoses>:",
            *_LUDB_DIAGNOSES,
        ]
        wfdb.wrsamp(
            record_name=rec_name,
            fs=_LUDB_FS,
            units=["mV"] * len(_LUDB_LEADS),
            sig_name=list(_LUDB_LEADS),
            p_signal=signal,
            fmt=["16"] * len(_LUDB_LEADS),
            write_dir=data_dir,
            comments=comments,
        )
        _write_ludb_annotations(data_dir, rec_name, rng)


def _synthesize_12lead_ecg(rng: np.random.Generator) -> np.ndarray:
    """Return (samples, 12) float array with ECG-ish waveforms."""
    t = np.arange(_LUDB_LEN) / _LUDB_FS
    # Base rhythm at ~1 Hz with QRS-like spikes
    base = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    spikes = np.zeros_like(t)
    for beat_t in np.arange(0.4, 10.0, 0.8):
        idx = int(beat_t * _LUDB_FS)
        if idx < _LUDB_LEN:
            spikes[idx] = 1.0
    # Small gaussian bumps around each spike for QRS shape
    kernel = np.exp(-((np.arange(-20, 21)) ** 2) / 40.0)
    qrs = np.convolve(spikes, kernel, mode="same")
    signal = np.stack([
        base + qrs * (0.8 + 0.1 * i) + rng.normal(0, 0.02, _LUDB_LEN)
        for i in range(len(_LUDB_LEADS))
    ], axis=1).astype(np.float32)
    return signal


def _write_ludb_annotations(
    data_dir: str, rec_name: str, rng: np.random.Generator,
) -> None:
    """Write one annotation file per lead with evenly spaced P/N/T waves.

    LUDB encodes each wave as a triplet of symbols ``(``, wave-type,
    ``)`` at onset, peak, offset sample indices, where wave-type is
    ``p`` (P wave), ``N`` (QRS complex), or ``t`` (T wave).
    """
    import wfdb

    # Produce ~10 beats across the 10 s clip, each with P/N/T triplet
    beat_centers = np.linspace(300, _LUDB_LEN - 300, 10, dtype=int)
    samples = []
    symbols = []
    for center in beat_centers:
        # P wave triplet (onset, symbol, offset) relative to beat center
        for offset_from_center, sym in (
            (-120, "p"), (-90, "p"), (-60, "p"),  # P wave
            (-20, "N"), (0, "N"), (20, "N"),  # QRS complex
            (80, "t"), (120, "t"), (160, "t"),  # T wave
        ):
            s = int(center + offset_from_center)
            if 0 <= s < _LUDB_LEN:
                # LUDB schema uses ( sym ) at the three sample positions
                # but wfdb wrann writes one symbol per sample. The parser
                # looks for '(' at onset, wave-type in middle, ')' at
                # offset. We emit them in triplets here.
                samples.append(s)
        # Build symbols matching the loop above
    # Recompute symbols aligned to samples
    samples = []
    symbols = []
    for center in beat_centers:
        for delta, sym in (
            (-120, "("), (-90, "p"), (-60, ")"),
            (-20, "("), (0, "N"), (20, ")"),
            (80, "("), (120, "t"), (160, ")"),
        ):
            s = int(center + delta)
            if 0 <= s < _LUDB_LEN:
                samples.append(s)
                symbols.append(sym)

    samples_arr = np.array(samples, dtype=np.int64)
    # wrann rejects extensions with digits (e.g., "v1", "v2"), but
    # LUDB's lead-name extensions include six of those. Write each
    # annotation with a letters-only placeholder extension, then
    # rename to the real lead extension afterwards.
    placeholder_stem = "xyzlead"
    for i, lead in enumerate(_LUDB_LEADS):
        placeholder = f"{placeholder_stem}{chr(ord('a') + i)}"
        wfdb.wrann(
            record_name=rec_name,
            extension=placeholder,
            sample=samples_arr,
            symbol=symbols,
            write_dir=data_dir,
        )
        src = os.path.join(data_dir, f"{rec_name}.{placeholder}")
        dst = os.path.join(data_dir, f"{rec_name}.{lead}")
        os.replace(src, dst)


# --------------------------------------------------------------------- #
# BIDMC: respiratory + ECG, 125 Hz, ~8 min, breath annotations
# --------------------------------------------------------------------- #

_BIDMC_FS = 125
_BIDMC_LEN = 60_001  # matches the real format's length
# BIDMC's real headers carry a trailing comma in each signal name
# (e.g., ``RESP,``). The dataset parser matches on the comma-suffixed
# form, so preserve it here.
_BIDMC_SIGS = ("RESP,", "PLETH,", "V,", "AVR,", "II,")


def synthesize_bidmc(
    dest_root: str,
    n_records: int = 2,
    seed: int = 0,
) -> None:
    """Write ``n_records`` synthetic BIDMC records into ``dest_root/``.

    Each record includes a RESP signal + 2 ECG leads + PLETH + AVR,
    plus a ``.breath`` annotation file with periodic breath markers.
    """
    import wfdb

    os.makedirs(dest_root, exist_ok=True)
    rng = np.random.default_rng(seed)

    for rec_idx in range(n_records):
        rec_name = f"bidmc{rec_idx + 1:02d}"
        t = np.arange(_BIDMC_LEN) / _BIDMC_FS
        resp = 0.5 * np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
        pleth = 0.3 * np.sin(2 * np.pi * 1.2 * t)
        ecg = 0.4 * np.sin(2 * np.pi * 1.2 * t)
        noise = lambda: rng.normal(0, 0.02, _BIDMC_LEN)
        sigs = np.stack([resp + noise(), pleth + noise(), ecg + noise(),
                         -ecg + noise(), ecg + noise()], axis=1).astype(np.float32)
        # Age + sex metadata in the first comment line — BIDMC parser
        # reads demographics from the header just like LUDB.
        comments = [
            f"<age>: {55 + 3 * rec_idx}",
            f"<sex>: {'M' if rec_idx % 2 == 0 else 'F'}",
        ]
        wfdb.wrsamp(
            record_name=rec_name,
            fs=_BIDMC_FS,
            units=["pm", "NU", "mV", "mV", "mV"],
            sig_name=list(_BIDMC_SIGS),
            p_signal=sigs,
            fmt=["16"] * len(_BIDMC_SIGS),
            write_dir=dest_root,
            comments=comments,
        )
        # Breath annotations: one mark per breath (0.25 Hz => 4 s apart)
        breath_samples = np.arange(2 * _BIDMC_FS, _BIDMC_LEN, 4 * _BIDMC_FS, dtype=np.int64)
        wfdb.wrann(
            record_name=rec_name,
            extension="breath",
            sample=breath_samples,
            symbol=["+"] * len(breath_samples),
            write_dir=dest_root,
        )


# --------------------------------------------------------------------- #
# MIT-BIH: 2-lead ECG at 360 Hz, beat annotations
# --------------------------------------------------------------------- #

_MITBIH_FS = 360
_MITBIH_LEN = 65_000  # short clip ≈ 3 min
_MITBIH_SIGS = ("MLII", "V5")


def synthesize_mitbih(
    dest_root: str,
    record_names: Iterable[str] = ("100", "101"),
    seed: int = 0,
) -> None:
    """Write synthetic MIT-BIH records with beat annotations.

    Each record has 2 ECG leads + ``.atr`` annotations containing a
    mix of normal (``N``) and abnormal (``V``, ``A``) beat symbols so
    ECGAnomalyDetection / ECGBoundaryDetection tasks see both classes.
    """
    import wfdb

    os.makedirs(dest_root, exist_ok=True)
    rng = np.random.default_rng(seed)

    for rec_idx, rec_name in enumerate(record_names):
        t = np.arange(_MITBIH_LEN) / _MITBIH_FS
        mlii = 0.6 * np.sin(2 * np.pi * 1.2 * t)
        v5 = 0.5 * np.sin(2 * np.pi * 1.2 * t + 0.3)
        noise = rng.normal(0, 0.02, (_MITBIH_LEN, 2)).astype(np.float32)
        sigs = np.stack([mlii, v5], axis=1).astype(np.float32) + noise
        # Age/sex encoded the way MIT-BIH headers do it: "# 69 M ..."
        comments = [f"# {60 + 3 * rec_idx} {'M' if rec_idx % 2 == 0 else 'F'}"]
        wfdb.wrsamp(
            record_name=rec_name,
            fs=_MITBIH_FS,
            units=["mV", "mV"],
            sig_name=list(_MITBIH_SIGS),
            p_signal=sigs,
            fmt=["16", "16"],
            write_dir=dest_root,
            comments=comments,
        )
        # Beat markers every ~1 s — mix N and V so tests see both classes
        beat_samples = np.arange(_MITBIH_FS, _MITBIH_LEN, _MITBIH_FS, dtype=np.int64)
        symbols = ["N" if i % 3 else "V" for i in range(len(beat_samples))]
        wfdb.wrann(
            record_name=rec_name,
            extension="atr",
            sample=beat_samples,
            symbol=symbols,
            write_dir=dest_root,
        )


# --------------------------------------------------------------------- #
# Entry point — regenerate the committed test-resources fixtures
# --------------------------------------------------------------------- #
#
# Run with ``python -m tests.core._synthetic_wfdb`` from the repo root.
# Regenerates the wfdb records under ``test-resources/core/{ludb,
# bidmc,mitbih}/`` from a fixed seed. Checked-in outputs are fully
# synthetic — never ship real patient data in tests.

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _regenerate_all(repo_root: str = _REPO_ROOT) -> None:
    """Rewrite all three dataset fixtures under ``test-resources/core/``."""
    import shutil

    def _wipe_and_recreate(path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    ludb_root = os.path.join(repo_root, "test-resources", "core", "ludb")
    _wipe_and_recreate(os.path.join(ludb_root, "data"))
    synthesize_ludb(ludb_root, n_records=2)

    bidmc_root = os.path.join(repo_root, "test-resources", "core", "bidmc")
    _wipe_and_recreate(bidmc_root)
    synthesize_bidmc(bidmc_root, n_records=2)

    mitbih_root = os.path.join(repo_root, "test-resources", "core", "mitbih")
    _wipe_and_recreate(mitbih_root)
    synthesize_mitbih(mitbih_root, record_names=["100", "101"])


if __name__ == "__main__":
    _regenerate_all()
    print(f"Regenerated synthetic test fixtures under {_REPO_ROOT}/test-resources/core/")
