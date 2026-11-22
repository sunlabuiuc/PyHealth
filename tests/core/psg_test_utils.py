"""Small synthetic sleep study files for the NCH and CHAT dataset tests."""

from pathlib import Path

import numpy as np
import pandas as pd


def write_edf(path: Path, labels, sfreq: int = 8, seconds: int = 120) -> None:
    """Write a 16-bit EDF of zeros with one-second records, readable by MNE.

    MNE cannot write EDF without the optional ``edfio`` package, so the
    header is written by hand.
    """

    def field(value, width):
        return str(value).ljust(width).encode("ascii")

    ns = len(labels)
    header = b"".join(
        [
            field(0, 8),
            field("X X X X", 80),
            field("Startdate 04-MAR-2017 X X X", 80),
            field("04.03.17", 8),
            field("21.30.00", 8),
            field(256 * (ns + 1), 8),
            field("", 44),
            field(seconds, 8),
            field(1, 8),
            field(ns, 4),
        ]
    )
    for width, values in (
        (16, labels),
        (80, [""] * ns),
        (8, ["uV"] * ns),
        (8, [-100] * ns),
        (8, [100] * ns),
        (8, [-32768] * ns),
        (8, [32767] * ns),
        (80, [""] * ns),
        (8, [sfreq] * ns),
        (32, [""] * ns),
    ):
        header += b"".join(field(v, width) for v in values)
    data = np.zeros((seconds, ns * sfreq), dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data.tobytes())


def write_tsv(path: Path, rows) -> None:
    """Write ``(onset, duration, description)`` rows as a tab-separated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["onset", "duration", "description"]).to_csv(
        path, sep="\t", index=False
    )
