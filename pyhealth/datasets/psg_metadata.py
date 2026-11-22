"""Shared helpers for indexing sleep studies (polysomnography).

Description: ``NCHSleepDataset`` and ``CHATDataset`` both describe each study
    by its EDF header (sampling rate, duration, channels) and by an
    apnea-hypopnea index counted from its scored annotations. Their files
    differ (NCH ships a TSV, NSRR ships XML) and so do their event names; the
    reading and counting are shared here.
"""

from collections.abc import Collection, Mapping
from pathlib import Path

import mne
import pandas as pd

#: Normalized sleep stages that count as sleep time.
SLEEP_STAGES: frozenset[str] = frozenset({"N1", "N2", "N3", "R"})


def read_edf_header(path: str | Path) -> dict[str, object]:
    """Read the sampling rate, duration and channel labels of an EDF file.

    Only the header is read, not the signals.

    Args:
        path: EDF file.

    Returns:
        ``sfreq`` (Hz), ``duration_sec`` and ``channels``, the channel
        labels as stored, joined with ``|``.

    Raises:
        Exception: Whatever MNE raises for a file it cannot read.

    Examples:
        >>> from pyhealth.datasets.psg_metadata import read_edf_header
        >>> read_edf_header("study.edf")  # doctest: +SKIP
        {'sfreq': 256.0, 'duration_sec': 34200.0, 'channels': 'EEG C3-M2|SpO2'}
    """
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    return {
        "sfreq": sfreq,
        "duration_sec": round(raw.n_times / sfreq, 3),
        "channels": "|".join(raw.ch_names),
    }


def summarize_events(
    annotations: pd.DataFrame,
    apnea_events: Collection[str],
    hypopnea_events: Collection[str],
    stage_names: Mapping[str, str],
) -> dict[str, int | float]:
    """Count apneas, hypopneas and sleep time in a table of annotations.

    Args:
        annotations: One row per annotation, with ``duration`` (seconds) and
            ``description`` columns.
        apnea_events: Descriptions counted as apneas.
        hypopnea_events: Descriptions counted as hypopneas.
        stage_names: Maps a stage description to ``"W"``, ``"N1"``,
            ``"N2"``, ``"N3"`` or ``"R"``. Annotations mapped to N1, N2, N3
            or R add their duration to the sleep time.

    Returns:
        ``n_apnea``, ``n_hypopnea``, ``sleep_hours`` and ``ahi``: apneas and
        hypopneas per hour of sleep, 0 when there is no sleep.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.psg_metadata import summarize_events
        >>> annotations = pd.DataFrame(
        ...     {
        ...         "duration": [1800.0, 12.0, 15.0],
        ...         "description": ["N2", "Apnea", "Hypopnea"],
        ...     }
        ... )
        >>> summarize_events(annotations, {"Apnea"}, {"Hypopnea"}, {"N2": "N2"})
        {'n_apnea': 1, 'n_hypopnea': 1, 'sleep_hours': 0.5, 'ahi': 4.0}
    """
    description = annotations["description"].astype(str)
    n_apnea = int(description.isin(apnea_events).sum())
    n_hypopnea = int(description.isin(hypopnea_events).sum())
    is_sleep = description.map(stage_names).isin(SLEEP_STAGES)
    duration = pd.to_numeric(annotations.loc[is_sleep, "duration"])
    sleep_hours = float(duration.sum()) / 3600
    ahi = (n_apnea + n_hypopnea) / sleep_hours if sleep_hours > 0 else 0.0
    return {
        "n_apnea": n_apnea,
        "n_hypopnea": n_hypopnea,
        "sleep_hours": round(sleep_hours, 4),
        "ahi": round(ahi, 2),
    }
