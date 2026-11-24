"""ISRUC-Sleep polysomnography dataset.

Author: Sameh Aissa (port to the PyHealth 2.0 API)

ISRUC-Sleep has three subgroups, each downloaded separately from
https://sleeptight.isr.uc.pt/:

- subgroup I, 100 subjects with sleep disorders, one recording each
- subgroup II, 8 subjects with two recordings each
- subgroup III, 10 healthy subjects, one recording each

A recording is an EDF file shipped with a ``.rec`` extension, sampled at
200 Hz, with the scorings of two experts next to it.

References:
    - Khalighi, S. et al. ISRUC-Sleep: A comprehensive public dataset for
      sleep researchers. Computer Methods and Programs in Biomedicine 124,
      2016. https://doi.org/10.1016/j.cmpb.2015.10.013
"""

import logging
import os
from pathlib import Path

import pandas as pd

from pyhealth.tasks import SleepStagingISRUC

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_cache_dir,
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_CSV_NAME = "isruc-pyhealth.csv"

_SIGNAL_SUFFIXES: tuple[str, ...] = (".edf", ".rec")


def _edf_path(rec_file: Path, root: Path, alias: str) -> Path | None:
    """Return a ``.edf`` path that opens ``rec_file``.

    MNE only reads EDF files with an ``.edf`` extension. The first choice
    is a hard link ``<id>.edf`` next to the ``.rec`` file, which takes no
    extra space. When that fails, for example on a read only folder, a
    symbolic link named ``alias`` goes in the home cache instead.

    Args:
        rec_file: Path to an ISRUC ``.rec`` recording.
        root: Dataset root directory, used for the cache directory.
        alias: File name for the link in the cache, unique per recording.

    Returns:
        The ``.edf`` path, or None if no link could be made.
    """
    sibling = rec_file.with_suffix(".edf")
    try:
        os.link(rec_file, sibling)
        return sibling
    except OSError:
        pass
    cached = metadata_cache_dir(root, "isruc") / "edf" / alias
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists():
            cached.symlink_to(rec_file.resolve())
        return cached
    except OSError:
        logger.warning(
            "Skipping %s, MNE needs an .edf extension and no link could be "
            "made. Copy or rename it to %s.",
            rec_file,
            sibling.name,
        )
        return None


def _scoring_file(signal_file: Path, annotator: int) -> str:
    """Return the path of one expert's scoring for a recording.

    Args:
        signal_file: Path to the ``.rec`` or ``.edf`` recording.
        annotator: 1 or 2.

    Returns:
        The resolved ``<id>_<annotator>.txt`` path, or an empty string when
        that file is missing.
    """
    path = signal_file.with_name(f"{signal_file.stem}_{annotator}.txt")
    return str(path.resolve()) if path.is_file() else ""


def _find_recordings(root: Path) -> list[Path]:
    """List the ISRUC recordings under a subgroup folder.

    A recording is ``<subject>/<subject>.rec`` (subgroups I and III) or
    ``<subject>/<session>/<session>.rec`` (subgroup II), with ``.edf`` in
    place of ``.rec`` also accepted. When both exist the ``.edf`` file is
    used.

    Args:
        root: Subgroup folder.

    Returns:
        One path per recording.
    """
    recordings: dict[Path, Path] = {}
    for pattern in ("*/*", "*/*/*"):
        for path in sorted(root.glob(pattern)):
            if path.suffix.lower() not in _SIGNAL_SUFFIXES:
                continue
            if path.stem != path.parent.name:
                continue
            key = path.with_suffix("")
            if key not in recordings or path.suffix.lower() == ".edf":
                recordings[key] = path
    return list(recordings.values())


def _sort_key(row: dict) -> tuple:
    """Sort subjects and sessions numerically when the ids are numbers.

    Args:
        row: Metadata row with ``patient_id`` and ``session_id``.

    Returns:
        A key that puts ``"2"`` before ``"10"``.
    """
    return tuple(
        (0, int(v)) if v.isdigit() else (1, v)
        for v in (row["patient_id"], row["session_id"])
    )


class ISRUCDataset(BaseDataset):
    """ISRUC-Sleep polysomnography (PSG) dataset.

    Point ``root`` at one subgroup folder. Subgroups I and III hold one
    folder per subject, ``<subject>/<subject>.rec``. Subgroup II adds a
    folder per recording, ``<subject>/<session>/<session>.rec``. Next to
    each recording are the scorings of the two experts, ``<id>_1.txt`` and
    ``<id>_2.txt``.

    ``prepare_metadata`` writes ``isruc-pyhealth.csv`` (one row per
    recording) in ``root``, or under ``~/.cache/pyhealth/isruc/<hash of
    root>`` when ``root`` is not writable. MNE only opens EDF files with
    an ``.edf`` extension, so each ``.rec`` file gets an ``.edf`` link the
    first time, a hard link next to it or a symbolic link in that cache.
    The default task is :class:`~pyhealth.tasks.SleepStagingISRUC`.

    Getting the data:
        - Source: https://sleeptight.isr.uc.pt/, page "Download Dataset"
        - Access: open, no account needed.
        - Download: one MEGA link per subgroup on that page, about 14 GB
          for subgroup I, 2.2 GB for II and 1.35 GB for III. Extract the
          subject archives in place.
        - ``root``: one extracted subgroup folder, holding ``1/``, ``2/``
          and so on.

    Args:
        root: Folder of one ISRUC-Sleep subgroup, for example the extracted
            ``ISRUC-Sleep-I`` folder.
        dataset_name: Name stored on the dataset. Defaults to ``"isruc"``.
        config_path: YAML config. Defaults to ``configs/isruc.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset
        >>> dataset = ISRUCDataset(root="/path/to/ISRUC-Sleep-I")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> samples[0]["signal"].shape
        torch.Size([6, 6000])
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "isruc.yaml"

        self.root = root
        self.prepare_metadata()

        super().__init__(
            root=resolve_metadata_root(root, "isruc", _CSV_NAME),
            tables=["recordings"],
            dataset_name=dataset_name or "isruc",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``isruc-pyhealth.csv`` if it is not already on disk.

        The CSV has one row per recording with ``patient_id``,
        ``session_id``, ``signal_file`` and the two scoring files
        ``label_file_1`` and ``label_file_2`` (empty when missing). In
        subgroups I and III the session is ``"1"``. An existing CSV, in
        ``root`` or in the cache, is left untouched.

        Returns:
            None. The CSV path is either ``<root>/isruc-pyhealth.csv`` or
            ``~/.cache/pyhealth/isruc/<hash of root>/isruc-pyhealth.csv``.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, "isruc", _CSV_NAME):
            return

        rows = []
        for path in _find_recordings(root):
            parts = path.relative_to(root).parts
            patient_id = parts[0]
            session_id = parts[1] if len(parts) == 3 else "1"
            signal_file = path
            if path.suffix.lower() == ".rec":
                signal_file = _edf_path(path, root, f"{patient_id}_{session_id}.edf")
                if signal_file is None:
                    continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "signal_file": str(signal_file.absolute()),
                    "label_file_1": _scoring_file(path, 1),
                    "label_file_2": _scoring_file(path, 2),
                }
            )
        if not rows:
            logger.warning("No ISRUC recordings found under %s", root)
            return

        rows.sort(key=_sort_key)
        write_metadata_csv(pd.DataFrame(rows), root, "isruc", _CSV_NAME)

    @property
    def default_task(self) -> SleepStagingISRUC:
        """Return the default ISRUC sleep staging task.

        Returns:
            A :class:`~pyhealth.tasks.SleepStagingISRUC` on the six EEG
            channels with 30 second epochs and the first expert's scoring.
        """
        return SleepStagingISRUC()
