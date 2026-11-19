"""
PyHealth dataset for BCI Competition IV dataset 2a (BNCI Horizon 001-2014).

Dataset link:
    https://bnci-horizon-2020.eu/database/data-sets (entry 001-2014)

Dataset description:
    C. Brunner, R. Leeb, G. R. Muller-Putz, A. Schlogl, and G. Pfurtscheller.
    "BCI Competition 2008 - Graz data set A."
    https://www.bbci.de/competition/iv/desc_2a.pdf

Dataset paper: (please cite if you use this dataset)
    M. Tangermann, K.-R. Muller, A. Aertsen, N. Birbaumer, C. Braun,
    C. Brunner, et al. "Review of the BCI Competition IV." Frontiers in
    Neuroscience 6, 55 (2012). https://doi.org/10.3389/fnins.2012.00055

Description:
    Implements ``BCICompetitionIV2aDataset``: one metadata row per session
    file (``A0sT.mat`` / ``A0sE.mat``) written to a cache directory, with the
    official session split. Trials are cut lazily by the task.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml
from filelock import FileLock

from .base_dataset import BaseDataset
from .configs.config import load_yaml_config
from .ptbxl import root_cache_key
from .utils import MODULE_CACHE_PATH

if TYPE_CHECKING:
    from pyhealth.tasks.bciiv2a_motor_imagery import (
        BCIIV2aMotorImageryClassification,
    )

logger = logging.getLogger(__name__)

BNCI_001_2014_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014"

#: Official protocol: session T (training) is the train part, session E
#: (evaluation) the test part, within each subject.
SESSION_SPLIT = {"T": "train", "E": "test"}

METADATA_COLUMNS = (
    "patient_id",
    "record_id",
    "subject_id",
    "session",
    "split",
    "signal_file",
)

_DEFAULT_METADATA_CACHE = Path(MODULE_CACHE_PATH) / "bciiv2a"


def _write_text_atomic(text: str, dest: Path) -> None:
    """Write ``text`` to ``dest`` through a temporary file and ``os.replace``."""
    tmp_path = dest.with_name(dest.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def bciiv2a_filename(subject: int, session: str) -> str:
    """Return the BNCI file name of one subject's session.

    Args:
        subject (int): Subject number, 1 to 9.
        session (str): ``"T"`` (training) or ``"E"`` (evaluation).

    Returns:
        str: For example ``"A01T.mat"``.

    Examples:
        >>> bciiv2a_filename(1, "E")
        'A01E.mat'
    """
    return f"A{int(subject):02d}{session.upper()}.mat"


class BCICompetitionIV2aDataset(BaseDataset):
    """BCI Competition IV dataset 2a: four-class motor imagery EEG.

    Nine subjects performed cued motor imagery of the left hand, right hand,
    both feet and tongue. Each subject has two sessions on different days, T
    (training) and E (evaluation), of 6 runs x 48 trials = 288 trials. EEG is
    22 channels (plus 3 EOG) at 250 Hz, band-passed 0.5-100 Hz with a 50 Hz
    notch. Each session file becomes one event of table ``sessions``; the
    task cuts the trials.

    Getting the data:
        - Source: https://bnci-horizon-2020.eu/database/data-sets, entry
          "1. Four class motor imagery (001-2014)", originally data set 2a
          of BCI Competition IV. Paradigm description:
          https://www.bbci.de/competition/iv/desc_2a.pdf. Paper: Tangermann
          et al. (2012), below.
        - Access: open download. No account, registration or data-use
          agreement.
        - Files: 18 MATLAB files ``A01T.mat``, ``A01E.mat`` ...
          ``A09T.mat``, ``A09E.mat``, 37-46 MB each (779,873,919 bytes, about
          780 MB, in all). ``T`` is the training session and ``E`` the
          evaluation session of each subject.
        - Download: pass ``download=True`` to fetch the requested files
          into ``root``, or run::

              mkdir -p bnci/001-2014 && cd bnci/001-2014
              for s in 1 2 3 4 5 6 7 8 9; do for x in T E; do
                wget https://bnci-horizon-2020.eu/database/data-sets/001-2014/A0${s}${x}.mat
              done; done

          (the links redirect to ``https://lampx.tugraz.at/~bci/database/001-2014/``).
        - Unpack: nothing to unpack; the ``.mat`` files are read directly.
        - ``root`` layout (only the files of the requested ``subjects`` and
          ``sessions`` need to be present)::

              <root>/
                  A01T.mat
                  A01E.mat
                  ...
                  A09T.mat
                  A09E.mat

          The derived metadata goes to ``metadata_cache_dir``, never into
          ``root``. The BNCI ``E`` files include the evaluation labels (the
          original competition GDF release does not), so the GDF files are
          not supported.

    Licence: Creative Commons Attribution-NoDerivatives 4.0 International
    (CC BY-ND 4.0, https://creativecommons.org/licenses/by-nd/4.0/),
    licensor Institute for Knowledge Discovery, Graz University of
    Technology. You may use and share the original files with attribution,
    but you must not distribute modified or preprocessed copies (for
    example the cached samples this class and its task produce). There is
    no non-commercial (NC) term. Cite Tangermann et al. (2012) and the
    dataset description above.

    Citation:
        M. Tangermann, K.-R. Muller, A. Aertsen, N. Birbaumer, C. Braun,
        C. Brunner, et al. "Review of the BCI Competition IV." Frontiers in
        Neuroscience 6, 55 (2012). https://doi.org/10.3389/fnins.2012.00055

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Official split:
        Each metadata row and each sample carries ``session`` (``"T"`` or
        ``"E"``), ``split`` (``"train"`` or ``"test"``) and
        ``session_fold`` (0 for T, 1 for E). The competition protocol trains
        on T and tests on E within each subject (cross-session). Because
        ``patient_id`` is the subject (``"A01"``) and is shared by both
        sessions, :class:`~pyhealth.datasets.PatientSplit` gives a
        cross-subject split instead; use
        :func:`~pyhealth.datasets.split_by_strat_fold` with
        ``fold_field="session_fold"`` for the official one.

    Args:
        root (str): Folder holding (or receiving) the ``.mat`` files.
        dataset_name (str | None): Optional name prefix. The selection and a
            content hash of the metadata are appended.
        config_path (str | Path | None): Optional YAML config; defaults to
            ``configs/bciiv2a.yaml``.
        subjects (Sequence[int] | None): Subjects to load. Defaults to 1-9.
        sessions (Sequence[str]): Sessions to load. Defaults to
            ``("T", "E")``.
        download (bool): Download missing files from BNCI. Defaults to
            ``False``.
        metadata_cache_dir (str | Path | None): Where the derived metadata
            CSV and YAML are written. Defaults to
            ``~/.cache/pyhealth/datasets/bciiv2a``.
        **kwargs: Forwarded to :class:`BaseDataset`.

    Raises:
        FileNotFoundError: If a requested file is missing and ``download`` is
            ``False``.

    Examples:
        >>> from pyhealth.datasets import BCICompetitionIV2aDataset
        >>> ds = BCICompetitionIV2aDataset(root="/data/bnci/001-2014", subjects=[1])  # doctest: +SKIP
        >>> ds.stats()  # doctest: +SKIP
        >>> samples = ds.set_task()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        subjects: Sequence[int] | None = None,
        sessions: Sequence[str] = ("T", "E"),
        download: bool = False,
        metadata_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.data_root = Path(root)
        self.subjects = self._check_subjects(subjects)
        self.sessions = self._check_sessions(sessions)
        self.download = download
        self.metadata_cache_dir = Path(
            metadata_cache_dir
            if metadata_cache_dir is not None
            else _DEFAULT_METADATA_CACHE
        )
        selection = (
            "s"
            + "-".join(f"{s:02d}" for s in self.subjects)
            + "_"
            + "".join(self.sessions)
        )
        self.metadata_file_name = (
            f"bciiv2a-pyhealth-{selection}-{root_cache_key(self.data_root)}.csv"
        )
        package_config = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).parent / "configs" / "bciiv2a.yaml"
        )
        self.prepare_metadata()
        resolved_config = self._write_resolved_config(package_config)
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        content_key = hashlib.sha1(csv_path.read_bytes()).hexdigest()[:10]
        super().__init__(
            root=str(self.data_root),
            tables=["sessions"],
            dataset_name=f"{dataset_name or 'bciiv2a'}_{selection}_{content_key}",
            config_path=str(resolved_config),
            **kwargs,
        )

    @staticmethod
    def _check_subjects(subjects: Iterable[int] | None) -> list[int]:
        values = sorted(
            {int(s) for s in (subjects if subjects is not None else range(1, 10))}
        )
        bad = [s for s in values if not 1 <= s <= 9]
        if bad or not values:
            raise ValueError(f"subjects must be in 1..9, got {list(subjects or [])}")
        return values

    @staticmethod
    def _check_sessions(sessions: Iterable[str]) -> list[str]:
        values = [s.upper() for s in sessions]
        if not values or any(s not in SESSION_SPLIT for s in values):
            raise ValueError(f"sessions must be 'T' and/or 'E', got {list(sessions)}")
        return [s for s in ("T", "E") if s in values]

    def load_table(self, table_name: str):
        """Read the derived metadata CSV from ``metadata_cache_dir``.

        Args:
            table_name (str): Table name from the YAML config.

        Returns:
            The dataframe produced by :meth:`BaseDataset.load_table`.
        """
        original_root = self.root
        self.root = str(self.metadata_cache_dir)
        try:
            return super().load_table(table_name)
        finally:
            self.root = original_root

    def _write_resolved_config(self, package_config: Path) -> Path:
        config = load_yaml_config(str(package_config))
        config.tables["sessions"].file_path = self.metadata_file_name
        out_path = self.metadata_cache_dir / self.metadata_file_name.replace(
            ".csv", ".yaml"
        )
        with FileLock(str(out_path) + ".lock"):
            _write_text_atomic(
                yaml.safe_dump(config.model_dump(), sort_keys=False), out_path
            )
        return out_path

    def _download(self, path: Path) -> None:
        from pyhealth.utils import download_file

        url = f"{BNCI_001_2014_URL}/{path.name}"
        logger.info("Downloading %s to %s", url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, str(path))

    def prepare_metadata(self) -> None:
        """Locate (or download) the requested files and write the metadata.

        Examples:
            >>> ds = BCICompetitionIV2aDataset(root="/data/bnci/001-2014")  # doctest: +SKIP
            >>> ds.prepare_metadata()  # doctest: +SKIP
        """
        rows = []
        for subject in self.subjects:
            for session in self.sessions:
                path = self.data_root / bciiv2a_filename(subject, session)
                if not path.is_file():
                    if not self.download:
                        raise FileNotFoundError(
                            f"Missing {path}. Download it from {BNCI_001_2014_URL}/"
                            f"{path.name} or pass download=True."
                        )
                    self._download(path)
                rows.append(
                    {
                        "patient_id": f"A{subject:02d}",
                        "record_id": path.stem,
                        "subject_id": subject,
                        "session": session,
                        "split": SESSION_SPLIT[session],
                        "signal_file": str(path.resolve()),
                    }
                )
        df = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(csv_path) + ".lock"):
            if csv_path.is_file():
                try:
                    if pd.read_csv(csv_path).equals(df):
                        return
                except (OSError, ValueError, pd.errors.ParserError):
                    pass
            _write_text_atomic(df.to_csv(index=False), csv_path)
        logger.info("Wrote %d BCI-IV-2a session rows to %s", len(df), csv_path)

    @property
    def default_task(self) -> BCIIV2aMotorImageryClassification:
        """The four-class motor imagery task with the [2, 6) s window.

        Returns:
            BCIIV2aMotorImageryClassification: Default task instance.
        """
        from pyhealth.tasks.bciiv2a_motor_imagery import (
            BCIIV2aMotorImageryClassification,
        )

        return BCIIV2aMotorImageryClassification()
