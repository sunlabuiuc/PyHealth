"""PyHealth dataset for the RESPect CCEP Dataset.

Dataset link:
    https://github.com/OpenNeuroDatasets/ds004080

This module provides the RESPectCCEPDataset class for loading and processing
Cortico-Cortical Evoked Potentials (CCEPs) data organized in BIDS format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class RESPectCCEPDataset(BaseDataset):
    """Dataset class for the RESPect CCEP dataset.

    This dataset consists of cortico-cortical evoked potentials (CCEPs)
    measured with electrocorticography (ECoG) during single pulse electrical
    stimulation (SPES) in 74 patients. The raw data is organized in BIDS.

    The dataset class flattens the BIDS hierarchy into a single metadata CSV
    with one row per run-level ``*_events.tsv`` file. Each row keeps patient
    demographics from ``participants.tsv`` together with file pointers to the
    run-specific and session-specific BIDS sidecars that downstream tasks can
    dereference.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the RESPectCCEPDataset.

        This initializer creates a metadata CSV index from the BIDS-organized
        data if it doesn't already exist, then initializes the BaseDataset with
        the indexed data.

        Args:
            root: Root directory containing the BIDS dataset with
                `participants.tsv` and `sub-*/` folders.
            tables: Additional tables to load beyond the default.
                Defaults to None.
            dataset_name: Custom name for the dataset. Defaults to
                "RESPectCCEPDataset".
            config_path: Path to the YAML configuration file. If None, uses the
                default config at `configs/respect_ccep.yaml`.
            **kwargs: Additional arguments passed to BaseDataset.

        Raises:
            FileNotFoundError: If `participants.tsv` is not found in root.
            ValueError: If `participants.tsv` doesn't contain a `participant_id` column.

        Example:
            >>> dataset = RESPectCCEPDataset(root="/path/to/ds004080/")
            >>> print(f"Patients: {len(dataset.unique_patient_ids)}")
        """
        if config_path is None:
            logger.info("No config path provided, using default config.")
            config_path = str(Path(__file__).parent / "configs" / "respect_ccep.yaml")

        self._pyhealth_csv = str(Path(root) / "respect_ccep_metadata-pyhealth.csv")
        if not Path(self._pyhealth_csv).exists():
            logger.info("Indexing BIDS structure and preparing PyHealth metadata...")
            self._prepare_metadata(root, self._pyhealth_csv)

        default_tables = ["respectccep"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "RESPectCCEPDataset",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def _extract_bids_entities(file_path: Path) -> Dict[str, Optional[str]]:
        """Extract BIDS entities from a BIDS-formatted filename.

        Parses the filename to extract participant ID, session ID, and run ID
        based on BIDS naming conventions (e.g., `sub-XXX_ses-YYY_run-ZZZ`).

        Args:
            file_path: Path object for a BIDS file.

        Returns:
            Dictionary with keys `participant_id`, `session_id`, and `run_id`,
            each with Optional[str] values (None if not found in filename).

        Example:
            >>> path = Path("sub-01_ses-1_task-SPES_run-01_events.tsv")
            >>> entities = RESPectCCEPDataset._extract_bids_entities(path)
            >>> entities["participant_id"]
            'sub-01'
        """
        entities: Dict[str, Optional[str]] = {
            "participant_id": None,
            "session_id": None,
            "run_id": None,
        }

        for part in file_path.stem.split("_"):
            if part.startswith("sub-"):
                entities["participant_id"] = part
            elif part.startswith("ses-"):
                entities["session_id"] = part
            elif part.startswith("run-"):
                entities["run_id"] = part

        return entities

    @staticmethod
    def _relative_or_none(path: Optional[Path], root_path: Path) -> Optional[str]:
        """Convert a path to a root-relative POSIX path if it exists.

        Args:
            path: Absolute path to check, or None.
            root_path: Root path to compute relative path from.

        Returns:
            Root-relative POSIX path as string if path exists, None otherwise.
        """
        if path is None or not path.exists():
            return None
        return path.relative_to(root_path).as_posix()

    @staticmethod
    def _find_session_electrodes(
        ieeg_dir: Path,
        participant_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional[Path]:
        """Find the session-level electrodes TSV file for an iEEG directory.

        Searches for electrodes.tsv files using multiple naming conventions,
        prioritizing more specific matches (with participant + session) before
        broader matches.

        Args:
            ieeg_dir: Directory to search for electrodes file.
            participant_id: BIDS participant ID (e.g., 'sub-01'), or None.
            session_id: BIDS session ID (e.g., 'ses-1'), or None.

        Returns:
            Path to the electrodes.tsv file if found, None otherwise.
        """
        candidates: List[Path] = []
        if participant_id and session_id:
            electrodes_file = f"{participant_id}_{session_id}_electrodes.tsv"
            candidates.append(ieeg_dir / electrodes_file)
        if participant_id:
            candidates.append(ieeg_dir / f"{participant_id}_electrodes.tsv")

        candidates.extend(sorted(ieeg_dir.glob("*_electrodes.tsv")))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _find_run_sidecar(events_path: Path, suffix: str) -> Optional[Path]:
        """Find a run-level sidecar file by replacing the `_events.tsv` suffix.

        Given a path to an `*_events.tsv` file, searches for a companion file
        with a different suffix (e.g., `_channels.tsv`, `_ieeg.vhdr`).

        Args:
            events_path: Path to the `*_events.tsv` file.
            suffix: Suffix to replace `_events.tsv` with (e.g., '_channels.tsv').

        Returns:
            Path to the sidecar file if it exists, None otherwise.
        """
        new_name = events_path.name.replace("_events.tsv", suffix)
        expected = events_path.with_name(new_name)
        if expected.exists():
            return expected
        return None

    def _prepare_metadata(self, root: str, output_csv: str) -> None:
        """Parse BIDS directory structure and generate metadata CSV.

        Scans the BIDS directory tree to find all run-level `*_events.tsv` files,
        extracts BIDS entities and file pointers from each, merges with patient
        demographics from `participants.tsv`, and writes the result to a single
        flattened metadata CSV file.

        Args:
            root: Root directory of the BIDS dataset.
            output_csv: Path where the output metadata CSV will be written.

        Raises:
            FileNotFoundError: If `participants.tsv` is not found in root.
            ValueError: If `participants.tsv` doesn't contain 'participant_id'
                column.

        Writes:
            CSV file at `output_csv` with columns:
                - participant_id: BIDS participant ID
                - session_id: BIDS session ID (if present)
                - run_id: BIDS run ID (if present)
                - *_file_path: Relative paths to BIDS sidecars
                - Additional columns from participants.tsv demographics
        """
        root_path = Path(root)
        participants_file = root_path / "participants.tsv"

        if not participants_file.exists():
            msg = f"Dataset root must contain 'participants.tsv'. Searched in {root}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        df_participants = pd.read_csv(participants_file, sep="\t")
        if "participant_id" not in df_participants.columns:
            msg = "participants.tsv must contain a 'participant_id' column"
            logger.error(msg)
            raise ValueError(msg)

        records: List[Dict[str, Optional[str]]] = []
        for events_path in sorted(root_path.rglob("*_events.tsv")):
            entities = self._extract_bids_entities(events_path)
            participant_id = entities["participant_id"]
            session_id = entities["session_id"]
            run_id = entities["run_id"]
            ieeg_dir = events_path.parent

            channels_path = self._find_run_sidecar(events_path, "_channels.tsv")
            vhdr_path = self._find_run_sidecar(events_path, "_ieeg.vhdr")
            electrodes_path = self._find_session_electrodes(
                ieeg_dir=ieeg_dir,
                participant_id=participant_id,
                session_id=session_id,
            )

            if participant_id is None:
                msg = f"Skipping events file without participant entity: {events_path}"
                logger.warning(msg)
                continue

            records.append({
                "participant_id": participant_id,
                "session_id": session_id,
                "run_id": run_id,
                "events_file_path": self._relative_or_none(
                    events_path, root_path
                ),
                "channels_file_path": self._relative_or_none(
                    channels_path, root_path
                ),
                "electrodes_file_path": self._relative_or_none(
                    electrodes_path, root_path
                ),
                "vhdr_file_path": self._relative_or_none(vhdr_path, root_path),
            })

        if records:
            df_records = pd.DataFrame(records)
            df_merged = pd.merge(
                df_records,
                df_participants,
                on="participant_id",
                how="left",
            )
        else:
            msg = "No *_events.tsv files found. Writing empty metadata index."
            logger.warning(msg)
            df_merged = pd.DataFrame(
                columns=[
                    "participant_id",
                    "session_id",
                    "run_id",
                    "events_file_path",
                    "channels_file_path",
                    "electrodes_file_path",
                    "vhdr_file_path",
                ]
                + list(df_participants.columns.drop("participant_id", errors="ignore"))
            )

        df_merged.to_csv(output_csv, index=False)
        logger.info("Indexed metadata saved to %s", output_csv)
