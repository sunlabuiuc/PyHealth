import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_dataset import BaseDataset
from ..tasks.sleep_staging import SleepStagingDREAMT

logger = logging.getLogger(__name__)


class DREAMTDataset(BaseDataset):
    """Base dataset wrapper for DREAMT sleep-staging data.

    This dataset is designed for reproducibility work around DREAMT-style sleep
    staging. It supports two common local layouts:

    1. The official PhysioNet DREAMT release rooted at a version directory
       containing ``participant_info.csv`` and one or both of ``data_64Hz`` and
       ``data_100Hz``.
    2. A processed per-subject directory containing files named with a subject
       identifier such as ``S002_record.csv`` or ``S002_features.npz``.

    For the official release, the dataset builds a metadata table with one
    event per subject. The downstream task then reads the referenced signal file
    and produces fixed-size windows for sleep-stage classification.

    Args:
        root: Root directory of DREAMT data. This can be the DREAMT version
            directory itself or a parent directory that contains exactly one
            DREAMT release.
        dataset_name: Optional dataset name. Defaults to ``"dreamt_sleep"``.
        config_path: Optional config path. Defaults to the bundled
            ``dreamt.yaml`` config.
        preferred_source: Preferred signal source for the generic
            ``signal_file`` column. One of ``"auto"``, ``"wearable"``, or
            ``"psg"``.
        cache_dir: Optional PyHealth cache directory.
        num_workers: Number of workers for PyHealth dataset operations.
        dev: Whether to enable PyHealth dev mode.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> dataset = DREAMTDataset(
        ...     root="/path/to/dreamt/2.1.0",
        ...     preferred_source="wearable",
        ... )
        >>> patient = dataset.get_patient("S002")
        >>> event = patient.get_events("dreamt_sleep")[0]
        >>> event.signal_file
        '/path/to/dreamt/2.1.0/data_64Hz/S002_whole_df.csv'
    """

    SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle", ".npz", ".npy"}
    _SUBJECT_PATTERN = re.compile(r"(S\d{3})", re.IGNORECASE)

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        preferred_source: str = "auto",
        cache_dir: str | Path | None = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        preferred_source = preferred_source.lower()
        if preferred_source not in {"auto", "wearable", "psg"}:
            raise ValueError(
                "preferred_source must be one of 'auto', 'wearable', or 'psg'."
            )

        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "dreamt.yaml"

        resolved_root = self._resolve_root(Path(root).expanduser().resolve())
        self.prepare_metadata(resolved_root, preferred_source=preferred_source)

        self.preferred_source = preferred_source

        super().__init__(
            root=str(resolved_root),
            tables=["dreamt_sleep"],
            dataset_name=dataset_name or "dreamt_sleep",
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def _extract_subject_id(path: Path) -> Optional[str]:
        match = DREAMTDataset._SUBJECT_PATTERN.search(path.name)
        if match is None:
            return None
        return match.group(1).upper()

    @classmethod
    def _is_signal_file(cls, path: Path) -> bool:
        if path.suffix.lower() not in cls.SUPPORTED_SUFFIXES:
            return False
        return cls._extract_subject_id(path) is not None

    @classmethod
    def _resolve_root(cls, root: Path) -> Path:
        """Resolve a user-provided root to a concrete DREAMT data directory."""
        if not root.exists():
            raise FileNotFoundError(f"DREAMT root does not exist: {root}")

        if (root / "participant_info.csv").exists():
            return root

        if any(
            cls._is_signal_file(path)
            for path in root.iterdir()
            if path.is_file()
        ):
            return root

        candidates = sorted(
            {
                path.parent.resolve()
                for path in root.rglob("participant_info.csv")
            }
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            candidate_text = ", ".join(str(path) for path in candidates[:5])
            raise ValueError(
                "Found multiple DREAMT roots under the provided path. Use an "
                f"explicit version directory instead. Candidates: {candidate_text}"
            )

        signal_candidates = sorted(
            {
                path.parent.resolve()
                for path in root.rglob("*")
                if path.is_file() and cls._is_signal_file(path)
            }
        )
        if len(signal_candidates) == 1:
            return signal_candidates[0]
        if len(signal_candidates) > 1:
            counts = {
                candidate: sum(
                    1
                    for child in candidate.iterdir()
                    if child.is_file() and cls._is_signal_file(child)
                )
                for candidate in signal_candidates
            }
            best = max(counts.items(), key=lambda item: item[1])[0]
            logger.info(
                "Resolved DREAMT root to %s based on detected subject files.",
                best,
            )
            return best

        raise FileNotFoundError(
            "Could not find a DREAMT data directory containing "
            "participant_info.csv or processed subject files."
        )

    @staticmethod
    def _coerce_float(value: object) -> object:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            return value

    @classmethod
    def _locate_subject_file(cls, directory: Path, patient_id: str) -> Optional[Path]:
        if not directory.exists():
            return None

        candidates = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and cls._extract_subject_id(path) == patient_id
            and path.suffix.lower() in cls.SUPPORTED_SUFFIXES
        )
        if not candidates:
            return None

        def score(path: Path) -> tuple[int, int, int]:
            name = path.name.lower()
            return (
                int("updated" in name),
                int("whole" in name or "psg" in name or "record" in name),
                -len(name),
            )

        return max(candidates, key=score)

    @staticmethod
    def _select_signal_file(
        file_64hz: Optional[Path],
        file_100hz: Optional[Path],
        preferred_source: str,
    ) -> tuple[Optional[Path], Optional[str], Optional[float]]:
        if preferred_source == "wearable":
            if file_64hz is not None:
                return file_64hz, "wearable", 64.0
            if file_100hz is not None:
                return file_100hz, "psg", 100.0
        elif preferred_source == "psg":
            if file_100hz is not None:
                return file_100hz, "psg", 100.0
            if file_64hz is not None:
                return file_64hz, "wearable", 64.0
        else:
            if file_64hz is not None:
                return file_64hz, "wearable", 64.0
            if file_100hz is not None:
                return file_100hz, "psg", 100.0
        return None, None, None

    @classmethod
    def _build_metadata_from_participant_info(
        cls,
        root: Path,
        preferred_source: str,
    ) -> pd.DataFrame:
        participant_info_path = root / "participant_info.csv"
        participant_info = pd.read_csv(participant_info_path)
        participant_info.columns = [
            str(col).strip() for col in participant_info.columns
        ]

        rename_map = {
            "SID": "patient_id",
            "AGE": "age",
            "Age": "age",
            "GENDER": "gender",
            "Gender": "gender",
            "BMI": "bmi",
            "OAHI": "oahi",
            "AHI": "ahi",
            "Mean_SaO2": "mean_sao2",
            "Arousal Index": "arousal_index",
            "MEDICAL_HISTORY": "medical_history",
            "Sleep_Disorders": "sleep_disorders",
        }
        participant_info = participant_info.rename(columns=rename_map)
        if "patient_id" not in participant_info.columns:
            raise ValueError(
                "participant_info.csv must contain a SID/patient_id column."
            )

        file_64_dir = root / "data_64Hz"
        if not file_64_dir.exists():
            legacy_dir = root / "data"
            file_64_dir = legacy_dir if legacy_dir.exists() else file_64_dir
        file_100_dir = root / "data_100Hz"

        participant_info["patient_id"] = participant_info["patient_id"].astype(str)
        participant_info["file_64hz"] = participant_info["patient_id"].apply(
            lambda pid: cls._locate_subject_file(file_64_dir, pid)
        )
        participant_info["file_100hz"] = participant_info["patient_id"].apply(
            lambda pid: cls._locate_subject_file(file_100_dir, pid)
        )

        resolved = participant_info.apply(
            lambda row: cls._select_signal_file(
                row["file_64hz"],
                row["file_100hz"],
                preferred_source=preferred_source,
            ),
            axis=1,
            result_type="expand",
        )
        resolved.columns = ["signal_file", "signal_source", "sampling_rate_hz"]
        participant_info = pd.concat([participant_info, resolved], axis=1)

        for column in ["age", "bmi", "oahi", "ahi", "mean_sao2", "arousal_index"]:
            if column in participant_info.columns:
                participant_info[column] = participant_info[column].apply(
                    cls._coerce_float
                )

        for column in ["file_64hz", "file_100hz", "signal_file"]:
            participant_info[column] = participant_info[column].apply(
                lambda value: str(value) if isinstance(value, Path) else value
            )

        participant_info["signal_format"] = participant_info["signal_file"].apply(
            lambda value: Path(value).suffix.lower() if isinstance(value, str) else None
        )
        participant_info["record_id"] = participant_info["patient_id"]

        available_mask = participant_info["signal_file"].notna()
        missing_count = int((~available_mask).sum())
        if missing_count:
            logger.info(
                "Dropping %s DREAMT participants without local signal files.",
                missing_count,
            )
        participant_info = participant_info.loc[available_mask].reset_index(drop=True)

        return participant_info

    @classmethod
    def _build_metadata_from_processed_files(
        cls,
        root: Path,
        preferred_source: str,
    ) -> pd.DataFrame:
        del preferred_source
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and cls._is_signal_file(path)
        )
        if not files:
            raise FileNotFoundError(
                "No DREAMT subject files were found under the provided root."
            )

        rows = []
        for path in files:
            patient_id = cls._extract_subject_id(path)
            if patient_id is None:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "record_id": patient_id,
                    "signal_file": str(path.resolve()),
                    "signal_source": "processed",
                    "sampling_rate_hz": None,
                    "signal_format": path.suffix.lower(),
                    "file_64hz": None,
                    "file_100hz": None,
                }
            )

        metadata = pd.DataFrame(rows)
        metadata = metadata.drop_duplicates(subset=["patient_id"], keep="first")
        return metadata.sort_values("patient_id").reset_index(drop=True)

    @classmethod
    def prepare_metadata(cls, root: str | Path, preferred_source: str = "auto") -> None:
        """Create ``dreamt-metadata.csv`` for a local DREAMT directory.

        The generated metadata always includes:

        - ``patient_id``
        - ``record_id``
        - ``signal_file``
        - ``signal_source``
        - ``sampling_rate_hz``
        - ``signal_format``
        - ``file_64hz``
        - ``file_100hz``

        For the official DREAMT release, participant metadata columns are also
        preserved when available.
        """
        root = Path(root).expanduser().resolve()
        output_path = root / "dreamt-metadata.csv"

        if (root / "participant_info.csv").exists():
            metadata = cls._build_metadata_from_participant_info(
                root,
                preferred_source=preferred_source,
            )
        else:
            metadata = cls._build_metadata_from_processed_files(
                root,
                preferred_source=preferred_source,
            )

        if metadata.empty:
            raise ValueError("No DREAMT subjects were found while preparing metadata.")

        if metadata["signal_file"].notna().sum() == 0:
            raise ValueError(
                "Metadata was created, but no usable signal files were detected. "
                "Expected DREAMT subject files such as S002_whole_df.csv, "
                "S002_PSG_df.csv, or processed subject files like S002_record.npz."
            )

        metadata.to_csv(output_path, index=False)

    @property
    def default_task(self) -> SleepStagingDREAMT:
        """Return the default DREAMT sleep-staging task."""
        return SleepStagingDREAMT()
