"""
PyHealth dataset for the ICBHI 2017 Respiratory Sound Database.

Dataset link:
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

Dataset paper: (please cite if you use this dataset)
    Rocha, Bruno M., et al. "An open access database for the evaluation of
    respiratory sound classification algorithms." Physiological Measurement
    40.3 (2019): 035001.

Paper link:
    https://doi.org/10.1088/1361-6579/ab03ea

Author:
    Andrew Zhao (andrew.zhao@aeroseal.com)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


# Human-readable expansions for filename codes. These are only used to build
# ``metadata_text``; the raw codes themselves are preserved in dedicated
# columns so nothing is invented or lost.
_CHEST_LOCATION_MAP: Dict[str, str] = {
    "Tc": "trachea",
    "Al": "anterior left",
    "Ar": "anterior right",
    "Pl": "posterior left",
    "Pr": "posterior right",
    "Ll": "lateral left",
    "Lr": "lateral right",
}

_ACQUISITION_MODE_MAP: Dict[str, str] = {
    "sc": "single-channel",
    "mc": "multi-channel",
}


class ICBHIDataset(BaseDataset):
    """Dataset class for the ICBHI 2017 Respiratory Sound Database.

    The dataset contains 920 recordings from 126 patients collected at two
    clinical sites (Portugal and Greece) using different acquisition
    equipment. Each recording is paired with a cycle-level annotation file
    that marks respiratory cycle boundaries and the presence of crackles
    and/or wheezes.

    Recordings are split into an official train set (~60%) and test set
    (~40%) defined in the ``ICBHI_challenge_train_and_test_txt`` directory
    that ships with the dataset download.

    The dataset is freely available after accepting the data-use terms at
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge.

    **Schema.** One event is emitted per respiratory cycle (not per
    recording). This preserves crackle / wheeze supervision in the event
    stream so downstream tasks can define arbitrary abnormality labels
    without re-parsing the raw ``.txt`` annotations. Every event carries:

    - ``patient_id``, ``recording_id``, ``cycle_id``
    - ``audio_path`` — absolute path to the source WAV
    - ``cycle_start``, ``cycle_end``, ``duration`` — seconds
    - ``has_crackles``, ``has_wheezes`` — 0 / 1
    - ``diagnosis`` — per-patient diagnosis string
    - ``chest_location``, ``acquisition_mode``, ``equipment``
    - ``age``, ``sex``, ``adult_bmi``, ``child_weight``, ``child_height``
      — demographics if ``ICBHI_Challenge_demographic_information.txt`` is
      available, otherwise ``NaN`` / empty string (never fabricated)
    - ``metadata_text`` — human-readable concatenation of the above;
      missing fields are omitted rather than filled with placeholders

    **Expected directory layout after download:**

    .. code-block:: text

        <root>/
        ├── 101_1u_Al_sc_Meditron.wav
        ├── 101_1u_Al_sc_Meditron.txt    # cycle annotations
        ├── ...
        ├── ICBHI_challenge_diagnosis.txt
        ├── ICBHI_Challenge_demographic_information.txt  (optional)
        └── ICBHI_challenge_train_and_test_txt/
            ├── ICBHI_challenge_train.txt
            └── ICBHI_challenge_test.txt

    Filename format:
        ``{patient_id}_{rec_index}_{chest_location}_{mode}_{equipment}``

    Cycle annotation format (tab-separated, one line per cycle):
        ``begin_time  end_time  crackle(0/1)  wheeze(0/1)``

    Args:
        root: Root directory of the raw ICBHI data.
        dataset_name: Optional override for the dataset name.
        config_path: Path to YAML config. Defaults to the bundled
            ``configs/icbhi.yaml``.
        subset: Which split(s) to load — ``"train"``, ``"test"``, or
            ``"both"``. Default ``"both"``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``dev``, ``cache_dir``, ``num_workers``).

    Attributes:
        task: Optional task name set by ``set_task()``.
        samples: Processed sample list after ``set_task()``.
        patient_to_index: Mapping from patient_id to sample indices.
        visit_to_index: Mapping from visit / record id to sample indices.

    Examples:
        >>> from pyhealth.datasets import ICBHIDataset
        >>> dataset = ICBHIDataset(
        ...     root="/data/ICBHI_final_database",
        ...     subset="train",
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subset: str = "both",
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = str(Path(__file__).parent / "configs" / "icbhi.yaml")

        self.root = root

        if subset == "train":
            tables = ["train"]
        elif subset == "test":
            tables = ["test"]
        elif subset == "both":
            tables = ["train", "test"]
        else:
            raise ValueError("subset must be one of 'train', 'test', or 'both'")

        self.prepare_metadata()

        root_path = Path(root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "icbhi"

        use_cache = False
        for table in tables:
            shared_csv = root_path / f"icbhi-{table}-pyhealth.csv"
            cache_csv = cache_dir / f"icbhi-{table}-pyhealth.csv"
            if not shared_csv.exists() and cache_csv.exists():
                use_cache = True
                break

        if use_cache:
            logger.info("Using cached metadata from %s", cache_dir)
            root = str(cache_dir)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "icbhi",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def _load_diagnosis_map(root: Path) -> Dict[str, str]:
        """Load ``patient_id -> diagnosis`` from ICBHI_challenge_diagnosis.txt.

        Returns an empty dict if the file is absent. Entries use the raw
        diagnosis strings from the release (e.g. ``Healthy``, ``URTI``,
        ``Pneumonia``).
        """
        diag_file = root / "ICBHI_challenge_diagnosis.txt"
        diagnosis_map: Dict[str, str] = {}
        if not diag_file.exists():
            return diagnosis_map
        for line in diag_file.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                diagnosis_map[parts[0]] = parts[1]
        return diagnosis_map

    @staticmethod
    def _load_demographic_map(root: Path) -> Dict[str, Dict[str, object]]:
        """Load demographics keyed by patient_id.

        Returns an empty dict if the file is absent. Real ICBHI uses ``NA``
        for missing values; those are converted to ``float('nan')`` /
        empty string so downstream code does not have to special-case
        literal ``"NA"`` strings.
        """
        demo_file = root / "ICBHI_Challenge_demographic_information.txt"
        demographic_map: Dict[str, Dict[str, object]] = {}
        if not demo_file.exists():
            return demographic_map

        def _to_float(tok: str) -> float:
            if tok == "" or tok.upper() == "NA":
                return float("nan")
            try:
                return float(tok)
            except ValueError:
                return float("nan")

        for line in demo_file.read_text().strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            pid = parts[0]
            age = _to_float(parts[1]) if len(parts) > 1 else float("nan")
            sex_raw = parts[2] if len(parts) > 2 else ""
            sex = "" if sex_raw.upper() == "NA" else sex_raw
            adult_bmi = _to_float(parts[3]) if len(parts) > 3 else float("nan")
            child_weight = _to_float(parts[4]) if len(parts) > 4 else float("nan")
            child_height = _to_float(parts[5]) if len(parts) > 5 else float("nan")
            demographic_map[pid] = {
                "age": age,
                "sex": sex,
                "adult_bmi": adult_bmi,
                "child_weight": child_weight,
                "child_height": child_height,
            }
        return demographic_map

    @staticmethod
    def _build_metadata_text(
        patient_id: str,
        diagnosis: str,
        chest_location: str,
        acquisition_mode: str,
        equipment: str,
        demographics: Dict[str, object],
    ) -> str:
        """Construct a human-readable description from real ICBHI fields.

        Missing fields are omitted. No placeholder text is invented when
        values are absent.
        """
        parts: List[str] = [f"Patient {patient_id}"]
        if diagnosis and diagnosis != "Unknown":
            parts.append(f"diagnosis: {diagnosis}")

        location_pretty = _CHEST_LOCATION_MAP.get(chest_location, chest_location)
        if location_pretty:
            parts.append(f"chest location: {location_pretty}")

        mode_pretty = _ACQUISITION_MODE_MAP.get(acquisition_mode, acquisition_mode)
        if mode_pretty and equipment:
            parts.append(f"acquisition: {mode_pretty} with {equipment}")
        elif mode_pretty:
            parts.append(f"acquisition: {mode_pretty}")
        elif equipment:
            parts.append(f"equipment: {equipment}")

        age = demographics.get("age")
        if isinstance(age, float) and age == age:  # not NaN
            parts.append(f"age: {age:g}")
        sex = demographics.get("sex", "")
        if sex:
            parts.append(f"sex: {sex}")

        return "; ".join(parts) + "."

    def prepare_metadata(self) -> None:
        """Build and save cycle-level metadata CSVs from raw ICBHI files.

        Scans ``<root>/*.wav`` plus the paired ``.txt`` annotation, joins
        per-patient diagnosis and (optionally) demographic information,
        and writes one row per respiratory cycle to:

        - ``<root>/icbhi-train-pyhealth.csv``
        - ``<root>/icbhi-test-pyhealth.csv``

        If a CSV already exists (in ``<root>`` or the user cache), that
        split is skipped so repeat instantiations are fast. Falls back to
        writing CSVs under ``~/.cache/pyhealth/icbhi/`` when the data
        directory is read-only.
        """
        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "icbhi"

        diagnosis_map = self._load_diagnosis_map(root)
        demographic_map = self._load_demographic_map(root)

        for split in ("train", "test"):
            shared_csv = root / f"icbhi-{split}-pyhealth.csv"
            cache_csv = cache_dir / f"icbhi-{split}-pyhealth.csv"
            if shared_csv.exists() or cache_csv.exists():
                continue

            split_file = (
                root
                / "ICBHI_challenge_train_and_test_txt"
                / f"ICBHI_challenge_{split}.txt"
            )
            if not split_file.exists():
                logger.warning("Split file not found: %s — skipping", split_file)
                continue

            split_stems = {
                line.strip()
                for line in split_file.read_text().strip().splitlines()
                if line.strip()
            }

            rows: List[dict] = []
            for wav_path in sorted(root.glob("*.wav")):
                stem = wav_path.stem
                if stem not in split_stems:
                    continue

                parts = stem.split("_")
                if len(parts) < 5:
                    logger.warning(
                        "Unexpected filename format: %s — skipping", wav_path
                    )
                    continue

                patient_id = parts[0]
                chest_location = parts[2]
                acquisition_mode = parts[3]
                equipment = parts[4]
                diagnosis = diagnosis_map.get(patient_id, "Unknown")
                demographics = demographic_map.get(
                    patient_id,
                    {
                        "age": float("nan"),
                        "sex": "",
                        "adult_bmi": float("nan"),
                        "child_weight": float("nan"),
                        "child_height": float("nan"),
                    },
                )

                ann_file = wav_path.with_suffix(".txt")
                if not ann_file.exists():
                    logger.warning(
                        "Annotation file missing for %s — skipping", wav_path
                    )
                    continue

                metadata_text = self._build_metadata_text(
                    patient_id=patient_id,
                    diagnosis=diagnosis,
                    chest_location=chest_location,
                    acquisition_mode=acquisition_mode,
                    equipment=equipment,
                    demographics=demographics,
                )

                for cycle_id, line in enumerate(
                    ann_file.read_text().strip().splitlines()
                ):
                    ann_parts = line.split()
                    if len(ann_parts) < 4:
                        continue
                    try:
                        cycle_start = float(ann_parts[0])
                        cycle_end = float(ann_parts[1])
                        has_crackles = int(ann_parts[2])
                        has_wheezes = int(ann_parts[3])
                    except ValueError:
                        logger.warning(
                            "Bad annotation line in %s: %r — skipping",
                            ann_file,
                            line,
                        )
                        continue

                    rows.append(
                        {
                            "patient_id": patient_id,
                            "recording_id": stem,
                            "cycle_id": cycle_id,
                            "audio_path": str(wav_path),
                            "cycle_start": cycle_start,
                            "cycle_end": cycle_end,
                            "duration": cycle_end - cycle_start,
                            "has_crackles": has_crackles,
                            "has_wheezes": has_wheezes,
                            "diagnosis": diagnosis,
                            "chest_location": chest_location,
                            "acquisition_mode": acquisition_mode,
                            "equipment": equipment,
                            "age": demographics["age"],
                            "sex": demographics["sex"],
                            "adult_bmi": demographics["adult_bmi"],
                            "child_weight": demographics["child_weight"],
                            "child_height": demographics["child_height"],
                            "metadata_text": metadata_text,
                        }
                    )

            if not rows:
                logger.warning("No %s cycles found in %s", split, root)
                continue

            df = pd.DataFrame(rows)
            df.sort_values(
                ["patient_id", "recording_id", "cycle_id"],
                inplace=True,
                na_position="last",
            )
            df.reset_index(drop=True, inplace=True)

            try:
                df.to_csv(shared_csv, index=False)
                logger.info("Wrote %s metadata to %s", split, shared_csv)
            except (PermissionError, OSError):
                cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_csv, index=False)
                logger.info("Wrote %s metadata to cache: %s", split, cache_csv)

    @property
    def default_task(self):
        """Return the default task"""
        from pyhealth.tasks import RespiratoryAbnormalityPredictionICBHI

        return RespiratoryAbnormalityPredictionICBHI()
