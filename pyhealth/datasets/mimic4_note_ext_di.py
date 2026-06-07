"""MIMIC-IV-Note-Ext-DI dataset for patient summary generation.

This module provides a PyHealth dataset class for the processed discharge
instruction datasets derived from MIMIC-IV-Note, as described in:

    Hegselmann, S., et al. (2024). A Data-Centric Approach To Generate
    Faithful and High Quality Patient Summaries with Large Language Models.
    Proceedings of Machine Learning Research, 248, 339-379.

The dataset maps Brief Hospital Course (BHC) clinical text to patient-facing
Discharge Instructions (DI), supporting research on faithful clinical text
summarization and hallucination reduction.

Data is available on PhysioNet (credentialed access required):
    https://doi.org/10.13026/m6hf-dq94
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ..tasks import PatientSummaryGeneration
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# Mapping from variant name to the JSONL file path relative to root.
_VARIANT_FILE_MAP = {
    # Main BHC-context datasets (mimic-iv-note-ext-di-bhc/dataset/)
    "bhc_all": "mimic-iv-note-ext-di-bhc/dataset/all.json",
    "bhc_train": "mimic-iv-note-ext-di-bhc/dataset/train.json",
    "bhc_valid": "mimic-iv-note-ext-di-bhc/dataset/valid.json",
    "bhc_test": "mimic-iv-note-ext-di-bhc/dataset/test.json",
    "bhc_train_100": "mimic-iv-note-ext-di-bhc/dataset/train_100.json",
    # Full-context datasets (mimic-iv-note-ext-di/dataset/)
    "full_all": "mimic-iv-note-ext-di/dataset/all.json",
    "full_train": "mimic-iv-note-ext-di/dataset/train.json",
    "full_valid": "mimic-iv-note-ext-di/dataset/valid.json",
    "full_test": "mimic-iv-note-ext-di/dataset/test.json",
    # Derived hallucination-reduction datasets (derived_datasets/)
    "original": "derived_datasets/hallucinations_mimic_di_original.json",
    "cleaned": "derived_datasets/hallucinations_mimic_di_cleaned.json",
    "cleaned_improved": (
        "derived_datasets/hallucinations_mimic_di_cleaned_improved.json"
    ),
    "original_validation": (
        "derived_datasets/hallucinations_mimic_di_validation_original.json"
    ),
    "cleaned_validation": (
        "derived_datasets/hallucinations_mimic_di_validation_cleaned.json"
    ),
    "cleaned_improved_validation": (
        "derived_datasets/"
        "hallucinations_mimic_di_validation_cleaned_improved.json"
    ),
}


def _jsonl_to_csv(jsonl_path: str, csv_path: str) -> str:
    """Convert a JSONL file with 'text' and 'summary' fields to CSV.

    Args:
        jsonl_path: Path to the source JSONL file.
        csv_path: Path where the CSV file will be written.

    Returns:
        The path to the created CSV file.
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records.append(
                    {"text": obj["text"], "summary": obj["summary"]}
                )
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    logger.info(
        f"Converted {len(records)} records from {jsonl_path} to {csv_path}"
    )
    return csv_path


class MimicIVNoteExtDIDataset(BaseDataset):
    """Processed discharge instruction dataset from MIMIC-IV-Note.

    This dataset provides context-summary pairs for patient summary
    generation. The context is clinical text (Brief Hospital Course or
    full discharge note) and the target is patient-facing Discharge
    Instructions written in layperson language.

    The dataset supports multiple variants corresponding to different
    subsets released by Hegselmann et al. (2024):

    **Main BHC-context datasets** (100,175 examples total):
        - ``bhc_all``: All 100,175 context-summary pairs
        - ``bhc_train``: Training split (80,140 examples)
        - ``bhc_valid``: Validation split (10,017 examples)
        - ``bhc_test``: Test split (10,018 examples)
        - ``bhc_train_100``: 100-example training subset

    **Full-context datasets** (all notes before DI as context):
        - ``full_all``, ``full_train``, ``full_valid``, ``full_test``

    **Derived datasets for hallucination-reduction experiments** (100 each):
        - ``original``: Doctor-written summaries with hallucinations
        - ``cleaned``: Summaries with hallucinations removed
        - ``cleaned_improved``: Cleaned and further improved summaries

    Args:
        root: Root directory of the PhysioNet data release. Should contain
            subdirectories ``mimic-iv-note-ext-di-bhc/``,
            ``mimic-iv-note-ext-di/``, and ``derived_datasets/``.
        variant: Which dataset variant to load. See class docstring for
            available variants. Defaults to ``"bhc_train"``.
        dataset_name: Name of the dataset. Defaults to
            ``"mimic4_note_ext_di"``.
        config_path: Path to the YAML configuration file. If None, uses
            the default config bundled with PyHealth.
        cache_dir: Directory for caching processed data.
        num_workers: Number of workers for parallel processing.
        dev: If True, limits to 1000 patients for development.

    Examples:
        >>> from pyhealth.datasets import MimicIVNoteExtDIDataset
        >>> dataset = MimicIVNoteExtDIDataset(
        ...     root="/path/to/physionet/data",
        ...     variant="bhc_train",
        ... )
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        variant: str = "bhc_train",
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir=None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if variant not in _VARIANT_FILE_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available variants: {sorted(_VARIANT_FILE_MAP.keys())}"
            )

        self.variant = variant
        jsonl_relpath = _VARIANT_FILE_MAP[variant]
        jsonl_path = os.path.join(root, jsonl_relpath)

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Expected JSONL file not found: {jsonl_path}. "
                f"Ensure 'root' points to the PhysioNet data release "
                f"directory."
            )

        # Convert JSONL to CSV in a subdirectory next to the JSONL file.
        # This allows the base class CSV loader to find the file.
        csv_dir = os.path.join(
            root, "_pyhealth_csv", variant.replace("/", "_")
        )
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "summaries.csv")

        if not os.path.exists(csv_path):
            logger.info(
                f"Converting JSONL to CSV for variant '{variant}'..."
            )
            _jsonl_to_csv(jsonl_path, csv_path)
        else:
            logger.info(
                f"Using cached CSV for variant '{variant}': {csv_path}"
            )

        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "mimic4_note_ext_di.yaml"
            )

        default_tables = ["summaries"]
        super().__init__(
            root=csv_dir,
            tables=default_tables,
            dataset_name=dataset_name or "mimic4_note_ext_di",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @property
    def default_task(self) -> PatientSummaryGeneration:
        """Returns the default task for this dataset."""
        return PatientSummaryGeneration()
