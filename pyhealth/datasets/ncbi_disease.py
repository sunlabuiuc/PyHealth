"""NCBI Disease corpus dataset for PyHealth.

This module provides a PyHealth dataset wrapper for the NCBI Disease corpus,
which is distributed in PubTator-compatible text files.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class NCBIDiseaseDataset(BaseDataset):
    """NCBI Disease corpus for disease mention recognition/normalization.

    The dataset contains PubMed titles and abstracts annotated with disease
    mention spans and concept identifiers. Each document is represented as one
    PyHealth patient/event pair so downstream tasks can operate on document-level
    text.

    Official data page:
    https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/

    Supported raw inputs placed under ``root``:
    - Extracted text files:
      ``NCBItrainset_corpus.txt``, ``NCBIdevelopset_corpus.txt``,
      ``NCBItestset_corpus.txt``
    - Official zip files with the same stems and ``.zip`` suffix

    Args:
        root: Directory containing the official corpus files.
        tables: Optional list of extra tables. The dataset uses ``documents`` by
            default.
        dataset_name: Name of the dataset. Defaults to ``"ncbi_disease"``.
        config_path: Optional custom YAML config path.

    Examples:
        >>> from pyhealth.datasets import NCBIDiseaseDataset
        >>> dataset = NCBIDiseaseDataset(root="/path/to/ncbi_disease")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0]["entities"][0]["text"])
    """

    RAW_FILES: Dict[str, str] = {
        "train": "NCBItrainset_corpus.txt",
        "dev": "NCBIdevelopset_corpus.txt",
        "test": "NCBItestset_corpus.txt",
    }

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ncbi_disease.yaml"

        standardized_csv = Path(root) / "ncbi_disease_pyhealth.csv"
        if not standardized_csv.exists():
            logger.info("Preparing NCBI Disease metadata...")
            self.prepare_metadata(root)

        default_tables = ["documents"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "ncbi_disease",
            config_path=str(config_path),
            **kwargs,
        )

    @classmethod
    def _find_raw_path(cls, root: Path, raw_filename: str) -> Optional[Path]:
        direct_path = root / raw_filename
        if direct_path.exists():
            return direct_path

        for candidate in root.rglob(raw_filename):
            if candidate.is_file():
                return candidate

        zip_name = raw_filename.replace(".txt", ".zip")
        direct_zip = root / zip_name
        if direct_zip.exists():
            return direct_zip

        for candidate in root.rglob(zip_name):
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _read_raw_payload(path: Path, expected_member: str) -> str:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                member = next(
                    (
                        name
                        for name in members
                        if Path(name).name == expected_member
                    ),
                    None,
                )
                if member is None:
                    raise FileNotFoundError(
                        f"Could not find {expected_member} inside archive {path}"
                    )
                with archive.open(member) as handle:
                    return handle.read().decode("utf-8")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_document_block(
        block: str, split: str, block_index: int
    ) -> Dict[str, str]:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError(
                f"Malformed NCBI Disease block at split={split}, index={block_index}"
            )

        title_parts = lines[0].split("|", 2)
        abstract_parts = lines[1].split("|", 2)
        if len(title_parts) != 3 or len(abstract_parts) != 3:
            raise ValueError(
                "Expected title/abstract lines in PubTator format "
                f"for split={split}, index={block_index}"
            )

        doc_id = title_parts[0].strip()
        title = title_parts[2].strip()
        abstract = abstract_parts[2].strip()
        text = title if not abstract else f"{title} {abstract}"

        mentions: List[Dict[str, object]] = []
        for line in lines[2:]:
            parts = line.split("\t")
            if len(parts) < 6:
                logger.warning(
                    "Skipping malformed annotation line for split=%s, index=%s: %s",
                    split,
                    block_index,
                    line,
                )
                continue

            mention_doc_id, start, end, mention_text, mention_type, concept_id = (
                parts[:6]
            )
            if mention_doc_id.strip() != doc_id:
                logger.warning(
                    "Mismatched annotation PMID %s for document %s; keeping line",
                    mention_doc_id,
                    doc_id,
                )

            mentions.append(
                {
                    "text": mention_text,
                    "type": mention_type,
                    "concept_id": concept_id,
                    "start": int(start),
                    "end": int(end),
                }
            )

        record_id = f"{split}_{block_index}_{doc_id}"
        return {
            "record_id": record_id,
            "doc_id": doc_id,
            "split": split,
            "title": title,
            "abstract": abstract,
            "text": text,
            "mentions_json": json.dumps(mentions, sort_keys=True),
        }

    @classmethod
    def _iter_records_for_split(
        cls, payload: str, split: str
    ) -> Iterable[Dict[str, str]]:
        blocks = [block for block in payload.split("\n\n") if block.strip()]
        for block_index, block in enumerate(blocks):
            yield cls._parse_document_block(block, split=split, block_index=block_index)

    @classmethod
    def prepare_metadata(cls, root: str) -> None:
        """Convert official raw files into a PyHealth-friendly CSV.

        Args:
            root: Dataset root containing extracted ``.txt`` files or official
                ``.zip`` downloads.
        """

        root_path = Path(root)
        records: List[Dict[str, str]] = []
        missing_splits: List[Tuple[str, str]] = []

        for split, raw_filename in cls.RAW_FILES.items():
            raw_path = cls._find_raw_path(root_path, raw_filename)
            if raw_path is None:
                missing_splits.append((split, raw_filename))
                continue

            payload = cls._read_raw_payload(raw_path, expected_member=raw_filename)
            records.extend(cls._iter_records_for_split(payload, split=split))

        if not records:
            missing_names = ", ".join(name for _, name in missing_splits)
            raise FileNotFoundError(
                "No NCBI Disease raw files were found. Place any of "
                f"{missing_names} or the corresponding official zip files under {root}."
            )

        if missing_splits:
            logger.warning(
                "Missing NCBI Disease splits: %s",
                ", ".join(split for split, _ in missing_splits),
            )

        df = pd.DataFrame.from_records(records)
        output_path = root_path / "ncbi_disease_pyhealth.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved %s documents to %s", len(df), output_path)

    @property
    def default_task(self):
        """Returns the default task for this dataset."""
        from pyhealth.tasks import NCBIDiseaseRecognition

        return NCBIDiseaseRecognition()
