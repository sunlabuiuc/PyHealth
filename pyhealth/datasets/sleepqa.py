from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class SleepQADataset(BaseDataset):
    """SleepQA dataset wrapper for PyHealth.

    This class prepares and exposes the SleepQA dataset for use with PyHealth.
    It converts the original JSON / JSONL question–answer files into a
    single flat CSV table (``sleepqa-pyhealth.csv``) and then delegates to
    ``BaseDataset`` for standard PyHealth handling.

    The original SleepQA dataset is described in:

        Bojic, I., Ong, Q. C., Thakkar, M., Kamran, E., Shua, I. Y. L.,
        Pang, J. R. E., Chen, J., Nayak, V., Joty, S., & Car, J. (2022).
        "SleepQA: A Health Coaching Dataset on Sleep for Extractive
        Question Answering."

    Expected raw data layout
    ------------------------

    The dataset is typically released with training / dev files such as:

        sleep-train.json
        sleep-dev.json
        sleep-test.json  (optional in some releases)

    Each record is expected to contain at least:

        {
            "id": "...",
            "question": "...",
            "context": "...",   # or "contents" in some variants
            "answer": "..."     # or {"text": "...", "start": int}
        }

    This wrapper is robust to minor schema differences:

        - Uses "context" if present, otherwise falls back to "contents".
        - Accepts "answer" as either a string or a dict with "text" and
          optional "start" fields.

    Generated metadata
    ------------------

    During initialization, if ``sleepqa-pyhealth.csv`` does not yet exist
    under ``root``, the dataset will scan for known SleepQA JSON/JSONL
    files (train, dev, test), normalize them, and write the combined
    metadata table with the following columns:

        - sample_id: unique integer index
        - split: "train", "dev", or "test"
        - qa_id: original question/passage id from the source file
        - question: question text
        - context: passage text
        - answer_text: gold answer span text (string, may be empty)
        - answer_start: integer character offset if available, else -1

    Args:
        root: Root directory of the raw SleepQA data. This should be a
            directory containing the SleepQA JSON/JSONL files directly or
            subdirectories such as "data/training/".
        dataset_name: Optional custom dataset name. Defaults to "sleepqa".
        config_path: Optional path to the dataset config YAML. If None, a
            default "sleepqa.yaml" in the same directory as this file is
            used.

    Example:
        >>> from pyhealth.datasets import SleepQADataset
        >>> dataset = SleepQADataset(root="/path/to/SleepQA/data/training")
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        # Default config path if not provided
        if config_path is None:
            logger.info("No config path provided, using default SleepQA config.")
            config_path = os.path.join(
                os.path.dirname(__file__),
                "configs",
                "sleepqa.yaml",
            )

        metadata_path = os.path.join(root, "sleepqa-pyhealth.csv")

        # Prepare metadata CSV if it does not exist yet
        if not os.path.exists(metadata_path):
            logger.info(
                "SleepQA metadata not found at '%s'; preparing metadata.",
                metadata_path,
            )
            self.prepare_metadata(root, metadata_path)

        super().__init__(
            root=root,
            tables=["sleepqa"],
            dataset_name=dataset_name or "sleepqa",
            config_path=config_path,
        )

    # -------------------------------------------------------------------------
    # Metadata preparation
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
        """Load records from a JSON or JSONL file.

        This helper supports two common formats:

        - JSON lines: one JSON object per line.
        - JSON array: a single list of objects.

        Args:
            path: File path.

        Returns:
            A list of dict records.
        """
        records: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            first_chars = f.read(1024)
            f.seek(0)

            # Simple heuristic: if file starts with '[' assume JSON array
            if first_chars.lstrip().startswith("["):
                data = json.load(f)
                if isinstance(data, list):
                    records = [r for r in data if isinstance(r, dict)]
                else:
                    raise ValueError(f"Unexpected JSON structure in '{path}'.")
            else:
                # JSONL case
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)

        return records

    @staticmethod
    def _candidate_paths(root: str) -> List[Tuple[str, str]]:
        """Return plausible (split, path) pairs for SleepQA files under root.

        This is intentionally permissive so the dataset can be initialized
        from slightly different directory layouts.

        Args:
            root: Root directory given by the user.

        Returns:
            List of (split, path) tuples.
        """
        candidates: List[Tuple[str, str]] = []

        # Directly in root
        for split, fname in [
            ("train", "sleep-train.json"),
            ("dev", "sleep-dev.json"),
            ("test", "sleep-test.json"),
        ]:
            candidates.append((split, os.path.join(root, fname)))

        # Common nested layout: root/data/training or root/training
        for split, fname in [
            ("train", "sleep-train.json"),
            ("dev", "sleep-dev.json"),
            ("test", "sleep-test.json"),
        ]:
            candidates.append(
                (split, os.path.join(root, "training", fname))
            )
            candidates.append(
                (split, os.path.join(root, "data", "training", fname))
            )

        # Also allow .jsonl naming
        extra: List[Tuple[str, str]] = []
        for split, path in candidates:
            if path.endswith(".json"):
                extra.append((split, path[:-5] + ".jsonl"))
        candidates.extend(extra)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[Tuple[str, str]] = []
        for split, path in candidates:
            key = f"{split}::{path}"
            if key not in seen:
                seen.add(key)
                unique.append((split, path))

        return unique

    @classmethod
    def prepare_metadata(cls, root: str, metadata_path: str) -> None:
        """Prepare the unified SleepQA metadata CSV.

        This scans for plausible SleepQA JSON/JSONL files, loads those that
        exist, normalizes each record, and writes a combined CSV.

        Args:
            root: Root directory containing the raw SleepQA files.
            metadata_path: Output CSV path (e.g., "sleepqa-pyhealth.csv").
        """
        candidates = cls._candidate_paths(root)

        all_rows: List[Dict[str, Any]] = []
        sample_id = 0

        for split, path in candidates:
            if not os.path.exists(path):
                continue

            logger.info("Loading SleepQA %s split from '%s'.", split, path)
            records = cls._load_json_or_jsonl(path)

            for rec in records:
                qa_id = str(rec.get("id", sample_id))

                # Question text
                question = rec.get("question", "")

                # Context / passage text
                context = rec.get("context") or rec.get("contents") or ""

                # Answer text + start index (if present)
                answer_val = rec.get("answer")
                answer_text = ""
                answer_start = -1

                if isinstance(answer_val, str):
                    answer_text = answer_val
                elif isinstance(answer_val, dict):
                    answer_text = str(answer_val.get("text", ""))
                    start_val = answer_val.get("start")
                    if isinstance(start_val, int):
                        answer_start = start_val

                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "qa_id": qa_id,
                    "question": question,
                    "context": context,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                }
                all_rows.append(row)
                sample_id += 1

        if not all_rows:
            raise FileNotFoundError(
                f"No SleepQA JSON/JSONL files were found under root='{root}'. "
                "Please ensure sleep-train.json / sleep-dev.json (or .jsonl) "
                "are present."
            )

        df = pd.DataFrame(all_rows)
        df.to_csv(metadata_path, index=False)
        logger.info("SleepQA metadata saved to '%s' (%d rows).", metadata_path, len(df))

    # -------------------------------------------------------------------------
    # Optional default task hook
    # -------------------------------------------------------------------------

    @property
    def default_task(self):
        """Default task placeholder for SleepQA.

        PyHealth allows datasets to expose a ``default_task`` property to
        indicate a recommended task definition. Since SleepQA is an extractive
        question answering dataset and PyHealth does not (yet) ship with a
        built-in SleepQA task, this property currently returns ``None``.

        Users are expected to define their own task (e.g., retrieval or
        QA modeling) on top of the SleepQADataset samples.

        Returns:
            None
        """
        return None
