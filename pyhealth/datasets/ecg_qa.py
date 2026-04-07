import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ECGQADataset(BaseDataset):
    """ECG Question Answering dataset.

    This dataset provides natural language question-answer pairs linked to
    ECG recordings via ecg_id. It is an annotation layer on top of ECG
    recordings from PTB-XL or MIMIC-IV-ECG.

    The QA data originates from the ECG-QA dataset (Oh et al., 2024),
    restructured for few-shot learning by Tang et al. (CHIL 2025).

    Dataset is available at https://github.com/Tang-Jia-Lu/FSL_ECG_QA

    Three question types are supported:
        - single-verify: yes/no questions about ECG findings
        - single-choose: multi-choice questions (answer is one option, "both", or "none")
        - single-query: open-ended questions with free-form answers

    Args:
        root: path to the paraphrased QA directory containing train/, valid/,
            test/ subdirectories with JSON files. Works with both PTB-XL
            (ecgqa/ptbxl/paraphrased/) and MIMIC-IV-ECG
            (ecgqa/mimic-iv-ecg/paraphrased/) data.
        dataset_name: name of the dataset. Default is "ecg_qa".
        config_path: path to the YAML config file. Default uses built-in config.

    Examples:
        >>> from pyhealth.datasets import ECGQADataset
        >>> dataset = ECGQADataset(
        ...     root="/path/to/ecgqa/ptbxl/paraphrased/",
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ecg_qa.yaml"

        self.root = root

        self.prepare_metadata()

        # Check if CSV is in cache rather than root
        root_path = Path(root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "ecg_qa"
        csv_name = "ecg-qa-pyhealth.csv"

        use_cache = False
        if not (root_path / csv_name).exists() and (cache_dir / csv_name).exists():
            use_cache = True

        if use_cache:
            logger.info(f"Using cached metadata from {cache_dir}")
            root = str(cache_dir)

        super().__init__(
            root=root,
            tables=["ecg_qa"],
            dataset_name=dataset_name or "ecg_qa",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Build and save a metadata CSV from all ECG-QA JSON files.

        Scans train/, valid/, test/ subdirectories under root, loads all
        JSON files, filters to single-* question types, and writes a
        single CSV with columns:
            patient_id, ecg_id, question, answer, question_type,
            attribute_type, template_id, question_id, sample_id, attribute
        """
        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "ecg_qa"
        csv_name = "ecg-qa-pyhealth.csv"

        shared_csv = root / csv_name
        cache_csv = cache_dir / csv_name
        if shared_csv.exists() or cache_csv.exists():
            return

        # Load all JSON files from all split directories
        data = []
        for split_dir in ("train", "valid", "test"):
            json_dir = root / split_dir
            if not json_dir.is_dir():
                logger.warning("JSON directory not found: %s", json_dir)
                continue
            for fpath in sorted(json_dir.glob("*.json")):
                with open(fpath, "r") as f:
                    data.extend(json.load(f))

        if not data:
            raise FileNotFoundError(
                f"No JSON files found in train/valid/test subdirectories of {root}"
            )

        # Filter to single-* question types and build rows
        rows: list[dict] = []
        for record in data:
            qt = record.get("question_type", "")
            if not qt.startswith("single-"):
                continue

            ecg_id = record["ecg_id"][0]
            answer = ";".join(record["answer"])
            attribute = ";".join(record.get("attribute", []))

            rows.append({
                "patient_id": str(ecg_id),
                "ecg_id": ecg_id,
                "question": record["question"],
                "answer": answer,
                "question_type": qt,
                "attribute_type": record.get("attribute_type", ""),
                "template_id": record.get("template_id", 0),
                "question_id": record.get("question_id", 0),
                "sample_id": record.get("sample_id", 0),
                "attribute": attribute,
            })

        if not rows:
            raise ValueError("No single-* question type records found in JSON data")

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id", "question_type", "template_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Try shared location first, fall back to cache
        try:
            shared_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(shared_csv, index=False)
            logger.info(f"Wrote metadata to {shared_csv}")
        except (PermissionError, OSError):
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_csv, index=False)
            logger.info(f"Wrote metadata to cache: {cache_csv}")

    @property
    def default_task(self):
        """Returns the default task for the ECG-QA dataset: ECGQA."""
        from pyhealth.tasks import ECGQA
        return ECGQA()
