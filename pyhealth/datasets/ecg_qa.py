import ast
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import dask.dataframe as dd
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ECGQADataset(BaseDataset):
    """ECG-QA base dataset for PTB-XL style question answering.

    Expected directory structure under `root`:

        root/
            answers.csv
            answers_for_each_template.csv
            train_ecgs.tsv
            valid_ecgs.tsv
            test_ecgs.tsv
            paraphrased/
                train/*.json
                valid/*.json
                test/*.json
            template/
                train/*.json
                valid/*.json
                test/*.json
    """

    VALID_SPLITS = {"train", "valid", "test"}
    VALID_SOURCES = {"paraphrased", "template"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        question_source: str = "paraphrased",
        question_types: Optional[Sequence[str]] = None,
        attribute_types: Optional[Sequence[str]] = None,
        single_ecg_only: bool = False,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}."
            )
        if question_source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid question_source '{question_source}'. "
                f"Expected one of {self.VALID_SOURCES}."
            )

        self.split = split
        self.question_source = question_source
        self.question_types = set(question_types) if question_types else None
        self.attribute_types = set(attribute_types) if attribute_types else None
        self.single_ecg_only = single_ecg_only

        super().__init__(
            root=root,
            tables=["qa"],
            dataset_name=dataset_name or "ecgqa_ptbxl",
            config_path=None,
            **kwargs,
        )

    def _split_file(self) -> Path:
        return Path(self.root) / f"{self.split}_ecgs.tsv"

    def _json_dir(self) -> Path:
        return Path(self.root) / self.question_source / self.split

    def _read_split_ecg_ids(self) -> set[str]:
        split_path = self._split_file()
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        split_df = pd.read_csv(
            split_path,
            sep="\t",
            header=None,
            names=["row_idx", "ecg_id"],
        )
        return set(split_df["ecg_id"].astype(str).tolist())

    def _read_answer_space(self) -> Dict[int, List[str]]:
        answer_space_path = Path(self.root) / "answers_for_each_template.csv"
        if not answer_space_path.exists():
            raise FileNotFoundError(
                f"Missing template answer file: {answer_space_path}"
            )

        df = pd.read_csv(answer_space_path)
        answer_space: Dict[int, List[str]] = {}
        for _, row in df.iterrows():
            template_id = int(row["template_id"])
            classes = ast.literal_eval(row["classes"])
            answer_space[template_id] = list(classes)
        return answer_space

    def _iter_json_records(self) -> Iterable[dict]:
        json_dir = self._json_dir()
        if not json_dir.exists():
            raise FileNotFoundError(f"Missing JSON shard directory: {json_dir}")

        for json_path in sorted(json_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    f"Expected list in {json_path}, got {type(data).__name__}"
                )

            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Expected dict item in {json_path}, got {type(item).__name__}"
                    )
                yield item

    def load_data(self) -> dd.DataFrame:
        allowed_ecg_ids = self._read_split_ecg_ids()
        answer_space = self._read_answer_space()

        rows: List[dict] = []
        for item in self._iter_json_records():
            question_type = item.get("question_type")
            attribute_type = item.get("attribute_type")
            ecg_ids = item.get("ecg_id", [])

            if self.question_types and question_type not in self.question_types:
                continue
            if self.attribute_types and attribute_type not in self.attribute_types:
                continue
            if self.single_ecg_only and len(ecg_ids) != 1:
                continue

            ecg_ids_as_str = [str(x) for x in ecg_ids]
            if not all(ecg_id in allowed_ecg_ids for ecg_id in ecg_ids_as_str):
                continue

            template_id = int(item["template_id"])
            answers = item.get("answer", [])
            attributes = item.get("attribute", None)

            patient_id = "__".join(ecg_ids_as_str)
            rows.append(
                {
                    "patient_id": patient_id,
                    "event_type": "qa",
                    "timestamp": pd.NaT,
                    "qa/template_id": template_id,
                    "qa/question_id": int(item["question_id"]),
                    "qa/sample_id": int(item["sample_id"]),
                    "qa/question_type": question_type,
                    "qa/attribute_type": attribute_type,
                    "qa/question": item.get("question", ""),
                    "qa/answer_text": " | ".join(answers),
                    "qa/answer_json": json.dumps(answers, ensure_ascii=False),
                    "qa/ecg_id_json": json.dumps(ecg_ids_as_str),
                    "qa/n_ecgs": len(ecg_ids_as_str),
                    "qa/attribute_json": json.dumps(attributes, ensure_ascii=False),
                    "qa/candidate_answers_json": json.dumps(
                        answer_space.get(template_id, []),
                        ensure_ascii=False,
                    ),
                    "qa/split": self.split,
                    "qa/question_source": self.question_source,
                }
            )

        if len(rows) == 0:
            raise ValueError(
                "No ECG-QA rows were loaded. Check your split/source filters."
            )

        df = pd.DataFrame(rows)
        npartitions = max(1, min(8, len(df) // 5000 + 1))
        return dd.from_pandas(df, npartitions=npartitions)