"""
Author(s): Ezra Hieh, Jeffrey Lee, Venkatesh Venkataramanan
NetID(s): iezralh2, jl187, vv33
Paper: SleepQA: Question Answering for Sleep and Health Knowledge
Paper Link: https://github.com/IvaBojic/SleepQA
Description: SleepQA Reader dataset compatible with PyHealth v1.1.6
"""

"""This module exposes :class:`SleepQAReaderDataset`, a subclass of
:class:`pyhealth.datasets.SampleDataset`, which:
Loads the JSON file.
Builds one sample per QA instance.
Preserves the DPR-style reader fields:
  ``question``, ``answers``, ``ctxs``.
Example:
    >>> from pyhealth.datasets import SleepQAReaderDataset
    >>> dataset = SleepQAReaderDataset(
    ...     root="path/to/data",
    ...     split="train",
    ... )
    >>> sample = dataset[0]
    >>> sample["question"]
    'what can lack of sleep in children impact?'
    >>> len(sample["ctxs"]) > 0
    True
Typical reader usage:
Model consumes ``sample["question"]`` and the ``"text"`` inside each
      context in ``sample["ctxs"]``.
Evaluation compares predictions against ``sample["answers"]`` and
      optionally uses ``ctx["has_answer"]`` for supervision.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pyhealth.datasets.sample_dataset import SampleDataset
class SleepQAReaderDataset(SampleDataset):
    """Reader-style SleepQA dataset using the official SleepQA schema.
    This dataset is designed for training DPR-style readers using the SleepQA
    data. Each sample corresponds to one question with:
``question``: the question text.
``answers``: a list of reference answer strings.
``ctxs``: a list of context dictionaries, each containing keys such as
      ``id``, ``title``, ``text``, ``score``, and ``has_answer``.
    The class builds on :class:`pyhealth.datasets.SampleDataset` to produce
    processed samples with appropriate feature processors:
``question``: processed with the ``text`` processor.
``answers``: passed through as-is using the ``raw`` processor.
``ctxs``: passed through as-is using the ``raw`` processor.
``question_id``: passed through as-is using the ``raw`` processor.
``patient_id`` and ``record_id``: passed through as-is using the
      ``raw`` processor.
    A "patient" in this dataset corresponds to a question, and the
    ``patient_id`` as well as the ``record_id`` are set to a stable question
    identifier. This keeps compatibility with utilities that expect
    patient/record maps without forcing an EHR structure.
    Args:
        root: Root directory where the SleepQA JSON file is stored.
        split: String identifying the split, such as "train", "dev", or
            "test". The default is "train".
        json_filename: Optional explicit filename. If not provided, it will
            default to ``f"sleepqa_reader_{split}.json"``.
        dataset_name: Name of the dataset. Defaults to "SleepQAReader".
        task_name: Name of the task. Defaults to "reader_qa".
    Raises:
        FileNotFoundError: If the JSON file cannot be found at the resolved
            path.
        ValueError: If the JSON file does not contain a list of entries.
    """
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        json_filename: Optional[str] = None,
        dataset_name: str = "SleepQAReader",
        task_name: str = "reader_qa",
    ) -> None:
        self.root = Path(root)
        self.split = split
        if json_filename is None:
            json_filename = f"sleepqa_reader_{split}.json"
        self.json_path = self.root / json_filename
        samples = self._load_samples(self.json_path)
        input_schema = {
            "question": "text",
            "answers": "raw",
            "ctxs": "raw",
            "question_id": "raw",
            "patient_id": "raw",
            "record_id": "raw",
        }
        # Reader evaluation is usually based on string metrics (EM/F1) rather
        # than scalar labels, so we keep output_schema empty here.
        output_schema: Dict[str, Any] = {}
        super().__init__(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name=dataset_name,
            task_name=task_name,
        )
    @staticmethod
    def _load_samples(json_path: Path) -> List[Dict[str, Any]]:
        """Load SleepQA reader JSON into a list of sample dicts.
        This helper reads the JSON file and converts each entry into a
        standardized dictionary that can be consumed by
        :class:`pyhealth.datasets.SampleDataset`.
        Each output sample has the following keys:
``patient_id``: string identifier for the "patient" (question).
``record_id``: string identifier for the record (equal to
          ``patient_id``).
``question_id``: string identifier derived from an explicit
          ``id`` field when present or from the zero-based index.
``question``: question text.
``answers``: list of answer strings.
``ctxs``: list of context dictionaries, each including keys such as
          ``id``, ``title``, ``text``, ``score``, and ``has_answer``.
        Args:
            json_path: Path to the JSON file with SleepQA reader data.
        Returns:
            A list of sample dictionaries ready to be passed into
            :class:`pyhealth.datasets.SampleDataset`.
        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the top-level JSON object is not a list.
        """
        if not json_path.exists():
            raise FileNotFoundError(
                f"SleepQA reader JSON file not found at: {json_path}"
            )
        with json_path.open("r", encoding="utf-8") as json_file:
            raw_data = json.load(json_file)
        if not isinstance(raw_data, list):
            raise ValueError(
                "SleepQA reader JSON must contain a list of entries, "
                f"got type {type(raw_data)!r}"
            )
        samples: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw_data):
            entry_id = entry.get("id")
            if entry_id is None:
                entry_id = str(index)
            else:
                entry_id = str(entry_id)
            question = entry.get("question", "")
            answers = entry.get("answers", [])
            ctxs = entry.get("ctxs", [])
            sample: Dict[str, Any] = {
                "patient_id": f"q_{entry_id}",
                "record_id": f"q_{entry_id}",
                "question_id": entry_id,
                "question": question,
                "answers": answers,
                "ctxs": ctxs,
            }
            samples.append(sample)
        return samples
