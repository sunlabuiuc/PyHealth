"""
Author(s): Ezra Hieh, Jeffrey Lee, Venkatesh Venkataramanan
NetID(s): iezralh2, jl187, vv33
Paper: SleepQA: Question Answering for Sleep and Health Knowledge
Paper Link: https://github.com/IvaBojic/SleepQA
Description: SleepQA Reader dataset compatible with PyHealth v1.1.6
"""

# pyhealth/datasets/sleepqa_dpr_dataset.py
"""SleepQA DPR dataset.
This module provides a PyHealth dataset wrapper for DPR-style SleepQA data.
The goal of this dataset is to make it easy to use SleepQA for training
bi-encoder retrievers such as Dense Passage Retrieval (DPR). Each sample
in the dataset corresponds to a single question and its associated
positive and negative contexts.
The expected JSON format follows the standard DPR style:
    [
      {
        "id": "0",  # optional; if missing, a sequential id will be used
        "question": "what can lack of sleep in children impact?",
        "answers": [
          "academic performance, behavior, and mood."
        ],
        "positive_ctxs": [
          {
            "title": "is your smartphone affecting your sleep",
            "text": "..."
          }
        ],
        "negative_ctxs": [
          {
            "title": "how does lack of sleep effect cognitive impairment",
            "text": "..."
          }
        ]
      },
      ...
    ]
This module exposes :class:`SleepQADPRDataset`, a subclass of
:class:`pyhealth.datasets.SampleDataset`, which:
Loads the JSON file.
Builds one sample per question.
Preserves all DPR fields (question, answers, positive_ctxs, negative_ctxs).
Uses the ``text`` processor for the question text and the ``raw`` processor
  for everything else, so downstream DPR models can implement their own
  tokenization and loss.
Example:
    >>> from pyhealth.datasets import SleepQADPRDataset
    >>> dataset = SleepQADPRDataset(
    ...     root="path/to/data",
    ...     split="train",
    ... )
    >>> sample = dataset[0]
    >>> sample["question"]
    'what can lack of sleep in children impact?'
    >>> len(sample["positive_ctxs"]) > 0
    True
Typical DPR usage:
Question encoder consumes ``sample["question"]``.
Context encoder consumes the ``"text"`` field inside each context in
      ``sample["positive_ctxs"]`` and ``sample["negative_ctxs"]``.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pyhealth.datasets.sample_dataset import SampleDataset
class SleepQADPRDataset(SampleDataset):
    """DPR-style SleepQA dataset.
    This dataset is designed specifically for training DPR-style bi-encoder
    retrievers on the SleepQA data. Each sample corresponds to one question
    together with its positive and negative contexts.
    The class builds on :class:`pyhealth.datasets.SampleDataset` to produce
    processed samples with appropriate feature processors:
``question``: processed with the ``text`` processor.
``answers``: passed through as-is using the ``raw`` processor.
``positive_ctxs``: passed through as-is using the ``raw`` processor.
``negative_ctxs``: passed through as-is using the ``raw`` processor.
    A "patient" in this dataset corresponds to a question, and the
    ``patient_id`` as well as the ``record_id`` are set to the question id.
    This keeps compatibility with utilities that expect patient/record maps
    without forcing an EHR structure.
    Args:
        root: Root directory where the SleepQA JSON file is stored.
        split: String identifying the split, such as "train", "dev", or
            "test". The default is "train".
        json_filename: Optional explicit filename. If not provided, it will
            default to ``f"sleepqa_retriever_{split}.json"``.
        dataset_name: Name of the dataset. Defaults to "SleepQADPR".
        task_name: Name of the task. Defaults to "dpr_retrieval".
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
        dataset_name: str = "SleepQADPR",
        task_name: str = "dpr_retrieval",
    ) -> None:
        self.root = Path(root)
        self.split = split
        if json_filename is None:
            json_filename = f"sleepqa_retriever_{split}.json"
        self.json_path = self.root / json_filename
        samples = self._load_samples(self.json_path)
        input_schema = {
            "question": "text",
            "answers": "raw",
            "positive_ctxs": "raw",
            "negative_ctxs": "raw",
            "question_id": "raw",
            "patient_id": "raw",
            "record_id": "raw",
        }
        # DPR does not have scalar labels in the classic formulation.
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
        """Load DPR-style SleepQA JSON into a list of sample dicts.
        This helper reads the JSON file and converts each entry into a
        standardized dictionary that can be consumed by
        :class:`pyhealth.datasets.SampleDataset`.
        Each output sample has the following keys:
``patient_id``: string identifier for the "patient" (question).
``record_id``: string identifier for the record (equal to
          ``patient_id``).
``question_id``: string identifier mirroring the original entry id.
``question``: question text.
``answers``: list of answer strings.
``positive_ctxs``: list of context dictionaries.
``negative_ctxs``: list of context dictionaries.
        Args:
            json_path: Path to the JSON file with DPR-style SleepQA data.
        Returns:
            A list of sample dictionaries ready to be passed into
            :class:`pyhealth.datasets.SampleDataset`.
        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the top-level JSON object is not a list.
        """
        if not json_path.exists():
            raise FileNotFoundError(
                f"SleepQA JSON file not found at: {json_path}"
            )
        with json_path.open("r", encoding="utf-8") as json_file:
            raw_data = json.load(json_file)
        if not isinstance(raw_data, list):
            raise ValueError(
                "SleepQA JSON must contain a list of entries, "
                f"got type {type(raw_data)!r}"
            )
        samples: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw_data):
            # Determine question id: use explicit "id" if present, otherwise
            # fall back to a zero-based index.
            entry_id = entry.get("id")
            if entry_id is None:
                entry_id = str(index)
            else:
                entry_id = str(entry_id)
            question = entry.get("question", "")
            answers = entry.get("answers", [])
            positive_ctxs = entry.get("positive_ctxs", [])
            negative_ctxs = entry.get("negative_ctxs", [])
            sample: Dict[str, Any] = {
                "patient_id": f"q_{entry_id}",
                "record_id": f"q_{entry_id}",
                "question_id": entry_id,
                "question": question,
                "answers": answers,
                "positive_ctxs": positive_ctxs,
                "negative_ctxs": negative_ctxs,
            }
            samples.append(sample)
        return samples
