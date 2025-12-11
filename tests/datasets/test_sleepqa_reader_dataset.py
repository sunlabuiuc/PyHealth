# tests/datasets/test_sleepqa_reader_dataset.py
# -*- coding: utf-8 -*-
"""Tests for the SleepQAReaderDataset."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from pyhealth.datasets import SleepQAReaderDataset
from pyhealth.processors import RawProcessor, TextProcessor
def _write_sleepqa_reader_json(tmp_path: Path) -> Path:
    """Write a small SleepQA reader JSON file for testing.
    Args:
        tmp_path: Temporary directory provided by pytest.
    Returns:
        Path to the created JSON file.
    """
    data: List[Dict[str, Any]] = [
        {
            "question": "what can lack of sleep in children impact?",
            "answers": [
                "academic performance, behavior, and mood."
            ],
            "ctxs": [
                {
                    "id": "sleep:126",
                    "title": "is your smartphone affecting your sleep",
                    "text": (
                        "children's eyes are even more sensitive to light, "
                        "and the blue light from screens can delay "
                        "melatonin production by up to two times as much for "
                        "children compared with adults. this can lead to "
                        "insomnia and poor quality of sleep, which can be "
                        "particularly harmful for children. quality rest is "
                        "crucial for children as they grow and develop. lack "
                        "of sleep in children can impact academic "
                        "performance, behavior, and mood. poor sleep in "
                        "children has also been associated with health "
                        "issues, such as obesity and depression."
                    ),
                    "score": "103.232",
                    "has_answer": True,
                }
            ],
        },
        {
            "question": (
                "what is the purpose of the light sensor in the watch?"
            ),
            "answers": [
                "can reveal whether someone's sleep problems might be due "
                "to an overly bright bedroom or insufficient light during "
                "the day"
            ],
            "ctxs": [
                {
                    "id": "sleep:909",
                    "title": "how is actigraphy used to evaluate sleep",
                    "text": (
                        "the watch may also have a light sensor, which can "
                        "reveal whether someone's sleep problems might be "
                        "due to an overly bright bedroom or insufficient "
                        "light during the day."
                    ),
                    "score": "103.232",
                    "has_answer": True,
                }
            ],
        },
    ]
    json_path = tmp_path / "sleepqa_reader_train.json"
    json_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )
    return json_path
def test_sleepqa_reader_dataset_loads_and_preserves_fields(
    tmp_path: Path,
) -> None:
    """Test that SleepQAReaderDataset loads data and preserves structure."""
    _write_sleepqa_reader_json(tmp_path)
    dataset = SleepQAReaderDataset(root=tmp_path, split="train")
    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    # Basic keys should be present.
    for sample in (first, second):
        for key in (
            "patient_id",
            "record_id",
            "question_id",
            "question",
            "answers",
            "ctxs",
        ):
            assert key in sample, f"Missing expected key {key!r} in sample"
    # IDs should be consistent.
    assert first["patient_id"] == "q_0"
    assert first["record_id"] == "q_0"
    assert first["question_id"] == "0"
    assert second["patient_id"] == "q_1"
    assert second["record_id"] == "q_1"
    assert second["question_id"] == "1"
    # Questions and answers should match.
    assert (
        first["question"]
        == "what can lack of sleep in children impact?"
    )
    assert (
        second["question"]
        == "what is the purpose of the light sensor in the watch?"
    )
    assert first["answers"] == [
        "academic performance, behavior, and mood."
    ]
    assert (
        "overly bright bedroom" in second["answers"][0]
    )
    # Ctxs should be lists with expected fields.
    for sample in (first, second):
        assert isinstance(sample["ctxs"], list)
        assert len(sample["ctxs"]) == 1
        ctx = sample["ctxs"][0]
        for key in ("id", "title", "text", "score", "has_answer"):
            assert key in ctx, f"Missing key {key!r} in ctx"
        assert isinstance(ctx["has_answer"], bool)
def test_sleepqa_reader_dataset_processors(tmp_path: Path) -> None:
    """Test that SleepQAReaderDataset configures processors correctly."""
    _write_sleepqa_reader_json(tmp_path)
    dataset = SleepQAReaderDataset(root=tmp_path, split="train")
    # Question should use TextProcessor.
    question_processor = dataset.input_processors["question"]
    assert isinstance(question_processor, TextProcessor)
    # Other fields should use RawProcessor.
    for key in ("answers", "ctxs", "patient_id", "record_id", "question_id"):
        processor = dataset.input_processors[key]
        assert isinstance(
            processor,
            RawProcessor,
        ), f"Expected RawProcessor for key {key!r}"
