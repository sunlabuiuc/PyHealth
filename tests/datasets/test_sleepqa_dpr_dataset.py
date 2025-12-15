# tests/datasets/test_sleepqa_dpr_dataset.py
# -*- coding: utf-8 -*-
"""Tests for the SleepQADPRDataset."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import pytest
from pyhealth.datasets import SleepQADPRDataset
from pyhealth.processors import RawProcessor, TextProcessor
def _write_sleepqa_json(tmp_path: Path) -> Path:
    """Write a small DPR-style SleepQA JSON file for testing.
    Args:
        tmp_path: Temporary directory provided by pytest.
    Returns:
        Path to the created JSON file.
    """
    data: List[Dict[str, Any]] = [
        {
            "id": "42",
            "question": "what can lack of sleep in children impact?",
            "answers": ["academic performance, behavior, and mood."],
            "positive_ctxs": [
                {
                    "title": "is your smartphone affecting your sleep",
                    "text": "lack of sleep in children can impact academic "
                    "performance, behavior, and mood.",
                }
            ],
            "negative_ctxs": [
                {
                    "title": "how does lack of sleep effect "
                    "cognitive impairment",
                    "text": "teens are considered to be especially high-risk "
                    "for detrimental effects of poor sleep.",
                }
            ],
        }
    ]
    json_path = tmp_path / "sleepqa_retriever_train.json"
    json_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )
    return json_path
def test_sleepqa_dpr_dataset_loads_and_preserves_fields(tmp_path: Path) -> None:
    """Test that SleepQADPRDataset loads data and preserves DPR structure."""
    _write_sleepqa_json(tmp_path)
    dataset = SleepQADPRDataset(root=tmp_path, split="train")
    assert len(dataset) == 1
    sample = dataset[0]
    # Basic keys should be present.
    for key in (
        "patient_id",
        "record_id",
        "question_id",
        "question",
        "answers",
        "positive_ctxs",
        "negative_ctxs",
    ):
        assert key in sample, f"Missing expected key {key!r} in sample"
    # IDs should be consistent.
    assert sample["patient_id"] == "q_42"
    assert sample["record_id"] == "q_42"
    assert sample["question_id"] == "42"
    # Question and answers should match the JSON.
    assert (
        sample["question"]
        == "what can lack of sleep in children impact?"
    )
    assert sample["answers"] == ["academic performance, behavior, and mood."]
    # Positive and negative contexts should be lists with one element each.
    assert isinstance(sample["positive_ctxs"], list)
    assert isinstance(sample["negative_ctxs"], list)
    assert len(sample["positive_ctxs"]) == 1
    assert len(sample["negative_ctxs"]) == 1
    pos_ctx = sample["positive_ctxs"][0]
    neg_ctx = sample["negative_ctxs"][0]
    assert "title" in pos_ctx and "text" in pos_ctx
    assert "title" in neg_ctx and "text" in neg_ctx
    assert "lack of sleep in children can impact" in pos_ctx["text"]
def test_sleepqa_dpr_dataset_processors(tmp_path: Path) -> None:
    """Test that SleepQADPRDataset configures processors correctly."""
    _write_sleepqa_json(tmp_path)
    dataset = SleepQADPRDataset(root=tmp_path, split="train")
    # Question should use TextProcessor.
    question_processor = dataset.input_processors["question"]
    assert isinstance(question_processor, TextProcessor)
    # All other DPR fields should use RawProcessor.
    for key in ("answers", "positive_ctxs", "negative_ctxs"):
        processor = dataset.input_processors[key]
        assert isinstance(
            processor,
            RawProcessor,
        ), f"Expected RawProcessor for key {key!r}"
    # ID-related fields should also be raw.
    for key in ("patient_id", "record_id", "question_id"):
        processor = dataset.input_processors[key]
        assert isinstance(
            processor,
            RawProcessor,
        ), f"Expected RawProcessor for key {key!r}"
