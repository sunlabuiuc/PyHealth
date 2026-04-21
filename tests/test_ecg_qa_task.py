import json
from pathlib import Path

import pytest

from pyhealth.datasets import ECGQADataset
from pyhealth.tasks import ECGQASingleChooseTask


def _write_fake_ecgqa_ptbxl(root: Path) -> None:
    (root / "paraphrased" / "train").mkdir(parents=True, exist_ok=True)
    (root / "paraphrased" / "valid").mkdir(parents=True, exist_ok=True)
    (root / "paraphrased" / "test").mkdir(parents=True, exist_ok=True)
    (root / "template" / "train").mkdir(parents=True, exist_ok=True)
    (root / "template" / "valid").mkdir(parents=True, exist_ok=True)
    (root / "template" / "test").mkdir(parents=True, exist_ok=True)

    (root / "answers.csv").write_text(
        "index,class\n0,no\n1,yes\n2,none\n",
        encoding="utf-8",
    )

    (root / "answers_for_each_template.csv").write_text(
        'template_id,classes\n'
        '1,"[\'no\', \'yes\']"\n'
        '2,"[\'none\', \'yes\']"\n',
        encoding="utf-8",
    )

    (root / "train_ecgs.tsv").write_text(
        "0\t10000\n1\t10001\n",
        encoding="utf-8",
    )
    (root / "valid_ecgs.tsv").write_text(
        "0\t20000\n",
        encoding="utf-8",
    )
    (root / "test_ecgs.tsv").write_text(
        "0\t30000\n",
        encoding="utf-8",
    )

    train_records = [
        {
            "template_id": 1,
            "question_id": 10,
            "sample_id": 100,
            "question_type": "single-choose",
            "attribute_type": "rhythm",
            "question": "Is there arrhythmia?",
            "answer": ["yes"],
            "ecg_id": [10000],
            "attribute": ["arrhythmia"],
        },
        {
            "template_id": 1,
            "question_id": 11,
            "sample_id": 101,
            "question_type": "single-choose",
            "attribute_type": "rhythm",
            "question": "Is there arrhythmia?",
            "answer": ["no"],
            "ecg_id": [10001],
            "attribute": ["arrhythmia"],
        },
        {
            "template_id": 2,
            "question_id": 12,
            "sample_id": 102,
            "question_type": "single-query",
            "attribute_type": "noise",
            "question": "What noise is present?",
            "answer": ["none"],
            "ecg_id": [10001],
            "attribute": ["noise"],
        },
    ]

    empty_records = []

    for source in ["paraphrased", "template"]:
        (root / source / "train" / "000000.json").write_text(
            json.dumps(train_records),
            encoding="utf-8",
        )
        (root / source / "valid" / "000000.json").write_text(
            json.dumps(empty_records),
            encoding="utf-8",
        )
        (root / source / "test" / "000000.json").write_text(
            json.dumps(empty_records),
            encoding="utf-8",
        )


def test_ecgqa_task_builds_samples(tmp_path):
    root = tmp_path / "ecgqa_ptbxl"
    _write_fake_ecgqa_ptbxl(root)

    dataset = ECGQADataset(
        root=str(root),
        split="train",
        question_source="paraphrased",
        single_ecg_only=True,
    )
    task = ECGQASingleChooseTask()
    sample_dataset = dataset.set_task(task, num_workers=1)

    assert len(sample_dataset) == 2

    sample = sample_dataset[0]
    assert "patient_id" in sample
    assert "visit_id" in sample
    assert "record_id" in sample
    assert "question" in sample
    assert "label" in sample

    assert hasattr(sample["question"], "ndim")
    assert sample["question"].ndim == 1
    assert len(sample["question"]) > 0
    assert int(sample["label"]) in {0, 1}

def test_ecgqa_task_drop_none_answers(tmp_path):
    root = tmp_path / "ecgqa_ptbxl"
    _write_fake_ecgqa_ptbxl(root)

    dataset = ECGQADataset(
        root=str(root),
        split="train",
        question_source="paraphrased",
        single_ecg_only=True,
    )
    task = ECGQASingleChooseTask(
        question_types=["single-query"],
        drop_none_answers=True,
    )

    with pytest.raises(ValueError, match="zero samples"):
        dataset.set_task(task, num_workers=1)