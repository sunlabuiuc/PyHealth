import json
from pathlib import Path

from pyhealth.datasets import ECGQADataset


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


def test_ecgqa_dataset_load_data_returns_expected_columns(tmp_path):
    root = tmp_path / "ecgqa_ptbxl"
    _write_fake_ecgqa_ptbxl(root)

    dataset = ECGQADataset(
        root=str(root),
        split="train",
        question_source="paraphrased",
        single_ecg_only=True,
    )

    df = dataset.load_data().compute()

    assert len(df) == 3
    assert "patient_id" in df.columns
    assert "qa/question" in df.columns
    assert "qa/answer_json" in df.columns
    assert "qa/candidate_answers_json" in df.columns
    assert set(df["patient_id"].tolist()) == {"10000", "10001"}


def test_ecgqa_dataset_filters_question_type(tmp_path):
    root = tmp_path / "ecgqa_ptbxl"
    _write_fake_ecgqa_ptbxl(root)

    dataset = ECGQADataset(
        root=str(root),
        split="train",
        question_source="paraphrased",
        question_types=["single-choose"],
        single_ecg_only=True,
    )

    df = dataset.load_data().compute()

    assert len(df) == 2
    assert set(df["qa/question_type"].tolist()) == {"single-choose"}