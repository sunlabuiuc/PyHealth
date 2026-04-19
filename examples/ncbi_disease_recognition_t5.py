"""T5-style example for the NCBI Disease recognition task.

This script converts NCBI Disease span annotations into text-to-text BIO
tagging pairs, matching the T5-style framing used in "Are Clinical T5 Models
Better for Clinical Text?" without downloading or fine-tuning a model.

The ablation compares two task configurations:
- full title+abstract context
- abstract-only context

For each configuration, the example trains a tiny disease-phrase memorization
baseline on the synthetic train split and evaluates token-level BIO F1 on the
dev/test splits. This keeps the example fast while showing how the input
configuration changes both the serialized T5 targets and a measurable score.
"""

import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from pyhealth.datasets import NCBIDiseaseDataset
from pyhealth.tasks import NCBIDiseaseRecognition


Phrase = Tuple[str, ...]


def serialize_t5_pair(sample: dict, task_prefix: str = "ncbi disease") -> dict:
    tokens, tags = NCBIDiseaseRecognition.entities_to_bio_tags(
        sample["text"], sample["entities"]
    )
    prefix = f"{task_prefix}: " if task_prefix else ""
    return {
        "source_text": prefix + " ".join(tokens),
        "target_text": " ".join(tags),
    }


def normalize_token(token: str) -> str:
    return re.sub(r"^\W+|\W+$", "", token).lower()


def normalize_phrase(tokens: Iterable[str]) -> Phrase:
    normalized: List[str] = []
    for token in tokens:
        token = normalize_token(token)
        if token:
            normalized.append(token)
    return tuple(normalized)


def build_phrase_bank(samples: Iterable[dict]) -> Set[Phrase]:
    phrases = set()
    for sample in samples:
        if sample["split"] != "train":
            continue
        for entity in sample["entities"]:
            tokens, _ = NCBIDiseaseRecognition.entities_to_bio_tags(
                entity["text"], []
            )
            phrase = normalize_phrase(tokens)
            if phrase:
                phrases.add(phrase)
    return phrases


def predict_bio_tags(tokens: List[str], phrase_bank: Set[Phrase]) -> List[str]:
    predictions = ["O"] * len(tokens)
    normalized_tokens = [normalize_token(token) for token in tokens]
    phrases = sorted(phrase_bank, key=len, reverse=True)

    for start in range(len(normalized_tokens)):
        if predictions[start] != "O":
            continue
        for phrase in phrases:
            end = start + len(phrase)
            if tuple(normalized_tokens[start:end]) != phrase:
                continue
            predictions[start] = "B-Disease"
            for index in range(start + 1, end):
                predictions[index] = "I-Disease"
            break

    return predictions


def score_bio_tags(gold_tags: List[str], predicted_tags: List[str]) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold, predicted in zip(gold_tags, predicted_tags):
        if predicted == gold and gold != "O":
            true_positive += 1
        elif predicted != gold:
            false_positive += int(predicted != "O")
            false_negative += int(gold != "O")

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_phrase_baseline(
    samples: Iterable[dict],
    phrase_bank: Set[Phrase],
) -> Dict[str, float]:
    gold_tags: List[str] = []
    predicted_tags: List[str] = []

    for sample in samples:
        if sample["split"] == "train":
            continue
        tokens, tags = NCBIDiseaseRecognition.entities_to_bio_tags(
            sample["text"], sample["entities"]
        )
        gold_tags.extend(tags)
        predicted_tags.extend(predict_bio_tags(tokens, phrase_bank))

    return score_bio_tags(gold_tags, predicted_tags)


def summarize(samples: List[dict], label: str) -> None:
    entity_counts = [len(sample["entities"]) for sample in samples]
    avg_entities = sum(entity_counts) / len(entity_counts)
    token_counts = [
        len(NCBIDiseaseRecognition.entities_to_bio_tags(sample["text"], [])[0])
        for sample in samples
    ]
    avg_tokens = sum(token_counts) / len(token_counts)
    phrase_bank = build_phrase_bank(samples)
    scores = evaluate_phrase_baseline(samples, phrase_bank)

    print(
        f"{label}: {len(samples)} samples, "
        f"avg entities={avg_entities:.2f}, avg tokens={avg_tokens:.2f}, "
        f"train phrases={len(phrase_bank)}, "
        f"dev/test precision={scores['precision']:.3f}, "
        f"recall={scores['recall']:.3f}, F1={scores['f1']:.3f}"
    )


def copy_demo_root(source_root: str, target_root: str) -> None:
    source_path = Path(source_root)
    target_path = Path(target_root)
    for path in source_path.glob("NCBI*_corpus.txt"):
        shutil.copy(path, target_path / path.name)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        copy_demo_root("test-resources/ncbi_disease", root)
        dataset = NCBIDiseaseDataset(root=root, dev=False)

        full_text_samples = dataset.set_task(
            NCBIDiseaseRecognition(text_source="full_text")
        )
        abstract_samples = dataset.set_task(
            NCBIDiseaseRecognition(text_source="abstract")
        )
        full_text_rows = list(full_text_samples)
        abstract_rows = list(abstract_samples)

        summarize(full_text_rows, "full_text")
        summarize(abstract_rows, "abstract")

        print("full_text T5 pair:", serialize_t5_pair(full_text_rows[0]))
        print("abstract T5 pair:", serialize_t5_pair(abstract_rows[0]))
