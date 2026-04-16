"""
Ablation Study: MedLingo task-variation comparison with one PyHealth model.

This script compares two task formulations for the MedLingo dataset using the
same existing PyHealth model architecture:

1. QuestionOnlyMedLingoClassification
   - input text = original MedLingo question only

2. ExplicitAbbreviationMedLingoClassification
   - input text = abbreviation + original question + explicit instruction

Both tasks predict the correct expansion as a multiclass label. The script:
- loads MedLingo through MedLingoDataset
- applies each task through dataset.set_task(...)
- uses the same patient-level train/val/test split for both tasks
- trains the same TransformersModel on both tasks
- compares test accuracy

This is intended to satisfy the example/ablation requirement for the
dataset + task project option while keeping the comparison focused on
task variation rather than model variation.

Current findings
----------------
In one reference run with a 70/10/20 patient split and Bio_ClinicalBERT,
the question-only formulation achieved 5% test accuracy while the
explicit-abbreviation formulation achieved 0% test accuracy. We interpret
this cautiously because MedLingo is very small and the task is framed as
a 100-class classification problem, which makes both variants difficult
to learn reliably from the available training examples.
"""

import os
import random
import tempfile
from typing import Dict, List, Tuple

from torch.utils.data import Subset, DataLoader

from pyhealth.datasets import MedLingoDataset
from pyhealth.datasets.utils import collate_fn_dict_with_padding
from pyhealth.models import TransformersModel
from pyhealth.tasks.base_task import BaseTask
from pyhealth.trainer import Trainer


DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test-resources",
    "MedLingo",
)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


class QuestionOnlyMedLingoClassification(BaseTask):
    """
    Baseline MedLingo classification task.

    Input:
        - original MedLingo question text only
    Output:
        - correct abbreviation expansion as a multiclass label
    """

    task_name: str = "medlingo_question_only_classification"
    input_schema: Dict[str, str] = {"text": "text"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient) -> List[Dict]:
        samples: List[Dict] = []

        events = patient.get_events("questions")
        if not events:
            return samples

        for event in events:
            question = str(getattr(event, "question", "") or "").strip()
            answer = str(getattr(event, "answer", "") or "").strip()

            if not question or not answer:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": patient.patient_id,
                    "text": question,
                    "label": answer,
                }
            )

        return samples


class ExplicitAbbreviationMedLingoClassification(BaseTask):
    """
    Ablated MedLingo classification task.

    Input:
        - explicit abbreviation
        - original MedLingo question
        - explicit instruction to give the full expansion

    Output:
        - correct abbreviation expansion as a multiclass label
    """

    task_name: str = "medlingo_explicit_abbreviation_classification"
    input_schema: Dict[str, str] = {"text": "text"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient) -> List[Dict]:
        samples: List[Dict] = []

        events = patient.get_events("questions")
        if not events:
            return samples

        for event in events:
            question = str(getattr(event, "question", "") or "").strip()
            answer = str(getattr(event, "answer", "") or "").strip()

            if not question or not answer:
                continue

            text = (
                f"Abbreviation: {patient.patient_id}\n"
                f"Question: {question}\n"
                f"Answer with the full expansion of the abbreviation."
            )

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": patient.patient_id,
                    "text": text,
                    "label": answer,
                }
            )

        return samples


def build_sample_dataset(task: BaseTask):
    """
    Build a SampleDataset for one task variation using a fresh cache_dir.
    """
    cache_dir = tempfile.mkdtemp(prefix="medlingo_ablation_cache_")
    dataset = MedLingoDataset(
        root=os.path.normpath(DATA_ROOT),
        cache_dir=cache_dir,
    )
    samples = dataset.set_task(task)
    return samples


def make_patient_splits(sample_dataset, seed: int = SEED) -> Tuple[List[str], List[str], List[str]]:
    """
    Create reproducible patient-level train/val/test splits.

    Because MedLingo is effectively one sample per abbreviation/patient, using
    patient-level splits gives a clean and aligned comparison across task variants.
    """
    patient_ids = sorted(sample_dataset.patient_to_index.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def ids_to_indices(sample_dataset, patient_ids: List[str]) -> List[int]:
    """
    Convert patient IDs into sample indices using SampleDataset.patient_to_index.
    """
    indices: List[int] = []
    for pid in patient_ids:
        indices.extend(sample_dataset.patient_to_index[pid])
    return indices


def _get_dataloader_from_subset(subset, batch_size: int):
    """
    Create a DataLoader from a Subset without calling set_shuffle.
    
    set_shuffle should be called on the underlying dataset before creating the subset.
    """
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        collate_fn=collate_fn_dict_with_padding,
    )
    return dataloader


def build_loaders(sample_dataset, train_ids: List[str], val_ids: List[str], test_ids: List[str]):
    """
    Build dataloaders for one task variation using the same patient splits.
    """
    train_idx = ids_to_indices(sample_dataset, train_ids)
    val_idx = ids_to_indices(sample_dataset, val_ids)
    test_idx = ids_to_indices(sample_dataset, test_ids)

    train_subset = Subset(sample_dataset, train_idx)
    val_subset = Subset(sample_dataset, val_idx)
    test_subset = Subset(sample_dataset, test_idx)

    # Set shuffle on the underlying dataset before passing subsets to get_dataloader
    sample_dataset.set_shuffle(True)
    train_loader = _get_dataloader_from_subset(train_subset, batch_size=BATCH_SIZE)
    sample_dataset.set_shuffle(False)
    val_loader = _get_dataloader_from_subset(val_subset, batch_size=BATCH_SIZE)
    test_loader = _get_dataloader_from_subset(test_subset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def train_and_evaluate(sample_dataset, train_ids: List[str], val_ids: List[str], test_ids: List[str], label: str):
    """
    Train one existing PyHealth model on one task variation and return test metrics.
    """
    print(f"\n=== Running variation: {label} ===")

    train_loader, val_loader, test_loader = build_loaders(
        sample_dataset,
        train_ids,
        val_ids,
        test_ids,
    )

    model = TransformersModel(
        model_name=MODEL_NAME,
        dataset=sample_dataset,
    )

    trainer = Trainer(
        model=model,
        metrics=["accuracy"],
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=EPOCHS,
        monitor="accuracy",
        optimizer_params={"lr": LR},
    )

    # In current PyHealth, evaluate() is the usual follow-up. If your local
    # version exposes test() instead, replace this line with trainer.test(...).
    test_metrics = trainer.evaluate(test_loader)

    return test_metrics


def extract_accuracy(metrics: Dict) -> float:
    """
    Pull accuracy out of the returned metrics dict.
    """
    for key in ("accuracy", "acc"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"Could not find accuracy in metrics: {metrics}")


def main():
    if not os.path.isfile(os.path.join(DATA_ROOT, "questions.csv")):
        raise FileNotFoundError(
            f"Could not find questions.csv under: {os.path.normpath(DATA_ROOT)}"
        )

    print("Building SampleDataset for baseline task...")
    baseline_samples = build_sample_dataset(QuestionOnlyMedLingoClassification())

    print("Building SampleDataset for explicit-abbreviation task...")
    explicit_samples = build_sample_dataset(ExplicitAbbreviationMedLingoClassification())

    # Use the same patient split for both task variations
    train_ids, val_ids, test_ids = make_patient_splits(baseline_samples, seed=SEED)

    print(f"\nPatient split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Model: {MODEL_NAME}")

    baseline_metrics = train_and_evaluate(
        baseline_samples,
        train_ids,
        val_ids,
        test_ids,
        label="Question only",
    )

    explicit_metrics = train_and_evaluate(
        explicit_samples,
        train_ids,
        val_ids,
        test_ids,
        label="Explicit abbreviation + instruction",
    )

    baseline_acc = extract_accuracy(baseline_metrics)
    explicit_acc = extract_accuracy(explicit_metrics)

    print("\n" + "=" * 72)
    print("MedLingo Task-Variation Ablation Results")
    print("=" * 72)
    print(f"{'Variation':<40} {'Test Accuracy':>15}")
    print("-" * 72)
    print(f"{'Question only':<40} {baseline_acc:>15.4f}")
    print(f"{'Explicit abbreviation + instruction':<40} {explicit_acc:>15.4f}")
    print("-" * 72)
    print(f"{'Delta (explicit - baseline)':<40} {explicit_acc - baseline_acc:>15.4f}")
    print("=" * 72)

    print(
        "\nInterpretation\n"
        "--------------\n"
        "This ablation compares two task formulations using the same existing PyHealth "
        "model. Any performance difference should therefore be attributed primarily to "
        "the input/task variation rather than to a change in model architecture. "
        "This is the intended focus of the MedLingo task-ablation comparison.\n\n"
        "In our current run, the question-only formulation performed slightly better "
        "than the explicit-abbreviation formulation, but both results were very weak. "
        "We interpret this cautiously because MedLingo is small and the task becomes "
        "a 100-class classification problem with very limited training data."
    )


if __name__ == "__main__":
    main()