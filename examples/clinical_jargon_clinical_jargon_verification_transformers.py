"""Clinical jargon verification example with `TransformersModel`.

This example is the course-facing ablation entrypoint for the public clinical
jargon benchmark contribution. It supports three lightweight task ablations
while keeping each source benchmark item in a single split:

- switch between ``medlingo``, ``casi``, and ``all`` benchmark subsets
- switch CASI between ``release62`` and ``paper59`` variants
- vary MedLingo distractor count through ``--medlingo-distractors``

Run this example from an environment where PyHealth is installed, such as
``pip install -e .`` from the repository root.

Example commands:
    python3 examples/clinical_jargon_clinical_jargon_verification_transformers.py \
        --benchmark medlingo --medlingo-distractors 1 --epochs 1

    python3 examples/clinical_jargon_clinical_jargon_verification_transformers.py \
        --benchmark casi --casi-variant paper59 --epochs 1
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from pyhealth.datasets import ClinicalJargonDataset, get_dataloader, split_by_patient
from pyhealth.models.transformers_model import TransformersModel
from pyhealth.tasks import ClinicalJargonVerification
from pyhealth.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the clinical jargon example.

    Returns:
        Parsed command-line arguments for dataset root, model choice, task
        configuration, and training hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Clinical jargon verification example with TransformersModel.",
        epilog=(
            "Ablation knobs: --benchmark changes the benchmark subset, "
            "--casi-variant changes the CASI filtering mode, and "
            "--medlingo-distractors changes the number of negative MedLingo "
            "candidates."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(
            PROJECT_ROOT / "test-resources" / "clinical_jargon"
        ),
        help="Dataset root containing clinical_jargon_examples.csv.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch the public source assets when the normalized CSV is missing.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="Hugging Face model name used by TransformersModel.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "medlingo", "casi"],
        default="medlingo",
        help="Benchmark subset used for the run.",
    )
    parser.add_argument(
        "--casi-variant",
        choices=["release62", "paper59"],
        default="release62",
        help="CASI filtering mode when --benchmark includes CASI.",
    )
    parser.add_argument(
        "--medlingo-distractors",
        type=int,
        default=3,
        help="Number of negative MedLingo candidates retained per sample.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    return parser.parse_args()


def main() -> None:
    """Run one clinical jargon verification configuration."""
    args = parse_args()
    dataset = ClinicalJargonDataset(root=args.root, download=args.download)
    task = ClinicalJargonVerification(
        benchmark=args.benchmark,
        casi_variant=args.casi_variant,
        medlingo_distractors=args.medlingo_distractors,
    )
    samples = dataset.set_task(task)
    print(
        {
            "benchmark": args.benchmark,
            "casi_variant": args.casi_variant,
            "medlingo_distractors": args.medlingo_distractors,
            "num_samples": len(samples),
            "model_name": args.model_name,
        }
    )
    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, [0.6, 0.2, 0.2], seed=42
    )
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TransformersModel(dataset=samples, model_name=args.model_name)
    trainer = Trainer(model=model, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=args.epochs,
    )
    scores = trainer.evaluate(test_loader)
    print(scores)


if __name__ == "__main__":
    main()
