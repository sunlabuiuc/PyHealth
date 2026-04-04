import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyhealth.datasets import ClinicalJargonDataset, get_dataloader, split_by_sample
from pyhealth.models.transformers_model import TransformersModel
from pyhealth.tasks import ClinicalJargonVerification
from pyhealth.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical jargon verification example with TransformersModel."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(
            PROJECT_ROOT / "test-resources" / "clinical_jargon"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "medlingo", "casi"],
        default="medlingo",
    )
    parser.add_argument(
        "--casi-variant",
        choices=["release62", "paper59"],
        default="release62",
    )
    parser.add_argument("--medlingo-distractors", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ClinicalJargonDataset(root=args.root)
    task = ClinicalJargonVerification(
        benchmark=args.benchmark,
        casi_variant=args.casi_variant,
        medlingo_distractors=args.medlingo_distractors,
    )
    samples = dataset.set_task(task)
    train_dataset, val_dataset, test_dataset = split_by_sample(
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
