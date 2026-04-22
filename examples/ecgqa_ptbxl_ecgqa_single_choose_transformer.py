import argparse

from pyhealth.datasets import ECGQADataset, get_dataloader, split_by_patient
from pyhealth.models import Transformer
from pyhealth.tasks import ECGQASingleChooseTask
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to ecgqa/ptbxl root directory",
    )
    parser.add_argument(
        "--question_source",
        type=str,
        default="paraphrased",
        choices=["paraphrased", "template"],
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use PyHealth dev mode for a smaller quick run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading ECG-QA dataset...")
    dataset = ECGQADataset(
        root=args.data_root,
        split="train",
        question_source=args.question_source,
        single_ecg_only=True,
        dev=args.dev,
    )

    print("Applying single-choose task...")
    task = ECGQASingleChooseTask()
    sample_dataset = dataset.set_task(task, num_workers=1)

    print("Splitting dataset...")
    train_ds, val_ds, test_ds = split_by_patient(
        sample_dataset,
        ratios=[0.7, 0.15, 0.15],
        seed=42,
    )

    print("Building dataloaders...")
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("Building Transformer model...")
    model = Transformer(dataset=sample_dataset)

    print("Training...")
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )

    print("Evaluating...")
    scores = trainer.evaluate(test_loader)
    print(scores)


if __name__ == "__main__":
    main()