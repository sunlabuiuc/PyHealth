"""End-to-end VQA-RAD MedFlamingo pipeline example.

This example demonstrates the PyHealth flow on the MedFlamingo fork branch:

1. load the VQA-RAD base dataset
2. apply the MedicalVQATask via ``set_task()``
3. split into train/validation/test sets
4. create dataloaders
5. train MedFlamingo with ``Trainer.train()``
6. evaluate with ``Trainer.evaluate()``
7. run one compact few-shot generation example

The default MedFlamingo constructor may download large Hugging Face weights on
its first run, so expect setup time and substantial memory use.
"""

import argparse

from pyhealth.datasets import (
    VQARADDataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import MedFlamingo
from pyhealth.tasks import MedicalVQATask
from pyhealth.trainer import Trainer


def choose_splitter(samples):
    """Prefer patient-level splitting when the sample dataset preserves it."""
    patient_to_index = getattr(samples, "patient_to_index", {})
    if patient_to_index:
        return split_by_patient, "patient"
    return split_by_sample, "sample"


def build_few_shot_text(sample):
    """Formats one processed sample as a simple in-context example."""
    return f"Q: {sample['question']}\nA: {sample['answer']}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedFlamingo on VQA-RAD")
    parser.add_argument("--root", required=True, help="path to the VQA-RAD root")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="optional cache directory for processed dataset artifacts",
    )
    parser.add_argument("--dataset-num-workers", type=int, default=1)
    parser.add_argument("--task-num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = VQARADDataset(
        root=args.root,
        cache_dir=args.cache_dir,
        num_workers=args.dataset_num_workers,
    )
    dataset.stats()

    task = MedicalVQATask()
    samples = dataset.set_task(task, num_workers=args.task_num_workers)

    splitter, split_name = choose_splitter(samples)
    print(f"using {split_name}-level split")
    train_dataset, val_dataset, test_dataset = splitter(
        samples,
        [0.7, 0.1, 0.2],
        seed=42,
    )

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MedFlamingo(dataset=samples)
    trainer = Trainer(model=model, metrics=["accuracy", "f1_macro"])

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )

    metrics = trainer.evaluate(test_loader)
    print("test metrics:", metrics)

    query_sample = test_dataset[0]
    context_sample = train_dataset[0]
    generation = model.generate(
        images=[query_sample["image"]],
        prompt=query_sample["question"],
        few_shot_examples=[
            {
                "image": context_sample["image"],
                "text": build_few_shot_text(context_sample),
            }
        ],
        max_new_tokens=args.max_new_tokens,
    )
    print("few-shot generation:", generation)

    samples.close()
