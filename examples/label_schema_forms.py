"""Output schema forms: string, processor class, and (name, kwargs) tuple.

A task's ``output_schema`` can name the label processor in several equivalent
ways. This script trains and evaluates an RNN on a tiny synthetic multilabel
dataset with each form and shows that ``model.mode`` and the evaluation
metrics come out the same. Runs on CPU in a few seconds; no data download.

Usage:
    python examples/label_schema_forms.py
"""

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.processors import MultiLabelProcessor
from pyhealth.trainer import Trainer

SCHEMA_FORMS = {
    "string": "multilabel",
    "processor class": MultiLabelProcessor,
    "(name, kwargs) tuple": ("multilabel", {}),
}


def make_samples(n: int = 30):
    codes = ["dx-1", "dx-2", "dx-3", "dx-4", "dx-5"]
    drugs = [["aspirin"], ["insulin"], ["aspirin", "insulin"]]
    return [
        {
            "patient_id": f"patient-{i}",
            "visit_id": f"visit-{i}",
            "conditions": codes[: 1 + i % len(codes)],
            "drugs": drugs[i % len(drugs)],
        }
        for i in range(n)
    ]


def main():
    samples = make_samples()
    for name, label_spec in SCHEMA_FORMS.items():
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence"},
            output_schema={"drugs": label_spec},
            dataset_name="label_schema_forms",
        )
        loader = get_dataloader(dataset, batch_size=10, shuffle=False)

        model = RNN(dataset=dataset)
        trainer = Trainer(
            model=model,
            metrics=["jaccard_samples", "f1_samples"],
            enable_logging=False,
        )
        trainer.train(train_dataloader=loader, epochs=1)
        scores = trainer.evaluate(loader)
        print(f"{name:22} model.mode={model.mode!r:14} scores={scores}")


if __name__ == "__main__":
    main()
