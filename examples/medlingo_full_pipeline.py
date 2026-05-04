"""
This script demonstrates a replication of the MedLingo clinical abbreviation expansion task.

It uses synthetic MedLingo-style samples defined inline, processes them into task-ready format, 
and evaluates a simple rule-based abbreviation lookup model.

Contributors:
    Tedra Birch (tbirch2@illinois.edu)

Paper:
    Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?
    https://arxiv.org/abs/2505.15024

"""

from pyhealth.datasets.medlingo import MedLingoDataset
from pyhealth.tasks.medlingo_task import MedLingoTask
from pyhealth.models.abbreviation_lookup import AbbreviationLookupModel



SYNTHETIC_MEDLINGO_SAMPLES = [
    {
        "abbr": "SOB",
        "context": "Patient presents with SOB.",
        "label": "shortness of breath",
        "source": "synthetic_demo",
    },
    {
        "abbr": "BP",
        "context": "BP remained stable overnight.",
        "label": "blood pressure",
        "source": "synthetic_demo",
    },
    {
        "abbr": "HTN",
        "context": "History of HTN.",
        "label": "hypertension",
        "source": "synthetic_demo",
    },
    {
        "abbr": "CHF",
        "context": "Known CHF with fluid overload.",
        "label": "congestive heart failure",
        "source": "synthetic_demo",
    },
    {
        "abbr": "DM",
        "context": "History of DM with elevated glucose.",
        "label": "diabetes mellitus",
        "source": "synthetic_demo",
    },
]


def main() -> None:
    dataset = MedLingoDataset(samples=SYNTHETIC_MEDLINGO_SAMPLES)
    records = dataset.process()

    task = MedLingoTask()
    processed = task.process(records)

    model = AbbreviationLookupModel(normalize=True)
    model.fit(
        [
            {"abbr": item["input"], "label": item["target"]}
            for item in processed
        ]
    )

    correct = 0
    total = len(processed)

    for item in processed:
        pred = model.predict(item["input"])
        if pred == item["target"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print("=== MedLingo Replication Pipeline ===")
    print(f"Loaded {len(records)} records")
    print(f"Processed {len(processed)} task samples")
    print(f"Accuracy: {accuracy:.3f}")
    print("Example sample:")
    print(processed[0])
    print("Example prediction:")
    print(model.predict(processed[0]['input']))


if __name__ == "__main__":
    main()