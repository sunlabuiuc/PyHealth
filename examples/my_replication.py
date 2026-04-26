from pyhealth.datasets.medlingo import MedLingoDataset
from pyhealth.tasks.medlingo_task import MedLingoTask
from pyhealth.models.abbreviation_lookup import AbbreviationLookupModel

"""
This script demonstrates a replication of the MedLingo clinical abbreviation expansion task.
It loads the MedLingo dataset, processes it into task-ready format, and evaluates a simple rule-based abbreviation lookup model.
Contributors:
    Tedra Birch (tbirch2@illinois.edu)

Paper:
    Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?
    https://arxiv.org/abs/2505.15024

"""
def main() -> None:
    dataset = MedLingoDataset(root="test-resources")
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