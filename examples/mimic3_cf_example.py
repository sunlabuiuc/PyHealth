from pyhealth.datasets import MIMIC3CirculatoryFailureDataset
from pyhealth.tasks import CirculatoryFailurePredictionTask


def main():
    dataset = MIMIC3CirculatoryFailureDataset(
        root="/path/to/mimic3"
    )

    task = CirculatoryFailurePredictionTask(prediction_window_hours=12)

    # apply task
    samples = dataset.set_task(task, max_patients=5)

    print(f"Total samples: {len(samples)}")

    if samples:
        print("Sample example:")
        print(samples[0])


if __name__ == "__main__":
    main()