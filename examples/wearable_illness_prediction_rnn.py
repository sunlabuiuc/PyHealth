from pyhealth.datasets.wearable_dataset import WearableDataset
from pyhealth.tasks.wearable_illness_task import WearableIllnessPrediction


def main():
    # Load the small wearable dataset used for testing
    dataset = WearableDataset(root="test_data")

    # Task predicts illness based on changes from a short baseline window
    task = WearableIllnessPrediction(baseline_window=2)

    # Apply the task to generate samples
    samples = dataset.set_task(task)

    print(f"Number of samples: {len(samples)}")
    print("\nExample samples:")
    for sample in samples[:5]:
        print(sample)


if __name__ == "__main__":
    main()