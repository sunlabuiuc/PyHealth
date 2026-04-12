from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks import PTBXLMIClassificationTask
import os

def main():
    root = os.path.expanduser(
        "~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
    dataset = PTBXLDataset(
        root=root,
        dev=True,
        use_high_resolution=False,  # False -> records100, True -> records500
    )

    task = PTBXLMIClassificationTask(
        root=root,
        signal_length=1000,   # 10 seconds at 100 Hz
        normalize=True,
    )
    task_dataset = dataset.set_task(task)

    print(task_dataset[0])
    print(f"Number of samples: {len(task_dataset)}")


if __name__ == "__main__":
    main()