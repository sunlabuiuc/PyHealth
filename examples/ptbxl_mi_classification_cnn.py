from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks import PTBXLMIClassificationTask


def main():
    root = "/Users/zaidalkhatib/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    dataset = PTBXLDataset(
        root=root,
        dev=True,
    )

    task = PTBXLMIClassificationTask(root=root)
    task_dataset = dataset.set_task(task)

    print(task_dataset[0])
    print(f"Number of samples: {len(task_dataset)}")


if __name__ == "__main__":
    main()