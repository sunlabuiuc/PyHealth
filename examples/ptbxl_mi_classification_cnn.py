from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks import PTBXLMIClassificationTask
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=os.getenv(
            "PTBXL_ROOT",
            os.path.expanduser("~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"),
        ),
        help="Path to PTB-XL root folder (contains ptbxl_database.csv, scp_statements.csv, records100/records500/). "
        "we can also set PTBXL_ROOT environment variable instead of passing --root.",
    )
    args = parser.parse_args()
    root = args.root
    #root = os.path.expanduser(
    #"~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
#)
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