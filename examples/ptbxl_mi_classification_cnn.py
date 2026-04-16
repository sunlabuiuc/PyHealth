from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks import PTBXLMIClassificationTask
import os
import argparse


def run_once(root: str, normalize: bool):
    dataset = PTBXLDataset(
        root=root,
        dev=True,
        use_high_resolution=False,  # False -> records100, True -> records500
    )

    task = PTBXLMIClassificationTask(
        root=root,
        signal_length=1000,   # 10 seconds at 100 Hz
        normalize=normalize,
    )

    task_dataset = dataset.set_task(task)

    sample = task_dataset[0]
    signal = sample["signal"]

    # Convert signal to numbers for printing mean/std
    try:
        signal_np = signal.detach().cpu().numpy()
    except Exception:
        signal_np = signal

    print("=" * 60)
    print(f"normalize={normalize}")
    print(f"sample label: {sample['label']}")
    print(f"signal shape: {signal_np.shape}")
    print(f"signal mean/std: {signal_np.mean():.4f} / {signal_np.std():.4f}")
    print(f"Number of samples: {len(task_dataset)}")


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
             "You can also set PTBXL_ROOT environment variable instead of passing --root.",
    )

    # Ablation flags
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Enable per-channel z-score normalization (default: True).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization.",
    )
    parser.add_argument(
        "--ablation-normalize",
        action="store_true",
        help="Running a tiny ablation: compare normalize=True vs normalize=False.",
    )

    args = parser.parse_args()
    root = args.root

    if args.ablation_normalize:
        run_once(root, normalize=True)
        run_once(root, normalize=False)
    else:
        run_once(root, normalize=args.normalize)


if __name__ == "__main__":
    main()
