from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks import PTBXLMIClassificationTask
import os
import argparse


def run_once(root: str, normalize: bool):
    metadata_file = os.path.join(root, "ptbxl_database.csv")

    if not os.path.exists(metadata_file):
        print("=" * 60)
        print("PTB-XL dataset not found. Running synthetic demo mode.")
        print(f"Expected file: {metadata_file}")

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.randn(10, 12, 1000)
        y = torch.randint(0, 2, (10,)).float()

        demo_dataset = TensorDataset(X, y)
        demo_loader = DataLoader(demo_dataset, batch_size=2, shuffle=False)

        first_batch = next(iter(demo_loader))
        demo_signal, demo_label = first_batch

        print(f"normalize={normalize}")
        print(f"demo batch signal shape: {demo_signal.shape}")
        print(f"demo batch labels: {demo_label}")
        print(f"demo signal mean/std: {demo_signal.mean():.4f} / {demo_signal.std():.4f}")
        print(f"Number of demo samples: {len(demo_dataset)}")
        return

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
            os.path.expanduser(
                "~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
            ),
        ),
        help=(
            "Path to PTB-XL root folder (contains ptbxl_database.csv, "
            "scp_statements.csv, records100/records500/). "
            "You can also set PTBXL_ROOT environment variable instead of passing --root."
        ),
    )

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
        help="Run a tiny ablation: compare normalize=True vs normalize=False.",
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