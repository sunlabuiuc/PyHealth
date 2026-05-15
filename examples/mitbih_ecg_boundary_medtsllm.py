"""MIT-BIH ECG Boundary Detection with MedTsLLM.

Demonstrates the MedTsLLM model (Chan et al., MLHC 2024) on the
MIT-BIH Arrhythmia Database for R-peak boundary detection using
2-channel ECG downsampled from 360 Hz to 120 Hz. Binary
segmentation task trained with BCE-with-logits loss.

Paper: https://arxiv.org/abs/2408.07773
Dataset: https://physionet.org/content/mitdb/1.0.0/

Usage:
    python examples/mitbih_ecg_boundary_medtsllm.py --synthetic
    python examples/mitbih_ecg_boundary_medtsllm.py \\
        --root /path/to/mitdb --backbone openai-community/gpt2

Ablation Study:
    The script exposes the paper's two main ablation axes as CLI
    flags so each run is a single ablation cell.

    1. LLM backbone swap -- ``--backbone <hf_id>``:
       compare GPT-2 vs. GPT-2-medium vs. DistilGPT-2 etc.

    2. Prompt components -- each piece of the text prompt can be
       disabled independently:
           --no-prompt-dataset   drops the dataset description
           --no-prompt-task      drops the task description
           --no-prompt-patient   drops the per-patient description
           --prompt-stats        adds the rolling signal stats

       python examples/mitbih_ecg_boundary_medtsllm.py \\
           --root /path/to/mitdb --no-prompt-patient
       python examples/mitbih_ecg_boundary_medtsllm.py \\
           --root /path/to/mitdb --no-prompt-dataset --prompt-stats
"""

import argparse

import numpy as np
import torch

from pyhealth.datasets import (
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import MedTsLLM
from pyhealth.trainer import Trainer


def make_synthetic_dataset(
    n_patients: int = 10, seq_len: int = 256, n_channels: int = 2
):
    """Synthetic 2-channel ECG data."""
    samples = []
    for i in range(n_patients):
        for w in range(5):
            signal = np.random.randn(seq_len, n_channels).astype(np.float32)
            label = np.zeros(seq_len, dtype=np.float32)
            # Sparse boundary labels (~3% density) to mimic real R-peaks
            idx = np.random.choice(seq_len, size=seq_len // 40, replace=False)
            label[idx] = 1
            samples.append({
                "patient_id": f"p{i}",
                "visit_id": f"v{w}",
                "signal": signal,
                "label": label,
                "description": "",
            })
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "tensor"},
        dataset_name="synthetic_mitbih",
    )


def make_real_dataset(root: str, seq_len: int = 256, step_size: int = 256):
    """Load MIT-BIH and apply ECGBoundaryDetection.

    Uses ``preprocess=True`` so the 30-minute wfdb records are decoded
    and downsampled once into ``{root}/processed/*.npz``. Subsequent
    runs skip wfdb entirely.
    """
    from pyhealth.datasets import MITBIHDataset
    from pyhealth.tasks import ECGBoundaryDetection

    dataset = MITBIHDataset(
        root=root,
        preprocess=True,
        downsample_factor=3,
        trim=True,
    )
    task = ECGBoundaryDetection(window_size=seq_len, step_size=step_size)
    return dataset.set_task(task)


def main():
    parser = argparse.ArgumentParser(
        description="MIT-BIH R-peak boundary detection with MedTsLLM"
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    # Ablation knobs — each flag disables one prompt component
    parser.add_argument("--no-prompt-dataset", action="store_true")
    parser.add_argument("--no-prompt-task", action="store_true")
    parser.add_argument("--no-prompt-patient", action="store_true")
    parser.add_argument("--prompt-stats", action="store_true")
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    seq_len = 256
    n_features = 2

    if args.synthetic or args.root is None:
        print("Using synthetic data")
        sample_dataset = make_synthetic_dataset(
            n_patients=10, seq_len=seq_len, n_channels=n_features
        )
        word_embeddings = torch.randn(100, 64)
        backbone = None
        epochs = 3
    else:
        print(f"Loading MIT-BIH from {args.root}")
        sample_dataset = make_real_dataset(args.root, seq_len=seq_len)
        word_embeddings = None
        backbone = args.backbone or "openai-community/gpt2"
        epochs = args.epochs

    train_ds, _, test_ds = split_by_patient(
        sample_dataset, ratios=[0.8, 0.0, 0.2]
    )
    train_loader = get_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    test_loader = get_dataloader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    model = MedTsLLM(
        dataset=sample_dataset,
        task="segmentation",
        seq_len=seq_len,
        n_features=n_features,
        covariate_mode="concat",
        d_model=32,
        d_ff=64,
        n_heads=8,
        num_tokens=1024,
        patch_len=16,
        stride=8,
        dataset_description=(
            "The MIT-BIH Arrhythmia Database contains excerpts of "
            "two-channel ambulatory ECG from a mixed population of "
            "inpatients and outpatients, digitized at 360 samples "
            "per second per channel."
        ),
        backbone=backbone,
        word_embeddings=word_embeddings,
        prompt_dataset=not args.no_prompt_dataset,
        prompt_task=not args.no_prompt_task,
        prompt_patient=not args.no_prompt_patient,
        prompt_stats=args.prompt_stats,
    )

    trainer = Trainer(model=model, device=device, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": args.lr},
    )


if __name__ == "__main__":
    main()
