from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlretrieve

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pyhealth.datasets import TCGACRCkDataset, get_dataloader
from pyhealth.models import TissueAwareSimCLR
from pyhealth.tasks import TCGACRCkMSIClassification
from pyhealth.trainer import Trainer


DATA_ROOT = Path.home() / "TCGA_CRCk"
CACHE_DIR = Path("/home/ubuntu/.cache/pyhealth_local")
CHECKPOINT_CACHE_DIR = Path.home() / ".cache" / "pyhealth_checkpoints"

# Keep downstream params close to the paper
BATCH_SIZE = 32
MAX_TILES = 1000
MAX_EPOCHS = 100
HIDDEN_DIM = 128
DROPOUT = 0.25
LR = 5e-3
MOMENTUM = 0.6
WEIGHT_DECAY = 1e-4
# PATIENCE = 50
POOLING = "attention"

# Runtime optimization only
TILE_CHUNK_SIZE = 1024
SEED = 42

# Ablation commands:
#
# 1) Main experiment: pretrained encoder + fine-tuning
# python /examples/tcga_crck_simclr/tcga_crck_msi_classification_tissue_aware_simclr.py \
#   --pretrain-from-checkpoint /path/to/checkpoint.ckpt
#
# 2) Ablation 1: no pretraining (random initialization)
# python /examples/tcga_crck_simclr/tcga_crck_msi_classification_tissue_aware_simclr.py
#
# 3) Ablation 2: pretrained encoder + frozen encoder
# python /examples/tcga_crck_simclr/tcga_crck_msi_classification_tissue_aware_simclr.py \
#   --pretrain-from-checkpoint /path/to/checkpoint.ckpt \
#   --freeze-encoder


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for downstream MSI classification.

    Returns:
        argparse.Namespace: Parsed arguments containing the optional
        pretrained checkpoint path and encoder freezing flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain-from-checkpoint",
        type=str,
        default=None,
        help="Local path or HTTP(S) URL for a pretrained encoder checkpoint.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the encoder during downstream training.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_spec: str | None) -> str | None:
    """Resolves a checkpoint spec into a usable local checkpoint path.

    If the input is an HTTP(S) URL, the checkpoint is downloaded into the
    local cache directory. If it is a local path, the file must already exist.

    Args:
        checkpoint_spec: Local filesystem path or HTTP(S) URL to a pretrained
            encoder checkpoint. If None, no checkpoint is used.

    Returns:
        str | None: Local path to the checkpoint, or None if no checkpoint
        was provided.

    Raises:
        FileNotFoundError: If a provided local checkpoint path does not exist.
    """
    if checkpoint_spec is None:
        return None

    parts = urlsplit(checkpoint_spec)
    if parts.scheme in {"http", "https"}:
        filename = Path(parts.path).name or "checkpoint.ckpt"
        CHECKPOINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = CHECKPOINT_CACHE_DIR / filename
        if not local_path.exists():
            print(f"Downloading checkpoint to {local_path}", flush=True)
            urlretrieve(checkpoint_spec, local_path)
        else:
            print(f"Using cached checkpoint at {local_path}", flush=True)
        return str(local_path)

    local_path = Path(checkpoint_spec).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {local_path}")
    return str(local_path)


def build_splits(sample_dataset):
    """Builds train, validation, and test splits from the task dataset.

    The function reads the `data_split` field from each sample, separates
    train and test data accordingly, and then creates a stratified validation
    split from the training partition.

    Args:
        sample_dataset: Task-specific PyHealth dataset produced by
            `set_task(...)`.

    Returns:
        tuple: A tuple of `(train_dataset, val_dataset, test_dataset)`.

    Raises:
        ValueError: If an unknown split label is encountered or if train/test
        samples are missing.
    """
    train_indices = []
    test_indices = []

    for i in range(len(sample_dataset)):
        split = str(sample_dataset[i]["data_split"]).strip().lower()
        if split in {"train", "training", "tr"}:
            train_indices.append(i)
        elif split in {"test", "testing", "te"}:
            test_indices.append(i)
        else:
            raise ValueError(f"Unknown data_split: {split}")

    if not train_indices or not test_indices:
        raise ValueError(
            f"Expected both train and test samples, got train={len(train_indices)}, "
            f"test={len(test_indices)}"
        )

    train_labels = [int(sample_dataset[i]["label"]) for i in train_indices]
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.2,
        random_state=SEED,
        stratify=train_labels,
    )

    train_dataset = sample_dataset.subset(train_indices)
    val_dataset = sample_dataset.subset(val_indices)
    test_dataset = sample_dataset.subset(test_indices)
    return train_dataset, val_dataset, test_dataset


def main() -> None:
    """Runs downstream MSI classification with a TissueAwareSimCLR encoder.

    This function:
    1. Parses command-line arguments.
    2. Sets random seeds.
    3. Checks for CUDA availability.
    4. Builds the TCGA-CRCk dataset and downstream MSI task.
    5. Splits the dataset into train/validation/test sets.
    6. Creates dataloaders, model, and trainer.
    7. Trains the model and evaluates on validation and test splits.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this run, but no GPU was found.")

    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint_path = resolve_checkpoint_path(args.pretrain_from_checkpoint)
    print(f"Resolved checkpoint: {checkpoint_path}", flush=True)

    base_dataset = TCGACRCkDataset(
        root=str(DATA_ROOT),
        cache_dir=str(CACHE_DIR),
    )

    sample_dataset = base_dataset.set_task(
        TCGACRCkMSIClassification(max_tiles=MAX_TILES)
    )
    print(f"Task dataset size: {len(sample_dataset)}", flush=True)

    train_dataset, val_dataset, test_dataset = build_splits(sample_dataset)
    print(
        f"Split sizes | train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )

    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TissueAwareSimCLR(
        dataset=train_dataset,
        checkpoint_path=checkpoint_path,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        freeze_encoder=args.freeze_encoder,
        pooling=POOLING,
        tile_chunk_size=TILE_CHUNK_SIZE,
        use_bf16=(device.type == "cuda"),
    ).to(device)

    trainer = Trainer(
        model=model,
        metrics=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        device=str(device),
        enable_logging=False,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=MAX_EPOCHS,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": LR, "betas": (MOMENTUM, 0.999)},
        weight_decay=WEIGHT_DECAY,
        monitor="balanced_accuracy",
        monitor_criterion="max",
#         patience=PATIENCE,
        load_best_model_at_last=True,
    )

    print("\nValidation metrics:", flush=True)
    print(trainer.evaluate(val_loader), flush=True)

    print("\nTest metrics:", flush=True)
    print(trainer.evaluate(test_loader), flush=True)


if __name__ == "__main__":
    main()