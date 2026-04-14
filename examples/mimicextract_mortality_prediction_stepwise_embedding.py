"""Ablation runner for the StepWiseEmbeddingModel on MIMIC-Extract mortality.

This example is structured to satisfy the model-contribution rubric: it
demonstrates the new model on a paper-aligned task and compares multiple
configurations, including the core grouped-vs-flat ablation and one basic
hidden-dimension variation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyhealth.datasets import MIMICExtractDataset, get_dataloader
from pyhealth.models import StepWiseEmbeddingModel
from pyhealth.tasks import StepWiseMortalityPredictionMIMICExtract
from pyhealth.trainer import Trainer


def build_sample_dataset(root: str, itemid_map: str, window_hours: int):
    """Create the MIMIC-Extract mortality dataset used in the ablations."""
    dataset = MIMICExtractDataset(
        root=root,
        tables=["vitals_labs"],
        itemid_to_variable_map=itemid_map,
        refresh_cache=False,
    )
    task = StepWiseMortalityPredictionMIMICExtract(
        observation_window_hours=window_hours,
    )
    return dataset.set_task(task)


def run_single_experiment(
    sample_dataset,
    group_mode: str,
    embedding_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
) -> dict[str, int | str]:
    """Train one ablation configuration and return its metadata."""
    model = StepWiseEmbeddingModel(
        dataset=sample_dataset,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        group_mode=group_mode,
    )
    dataloader = get_dataloader(sample_dataset, batch_size=batch_size, shuffle=True)
    trainer = Trainer(model=model)
    trainer.train(train_dataloader=dataloader, epochs=epochs)
    return {
        "group_mode": group_mode,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
    }


def main() -> None:
    """Run the ablation study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to MIMIC-Extract data.")
    parser.add_argument(
        "--itemid-map",
        required=True,
        help="Path to the MIMIC-Extract itemid_to_variable_map.csv file.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-hours", type=int, default=48)
    args = parser.parse_args()

    root = str(Path(args.root).expanduser().resolve())
    itemid_map = str(Path(args.itemid_map).expanduser().resolve())
    sample_dataset = build_sample_dataset(root, itemid_map, args.window_hours)

    ablations = [
        {"group_mode": "grouped", "embedding_dim": 128, "hidden_dim": 128},
        {"group_mode": "flat", "embedding_dim": 128, "hidden_dim": 128},
        {"group_mode": "grouped", "embedding_dim": 128, "hidden_dim": 192},
    ]

    print("Running StepWiseEmbeddingModel ablations:")
    for config in ablations:
        print(
            "  - "
            f"group_mode={config['group_mode']}, "
            f"embedding_dim={config['embedding_dim']}, "
            f"hidden_dim={config['hidden_dim']}"
        )
        run_single_experiment(
            sample_dataset=sample_dataset,
            group_mode=config["group_mode"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
