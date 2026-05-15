"""Ablation script for ShiftLSTM on synthetic time-varying data.

This example is designed to satisfy the course project's
"Ablation Study / Example Usage" requirement while staying lightweight enough
for local smoke runs.

It compares ShiftLSTM with different segment counts K on synthetic data
generated following Section 4.1 of Oh et al. (2019):

  - K = 1 acts as the shared-parameter LSTM baseline
  - K > 1 relaxes parameter sharing over time
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.models import ShiftLSTM
from pyhealth.trainer import Trainer


THIS_DIR = Path(__file__).resolve().parent
SYNTHETIC_MODULE_PATH = THIS_DIR / "synthetic" / "shift_lstm_synthetic_data.py"


def load_synthetic_module():
    """Loads the synthetic generator module from examples/synthetic."""

    spec = importlib.util.spec_from_file_location(
        "shift_lstm_synthetic_data", SYNTHETIC_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_sample_dataset(
    num_samples: int,
    seq_len: int,
    num_features: int,
    lookback: int,
    delta: float,
    seed: int,
):
    """Builds a PyHealth SampleDataset from synthetic arrays."""

    synth = load_synthetic_module()
    config = synth.SyntheticConfig(
        N=num_samples,
        T=seq_len,
        d=num_features,
        l=lookback,
        delta=delta,
        seed=seed,
    )
    bundle = synth.generate_synthetic_arrays(config)
    samples = synth.to_pyhealth_samples(bundle["x"], bundle["y"])
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"signal": "timeseries"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_shift_lstm",
    )
    return dataset


def run_single_experiment(
    dataset,
    num_segments: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
):
    """Runs one ShiftLSTM configuration and returns validation/test metrics."""

    train_dataset, val_dataset, test_dataset = split_by_patient(
        dataset, [0.7, 0.15, 0.15], seed=seed
    )
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ShiftLSTM(
        dataset=dataset,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_segments=num_segments,
        dropout=dropout,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            metrics=["accuracy", "roc_auc", "pr_auc", "f1"],
            enable_logging=False,
            output_path=tmpdir,
            exp_name=f"shift_lstm_k{num_segments}",
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            optimizer_params={"lr": learning_rate},
            monitor="roc_auc",
            monitor_criterion="max",
            load_best_model_at_last=False,
        )
        val_scores = trainer.evaluate(val_loader)
        test_scores = trainer.evaluate(test_loader)

    return {
        "num_segments": num_segments,
        "val": val_scores,
        "test": test_scores,
    }


def format_results_table(results: list[dict]) -> str:
    """Formats a compact human-readable table."""

    header = (
        f"{'Model':<18} {'K':<4} {'Val AUROC':<10} "
        f"{'Test AUROC':<11} {'Test AUPRC':<11} {'Test Acc':<9}"
    )
    rows = [header, "-" * len(header)]
    for result in results:
        k = result["num_segments"]
        name = "LSTM baseline" if k == 1 else "ShiftLSTM"
        val = result["val"]
        test = result["test"]
        rows.append(
            f"{name:<18} {k:<4} {val['roc_auc']:<10.4f} "
            f"{test['roc_auc']:<11.4f} {test['pr_auc']:<11.4f} {test['accuracy']:<9.4f}"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ShiftLSTM ablations on synthetic sequence classification."
    )
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--num-features", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Segment counts K to compare.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save the full metrics as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = build_sample_dataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_features=args.num_features,
        lookback=args.lookback,
        delta=args.delta,
        seed=args.seed,
    )

    results = []
    for num_segments in args.segments:
        result = run_single_experiment(
            dataset=dataset,
            num_segments=num_segments,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
        )
        results.append(result)

    print(format_results_table(results))

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to: {save_path}")


if __name__ == "__main__":
    main()
