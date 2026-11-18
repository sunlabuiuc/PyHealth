"""Run paired DeepHit NeuralFG/Fine-Gray cross-validation from PyHealth root.

Full five-fold reproduction:
    python examples/deephit_competing_risks_cv.py

Short two-fold integration test (does not produce final performance estimates):
    python examples/deephit_competing_risks_cv.py --smoke
"""

from __future__ import annotations

import argparse

from neuralfg.cross_validation import CVConfig, run_cross_validation
from neuralfg.data import load_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 folds / 600 rows / 2 epochs")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-root", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = CVConfig(
        max_samples=600 if args.smoke else args.max_samples,
        n_splits=2 if args.smoke else args.folds,
        epochs=2 if args.smoke else args.epochs,
        patience=2 if args.smoke else args.patience,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        seed=args.seed,
        batch_size=args.batch_size,
        output_root=args.output_root,
    )
    print("Cross-validation configuration:", config)
    df = load_data()
    run_cross_validation(df, config)


if __name__ == "__main__":
    main()
