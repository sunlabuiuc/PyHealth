"""Run paper-aligned NeuralFG experiments without touching prior CV outputs.

Put at PyHealth/examples/deephit_paper_reproduction.py.

Examples:
  python examples/deephit_paper_reproduction.py --pilot
  python examples/deephit_paper_reproduction.py --mode fixed
  python examples/deephit_paper_reproduction.py --mode search --trials 5
  python examples/deephit_paper_reproduction.py --mode search --trials 100

WARNING: --mode search --trials 100 runs 500 neural fits with 5 folds,
which can be extremely expensive on a Mac CPU. Pilot first.
"""
from __future__ import annotations

import argparse
from neuralfg.data import load_data
from neuralfg.paper_reproduction import PaperRunConfig, run_paper_reproduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--mode", choices=["fixed", "search"], default="fixed")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = PaperRunConfig(
        max_samples=600 if args.pilot else args.max_samples,
        folds=2 if args.pilot else args.folds,
        mode="search" if args.pilot else args.mode,
        trials=2 if args.pilot else args.trials,
        max_epochs=2 if args.pilot else args.epochs,
        patience=2 if args.pilot else args.patience,
        seed=args.seed,
        output_root=args.output_root,
    )
    print("Paper-aligned configuration:", config)
    run_paper_reproduction(load_data(), config)


if __name__ == "__main__":
    main()
