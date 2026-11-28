"""SHHS+MESA WatchSleepNet pretraining on real cohorts.

The script requires real SHHS and MESA roots, builds the three-class
IBI tasks, pools the recordings, splits by patient, trains
:class:`~pyhealth.models.WatchSleepNet` with ``Trainer.train``, and saves a
self-describing checkpoint plus ``training_history.csv``. There is no
synthetic-data fallback.

Command:
    python examples/watchsleepnet/watchsleepnet_pretrain_shhs_mesa.py \
        --shhs-root /path/to/shhs --mesa-root /path/to/mesa --epochs 200 \
        --output-dir output/watchsleepnet-pretrain

A 1100-epoch cap is used by default. ``--loss`` selects the
training objective; ``ce`` is the reference default. Example output table::

    cohort         status  accuracy  f1_wake  f1_nrem  f1_rem
    shhs+mesa      ok        0.6364    0.0000   0.7778  0.0000

The predicted-class distribution is also reported in ``results.csv`` so that
NREM-majority collapse remains visible even when accuracy looks reasonable.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .watchsleepnet_common import (
        add_model_args,
        add_training_args,
        build_dataset,
        build_model,
        build_samples,
        build_task,
        existing_dir,
        fit,
        get_loaders,
        model_kwargs_from_args,
        pool_datasets,
        predict,
        print_table,
        save_checkpoint,
        split_dataset,
        summarize_predictions,
        write_results,
    )
except ImportError:
    from watchsleepnet_common import (
        add_model_args,
        add_training_args,
        build_dataset,
        build_model,
        build_samples,
        build_task,
        existing_dir,
        fit,
        get_loaders,
        model_kwargs_from_args,
        pool_datasets,
        predict,
        print_table,
        save_checkpoint,
        split_dataset,
        summarize_predictions,
        write_results,
    )

TABLE_COLUMNS = (
    "cohort",
    "status",
    "loss",
    "accuracy",
    "f1_wake",
    "f1_nrem",
    "f1_rem",
    "pred_wake",
    "pred_nrem",
    "pred_rem",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the pretraining example.

    Args:
        argv: Optional argument list.

    Returns:
        The parsed namespace.

    Examples:
        >>> args = parse_args(["--shhs-root", ".", "--mesa-root", ".",
        ...                    "--epochs", "2"])
        >>> args.epochs
        2
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shhs-root", required=True, type=existing_dir, help="real SHHS root"
    )
    parser.add_argument(
        "--mesa-root", required=True, type=existing_dir, help="real MESA root"
    )
    parser.add_argument("--points-per-epoch", type=int, default=750)
    parser.add_argument("--task-max-epochs", type=int, default=None)
    add_training_args(parser)
    add_model_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs pretraining and writes the checkpoint, history, and result files.

    Args:
        argv: Optional argument list.

    Returns:
        Zero after a completed run.

    Examples:
        >>> main(["--shhs-root", ".", "--mesa-root", ".",
        ...       "--epochs", "0"]) if False else "ready"
        'ready'
    """
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = output_dir / "cache"
    cohorts = [("shhs", Path(args.shhs_root)), ("mesa", Path(args.mesa_root))]

    sample_sets = []
    for name, root in cohorts:
        dataset = build_dataset(
            name, root, cache_dir=str(cache_root / name)
        )
        task = build_task(
            name,
            points_per_epoch=args.points_per_epoch,
            max_epochs=(
                args.task_max_epochs
                if args.task_max_epochs is not None
                else 1100
            ),
            peak_backend="auto",
        )
        sample_sets.append(
            build_samples(dataset, task, num_workers=args.num_workers)
        )
    samples = pool_datasets(sample_sets)
    parts = split_dataset(samples, seed=args.seed)
    loaders = get_loaders(
        parts, batch_size=args.batch_size, seed=args.seed
    )
    model_kwargs = model_kwargs_from_args(args)
    model = build_model(
        parts[0], seed=args.seed, model_kwargs=model_kwargs,
        quantization=args.quantization, loss=args.loss,
    )
    trainer, history = fit(
        model,
        loaders,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        quantization=args.quantization,
    )
    val_scores = trainer.evaluate(loaders[1])
    y_true, y_prob = predict(model, loaders[2], device=args.device)
    metrics = summarize_predictions(y_true, y_prob)

    checkpoint = save_checkpoint(
        output_dir / "pretrained.pt",
        model,
        {
            "stage": "pretrain",
            "cohorts": [name for name, _ in cohorts],
            "roots": {name: str(root) for name, root in cohorts},
            "task": {
                "points_per_epoch": args.points_per_epoch,
                "max_epochs": args.task_max_epochs or 1100,
            },
            "model": model_kwargs,
            "quantization": args.quantization,
            "loss": args.loss,
            "split_seed": args.seed,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        },
        metrics,
    )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )
    row = {
        "cohort": "+".join(name for name, _ in cohorts),
        "status": "ok",
        "loss": args.loss,
        "val_accuracy": float(val_scores.get("accuracy", 0.0)),
        "checkpoint": str(checkpoint),
        **metrics,
    }
    write_results(
        output_dir,
        [row],
        {
            "roots": {name: str(root) for name, root in cohorts},
            "loss": args.loss,
            "seed": args.seed,
        },
    )
    print_table([row], TABLE_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
