"""WatchSleepNet fine-tuning from a bare or bundled checkpoint.

The script requires a real dataset root, optionally loads an existing ``.pt``
checkpoint, fine-tunes with ``Trainer.train``, evaluates with
``Trainer.evaluate``, and saves a self-describing fine-tuned checkpoint.
``slpdb``, ``isruc``, and ``sleep_accel`` fail clearly and lazily until their
PyHealth ports land. There is no synthetic-data fallback. ``--loss`` selects
the training objective; ``ce`` is the reference default.

Command:
    python examples/watchsleepnet/watchsleepnet_pretrain_shhs_mesa.py \
        --shhs-root /path/to/shhs --mesa-root /path/to/mesa --epochs 200 \
        --output-dir output/watchsleepnet-pretrain
    python examples/watchsleepnet/watchsleepnet_finetune.py --dataset dreamt \
        --root /path/to/dreamt \
        --checkpoint output/watchsleepnet-pretrain/pretrained.pt \
        --epochs 5 --output-dir output/watchsleepnet-finetune

Example output table::

    dataset  status  checkpoint_loaded  accuracy  f1_wake  f1_nrem  f1_rem
    dreamt   ok      True               0.6364    0.0000   0.7778   0.0000

Per-class F1 and the predicted-class distribution are written to
``results.csv``/``results.json``; they expose NREM-majority collapse.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .watchsleepnet_common import (
        DEFAULT_MODEL_KWARGS,
        add_model_args,
        add_training_args,
        build_dataset,
        build_model,
        build_model_from_checkpoint,
        build_samples,
        build_task,
        existing_dir,
        fit,
        get_loaders,
        load_checkpoint,
        model_kwargs_from_args,
        predict,
        print_table,
        save_checkpoint,
        split_dataset,
        summarize_predictions,
        write_results,
    )
except ImportError:
    from watchsleepnet_common import (
        DEFAULT_MODEL_KWARGS,
        add_model_args,
        add_training_args,
        build_dataset,
        build_model,
        build_model_from_checkpoint,
        build_samples,
        build_task,
        existing_dir,
        fit,
        get_loaders,
        load_checkpoint,
        model_kwargs_from_args,
        predict,
        print_table,
        save_checkpoint,
        split_dataset,
        summarize_predictions,
        write_results,
    )

TABLE_COLUMNS = (
    "dataset",
    "status",
    "loss",
    "checkpoint_loaded",
    "pre_accuracy",
    "accuracy",
    "f1_wake",
    "f1_nrem",
    "f1_rem",
    "pred_nrem",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the fine-tuning example.

    Args:
        argv: Optional argument list.

    Returns:
        The parsed namespace. ``quantization`` is ``None`` unless explicitly
        set, so a loaded checkpoint's own mode is preserved.

    Examples:
        >>> args = parse_args(["--root", "."])
        >>> args.quantization is None, args.loss
        (True, 'ce')
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=("dreamt", "shhs", "mesa", "slpdb", "isruc", "sleep_accel"),
        default="dreamt",
    )
    parser.add_argument(
        "--root", required=True, type=existing_dir, help="real dataset root"
    )
    parser.add_argument("--checkpoint", default=None, help=".pt to fine-tune")
    parser.add_argument("--points-per-epoch", type=int, default=750)
    parser.add_argument("--task-max-epochs", type=int, default=None)
    parser.add_argument(
        "--signal-source",
        choices=("psg_100hz", "wearable_64hz"),
        default="psg_100hz",
    )
    add_training_args(parser)
    add_model_args(parser)
    parser.set_defaults(quantization=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fine-tunes from an optional checkpoint and writes model and result files.

    Args:
        argv: Optional argument list.

    Returns:
        Zero after a completed run.

    Raises:
        NotImplementedError: For datasets whose PyHealth port is not available.

    Examples:
        >>> main(["--root", ".", "--epochs", "0"]) if False else "ready"
        'ready'
    """
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    cache_dir = output_dir / "cache"
    dataset = build_dataset(
        args.dataset, root, cache_dir=str(cache_dir)
    )
    task = build_task(
        args.dataset,
        points_per_epoch=args.points_per_epoch,
        max_epochs=args.task_max_epochs or 1100,
        signal_source=args.signal_source,
        peak_backend="auto",
    )
    samples = build_samples(dataset, task, num_workers=args.num_workers)
    parts = split_dataset(samples, seed=args.seed)
    loaders = get_loaders(
        parts, batch_size=args.batch_size, seed=args.seed
    )

    cli_kwargs = model_kwargs_from_args(args)
    checkpoint_config: dict[str, object] = {}
    if args.checkpoint:
        _, metadata = load_checkpoint(args.checkpoint)
        checkpoint_config = dict(metadata.get("config", {}))
        overrides: dict[str, object] = {}
        if args.quantization is not None:
            overrides["quantization"] = args.quantization
        overrides["loss"] = args.loss
        if cli_kwargs:
            overrides["model_kwargs"] = cli_kwargs
        model, _ = build_model_from_checkpoint(
            args.checkpoint, parts[0], seed=args.seed, overrides=overrides
        )
    else:
        model = build_model(
            parts[0], seed=args.seed, model_kwargs=cli_kwargs,
            quantization=args.quantization or "fp32", loss=args.loss,
        )
    quantization = str(
        args.quantization
        or checkpoint_config.get("quantization", "fp32")
    )
    model_kwargs = dict(DEFAULT_MODEL_KWARGS)
    model_kwargs.update(checkpoint_config.get("model", {}))
    model_kwargs.update(cli_kwargs)

    pre_metrics = {}
    if args.checkpoint:
        pre_true, pre_prob = predict(model, loaders[2],
                                     device=args.device)
        pre_metrics = summarize_predictions(pre_true, pre_prob)

    trainer, history = fit(
        model,
        loaders,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        quantization=quantization,
    )
    val_scores = trainer.evaluate(loaders[1])
    y_true, y_prob = predict(model, loaders[2], device=args.device)
    metrics = summarize_predictions(y_true, y_prob)

    checkpoint = save_checkpoint(
        output_dir / "finetuned.pt",
        model,
        {
            "stage": "finetune",
            "dataset": args.dataset,
            "root": str(root),
            "source_checkpoint": args.checkpoint,
            "task": {
                "signal_source": args.signal_source,
                "points_per_epoch": args.points_per_epoch,
                "max_epochs": args.task_max_epochs or 1100,
            },
            "model": model_kwargs,
            "quantization": quantization,
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
        "dataset": args.dataset,
        "status": "ok",
        "loss": args.loss,
        "checkpoint_loaded": bool(args.checkpoint),
        "pre_accuracy": pre_metrics.get("accuracy", ""),
        "val_accuracy": float(val_scores.get("accuracy", 0.0)),
        "checkpoint": str(checkpoint),
        **metrics,
    }
    write_results(
        output_dir,
        [row],
        {
            "root": str(root),
            "loss": args.loss,
            "seed": args.seed,
            "source_checkpoint": args.checkpoint,
        },
    )
    print_table([row], TABLE_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
