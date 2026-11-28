"""WatchSleepNet sleep staging with one-factor-at-a-time ablations.

The script requires a real DREAMT, SHHS, or MESA cohort via ``--root`` and
exercises the full PyHealth path: dataset, IBI task, patient split,
:class:`~pyhealth.models.WatchSleepNet`, ``Trainer.train``, ``Trainer.evaluate``,
checkpoint save/load, and re-evaluation. There is no synthetic-data fallback.

Command:
    python examples/watchsleepnet/watchsleepnet_sleep_staging.py --dataset dreamt \
        --root /path/to/dreamt --ablation all --epochs 5 \
        --output-dir output/watchsleepnet-staging
    python examples/watchsleepnet/watchsleepnet_sleep_staging.py --dataset shhs \
        --root /path/to/shhs --ablation none --epochs 200

Each ablation reuses the seed and the patient split. The 1100/500
``max_epochs`` values are passed to the task unchanged. ``--loss`` selects the
training objective; ``ce`` is the reference default.

The per-class F1 and predicted-class distribution expose NREM-majority
collapse: accuracy can look reasonable while Wake and REM F1 are zero. Results
are written to ``results.csv`` and ``results.json`` in ``--output-dir``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .watchsleepnet_common import (
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
        model_kwargs_from_args,
        predict,
        print_table,
        resample_dataset,
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
        build_model_from_checkpoint,
        build_samples,
        build_task,
        existing_dir,
        fit,
        get_loaders,
        model_kwargs_from_args,
        predict,
        print_table,
        resample_dataset,
        save_checkpoint,
        split_dataset,
        summarize_predictions,
        write_results,
    )

TABLE_COLUMNS = (
    "condition",
    "status",
    "loss",
    "accuracy",
    "f1_wake",
    "f1_nrem",
    "f1_rem",
    "pred_wake",
    "pred_nrem",
    "pred_rem",
    "reload_max_diff",
)
METRIC_KEYS = (
    "accuracy",
    "f1_macro",
    "f1_wake",
    "f1_nrem",
    "f1_rem",
    "pred_wake",
    "pred_nrem",
    "pred_rem",
)
ABLATIONS = (
    "none",
    "signal",
    "points",
    "length",
    "architecture",
    "quantization",
    "all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the staging ablation example.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.

    Examples:
        >>> parse_args(["--root", ".", "--ablation", "none"]).ablation
        'none'
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=("dreamt", "shhs", "mesa", "isruc", "sleep_accel"),
        default="dreamt",
    )
    parser.add_argument(
        "--root", required=True, type=existing_dir, help="real dataset root"
    )
    parser.add_argument("--task-max-epochs", type=int, default=None)
    parser.add_argument(
        "--signal-source",
        choices=("psg_100hz", "wearable_64hz"),
        default="psg_100hz",
    )
    parser.add_argument("--points-per-epoch", type=int, default=750)
    parser.add_argument(
        "--scorer", choices=("expert1", "expert2"), default="expert1"
    )
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    add_training_args(parser)
    add_model_args(parser)
    return parser.parse_args(argv)


def _task_length(args: argparse.Namespace, default_value: int) -> int:
    """Returns the epoch cap passed to the IBI task.

    Args:
        args: Parsed namespace with an optional ``--task-max-epochs``.
        default_value: The default 1100 or 500 epoch cap.

    Returns:
        The explicit override, or the default value.

    Examples:
        >>> args = parse_args(["--root", "."])
        >>> _task_length(args, 1100), _task_length(args, 500)
        (1100, 500)
    """
    if args.task_max_epochs is not None:
        return args.task_max_epochs
    return default_value


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Builds the one-factor-at-a-time ablation conditions.

    Args:
        args: Parsed namespace selecting one axis or ``"all"``.

    Returns:
        Condition dictionaries with task, model, and quantization settings.
        Duplicate effective settings are removed.

    Examples:
        >>> conditions = build_conditions(parse_args(
        ...     ["--root", ".", "--ablation", "points"]
        ... ))
        >>> [c["points"] for c in conditions]
        [750, 375, 150]
    """
    base = {
        "name": "baseline",
        "signal_source": args.signal_source,
        "points": args.points_per_epoch,
        "max_epochs": _task_length(args, 1100),
        "scorer": args.scorer,
        "model_kwargs": model_kwargs_from_args(args),
        "quantization": args.quantization,
        "loss": args.loss,
    }
    if args.ablation == "none":
        return [base]

    candidates = []
    if args.ablation in ("signal", "all") and args.dataset == "dreamt":
        for source in ("psg_100hz", "wearable_64hz"):
            candidates.append({**base, "name": f"signal_{source}",
                               "signal_source": source})
    if args.ablation in ("points", "all"):
        for points in (750, 375, 150):
            candidates.append({**base, "name": f"points_{points}",
                               "points": points})
    if args.ablation in ("length", "all"):
        for default_value in (1100, 500):
            candidates.append({
                **base,
                "name": f"max_epochs_{default_value}",
                "max_epochs": _task_length(args, default_value),
            })
    if args.ablation in ("architecture", "all"):
        variants = (
            ("hidden_dim_64", {"hidden_dim": 64}),
            ("hidden_dim_256", {"hidden_dim": 256}),
            ("tcn_layers_1", {"tcn_layers": 1}),
            ("tcn_layers_3", {"tcn_layers": 3}),
            ("attention_off", {"use_attention": False}),
            ("attention_on", {"use_attention": True}),
        )
        for name, changes in variants:
            candidates.append({
                **base,
                "name": name,
                "model_kwargs": {**base["model_kwargs"], **changes},
            })
    if args.ablation in ("quantization", "all"):
        for mode in ("fp32", "bf16", "fp16"):
            candidates.append({**base, "name": f"quantization_{mode}",
                               "quantization": mode})

    conditions = [base]
    seen = {_signature(base)}
    for condition in candidates:
        signature = _signature(condition)
        if signature not in seen:
            seen.add(signature)
            conditions.append(condition)
    return conditions


def _signature(condition: dict[str, Any]) -> str:
    """Returns a hashable signature for one condition.

    Args:
        condition: Condition dictionary from :func:`build_conditions`.

    Returns:
        A JSON string of the effective settings, excluding the display name.

    Examples:
        >>> _signature({"name": "a", "points": 750}) == _signature(
        ...     {"name": "b", "points": 750})
        True
    """
    payload = {k: v for k, v in condition.items() if k != "name"}
    return str(payload)


def _safe_name(name: str) -> str:
    """Converts a condition name into a filesystem-safe stem.

    Args:
        name: Human-readable condition name.

    Returns:
        A name containing only alphanumerics, dashes, and underscores.

    Examples:
        >>> _safe_name("signal_psg_100hz")
        'signal_psg_100hz'
    """
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def run_condition(
    args: argparse.Namespace,
    condition: dict[str, Any],
    root: Path,
    cache_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Trains, evaluates, checkpoints, and reloads one ablation condition.

    Args:
        args: Parsed namespace with training flags.
        condition: One condition from :func:`build_conditions`.
        root: Real dataset root.
        cache_dir: Writable PyHealth cache location.
        output_dir: Directory for checkpoints.

    Returns:
        The metric row and per-epoch validation history.

    Examples:
        >>> run_condition(None, {}, *[Path("x")] * 3) if False else "ready"
        'ready'
    """
    quantization = condition["quantization"]
    base_dataset = build_dataset(
        args.dataset, root, cache_dir=str(cache_dir)
    )
    task = build_task(
        args.dataset,
        points_per_epoch=condition["points"],
        max_epochs=condition["max_epochs"],
        signal_source=condition["signal_source"],
        peak_backend="auto",
        scorer=condition["scorer"],
    )
    samples = build_samples(base_dataset, task, num_workers=args.num_workers)
    if condition["points"] != 750:
        samples = resample_dataset(samples, 750)
    parts = split_dataset(samples, seed=args.seed)
    loaders = get_loaders(
        parts, batch_size=args.batch_size, seed=args.seed
    )
    model_kwargs = dict(condition["model_kwargs"])
    model = build_model(
        parts[0], seed=args.seed, model_kwargs=model_kwargs,
        quantization=quantization, loss=condition["loss"],
    )
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

    config = {
        "dataset": args.dataset,
        "root": str(root),
        "task": {
            "signal_source": condition["signal_source"],
            "points_per_epoch": condition["points"],
            "max_epochs": condition["max_epochs"],
            "scorer": condition["scorer"],
        },
        "model": model_kwargs,
        "quantization": quantization,
        "loss": condition["loss"],
        "split_seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    checkpoint = save_checkpoint(
        output_dir / "checkpoints" / f"{_safe_name(condition['name'])}.pt",
        model,
        config,
        metrics,
    )
    reloaded, _ = build_model_from_checkpoint(
        checkpoint, parts[0], seed=args.seed,
        overrides={"loss": condition["loss"]},
    )
    reloaded_true, reloaded_prob = predict(
        reloaded, loaders[2], device=args.device
    )
    reloaded_metrics = summarize_predictions(reloaded_true, reloaded_prob)
    reload_diff = max(
        abs(float(metrics[key]) - float(reloaded_metrics[key]))
        for key in METRIC_KEYS
    )

    row: dict[str, Any] = {
        "condition": condition["name"],
        "status": "ok",
        "signal_source": condition["signal_source"],
        "scorer": condition["scorer"],
        "points_per_epoch": condition["points"],
        "task_max_epochs": condition["max_epochs"],
        "hidden_dim": model_kwargs.get("hidden_dim", 256),
        "tcn_layers": model_kwargs.get("tcn_layers", 3),
        "use_attention": model_kwargs.get("use_attention", True),
        "quantization": quantization,
        "loss": condition["loss"],
        "val_accuracy": float(val_scores.get("accuracy", 0.0)),
        "reload_max_diff": reload_diff,
        "checkpoint": str(checkpoint),
        **metrics,
    }
    return row, history


def main(argv: list[str] | None = None) -> int:
    """Runs the real-data staging ablation and writes result files.

    Args:
        argv: Optional argument list.

    Returns:
        Zero when at least one condition completes, otherwise one.

    Examples:
        >>> main(["--root", ".", "--epochs", "0",
        ...       "--ablation", "none"]) if False else "ready"
        'ready'
    """
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    histories: list[tuple[str, list[dict[str, float]]]] = []

    root = Path(args.root)
    cache_dir = output_dir / "cache"
    for condition in build_conditions(args):
        try:
            row, history = run_condition(
                args, condition, root, cache_dir, output_dir
            )
        except Exception as exc:  # noqa: BLE001 - record condition errors.
            name = condition["name"]
            rows.append({
                "condition": name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "quantization": condition["quantization"],
                "loss": condition["loss"],
            })
            continue
        rows.append(row)
        histories.append((condition["name"], history))

    history_rows = [
        {"condition": name, **entry} for name, history in histories
        for entry in history
    ]
    pd.DataFrame(history_rows, columns=["condition", "epoch", "accuracy",
                                        "f1_macro", "cohen_kappa", "loss"]).to_csv(
        output_dir / "training_history.csv", index=False
    )
    write_results(
        output_dir,
        rows,
        {
            "dataset": args.dataset,
            "root": str(root),
            "scorer": args.scorer,
            "loss": args.loss,
            "seed": args.seed,
            "ablation": args.ablation,
            "task_max_epochs": {
                "1100": _task_length(args, 1100),
                "500": _task_length(args, 500),
            },
        },
    )
    print_table(rows, TABLE_COLUMNS)
    return 0 if any(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
