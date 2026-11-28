"""Evaluates a trained WatchSleepNet checkpoint on a real staging cohort.

The script rebuilds the IBI task from a real dataset root, loads a
bare ``.pt`` state dict or a self-describing checkpoint, evaluates on the
held-out patient split, prints the metric table, and writes ``results.csv``,
``results.json``, and a three-class ``confusion_matrix.png`` to ``--output-dir``.
DREAMT, SHHS, MESA, ISRUC and Sleep-Accel roots are supported; ISRUC uses the
checkpoint's scorer unless ``--scorer`` overrides it.

Command:
    python examples/watchsleepnet/watchsleepnet_evaluate.py \
        --checkpoint output/watchsleepnet-finetune/finetuned.pt \
        --dataset dreamt --root /path/to/dreamt \
        --output-dir output/watchsleepnet-evaluate

Metrics include accuracy, macro one-vs-rest AUROC, Cohen's kappa, macro F1,
weighted F1, and Wake/NREM/REM F1. Classes with no positive examples are skipped
when averaging AUROC, so a small held-out split remains scorable.
"""

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    from .watchsleepnet_common import (
        CLASS_NAMES,
        build_dataset,
        build_model_from_checkpoint,
        build_samples,
        build_task,
        existing_dir,
        get_loaders,
        load_checkpoint,
        predict,
        print_table,
        resample_dataset,
        split_dataset,
        summarize_predictions,
        write_results,
    )
except ImportError:
    from watchsleepnet_common import (
        CLASS_NAMES,
        build_dataset,
        build_model_from_checkpoint,
        build_samples,
        build_task,
        existing_dir,
        get_loaders,
        load_checkpoint,
        predict,
        print_table,
        resample_dataset,
        split_dataset,
        summarize_predictions,
        write_results,
    )

REPORT_COLUMNS = (
    "dataset",
    "n_epochs",
    "accuracy",
    "auroc",
    "cohen_kappa",
    "f1_macro",
    "f1_weighted",
    "f1_wake",
    "f1_nrem",
    "f1_rem",
)


def existing_file(value: str) -> Path:
    """Returns an existing checkpoint file supplied on the command line.

    Args:
        value: User-supplied checkpoint path.

    Returns:
        The expanded path.

    Raises:
        argparse.ArgumentTypeError: If the path is not an existing file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "model.pt"
        ...     _ = path.write_bytes(b"")
        ...     existing_file(str(path)).name
        'model.pt'
    """
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"{value!r} does not point to an existing checkpoint file"
        )
    return path


def macro_ovr_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes macro one-vs-rest AUROC over the three staging classes.

    Args:
        y_true: Integer labels in ``0..2``.
        y_prob: Class probabilities with shape ``(n, 3)``.

    Returns:
        The mean AUROC over classes with both positive and negative examples,
        or ``0.0`` when no such class exists.

    Examples:
        >>> import numpy as np
        >>> probs = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]])
        >>> macro_ovr_auroc(np.array([0, 1]), probs)
        1.0
    """
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_prob, dtype=np.float64).reshape(-1, len(CLASS_NAMES))
    if not truth.size:
        return 0.0
    values = []
    for index in range(len(CLASS_NAMES)):
        binary = (truth == index).astype(np.int64)
        if np.unique(binary).size < 2:
            continue
        values.append(float(roc_auc_score(binary, scores[:, index])))
    return float(np.mean(values)) if values else 0.0


def evaluation_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Computes the full evaluation table from flattened predictions.

    Args:
        y_true: Integer labels in ``0..2``.
        y_prob: Class probabilities with shape ``(n, 3)``.

    Returns:
        A flat dictionary with accuracy, AUROC, kappa, macro/weighted F1, and
        per-class F1. Empty inputs produce zeros.

    Examples:
        >>> import numpy as np
        >>> row = evaluation_metrics(np.array([0, 1, 2]), np.eye(3))
        >>> row["accuracy"], row["auroc"], row["cohen_kappa"]
        (1.0, 1.0, 1.0)
        >>> row["f1_macro"], row["f1_weighted"]
        (1.0, 1.0)
        >>> row["f1_wake"], row["f1_nrem"], row["f1_rem"]
        (1.0, 1.0, 1.0)
    """
    metrics = dict(summarize_predictions(y_true, y_prob))
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_prob, dtype=np.float64).reshape(-1, len(CLASS_NAMES))
    predicted = (
        scores.argmax(axis=1) if scores.size else np.empty(0, dtype=np.int64)
    )
    labels = list(range(len(CLASS_NAMES)))
    if not truth.size:
        metrics.update(
            auroc=0.0,
            cohen_kappa=0.0,
            f1_weighted=0.0,
        )
        return metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(truth, predicted, labels=labels)
    metrics["auroc"] = macro_ovr_auroc(truth, scores)
    metrics["cohen_kappa"] = float(kappa) if np.isfinite(kappa) else 0.0
    metrics["f1_weighted"] = float(
        f1_score(truth, predicted, labels=labels, average="weighted",
                 zero_division=0)
    )
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: str | Path
) -> Path:
    """Saves a labeled three-class confusion matrix as a PNG.

    Args:
        y_true: Integer labels in ``0..2``.
        y_pred: Predicted labels in ``0..2``.
        output_dir: Report directory to write ``confusion_matrix.png`` into.

    Returns:
        The saved image path.

    Examples:
        >>> import numpy as np
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = save_confusion_matrix(
        ...         np.array([0, 1]), np.array([0, 2]), tmp
        ...     )
        ...     path.name
        'confusion_matrix.png'
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        labels=list(range(len(CLASS_NAMES))),
    )
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("WatchSleepNet confusion matrix")
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path = output_dir / "confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for checkpoint evaluation.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace. ``seed`` defaults to the checkpoint's split seed.

    Examples:
        >>> args = parse_args([
        ...     "--checkpoint", __file__, "--root", ".",
        ...     "--output-dir", "out",
        ... ])
        >>> args.dataset, args.batch_size
        ('dreamt', 16)
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, type=existing_file,
        help="trained .pt state dict or bundled checkpoint",
    )
    parser.add_argument(
        "--dataset",
        choices=("dreamt", "shhs", "mesa", "isruc", "sleep_accel"),
        default="dreamt",
    )
    parser.add_argument(
        "--root", required=True, type=existing_dir, help="real dataset root"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="report directory"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="torch device; default auto")
    parser.add_argument("--signal-source", default=None)
    parser.add_argument("--signal-channel", default=None)
    parser.add_argument(
        "--scorer", choices=("expert1", "expert2"), default=None
    )
    parser.add_argument("--points-per-epoch", type=int, default=None)
    parser.add_argument("--task-max-epochs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluates one checkpoint and writes a complete report folder.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code, ``0`` on success.

    Examples:
        >>> main([
        ...     "--checkpoint", __file__, "--root", ".",
        ...     "--output-dir", "out",
        ... ]) if False else "ready"
        'ready'
    """
    args = parse_args(argv)
    _, metadata = load_checkpoint(args.checkpoint)
    config: dict[str, Any] = dict(metadata.get("config", {}))
    task_config: dict[str, Any] = dict(config.get("task", {}))
    seed = args.seed if args.seed is not None else int(config.get("split_seed", 0))
    points = (
        args.points_per_epoch
        if args.points_per_epoch is not None
        else int(task_config.get("points_per_epoch", 750))
    )
    max_epochs = (
        args.task_max_epochs
        if args.task_max_epochs is not None
        else int(task_config.get("max_epochs", 1100))
    )
    signal_source = args.signal_source or str(
        task_config.get("signal_source", "psg_100hz")
    )
    scorer = args.scorer or str(task_config.get("scorer", "expert1"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = build_dataset(args.dataset, args.root)
    task = build_task(
        args.dataset,
        points_per_epoch=points,
        max_epochs=max_epochs,
        signal_source=signal_source,
        signal_channel=args.signal_channel,
        scorer=scorer,
    )
    samples = build_samples(dataset, task, num_workers=args.num_workers)
    if points != 750:
        samples = resample_dataset(samples, 750)
    parts = split_dataset(samples, seed=seed)
    loader = get_loaders(parts, batch_size=args.batch_size, seed=seed)[2]

    model, _ = build_model_from_checkpoint(
        args.checkpoint, parts[0], seed=seed
    )
    target = args.device or str(model.device)
    model.to(target)
    y_true, y_prob = predict(model, loader, device=target)
    y_pred = (
        y_prob.argmax(axis=1) if y_prob.size else np.empty(0, dtype=np.int64)
    )

    row = {"dataset": args.dataset, **evaluation_metrics(y_true, y_prob)}
    image_path = save_confusion_matrix(y_true, y_pred, args.output_dir)
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    ).tolist()
    write_results(
        args.output_dir,
        [row],
        {
            "checkpoint": str(args.checkpoint),
            "root": str(args.root),
            "split": "test",
            "seed": seed,
            "task": {
                "signal_source": signal_source,
                "points_per_epoch": points,
                "max_epochs": max_epochs,
                "scorer": scorer,
            },
            "confusion_matrix": matrix,
            "confusion_matrix_image": image_path.name,
        },
    )
    print_table([row], REPORT_COLUMNS)
    print(f"confusion matrix: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
