"""DREAMT-only WatchSleepNet training example.

This example demonstrates patient-level cross-validation for three-class
sleep staging (Wake, NREM, REM) using 100 Hz DREAMT PSG files.

It does not reproduce the SHHS/MESA pretraining pipeline from the paper.

This example uses a lightweight model configuration for demonstration.
It is not the full research training configuration.

Architectural options support controlled comparisons using --use-attention,
--no-tcn, --residual-attention, and configurable model dimensions.
Use --loss unweighted to disable inverse-frequency class weighting.
These options enable new experiments; they do not automatically reproduce
the complete research evaluation protocol.

Use --context-seconds 5 to enable experimental retrospective
context-aware IBI extraction. The default of zero preserves
independent epochwise extraction.

Example:
    python examples/dreamt_sleep_staging_watchsleepnet.py \\
        --root /path/to/dreamt-2.2.0 \\
        --cache-dir /path/to/cache \\
        --patient-ids S002 S003 S004 S005 \\
        --epochs 1
"""

import argparse
import copy

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from pyhealth.datasets import DREAMTDataset, create_sample_dataset, get_dataloader
from pyhealth.models import WatchSleepNet
from pyhealth.tasks.sleep_staging_dreamt_sequence import SleepStagingDREAMTSequence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="DREAMT dataset directory")
    parser.add_argument("--cache-dir", required=True, help="PyHealth cache directory")
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        default=[f"S{i:03d}" for i in range(2, 10)],
        help="DREAMT participant IDs",
    )
    parser.add_argument(
        "--context-seconds",
        type=int,
        default=0,
        help=(
            "Surrounding BVP context in seconds on each side "
            "of an epoch (0 preserves epochwise extraction; "
            "5 enables the experimental retrospective extension)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--tcn-layers", type=int, default=1)
    parser.add_argument("--no-tcn", action="store_true")
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--residual-attention", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["weighted", "unweighted"],
        default="weighted",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def make_chunks(task, dataset, patient_id, chunk_size):
    samples = task(dataset.get_patient(patient_id))
    if not samples:
        raise RuntimeError(f"No valid samples for {patient_id}")

    sample = samples[0]
    signals = sample["signal"]
    times = np.asarray(sample["label"]["time"], dtype=float)
    labels = np.asarray(sample["label"]["values"], dtype=int)

    if not (len(signals) == len(times) == len(labels)):
        raise ValueError(f"Inconsistent signal and label lengths: {patient_id}")

    boundaries = np.r_[
        0,
        np.where(~np.isclose(np.diff(times), 30.0, atol=0.01))[0] + 1,
        len(times),
    ]

    chunks = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        for i in range(start, stop, chunk_size):
            j = min(i + chunk_size, stop)
            if j - i < 2:
                continue
            chunks.append(
                {
                    "patient_id": patient_id,
                    "signal": signals[i:j].astype(np.float32),
                    "label": {
                        "time": times[i:j].tolist(),
                        "values": labels[i:j].tolist(),
                    },
                }
            )

    if not chunks:
        raise RuntimeError(f"No usable sequence chunks for {patient_id}")
    print(f"{patient_id}: {len(chunks)} chunks", flush=True)
    return chunks


def make_dataset(ids, name, patient_chunks, task):
    samples = [chunk for pid in ids for chunk in patient_chunks[pid]]
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=name,
    )


def evaluate(model, loader):
    model.eval()
    true_labels, predictions, probabilities = [], [], []

    with torch.no_grad():
        for batch in loader:
            output = model(**batch)
            true_labels.extend(batch["label"][0].cpu().tolist())
            predictions.extend(output["y_prob"].argmax(dim=1).cpu().tolist())
            probabilities.extend(output["y_prob"].cpu().tolist())

    auroc = float("nan")
    if set(true_labels) == {0, 1, 2}:
        auroc = float(
            roc_auc_score(
                true_labels,
                np.asarray(probabilities),
                labels=[0, 1, 2],
                multi_class="ovr",
                average="macro",
            )
        )

    return {
        "auroc_macro_ovr": auroc,
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "f1_macro": float(
            f1_score(
                true_labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(true_labels, predictions)),
        "rem_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=[2],
                average="macro",
                zero_division=0,
            )
        ),
        "true_labels": true_labels,
        "predictions": predictions,
    }


def main():
    args = parse_args()

    if args.context_seconds < 0:
        raise ValueError("context-seconds must be non-negative")

    if args.epochs < 1 or args.chunk_size < 2 or args.max_folds < 1:
        raise ValueError("epochs and max-folds must be >= 1; chunk-size >= 2")
    if len(args.patient_ids) < 3:
        raise ValueError("At least three patients are needed for train/val/test")
    if len(set(args.patient_ids)) != len(args.patient_ids):
        raise ValueError("Patient IDs must be unique")

    dimensions = [
        args.num_channels,
        args.kernel_size,
        args.hidden_dim,
        args.num_heads,
        args.num_layers,
        args.tcn_layers,
    ]
    if any(value < 1 for value in dimensions):
        raise ValueError("Model dimensions must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if args.residual_attention and not args.use_attention:
        raise ValueError("residual-attention requires use-attention")

    print("Experiment configuration:", vars(args))

    torch.set_num_threads(2)
    dataset = DREAMTDataset(root=args.root, cache_dir=args.cache_dir)
    task = SleepStagingDREAMTSequence(
        context_seconds=args.context_seconds,
    )
    print("IBI context seconds:", args.context_seconds)

    patient_ids = args.patient_ids
    patient_chunks = {
        pid: make_chunks(task, dataset, pid, args.chunk_size) for pid in patient_ids
    }

    folds = [
        (test_id, patient_ids[(i + 1) % len(patient_ids)])
        for i, test_id in enumerate(patient_ids)
    ]
    folds = folds[: args.max_folds]
    all_results = []

    for fold_idx, (test_id, val_id) in enumerate(folds, start=1):
        train_ids = [pid for pid in patient_ids if pid not in (test_id, val_id)]

        print(f"\nFold {fold_idx}/{len(folds)}")
        print(f"Train: {train_ids}; validation: {val_id}; test: {test_id}")

        torch.manual_seed(args.seed + fold_idx)
        np.random.seed(args.seed + fold_idx)

        train_ds = make_dataset(
            train_ids, f"watchsleepnet_train_{fold_idx}", patient_chunks, task
        )
        val_ds = make_dataset(
            [val_id], f"watchsleepnet_val_{fold_idx}", patient_chunks, task
        )
        test_ds = make_dataset(
            [test_id], f"watchsleepnet_test_{fold_idx}", patient_chunks, task
        )

        train_loader = get_dataloader(
            train_ds, batch_size=1, shuffle=True, seed=args.seed + fold_idx
        )
        val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False)

        counts = np.zeros(3, dtype=np.int64)
        for pid in train_ids:
            for chunk in patient_chunks[pid]:
                counts += np.bincount(chunk["label"]["values"], minlength=3)[:3]

        total_epochs = int(counts.sum())
        class_weights = [total_epochs / (3 * max(int(count), 1)) for count in counts]

        print("Training class counts:", counts.tolist())

        model = WatchSleepNet(
            dataset=train_ds,
            num_channels=args.num_channels,
            kernel_size=args.kernel_size,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            tcn_layers=args.tcn_layers,
            use_tcn=not args.no_tcn,
            use_attention=args.use_attention,
            residual_attention=args.residual_attention,
            class_weights=(
                class_weights if args.loss == "weighted" else None
            ),
        ).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_f1 = float("-inf")
        best_state = None
        best_epoch = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []

            for batch in train_loader:
                optimizer.zero_grad()
                output = model(**batch)
                loss = output["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss in fold {fold_idx}, epoch {epoch}"
                    )
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            val_f1 = evaluate(model, val_loader)["f1_macro"]
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"loss={np.mean(losses):.4f}, val_macro_f1={val_f1:.4f}",
                flush=True,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        result = evaluate(model, test_loader)

        print(f"Best epoch: {best_epoch}")
        print(f"Test accuracy: {result['accuracy']:.4f}")
        print(f"Test macro-F1: {result['f1_macro']:.4f}")
        print(f"Test macro-OVR AUROC: {result['auroc_macro_ovr']:.4f}")
        print(f"Test REM F1: {result['rem_f1']:.4f}")
        print(
            "Confusion matrix:\n",
            confusion_matrix(
                result["true_labels"], result["predictions"], labels=[0, 1, 2]
            ),
        )

        all_results.append(result)

    print(f"\nCompleted {len(all_results)} patient-level fold(s).")
    metrics = [
        "accuracy",
        "f1_macro",
        "auroc_macro_ovr",
        "balanced_accuracy",
        "cohen_kappa",
        "rem_f1",
    ]
    for metric in metrics:
        values = np.asarray([result[metric] for result in all_results])
        valid_values = values[np.isfinite(values)]
        if len(valid_values) == 0:
            print(f"{metric}: undefined (no valid folds)")
        elif len(valid_values) == 1:
            print(f"{metric}: {valid_values[0]:.4f}")
        else:
            print(
                f"{metric}: {valid_values.mean():.4f} ± {valid_values.std(ddof=1):.4f}"
            )
        if len(valid_values) != len(values):
            print(f"  Valid folds: {len(valid_values)}/{len(values)}")


if __name__ == "__main__":
    main()
