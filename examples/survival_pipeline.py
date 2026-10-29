"""Authors: Aritra Saha and Ugochukwu Njoku.

GBSG/SUPPORT loader-to-model compatibility with PyHealth's accepted SurvivalMDN.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html

Run from a source checkout with separately obtained, local public inputs:
    python -m examples.survival_pipeline --dataset gbsg --root /data/gbsg \
        --released-dir /data/Survival-MDN/data/gbsg --out /tmp/gbsg-native

Use --dataset support2 with its CSV/released directories for SUPPORT. These
short pilots verify interfaces; they are not paper reproduction experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from pyhealth.datasets import (
    GBSGDataset,
    SampleDataset,
    Support2Dataset,
    gbsg_paper_time_transform,
    gbsg_released_split_ids,
    get_dataloader,
    support2_paper_time_transform,
    support2_released_split_ids,
)
from pyhealth.metrics import d_calibration, survival_curve_metrics_fn
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventGBSG, SurvivalTimeToEventSupport2
from pyhealth.trainer import Trainer
from pyhealth.trainer import logger as trainer_logger

PARTITIONS = ("train", "valid", "test")


def load_split(
    dataset: str,
    root: Path,
    released_dir: Path,
    split: int,
    cache_dir: Path,
) -> tuple[tuple[SampleDataset, ...], dict]:
    """Load native samples in released order with an explicit time convention.

    GBSG keeps raw features and months/cohort-maximum + 0.002. SUPPORT uses
    days/365.25 + 0.001 and feature scaling fitted on training patients once.
    Both tasks emit duration/event tensors; neither assumes model time_scale.

    Args:
        dataset: ``gbsg`` or ``support2``.
        root: Prepared CSV directory, never written by this function.
        released_dir: Authors' separately obtained local NPZ directory.
        split: Released partition number, 1 through 10.
        cache_dir: Writable destination for native PyHealth task caches.

    Returns:
        Training/validation/test datasets and reconstruction metadata.

    Examples:
        >>> parts, metadata = load_split(  # doctest: +SKIP
        ...     "gbsg", Path("/data/gbsg"), Path("/data/released/gbsg"),
        ...     1, Path("/tmp/native-cache"))
        >>> parts[0].output_schema  # doctest: +SKIP
        {'duration': 'tensor', 'event': 'tensor'}
    """
    if split not in range(1, 11):
        raise ValueError("split must be 1 through 10")
    if dataset == "gbsg":
        ids = gbsg_released_split_ids(root, released_dir, split)
        native = GBSGDataset(str(root), cache_dir=cache_dir, num_workers=1)
        transform = gbsg_paper_time_transform(root)
        samples = native.set_task(SurvivalTimeToEventGBSG(**transform))
        preprocessing = {"features": "raw", **transform}
        time_unit = "months / cohort maximum + 0.002"
    elif dataset == "support2":
        ids = support2_released_split_ids(root, released_dir, split)
        native = Support2Dataset(
            str(root), tables=["support2"], cache_dir=cache_dir, num_workers=1
        )
        transform = support2_paper_time_transform()
        raw = native.set_task(SurvivalTimeToEventSupport2(**transform))
        raw_by_id = {s["patient_id"]: s["features"].numpy() for s in raw}
        train_x = np.stack([raw_by_id[pid] for pid in ids["train"]]).astype(float)
        mean, scale = train_x.mean(0), train_x.std(0)
        scale[scale == 0] = 1.0
        samples = native.set_task(
            SurvivalTimeToEventSupport2(mean.tolist(), scale.tolist(), **transform)
        )
        preprocessing = {
            "features": "standardized on training patients once",
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            **transform,
        }
        time_unit = "days / 365.25 + 0.001"
    else:
        raise ValueError("dataset must be gbsg or support2")
    parts = tuple(
        samples.subset([i for pid in ids[name] for i in samples.patient_to_index[pid]])
        for name in PARTITIONS
    )
    paths = {"cohort": root / f"{dataset}.csv"}
    paths.update({name: released_dir / f"{name}_{split}.npz" for name in PARTITIONS})
    return parts, {
        "dataset": dataset,
        "split": split,
        "protocol": "authors-released-splits",
        "time_unit": time_unit,
        "event": "1 observed; 0 censored",
        "preprocessing": preprocessing,
        "patient_ids": ids,
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }


def restore_pilot(run_dir: Path, dataset: SampleDataset) -> SurvivalMDN:
    """Reconstruct the recorded model configuration and load its best weights.

    Args:
        run_dir: Completed pilot directory, including record.json and trainer/.
        dataset: Matching training schema and feature representation.

    Returns:
        Restored accepted SurvivalMDN in CPU evaluation mode.

    Examples:
        >>> model = restore_pilot(Path("/tmp/gbsg-native"), train)  # doctest: +SKIP
    """
    record = json.loads((run_dir / "record.json").read_text())
    model = SurvivalMDN(dataset, **record["model"])
    model.load_state_dict(
        torch.load(run_dir / "trainer/best.ckpt", map_location="cpu", weights_only=True)
    )
    return model.eval()


def run_pilot(
    parts: tuple[SampleDataset, ...],
    metadata: dict,
    run_dir: Path,
    epochs: int = 5,
    seed: int = 1,
    time_transform: str = "softplus",
) -> dict:
    """Exercise the actual standard Trainer, curve metrics and checkpoint reload.

    Args:
        parts: Native train/validation/test datasets with duration/event fields.
        metadata: Source and preprocessing information to retain with weights.
        run_dir: New output directory; existing runs are never overwritten.
        epochs: Small compatibility training budget, not a tuning search.
        seed: Model initialization and training shuffle seed.
        time_transform: Accepted model's softplus or exp transformation.

    Returns:
        Saved compatibility evidence. Invalid losses/gradients abort the run.

    Examples:
        >>> record = run_pilot(parts, metadata, Path("/tmp/pilot"))  # doctest: +SKIP
        >>> record["checkpoint_max_error"]  # doctest: +SKIP
        0.0
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train, valid, test = parts
    if len(train) < 2 or not len(valid) or not len(test):
        raise ValueError("need at least two training and one validation/test patient")
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_times = [float(s["duration"][0]) for s in train]
    # The accepted model validates strictly positive prediction times.
    grid = np.linspace(min(train_times), max(train_times), 40, dtype=np.float32)
    config = {
        "hidden_dim": 32,
        "num_components": 10,
        "num_hidden_layers": 3,
        "time_transform": time_transform,
        "include_jacobian": True,
        "reference_initialization": True,
        "time_grid": grid.tolist(),
    }
    model = SurvivalMDN(train, **config)
    gradient_checks, training_losses = [], []

    def check_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite training gradient")
        gradient_checks.append(True)
        return gradient

    def check_output(module, inputs, output):
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError("nonfinite loss")
        if module.training:
            training_losses.append(float(output["loss"].detach()))

    hooks = [p.register_hook(check_gradient) for p in model.parameters()]
    hooks.append(model.register_forward_hook(check_output))
    previous_handlers = set(trainer_logger.handlers)
    try:
        trainer = Trainer(
            model, device="cpu", output_path=str(run_dir), exp_name="trainer"
        )
        validation = get_dataloader(valid, batch_size=len(valid))
        initial = float(trainer.evaluate(validation)["loss"])
        batch_size = min(128, len(train))
        while batch_size > 2 and len(train) % batch_size == 1:
            batch_size -= 1
        if len(train) % batch_size == 1:
            batch_size = len(train)
        trainer.train(
            get_dataloader(train, batch_size=batch_size, shuffle=True, seed=seed),
            validation,
            epochs=epochs,
            optimizer_class=torch.optim.RMSprop,
            optimizer_params={"lr": 0.001},
            weight_decay=0.0,
            max_grad_norm=100.0,
            monitor="loss",
            monitor_criterion="min",
            load_best_model_at_last=True,
        )
        final = float(trainer.evaluate(validation)["loss"])
        vbatch = next(iter(get_dataloader(valid, batch_size=len(valid))))
        model.eval()
        with torch.no_grad():
            params = model.network.mixture_parameters(vbatch["features"])
            per_patient = model.network.censored_nll(
                params, vbatch["duration"], vbatch["event"], reduction="none"
            )
        patient_mean = float(per_patient.double().mean())
        np.testing.assert_allclose(final, patient_mean, rtol=1e-6, atol=1e-6)
        batch = next(iter(get_dataloader(test, batch_size=len(test))))
        times = batch["duration"][:, 0].numpy()
        events = batch["event"][:, 0].numpy()

        def predict(t_grid):
            return model.predict_survival(batch["features"], t_grid).cpu().numpy()

        curves = predict(grid)
        assert np.isfinite(curves).all() and ((curves >= 0) & (curves <= 1)).all()
        assert (np.diff(curves, axis=1) <= 1e-6).all()
        unique, columns = np.unique(times, return_inverse=True)
        observed = predict(unique)[np.arange(len(times)), columns]
        metrics = survival_curve_metrics_fn(times, events, predict)
        assert all(np.isfinite(value) for value in metrics.values())
        assert gradient_checks and training_losses
        np.savez_compressed(
            run_dir / "predictions.npz",
            patient_id=np.asarray(batch["patient_id"]),
            time=times,
            event=events,
            grid=grid,
            survival=curves,
            survival_at_observed_time=observed,
        )
        record = {
            **metadata,
            "model": config,
            "epochs": epochs,
            "seed": seed,
            "counts": {name: len(part) for name, part in zip(PARTITIONS, parts)},
            "validation_batch_size": len(valid),
            "training_batch_size": batch_size,
            "initial_validation_nll": initial,
            "best_validation_nll": final,
            "direct_patient_mean_nll": patient_mean,
            "gradient_checks": len(gradient_checks),
            "training_losses": training_losses,
            "test_metrics": metrics,
            "d_calibration": d_calibration(observed, events),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
            "source": {
                name: {
                    "path": inspect.getfile(cls),
                    "sha256": hashlib.sha256(
                        Path(inspect.getfile(cls)).read_bytes()
                    ).hexdigest(),
                }
                for name, cls in [("model", SurvivalMDN), ("trainer", Trainer)]
            },
        }
        record_path = run_dir / "record.json"
        record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        restored = restore_pilot(run_dir, train)
        replay = restored.predict_survival(batch["features"], grid).numpy()
        np.testing.assert_array_equal(curves, replay)
        observed_replay = restored.predict_survival(batch["features"], unique).numpy()
        np.testing.assert_array_equal(
            observed, observed_replay[np.arange(len(times)), columns]
        )
        record["checkpoint_max_error"] = float(np.max(np.abs(curves - replay)))
        record["passed"] = True
        record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return record
    finally:
        for hook in hooks:
            hook.remove()
        for handler in set(trainer_logger.handlers) - previous_handlers:
            trainer_logger.removeHandler(handler)
            handler.close()


def main() -> None:
    """Run one local compatibility pilot using an explicit output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gbsg", "support2"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--released-dir", type=Path, required=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    parts, metadata = load_split(
        args.dataset,
        args.root,
        args.released_dir,
        args.split,
        args.out.parent / "cache",
    )
    result = run_pilot(parts, metadata, args.out, epochs=args.epochs)
    print(
        json.dumps(
            {"passed": result["passed"], "record": str(args.out / "record.json")}
        )
    )


if __name__ == "__main__":
    main()
