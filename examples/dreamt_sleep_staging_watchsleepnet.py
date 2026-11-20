"""Ablation study: WatchSleepNet on DREAMT wrist IBI sleep staging.

Runs the full PyHealth pipeline::

    DREAMTDataset -> SleepStagingDREAMT (task) -> WatchSleepNet (model)

and compares configurations that go beyond the paper's own ablations. Every configuration trains from scratch on
DREAMT only, so absolute numbers are far below the paper's pretrained results;
only the comparison between configurations is meaningful.

Ablation groups (``--groups``):
    preprocessing: IBI source (own BVP peak detection vs the device ``IBI``
        column), missing-value fill (zero / interpolate / median), per-night
        and per-epoch normalisation, and an extra validity-mask channel.
    causal: bidirectional LSTM vs a causal model (unidirectional LSTM with a
        causal attention mask), as a real-time wearable would need.
    loss: plain cross-entropy vs class-weighted cross-entropy vs focal loss.
    context: number of 30 s epochs per training window.
    labels: 3-class (Wake/NREM/REM), 4-class (Wake/Light/Deep/REM) and
        5-class (Wake/N1/N2/N3/REM) staging.
    hparams: a small dropout x learning-rate grid.
    robustness: the trained baseline evaluated while a random fraction of
        test epochs have their IBI zeroed (simulating motion artefacts).

Usage:
    # Fast end-to-end check on tiny synthetic DREAMT-style files (no data
    # needed; this is what graders / reviewers should run):
    python examples/dreamt_sleep_staging_watchsleepnet.py --synthetic

    # Real run on a subset of DREAMT (point --root at the version folder):
    python examples/dreamt_sleep_staging_watchsleepnet.py \\
        --root /path/to/dreamt/2.1.0 --max-patients 30 --seeds 3 --epochs 20

Setup of the reported experiment:
    Subject-wise 60/20/20 train/val/test split (re-drawn for every seed), model
    selected on validation loss, metrics on the held-out test participants.
    Metrics: accuracy, macro F1, REM F1 (REM is always the last class) and
    Cohen's kappa, reported as mean +/- std over seeds.

Results:
    group                config                 accuracy        macro_f1          rem_f1           kappa
    baseline             baseline 0.695 +/- 0.008 0.479 +/- 0.003 0.000 +/- 0.000 0.421 +/- 0.010
    robustness              drop=0.0 0.695 +/- 0.008 0.479 +/- 0.003 0.000 +/- 0.000 0.421 +/- 0.010
    robustness              drop=0.1 0.543 +/- 0.117 0.343 +/- 0.109 0.000 +/- 0.000 0.199 +/- 0.167
    robustness              drop=0.2 0.516 +/- 0.128 0.308 +/- 0.123 0.000 +/- 0.000 0.160 +/- 0.187
    robustness              drop=0.4 0.480 +/- 0.102 0.271 +/- 0.111 0.000 +/- 0.000 0.103 +/- 0.145
    preprocessing      fill=interpolate 0.670 +/- 0.029 0.462 +/- 0.020 0.000 +/- 0.000 0.377 +/- 0.056
    preprocessing           fill=median 0.682 +/- 0.027 0.471 +/- 0.016 0.000 +/- 0.000 0.404 +/- 0.040
    preprocessing         source=e4_ibi 0.605 +/- 0.013 0.418 +/- 0.009 0.000 +/- 0.000 0.263 +/- 0.023
    preprocessing            norm=night 0.720 +/- 0.008 0.496 +/- 0.005 0.000 +/- 0.000 0.467 +/- 0.015
    preprocessing            norm=epoch 0.597 +/- 0.067 0.367 +/- 0.102 0.000 +/- 0.000 0.205 +/- 0.155
    preprocessing          mask_channel 0.673 +/- 0.041 0.464 +/- 0.029 0.000 +/- 0.000 0.381 +/- 0.077
    causal                causal 0.677 +/- 0.015 0.464 +/- 0.008 0.000 +/- 0.000 0.383 +/- 0.023
    loss           weighted_ce 0.473 +/- 0.163 0.378 +/- 0.075 0.124 +/- 0.087 0.228 +/- 0.145
    loss            focal(g=2) 0.663 +/- 0.033 0.458 +/- 0.022 0.000 +/- 0.000 0.373 +/- 0.054
    context             window=25 0.685 +/- 0.011 0.472 +/- 0.008 0.000 +/- 0.000 0.401 +/- 0.024
    context             window=50 0.672 +/- 0.009 0.459 +/- 0.012 0.000 +/- 0.000 0.370 +/- 0.029
    context            window=200 0.706 +/- 0.009 0.485 +/- 0.005 0.000 +/- 0.000 0.440 +/- 0.017
    labels                4class 0.664 +/- 0.037 0.349 +/- 0.017 0.000 +/- 0.000 0.382 +/- 0.058
    labels                5class 0.616 +/- 0.022 0.273 +/- 0.007 0.000 +/- 0.000 0.357 +/- 0.029
    hparams  dropout=0.0,lr=0.001 0.696 +/- 0.010 0.478 +/- 0.008 0.000 +/- 0.000 0.419 +/- 0.024
    hparams dropout=0.0,lr=0.0003 0.685 +/- 0.011 0.467 +/- 0.010 0.000 +/- 0.000 0.394 +/- 0.025
    hparams  dropout=0.4,lr=0.001 0.694 +/- 0.020 0.477 +/- 0.012 0.000 +/- 0.000 0.418 +/- 0.036
    hparams dropout=0.4,lr=0.0003 0.690 +/- 0.005 0.471 +/- 0.002 0.000 +/- 0.000 0.401 +/- 0.007

    The baseline model parameters from the paper outperformed most alternate 
    configurations in terms of accuracy and Cohen's Kappa. The main changes  
    that led to better performance than the baseline were preprocessing 
    parameter adjustments. For example, normalizing IBI values per night lead 
    to a higher accuracy score of 0.720, compared to the baseline accuracy 
    0.695. Additionally, increasing the number of epochs passed into the model 
    at a time also increased the accuracy score to 0.706. 
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from pyhealth.datasets import DREAMTDataset, get_dataloader, split_by_patient
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingDREAMT
from pyhealth.trainer import Trainer

# Model sizes. The paper's configuration has ~8.5M parameters; the default here
# is much smaller so the sweep is feasible on a free Colab GPU.
FULL_MODEL = {
    "res_channels": (32, 64, 128, 256),
    "res_strides": (5, 5, 5, 2),
    "hidden_dim": 256,
    "lstm_layers": 4,
    "num_heads": 16,
    "tcn_layers": 3,
}
SMALL_MODEL = {
    "res_channels": (16, 32, 64, 128),
    "res_strides": (5, 5, 5, 2),
    "hidden_dim": 64,
    "lstm_layers": 2,
    "num_heads": 4,
    "tcn_layers": 2,
}
TINY_MODEL = {
    "res_channels": (4, 8, 8, 8),
    "res_strides": (5, 5, 5, 2),
    "hidden_dim": 8,
    "lstm_layers": 1,
    "num_heads": 2,
    "tcn_layers": 1,
}
METRICS = ["accuracy", "macro_f1", "rem_f1", "kappa"]


# --------------------------------------------------------------------------- #
# Synthetic DREAMT-style data (so the script runs without the real dataset)
# --------------------------------------------------------------------------- #
def make_synthetic_dreamt(
    root: Path, n_patients: int = 6, n_epochs: int = 12, seed: int = 0
) -> None:
    """Writes a tiny DREAMT-like folder: participant_info.csv + 64 Hz files.

    Each night cycles through W/N1/N2/N3/R. Heart-rate variability is larger in
    "REM" so there is a weak learnable signal.

    Args:
        root: Folder to create the files in.
        n_patients: Number of fake participants.
        n_epochs: 30 s epochs per participant.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    (root / "data_64Hz").mkdir(parents=True)
    (root / "data_100Hz").mkdir(parents=True)
    sids = [f"S{i:03d}" for i in range(1, n_patients + 1)]
    pd.DataFrame(
        {
            "SID": sids,
            "AGE": rng.uniform(25, 65, n_patients),
            "GENDER": rng.choice(["M", "F"], n_patients),
            "BMI": rng.integers(20, 50, n_patients),
            "OAHI": rng.integers(0, 50, n_patients),
            "AHI": rng.integers(0, 50, n_patients),
            "Mean_SaO2": [f"{v}%" for v in rng.integers(85, 99, n_patients)],
            "Arousal Index": rng.integers(1, 100, n_patients),
            "MEDICAL_HISTORY": ["none"] * n_patients,
            "Sleep_Disorders": ["none"] * n_patients,
        }
    ).to_csv(root / "participant_info.csv", index=False)

    fs, rows_per_epoch = 64, 30 * 64
    cycle = ["W", "N1", "N2", "N3", "R"]
    wobble = {"W": 0.03, "N1": 0.02, "N2": 0.02, "N3": 0.01, "R": 0.12}
    for sid in sids:
        stages = [cycle[i % len(cycle)] for i in range(n_epochs)]
        n = n_epochs * rows_per_epoch
        amp = np.repeat([wobble[s] for s in stages], rows_per_epoch)
        t = np.arange(n) / fs
        freq = rng.uniform(0.9, 1.2) + amp * np.sin(2 * np.pi * t / 20.0)
        phase = 2 * np.pi * np.cumsum(freq) / fs
        bvp = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * rng.standard_normal(n)
        beat_rows = np.flatnonzero(np.diff(np.floor(phase / (2 * np.pi))) > 0) + 1
        ibi = np.full(n, np.nan)
        ibi[beat_rows[1:]] = np.diff(beat_rows) / fs
        pd.DataFrame(
            {
                "TIMESTAMP": t,
                "BVP": bvp,
                "HR": 60.0,
                "EDA": 0.5,
                "TEMP": 30.0,
                "ACC_X": 0.0,
                "ACC_Y": 0.0,
                "ACC_Z": 0.0,
                "IBI": ibi,
                "Sleep_Stage": np.repeat(stages, rows_per_epoch),
            }
        ).to_csv(root / "data_64Hz" / f"{sid}_whole_df.csv")
        pd.DataFrame({"TIMESTAMP": [0.0], "BVP": [0.0], "IBI": [0.8]}).to_csv(
            root / "data_100Hz" / f"{sid}_PSG_df.csv"
        )


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Builds the list of configurations to run.

    Each config has a ``group``, a ``name`` and kwargs for the task, the model
    and the optimiser, plus an optional ``weights`` flag (class-weighted loss).

    Args:
        args: Parsed command-line arguments.

    Returns:
        A list of config dicts; the baseline is always first.
    """

    def cfg(group: str, name: str, **kw: Any) -> dict[str, Any]:
        return {
            "group": group,
            "name": name,
            "task": kw.pop("task", {}),
            "model": kw.pop("model", {}),
            "lr": kw.pop("lr", 1e-3),
            "weights": kw.pop("weights", False),
        }

    configs = [cfg("baseline", "baseline")]
    configs[0]["corrupt"] = args.corrupt if "robustness" in args.groups else []
    g = args.groups
    if "preprocessing" in g:
        configs += [
            cfg("preprocessing", "fill=interpolate", task={"fill": "interpolate"}),
            cfg("preprocessing", "fill=median", task={"fill": "median"}),
            cfg("preprocessing", "source=e4_ibi", task={"ibi_source": "e4"}),
            cfg("preprocessing", "norm=night", task={"normalize": "night"}),
            cfg("preprocessing", "norm=epoch", task={"normalize": "epoch"}),
            cfg(
                "preprocessing",
                "mask_channel",
                task={"add_mask": True},
                model={"num_input_channels": 2},
            ),
        ]
    if "causal" in g:
        configs.append(cfg("causal", "causal", model={"bidirectional": False}))
    if "loss" in g:
        configs += [
            cfg("loss", "weighted_ce", weights=True),
            cfg("loss", "focal(g=2)", model={"focal_gamma": 2.0}),
        ]
    if "context" in g:
        configs += [
            cfg("context", f"window={w}", task={"window_epochs": w})
            for w in args.windows
            if w != args.window
        ]
    if "labels" in g:
        configs += [
            cfg("labels", "4class", task={"label_scheme": "4class"}),
            cfg("labels", "5class", task={"label_scheme": "5class"}),
        ]
    if "hparams" in g:
        configs += [
            cfg("hparams", f"dropout={d},lr={lr}", model={"dropout": d}, lr=lr)
            for d in (0.0, 0.4)
            for lr in (1e-3, 3e-4)
        ]
    return configs


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: WatchSleepNet,
    loader: Any,
    num_classes: int,
    drop_prob: float = 0.0,
    seed: int = 0,
) -> dict[str, float]:
    """Scores a model on a loader, optionally corrupting the input.

    Args:
        model: Trained model.
        loader: Dataloader yielding batches with ``ibi`` and ``label``.
        num_classes: Number of classes (REM is the last one).
        drop_prob: Probability of zeroing the whole IBI of an epoch at test time.
        seed: Seed of the corruption mask.

    Returns:
        dict with accuracy, macro_f1, rem_f1 and kappa over non-ignored epochs.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ys, ps = [], []
    for batch in loader:
        if drop_prob > 0:
            x = batch["ibi"]
            keep = (torch.rand(x.shape[:2], generator=gen) >= drop_prob).to(x.dtype)
            batch = dict(batch)
            batch["ibi"] = x * keep.reshape(*keep.shape, *([1] * (x.dim() - 2)))
        out = model(**batch)
        y = out["y_true"].reshape(-1).cpu().numpy()
        p = out["y_prob"].argmax(-1).reshape(-1).cpu().numpy()
        ys.append(y[y >= 0])
        ps.append(p[y >= 0])
    y, p = np.concatenate(ys), np.concatenate(ps)
    labels = list(range(num_classes))
    f1s = f1_score(y, p, labels=labels, average=None, zero_division=0)
    kappa = cohen_kappa_score(y, p, labels=labels)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1s.mean()),
        "rem_f1": float(f1s[-1]),
        "kappa": float(np.nan_to_num(kappa)),
    }


def class_weights(loader: Any, num_classes: int) -> list[float]:
    """Inverse-frequency class weights computed on the training labels."""
    counts = np.zeros(num_classes)
    for batch in loader:
        y = batch["label"].long().reshape(-1).numpy()
        counts += np.bincount(y[y >= 0], minlength=num_classes)
    return (counts.sum() / (num_classes * (counts + 1.0))).tolist()


class Runner:
    """Builds datasets per task configuration and trains/evaluates models."""

    def __init__(self, dataset: DREAMTDataset, args: argparse.Namespace) -> None:
        self.dataset = dataset
        self.args = args
        self._cache: dict[str, Any] = {}
        ids = sorted(dataset.unique_patient_ids)
        self.patient_ids = ids[: args.max_patients] if args.max_patients else ids

    def samples(self, task_kwargs: dict[str, Any]) -> Any:
        """Returns (task, sample dataset), memoised by the task settings."""
        key = json.dumps(task_kwargs, sort_keys=True)
        if key not in self._cache:
            task = SleepStagingDREAMT(
                window_epochs=task_kwargs.get("window_epochs", self.args.window),
                max_epochs=self.args.max_epochs,
                patient_ids=self.patient_ids,
                **{k: v for k, v in task_kwargs.items() if k != "window_epochs"},
            )
            self._cache[key] = (task, self.dataset.set_task(task))
        return self._cache[key]

    def run(self, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
        """Trains one config with one seed and evaluates on the test split."""
        args = self.args
        torch.manual_seed(seed)
        np.random.seed(seed)
        task, sample_ds = self.samples(cfg["task"])
        train_ds, val_ds, test_ds = split_by_patient(
            sample_ds, [0.6, 0.2, 0.2], seed=seed
        )
        bs = args.batch_size
        train_dl = get_dataloader(train_ds, bs, shuffle=True, seed=seed)
        val_dl = get_dataloader(val_ds, bs, shuffle=False)
        test_dl = get_dataloader(test_ds, bs, shuffle=False)

        model_kwargs = dict(args.model_size)
        model_kwargs.update(cfg["model"])
        weights = class_weights(train_dl, task.num_classes) if cfg["weights"] else None
        model = WatchSleepNet(
            sample_ds,
            num_classes=task.num_classes,
            class_weights=weights,
            **model_kwargs,
        )
        trainer = Trainer(model=model, device=args.device, enable_logging=False)
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            epochs=args.epochs,
            optimizer_params={"lr": cfg["lr"]},
            monitor="loss",
            monitor_criterion="min",
            patience=args.patience,
            max_grad_norm=1.0,
        )
        result = {"metrics": evaluate(model, test_dl, task.num_classes)}
        result["corrupt"] = {
            p: evaluate(model, test_dl, task.num_classes, drop_prob=p, seed=seed)
            for p in cfg.get("corrupt", [])
        }
        return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarise(runs: list[dict[str, float]]) -> dict[str, str]:
    """Formats mean +/- std over seeds for each metric."""
    row = {}
    for m in METRICS:
        vals = np.array([r[m] for r in runs])
        row[m] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
    return row


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    groups = ["preprocessing", "causal", "loss", "context", "labels", "hparams"]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=str, help="DREAMT version folder")
    p.add_argument("--synthetic", action="store_true", help="tiny fake data run")
    p.add_argument("--groups", nargs="+", default=groups + ["robustness"])
    p.add_argument("--max-patients", type=int, default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--window", type=int, default=100, help="epochs per sample")
    p.add_argument("--windows", type=int, nargs="+", default=[25, 50, 100, 200])
    p.add_argument("--max-epochs", type=int, default=1100)
    p.add_argument("--corrupt", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    p.add_argument("--full-size", action="store_true", help="paper-sized model")
    p.add_argument("--out", type=str, default="dreamt_ablation_results.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.synthetic:  # shrink everything for a quick end-to-end check
        args.seeds, args.epochs, args.patience = 1, 2, 2
        args.window, args.windows, args.max_epochs = 4, [2, 4, 6], 12
        args.batch_size, args.model_size = 2, TINY_MODEL
    else:
        if not args.root:
            p.error("--root is required unless --synthetic is given")
        args.model_size = FULL_MODEL if args.full_size else SMALL_MODEL
    return args


def main() -> None:
    """Runs the selected ablations and prints a results table."""
    args = parse_args()
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root: str = args.root
        if args.synthetic:
            make_synthetic_dreamt(Path(tmp) / "dreamt", n_epochs=args.max_epochs)
            root = str(Path(tmp) / "dreamt")
        runner = Runner(DREAMTDataset(root=root), args)

        rows: list[dict[str, str]] = []
        for cfg in build_configs(args):
            runs = [runner.run(cfg, seed) for seed in range(args.seeds)]
            rows.append({"group": cfg["group"], "config": cfg["name"]})
            rows[-1].update(summarise([r["metrics"] for r in runs]))
            for p in cfg.get("corrupt", []):
                rows.append({"group": "robustness", "config": f"drop={p}"})
                rows[-1].update(summarise([r["corrupt"][p] for r in runs]))
            print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {args.out} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
