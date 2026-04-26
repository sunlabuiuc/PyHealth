"""DREAMT Sleep Staging — Ablation Study with PyHealth RNN.

This script runs two ablation experiments on the DREAMT
sleep staging task using ``SleepStagingDREAMT``:

1. **Signal-subset ablation** (binary wake/sleep):
   ACC-only vs BVP/HRV-only vs EDA+TEMP-only vs ALL signals.

2. **Label-granularity ablation** (ALL signals):
   2-class (wake/sleep) vs 5-class (W/N1/N2/N3/R).

The model is PyHealth's built-in ``RNN`` (LSTM variant), trained
using PyHealth's ``Trainer`` with **patient-level** splits.

**Why summary statistics instead of raw 64 Hz windows here**

The task emits rich ``(n_channels, 1920)`` tensors. This script
collapses each epoch to **four scalar statistics per channel**
(mean, std, min, max) so the built-in ``RNN`` (which expects a
fixed-size vector per epoch via ``TensorProcessor``) can run a
lightweight baseline without a custom sequence encoder. That is a
deliberate trade-off for a short tutorial script, **not** a
recommended production pipeline for waveform sleep staging.

For an example that trains **SparcNet on the raw multichannel
signal**, see ``dreamt_sleep_staging_sparcnet.py`` in this folder.

**Evaluation protocol**

By default the script uses **one** random patient-level split
(70% / 10% / 20% train / val / test). Pass ``--n-folds 5`` (or
``run.n_folds: 5`` in YAML) to run **5-fold** participant-level
cross-validation; metrics are averaged over folds (``*_std`` keys
when more than one fold). This matches the common "5-fold CV"
wording when that option is enabled.

Usage — YAML config (dataset + task + training)::

    python dreamt_sleep_staging_rnn.py --config dreamt_sleep_staging_rnn.example.yaml

CLI arguments override the YAML file when given (for example ``--epochs 10``).

Usage — full DREAMT run::

    python dreamt_sleep_staging_rnn.py --root /path/to/dreamt

Usage — synthetic demo (no dataset required)::

    python dreamt_sleep_staging_rnn.py --demo

Metrics: F1 (macro), Accuracy, Cohen's Kappa.

Results / Findings
------------------

**Demo mode** (synthetic data, 6 patients, 2 training epochs):

Results are non-meaningful and serve only to verify that the
full pipeline (epoching -> feature extraction -> PyHealth RNN
training -> evaluation) runs end-to-end without error.  Expected
output is near-random performance.

**Paper reference** (Wang et al. CHIL 2024, Table 2):

The original paper reports wake/sleep (2-class) detection on
80 participants (after artifact QC) using LightGBM / GPBoost
with hand-crafted features from all 8 E4 channels and 5-fold
participant-level CV in the paper.  Key results from Table 2:

- Baseline LightGBM:  F1 = 0.777, Acc = 0.816, Kappa = 0.605
- Best (GPBoost + Apnea Severity RE + LSTM post-processing):
  F1 = 0.823, Acc = 0.857, Kappa = 0.694

The paper does not report per-signal-subset ablations; those
are original to this script.  This ablation also uses a simpler
feature set (4 summary stats per channel) and a neural model
(LSTM via PyHealth RNN) with the split / CV settings selected
above, so results are expected to differ from the paper.

Reference:
    Wang et al. "Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep
    disorders." CHIL 2024, PMLR 248:380-396.
"""

import argparse
import copy
import os
import warnings
from typing import Any, Dict, List, MutableMapping, Optional

import numpy as np
import yaml
from sklearn.model_selection import KFold

from dreamt_sleep_staging_demo_utils import generate_demo_samples

from pyhealth.datasets import (
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import RNN
from pyhealth.tasks.sleep_staging_dreamt import (
    ALL_SIGNAL_COLUMNS,
    SleepStagingDREAMT,
)
from pyhealth.trainer import Trainer

warnings.filterwarnings("ignore", category=FutureWarning)

SIGNAL_SUBSETS: Dict[str, List[str]] = {
    "ACC": ["ACC_X", "ACC_Y", "ACC_Z"],
    "BVP_HRV": ["BVP", "HR", "IBI"],
    "EDA_TEMP": ["EDA", "TEMP"],
    "ALL": list(ALL_SIGNAL_COLUMNS),
}

DEFAULT_YAML_CONFIG: Dict[str, Any] = {
    "demo": False,
    "dataset": {
        "root": None,
        "dataset_name": None,
        "config_path": None,
    },
    "task": {
        "n_classes": 5,
        "signal_subset": "ALL",
        "signal_columns": None,
        "epoch_seconds": 30.0,
        "sampling_rate": 64,
        "apply_filters": True,
    },
    "training": {
        "epochs": 30,
        "hidden_dim": 64,
        "device": "cpu",
    },
    "run": {
        "mode": "ablations",
        "n_folds": 1,
    },
}


def _deep_merge(
    base: MutableMapping[str, Any],
    override: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data


def _resolve_signal_columns(task_cfg: Dict[str, Any]) -> List[str]:
    """Return channel list from ``signal_columns`` or ``signal_subset``."""
    cols = task_cfg.get("signal_columns")
    if cols is not None:
        if not isinstance(cols, list) or not cols:
            raise ValueError("task.signal_columns must be a non-empty list when set")
        return [str(c) for c in cols]
    subset = str(task_cfg.get("signal_subset", "ALL")).upper()
    if subset not in SIGNAL_SUBSETS:
        allowed = ", ".join(sorted(SIGNAL_SUBSETS))
        raise ValueError(
            f"Unknown task.signal_subset {subset!r}; use one of: {allowed}"
        )
    return list(SIGNAL_SUBSETS[subset])


def _task_from_config(task_cfg: Dict[str, Any]) -> SleepStagingDREAMT:
    """Build ``SleepStagingDREAMT`` from a ``task`` config block."""
    return SleepStagingDREAMT(
        n_classes=int(task_cfg["n_classes"]),
        signal_columns=_resolve_signal_columns(task_cfg),
        epoch_seconds=float(task_cfg.get("epoch_seconds", 30.0)),
        sampling_rate=int(task_cfg.get("sampling_rate", 64)),
        apply_filters=bool(task_cfg.get("apply_filters", True)),
    )


def _build_merged_config(
    yaml_path: Optional[str],
    cli: argparse.Namespace,
) -> Dict[str, Any]:
    """Defaults, then YAML, then explicit CLI overrides."""
    merged = copy.deepcopy(DEFAULT_YAML_CONFIG)
    if yaml_path:
        merged = _deep_merge(merged, _load_yaml_config(yaml_path))
    if cli.demo is not None:
        merged["demo"] = cli.demo
    ds = merged["dataset"]
    if cli.root is not None:
        ds["root"] = cli.root
    tr = merged["training"]
    if cli.epochs is not None:
        tr["epochs"] = cli.epochs
    if cli.hidden_dim is not None:
        tr["hidden_dim"] = cli.hidden_dim
    if cli.device is not None:
        tr["device"] = cli.device
    if cli.run_mode is not None:
        merged["run"]["mode"] = cli.run_mode
    if getattr(cli, "n_folds", None) is not None:
        merged["run"]["n_folds"] = int(cli.n_folds)
    return merged


def _make_dreamt_dataset(ds_cfg: Dict[str, Any], root: str) -> Any:
    """Instantiate ``DREAMTDataset`` from a ``dataset`` config block."""
    from pyhealth.datasets import DREAMTDataset

    kwargs: Dict[str, Any] = {"root": root}
    name = ds_cfg.get("dataset_name")
    if name:
        kwargs["dataset_name"] = name
    cfg_path = ds_cfg.get("config_path")
    if cfg_path:
        kwargs["config_path"] = cfg_path
    return DREAMTDataset(**kwargs)


def _epoch_features(signal: np.ndarray) -> List[float]:
    """Convert a raw epoch signal to a compact feature vector.

    Computes mean, std, min, and max per channel.

    Args:
        signal: Array of shape ``(n_channels, epoch_len)``.

    Returns:
        Flat list of length ``4 * n_channels``.
    """
    feats: List[float] = []
    for ch in range(signal.shape[0]):
        s = signal[ch].astype(np.float64)
        feats.extend([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
        ])
    return feats


def _prepare_samples(
    raw_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Transform task-output samples into feature-vector samples.

    Each raw sample has ``signal`` (n_channels, epoch_len).
    This replaces it with a flat feature vector suitable for
    PyHealth's ``TensorProcessor``.

    Args:
        raw_samples: Output of ``SleepStagingDREAMT(patient)``.

    Returns:
        List of dicts with ``patient_id``, ``features``, ``label``.
    """
    return [
        {
            "patient_id": s["patient_id"],
            "features": _epoch_features(s["signal"]),
            "label": s["label"],
        }
        for s in raw_samples
    ]


_SCHEMA = {
    "input_schema": {"features": "tensor"},
    "output_schema": {"label": "multiclass"},
    "dataset_name": "dreamt",
    "task_name": "sleep_staging",
}


def build_feature_sample_dataset(
    prepared: List[Dict[str, Any]],
) -> Any:
    """Build a :class:`~pyhealth.datasets.SampleDataset` over feature vectors."""
    return create_sample_dataset(samples=prepared, **_SCHEMA)


def _feature_subset_dataset(
    prepared: List[Dict[str, Any]],
    template: Any,
) -> Any:
    """Same processors as ``template``, restricted to ``prepared`` rows."""
    return create_sample_dataset(
        samples=prepared,
        input_processors=template.input_processors,
        output_processors=template.output_processors,
        **_SCHEMA,
    )


def train_and_evaluate(
    model_dataset: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    *,
    device: str,
    epochs: int,
    hidden_dim: int,
) -> Dict[str, float]:
    """Construct an RNN, train, and return test-set metrics."""
    model = RNN(
        dataset=model_dataset,
        embedding_dim=hidden_dim,
        hidden_dim=hidden_dim,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0.0,
    )
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_macro", "cohen_kappa"],
        device=device,
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
    )
    return trainer.evaluate(test_loader)


def run_training_experiment(
    raw_samples: List[Dict[str, Any]],
    *,
    device: str = "cpu",
    epochs: int = 5,
    hidden_dim: int = 64,
    split_ratios: Optional[List[float]] = None,
    n_folds: int = 1,
) -> Dict[str, Any]:
    """Train/evaluate on summary features: one split or K-fold by patient.

    Pipeline: :func:`_prepare_samples` → dataset → split(s) →
    :func:`train_and_evaluate`.
    """
    if split_ratios is None:
        split_ratios = [0.7, 0.1, 0.2]

    prepared = _prepare_samples(raw_samples)

    if n_folds < 2:
        dataset = build_feature_sample_dataset(prepared)
        train_ds, val_ds, test_ds = split_by_patient(
            dataset, split_ratios, seed=42,
        )
        train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
        return train_and_evaluate(
            dataset,
            train_loader,
            val_loader,
            test_loader,
            device=device,
            epochs=epochs,
            hidden_dim=hidden_dim,
        )

    pids = np.array(sorted({s["patient_id"] for s in prepared}))
    if len(pids) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} distinct patients for "
            f"{n_folds}-fold CV, got {len(pids)}"
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results: List[Dict[str, float]] = []

    for fold_idx, (tr_va_idx, te_idx) in enumerate(kf.split(pids)):
        test_pids = set(pids[te_idx])
        tr_va = pids[tr_va_idx]
        rng = np.random.RandomState(42 + fold_idx)
        order = rng.permutation(len(tr_va))
        n_val = max(1, int(round(0.1 * len(tr_va))))
        val_pids = set(tr_va[order[:n_val]])
        train_pids = set(tr_va[order[n_val:]])

        train_s = [s for s in prepared if s["patient_id"] in train_pids]
        val_s = [s for s in prepared if s["patient_id"] in val_pids]
        test_s = [s for s in prepared if s["patient_id"] in test_pids]
        combined = train_s + val_s + test_s
        if not train_s or not val_s or not test_s:
            continue

        template_ds = build_feature_sample_dataset(combined)
        train_ds = _feature_subset_dataset(train_s, template_ds)
        val_ds = _feature_subset_dataset(val_s, template_ds)
        test_ds = _feature_subset_dataset(test_s, template_ds)

        train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

        fold_results.append(
            train_and_evaluate(
                template_ds,
                train_loader,
                val_loader,
                test_loader,
                device=device,
                epochs=epochs,
                hidden_dim=hidden_dim,
            )
        )

    if not fold_results:
        return {}

    keys = fold_results[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [
            float(r[k]) for r in fold_results
            if k in r and isinstance(r[k], (int, float))
            and r[k] == r[k]
        ]
        if not vals:
            continue
        out[k] = float(np.mean(vals))
        if len(vals) > 1:
            out[f"{k}_std"] = float(np.std(vals))
    return out


# -----------------------------------------------------------
# Ablation runners
# -----------------------------------------------------------

DEFAULT_ROOT = os.path.expanduser("~/.pyhealth/dreamt")


def _resolve_root(root_arg: Optional[str]) -> str:
    """Find a valid DREAMT root, or exit with guidance.

    Args:
        root_arg: User-supplied ``--root`` value, or None.

    Returns:
        Absolute path to the DREAMT version directory.

    Raises:
        SystemExit: If no valid directory is found.
    """
    candidates = (
        [root_arg]
        if root_arg
        else [
            DEFAULT_ROOT,
            os.path.expanduser("~/data/dreamt"),
            os.path.expanduser("~/dreamt"),
        ]
    )
    for path in candidates:
        if path and os.path.isdir(path):
            info = os.path.join(path, "participant_info.csv")
            if os.path.isfile(info):
                return path
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isdir(subpath) and os.path.isfile(
                    os.path.join(subpath, "participant_info.csv")
                ):
                    return subpath
    print(
        "ERROR: Could not find the DREAMT dataset.\n"
        "\n"
        "Download from PhysioNet (credentialed access):\n"
        "  https://physionet.org/content/dreamt/\n"
        "\n"
        "Then either:\n"
        f"  - Extract to {DEFAULT_ROOT}/\n"
        "  - Or pass --root /path/to/dreamt/version/\n"
        "\n"
        "The directory must contain participant_info.csv\n"
        "and a data_64Hz/ folder with per-participant CSVs."
    )
    raise SystemExit(1)


def _run_ablations_real(cfg: Dict[str, Any]) -> None:
    """Run ablations on the real DREAMT dataset."""
    ds_cfg = cfg["dataset"]
    tr = cfg["training"]
    n_folds = int(cfg.get("run", {}).get("n_folds", 1))
    root = _resolve_root(ds_cfg.get("root"))
    print(f"Loading DREAMT dataset from {root} ...")
    dataset = _make_dreamt_dataset(ds_cfg, root)

    print("\n" + "=" * 60)
    print("ABLATION 1: Signal Subset (2-class wake/sleep)")
    print("=" * 60)

    for subset_name, columns in SIGNAL_SUBSETS.items():
        print(f"\n--- Signal subset: {subset_name} ---")
        task = SleepStagingDREAMT(
            n_classes=2,
            signal_columns=columns,
        )
        sample_ds = dataset.set_task(task)
        raw = [sample_ds[i] for i in range(len(sample_ds))]
        print(f"  Total samples: {len(raw)}")
        results = run_training_experiment(
            raw,
            epochs=tr["epochs"],
            hidden_dim=tr["hidden_dim"],
            device=tr["device"],
            n_folds=n_folds,
        )
        print(f"  Results: {results}")

    print("\n" + "=" * 60)
    print("ABLATION 2: Label Granularity (ALL signals)")
    print("=" * 60)

    for nc in [2, 5]:
        print(f"\n--- {nc}-class ---")
        task = SleepStagingDREAMT(n_classes=nc)
        sd = dataset.set_task(task)
        raw = [sd[i] for i in range(len(sd))]
        print(f"  Total samples: {len(raw)}")
        results = run_training_experiment(
            raw,
            epochs=tr["epochs"],
            hidden_dim=tr["hidden_dim"],
            device=tr["device"],
            n_folds=n_folds,
        )
        print(f"  Results: {results}")


def _run_ablations_demo(cfg: Dict[str, Any]) -> None:
    """Run ablations on synthetic demo data."""
    tr = cfg["training"]
    n_folds = int(cfg.get("run", {}).get("n_folds", 1))
    print("=== DEMO MODE (synthetic data) ===\n")
    print("Generating 6 synthetic patients (15 epochs each) ...")

    demo_epochs = min(int(tr["epochs"]), 2)

    print("\n" + "=" * 60)
    print("ABLATION 1: Signal Subset (2-class, demo)")
    print("=" * 60)

    for subset_name, columns in SIGNAL_SUBSETS.items():
        print(f"\n--- Signal subset: {subset_name} ---")
        seed = abs(hash(subset_name)) % (2**31)
        raw = generate_demo_samples(
            n_classes=2,
            signal_columns=columns,
            n_patients=6,
            seed=seed,
        )
        print(f"  Total samples: {len(raw)}")
        results = run_training_experiment(
            raw,
            epochs=demo_epochs,
            hidden_dim=tr["hidden_dim"],
            device=tr["device"],
            split_ratios=[0.5, 0.17, 0.33],
            n_folds=n_folds,
        )
        print(f"  Results: {results}")

    print("\n" + "=" * 60)
    print("ABLATION 2: Label Granularity (demo)")
    print("=" * 60)

    for nc in [2, 5]:
        print(f"\n--- {nc}-class ---")
        raw = generate_demo_samples(
            n_classes=nc, n_patients=6, seed=123,
        )
        print(f"  Total samples: {len(raw)}")
        results = run_training_experiment(
            raw,
            epochs=demo_epochs,
            hidden_dim=tr["hidden_dim"],
            device=tr["device"],
            split_ratios=[0.5, 0.17, 0.33],
            n_folds=n_folds,
        )
        print(f"  Results: {results}")

    print("\nDemo complete.")


def _run_single_real(cfg: Dict[str, Any]) -> None:
    """One task configuration on the real DREAMT dataset."""
    ds_cfg = cfg["dataset"]
    task_cfg = cfg["task"]
    tr = cfg["training"]
    n_folds = int(cfg.get("run", {}).get("n_folds", 1))
    root = _resolve_root(ds_cfg.get("root"))
    print(f"Loading DREAMT dataset from {root} ...")
    dataset = _make_dreamt_dataset(ds_cfg, root)

    task = _task_from_config(task_cfg)
    print(
        f"\nSingle run: n_classes={task_cfg['n_classes']}, "
        f"channels={task.signal_columns}"
    )
    sample_ds = dataset.set_task(task)
    raw = [sample_ds[i] for i in range(len(sample_ds))]
    print(f"  Total samples: {len(raw)}")
    results = run_training_experiment(
        raw,
        epochs=tr["epochs"],
        hidden_dim=tr["hidden_dim"],
        device=tr["device"],
        n_folds=n_folds,
    )
    print(f"  Results: {results}")


def _run_single_demo(cfg: Dict[str, Any]) -> None:
    """One task configuration on synthetic demo data."""
    task_cfg = cfg["task"]
    tr = cfg["training"]
    n_folds = int(cfg.get("run", {}).get("n_folds", 1))
    print("=== DEMO MODE (single run, synthetic data) ===\n")

    demo_epochs = min(int(tr["epochs"]), 2)
    raw = generate_demo_samples(
        n_classes=int(task_cfg["n_classes"]),
        signal_columns=_resolve_signal_columns(task_cfg),
        n_patients=6,
        seed=123,
    )
    print(f"  Total samples: {len(raw)}")
    results = run_training_experiment(
        raw,
        epochs=demo_epochs,
        hidden_dim=tr["hidden_dim"],
        device=tr["device"],
        split_ratios=[0.5, 0.17, 0.33],
        n_folds=n_folds,
    )
    print(f"  Results: {results}")
    print("\nDemo complete.")


def _dispatch(cfg: Dict[str, Any]) -> None:
    """Run according to ``run.mode`` and ``demo`` flag."""
    mode = str(cfg.get("run", {}).get("mode", "ablations")).lower()
    if mode not in {"ablations", "single"}:
        raise SystemExit(
            f"run.mode must be 'ablations' or 'single', got {mode!r}"
        )
    demo = bool(cfg.get("demo"))
    if mode == "ablations":
        if demo:
            _run_ablations_demo(cfg)
        else:
            _run_ablations_real(cfg)
        return
    if demo:
        _run_single_demo(cfg)
    else:
        _run_single_real(cfg)


def main() -> None:
    """Entry point for the DREAMT sleep staging ablation study."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="YAML file with dataset, task, and training settings",
    )
    pre_args, argv_rest = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description="DREAMT sleep staging ablation (PyHealth RNN)",
        parents=[pre],
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Path to DREAMT dataset (overrides YAML). "
            f"When unset, search paths include {DEFAULT_ROOT}"
        ),
    )
    parser.add_argument(
        "--demo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Synthetic demo data instead of DREAMT (or --no-demo to force off)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (overrides YAML; default from YAML or 30)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="RNN hidden dimension (overrides YAML)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu or cuda (overrides YAML)",
    )
    parser.add_argument(
        "--run-mode",
        dest="run_mode",
        choices=("ablations", "single"),
        default=None,
        help="Run preset: ablations (two blocks) or single (one YAML task)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        dest="n_folds",
        metavar="K",
        help=(
            "Patient-level K-fold CV (>=2), or 1 for a single 70/10/20 split "
            "(overrides YAML run.n_folds)"
        ),
    )
    args = parser.parse_args(argv_rest)

    merged = _build_merged_config(pre_args.config, args)
    _dispatch(merged)


if __name__ == "__main__":
    main()
