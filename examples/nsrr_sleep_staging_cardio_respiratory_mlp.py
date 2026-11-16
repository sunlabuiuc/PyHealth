"""Ablation of cardio-respiratory sleep staging on NSRR data with an MLP.

Reproduction target: wav2sleep (Carter & Tarassenko, ML4H 2024,
https://arxiv.org/abs/2411.04644) stages sleep from ECG, PPG and
thoracic/abdominal respiratory effort, and reports that combining modalities
helps. This script exercises the PyHealth contribution (``NSRRSleepDataset``
+ ``SleepStagingCardioRespiratory``) and runs two small ablations with the
existing ``MLP`` model:

  1. Signal modalities: ECG only / respiration only (THX+ABD) / all three.
  2. Label definition: 4-class (Wake/Light/Deep/REM), 5-class (AASM) and
     binary sleep/wake.

By default the script is fully self-contained: it writes a few tiny synthetic
NSRR-style recordings (EDF + profusion XML) to a temporary directory. The toy
signals are built so that ECG separates {Wake, REM} from {Light, Deep} and
respiration separates {Wake, Deep} from {Light, REM}; the numbers therefore
only demonstrate that the pipeline and the ablation work, NOT the paper's
results. To run on real data, download SHHS or MESA from https://sleepdata.org
(an approved data-use request is required) and pass ``--root``.

Examples:
    python examples/nsrr_sleep_staging_cardio_respiratory_mlp.py
    python examples/nsrr_sleep_staging_cardio_respiratory_mlp.py \\
        --root /path/to/shhs_sample --cohort shhs --cache-dir /path/to/cache
"""

import argparse
import os
import tempfile

import numpy as np

from pyhealth.datasets import NSRRSleepDataset, get_dataloader, split_by_patient
from pyhealth.models import MLP
from pyhealth.tasks import SleepStagingCardioRespiratory
from pyhealth.trainer import Trainer

# (class id in NSRR profusion codes) -> offset of the toy signals. Codes:
# 0 Wake, 2 N2 (Light), 3 N3 (Deep), 5 REM.
ECG_LEVEL = {0: 1.0, 2: -1.0, 3: -1.0, 5: 1.0}
RESP_LEVEL = {0: 1.0, 2: -1.0, 3: 1.0, 5: -1.0}


def _field(value, width):
    return str(value).ljust(width)[:width].encode("ascii")


def write_edf(path, channels, n_seconds):
    """Writes a minimal EDF file. ``channels`` is [(label, rate, signal)]."""
    ns = len(channels)
    header = (
        _field("0", 8)
        + _field("synthetic", 160)
        + _field("01.01.0000.00.00", 16)
        + _field(256 * (ns + 1), 8)
        + _field("", 44)
        + _field(n_seconds, 8)
        + _field(1, 8)
        + _field(ns, 4)
    )
    per_signal = [
        (16, lambda c: c[0]),
        (80, lambda c: ""),
        (8, lambda c: "uV"),
        (8, lambda c: -100),
        (8, lambda c: 100),
        (8, lambda c: -32768),
        (8, lambda c: 32767),
        (80, lambda c: ""),
        (8, lambda c: c[1]),
        (32, lambda c: ""),
    ]  # label, transducer, unit, pmin, pmax, dmin, dmax, prefilter, nsamp, rsv
    signal_header = b"".join(_field(f(c), w) for w, f in per_signal for c in channels)
    columns = []
    for _, rate, signal in channels:
        digital = np.clip((signal + 100.0) / 200.0 * 65535.0 - 32768.0, -32768, 32767)
        columns.append(digital.astype("<i2").reshape(n_seconds, rate))
    with open(path, "wb") as f:
        f.write(header + signal_header)
        f.write(np.concatenate(columns, axis=1).tobytes())


def write_synthetic_cohort(root, n_patients, n_epochs, seed=0):
    """Writes ``n_patients`` toy SHHS-style nights under ``root``."""
    rng = np.random.default_rng(seed)
    for p in range(n_patients):
        # Stages come in blocks of 3-6 epochs.
        stages = []
        while len(stages) < n_epochs:
            stages += [int(rng.choice([0, 2, 3, 5]))] * int(rng.integers(3, 7))
        stages = stages[:n_epochs]
        n_seconds = 30 * n_epochs

        def toy(rate, levels):
            per_epoch = np.repeat([levels[s] for s in stages], 30 * rate)
            return 10.0 * (per_epoch + rng.normal(0, 1.0, n_epochs * 30 * rate))

        stem = f"shhs1-{200001 + p}"
        write_edf(
            os.path.join(root, f"{stem}.edf"),
            [
                ("ECG", 125, toy(125, ECG_LEVEL)),
                ("THOR RES", 10, toy(10, RESP_LEVEL)),
                ("ABDO RES", 10, toy(10, RESP_LEVEL)),
            ],
            n_seconds,
        )
        body = "".join(f"<SleepStage>{s}</SleepStage>" for s in stages)
        with open(os.path.join(root, f"{stem}-profusion.xml"), "w") as f:
            f.write(
                "<?xml version='1.0'?><CMPStudyConfig><EpochLength>30</EpochLength>"
                f"<SleepStages>{body}</SleepStages></CMPStudyConfig>"
            )


def run_experiment(dataset, modalities, label_scheme, epochs, seed):
    """Trains an MLP for one task configuration and returns test metrics."""
    task = SleepStagingCardioRespiratory(
        modalities=modalities, label_scheme=label_scheme
    )
    samples = dataset.set_task(task)
    train_ds, val_ds, test_ds = split_by_patient(samples, [0.6, 0.2, 0.2], seed=seed)
    loaders = [
        get_dataloader(ds, batch_size=64, shuffle=(i == 0))
        for i, ds in enumerate((train_ds, val_ds, test_ds))
    ]
    model = MLP(dataset=samples, embedding_dim=32, hidden_dim=32)
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "cohen_kappa"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=loaders[0],
        val_dataloader=loaders[1],
        epochs=epochs,
        monitor="accuracy",
    )
    return trainer.evaluate(loaders[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="folder with NSRR EDF + profusion XML files")
    parser.add_argument("--cohort", default="shhs", choices=["shhs", "mesa"])
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument("--patients", type=int, default=8, help="synthetic nights")
    parser.add_argument("--night-epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = args.root
        if root is None:
            root = os.path.join(tmp, "synthetic_shhs")
            os.makedirs(root)
            write_synthetic_cohort(root, args.patients, args.night_epochs, args.seed)
            print(f"Wrote {args.patients} synthetic nights (toy signals).")
        dataset = NSRRSleepDataset(
            root, cohort=args.cohort, cache_dir=args.cache_dir or os.path.join(tmp, "c")
        )

        # Ablation 1: signal modalities (label scheme fixed to 4-class).
        # Ablation 2: label definition (all SHHS modalities).
        all_modalities = ("ecg", "thx", "abd")
        if args.cohort == "mesa":
            all_modalities += ("ppg",)
        configs = [
            ("ECG only", ("ecg",), "four_class"),
            ("Respiration only (THX+ABD)", ("thx", "abd"), "four_class"),
            ("All modalities", all_modalities, "four_class"),
            ("All modalities, 5-class", all_modalities, "five_class"),
            ("All modalities, sleep/wake", all_modalities, "sleep_wake"),
        ]
        results = []
        for name, modalities, scheme in configs:
            print(f"\n=== {name} ({scheme}) ===")
            metrics = run_experiment(
                dataset, modalities, scheme, args.epochs, args.seed
            )
            results.append((name, scheme, metrics))

        print(
            "\n{:<30}{:<12}{:>10}{:>14}".format(
                "Configuration", "Labels", "Acc", "Cohen kappa"
            )
        )
        for name, scheme, m in results:
            print(
                "{:<30}{:<12}{:>10.3f}{:>14.3f}".format(
                    name, scheme, m["accuracy"], m["cohen_kappa"]
                )
            )


if __name__ == "__main__":
    # BaseDataset spawns worker processes; keep the main-module guard.
    main()
