"""TaskAug on PTB-XL: learned data augmentation for binary ECG diagnosis.

Reproduces the PTB-XL experiment of Raghu et al., "Data Augmentation for
Electrocardiograms" (CHIL 2022, https://proceedings.mlr.press/v174/raghu22a.html)
with ``PTBXLDataset``, ``PTBXLBinaryClassification`` and ``TaskAug``.

Authors:
    Allen Zhang             (allenz3@illinois.edu)
    Bjorn Thorleifsson      (bthor5@illinois.edu)
    Rohan Vanjani           (vanjani3@illinois.edu)

Usage
-----
On synthetic signals, no download needed (a few minutes on a laptop CPU)::

    python examples/ecg/ptbxl/ptbxl_binary_classification_taskaug.py --synthetic

On PTB-XL v1.0.3 (https://physionet.org/content/ptb-xl/1.0.3/, open access;
``pip install 'pyhealth[ptbxl]'``)::

    python examples/ecg/ptbxl/ptbxl_binary_classification_taskaug.py \\
        --root /data/ptb-xl/1.0.3 --experiment main hparam transfer

Each finished run is a line of ``results.jsonl`` in ``--output-dir``, and the
run in progress is checkpointed every epoch. Rerunning the command skips
finished runs and resumes the unfinished one.

Setup
-----
- Data: 12-lead ECGs at 250 Hz (2496 samples), with a binary label for one
  diagnostic superclass (``--target``, default MI). Fold 10 is the test set.
  For each seed, 1000 ECGs (``--n-dev``) are drawn by patient from folds 1 to
  9 and split 80/20 by patient into training and validation. Signals are
  standardized with training-set statistics.
- Training: ``train_taskaug`` for every arm. Adam at 1e-3, batch 128, 100
  epochs, keeping the epoch with the lowest unaugmented validation loss.
  TaskAug's policy: 2 stages, RMSprop at 1e-2, after 5 warm-up epochs.
- Metrics: test AUROC and AUPRC from ``Trainer.evaluate``.

Experiments (``--experiment``):

- ``main``: the paper's ResNet with no augmentation (``noaug``), a random 10%
  time mask (``timemask``), the TaskAug policy frozen at its initial values
  (``initaug``) and the learned policy (``taskaug``).
- ``hparam``: TaskAug with 1 to 3 stages and a policy learning rate of 1e-3,
  1e-2 or 1e-1. The paper fixes both.
- ``transfer`` (our extension): SparcNet trained with no augmentation, the
  initial policy, or the policy the ResNet learned on the same split.

Results
-------
MI, N = 1000, two Kaggle T4s. Percent, mean ± standard error over seeds.

Main, 20 seeds, against the paper's Tables 2 and 5 (5 splits)::

    arm        AUROC        AUPRC        paper AUROC   paper AUPRC
    noaug      80.6 ± 0.6   60.4 ± 0.8   80.0 ± 0.8    59.2 ± 2.1
    timemask   81.0 ± 0.4   60.8 ± 0.8   81.1 ± 1.3    60.3 ± 1.3
    initaug    80.0 ± 0.5   60.3 ± 0.7
    taskaug    80.5 ± 0.4   60.5 ± 0.8   82.3 ± 0.5    63.1 ± 1.7

Hyperparameter ablation, AUROC, seeds 0 to 2 (no augmentation: 81.3 ± 0.9)::

               policy lr 1e-3   1e-2         1e-1
    K = 1      82.6 ± 0.8       82.1 ± 0.8   81.3 ± 0.1
    K = 2      81.8 ± 0.3       81.5 ± 0.6   81.1 ± 2.1
    K = 3      81.0 ± 0.8       79.6 ± 0.2   81.8 ± 0.9

Transfer to SparcNet, AUROC, seeds 0 to 2: no augmentation 73.0 ± 2.7, initial
policy 74.5 ± 2.0, transferred policy 76.4 ± 2.9.

Findings:

- The baseline reproduces, but TaskAug's gain does not. Paired by seed,
  TaskAug minus no augmentation is -0.0 ± 0.7 AUROC points (95% CI -1.4 to
  +1.3); the paper reports +2.3. Time masking gives +0.4 ± 0.5 (paper +1.1).
- Seeds matter more than arms: AUROC varies by 2.5 points between splits.
  Our first 5 seeds put the baseline at 82.2, two points above the paper.
- In the ablation no setting clearly beats the default. Fewer stages tend to
  do better; the best cell (K = 1, lr 1e-3) is the best of nine on 3 seeds.
- The transferred policy helps SparcNet by +3.4 ± 4.7 points, paired:
  the right direction, but inconclusive with 3 seeds.
- Cost on a T4: 2.6 min for a fixed-policy run, 20.6 min for TaskAug.
- Protocol differences from the reference code: patient-level splits (it
  splits records at random), training-set normalization only, a 10% time
  mask (its policy uses 5%), and the policy step reuses the training batch.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ResNet1D, SparcNet, TaskAug, TaskAugPolicy, train_taskaug
from pyhealth.trainer import Trainer

# The arms that train under a policy that is never updated.
FIXED_POLICIES = {
    "noaug": lambda: TaskAugPolicy(operations=("noop",), num_stages=1),
    "timemask": lambda: TaskAugPolicy(operations=("mask",), num_stages=1),
    "initaug": lambda: TaskAugPolicy(),
}


def synthetic_splits(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Noisy 12-lead signals; positives carry a short burst at a random time."""
    rng = np.random.default_rng(seed)

    def split(name: str, n_samples: int) -> list[dict]:
        samples = []
        for i in range(n_samples):
            signal = rng.normal(size=(12, 256)).astype(np.float32)
            if i % 2:
                start = rng.integers(0, 256 - 32)
                signal[:, start : start + 32] += np.hanning(32).astype(np.float32)
            samples.append(
                {"patient_id": f"{name}-{i}", "signal": signal, "label": i % 2}
            )
        return samples

    return split("train", 128), split("val", 64), split("test", 256)


class PTBXLSplits:
    """Patient-level development and test splits of PTB-XL for one target."""

    def __init__(self, root: Path, target: str, n_dev: int, cache_dir: str | None):
        import pandas as pd

        from pyhealth.datasets import PTBXLDataset
        from pyhealth.tasks import PTBXLBinaryClassification

        dataset = PTBXLDataset(root=str(root), sampling_rate=500, cache_dir=cache_dir)
        task = PTBXLBinaryClassification(dataset.scp_statements_path, target=target)
        self.samples = dataset.set_task(task)
        # Splits are made from the metadata, so no signal is decoded for them.
        meta = pd.read_csv(
            root / "ptbxl_database.csv",
            usecols=["ecg_id", "patient_id", "strat_fold"],
        )
        self.dev = meta[meta.strat_fold != 10]
        self.n_dev = n_dev
        test = meta[meta.strat_fold == 10]
        self.test_patients = set(test.patient_id)
        self.test = self._read(test)  # the same for every seed

    def _read(self, records) -> list[dict]:
        rows = []
        for ecg_id, patient_id in zip(records.ecg_id, records.patient_id):
            (index,) = self.samples.record_to_index[str(ecg_id)]
            sample = self.samples[index]
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "signal": sample["signal"],
                    "label": int(sample["label"]),
                }
            )
        return rows

    def __call__(self, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
        """Draws a development set and normalizes all splits with its train part."""
        rng = np.random.default_rng(seed)
        patients = rng.permutation(self.dev.patient_id.unique())
        seen = self.dev.groupby("patient_id").size().loc[patients].cumsum()
        n_train = 0.8 * self.n_dev
        train_patients = set(seen.index[seen <= n_train])
        val_patients = set(seen.index[(seen > n_train) & (seen <= self.n_dev)])
        assert not train_patients & val_patients
        assert not (train_patients | val_patients) & self.test_patients
        train = self._read(self.dev[self.dev.patient_id.isin(train_patients)])
        val = self._read(self.dev[self.dev.patient_id.isin(val_patients)])

        signals = torch.stack([sample["signal"] for sample in train])
        mean, std = signals.mean(), signals.std()

        def normalize(samples: list[dict]) -> list[dict]:
            return [
                {**sample, "signal": (sample["signal"] - mean) / std}
                for sample in samples
            ]

        return normalize(train), normalize(val), normalize(self.test)


def run(
    datasets, policy, args, seed, checkpoint, backbone="resnet", policy_lr=1e-2
) -> dict:
    """Trains one model, resuming from ``checkpoint``, and scores it on the test set."""
    train, val, test = datasets
    torch.manual_seed(seed)
    if backbone == "sparcnet":
        network = SparcNet(train)
    elif args.synthetic:  # narrow, so the demo is fast on a CPU
        network = ResNet1D(train, kernel_size=7, channels=(8, 16, 32), blocks=(1, 1, 1))
    else:
        network = ResNet1D(train)
    model = TaskAug(train, backbone=network, policy=policy).to(args.device)
    train_loader = get_dataloader(train, args.batch_size, shuffle=True, seed=seed)
    val_loader = get_dataloader(val, args.batch_size, shuffle=True, seed=seed)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = train_taskaug(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        policy_lr=policy_lr,
        warmup_epochs=args.warmup_epochs,
        checkpoint_path=checkpoint,
    )
    trainer = Trainer(
        model, metrics=["roc_auc", "pr_auc"], device=args.device, enable_logging=False
    )
    scores = trainer.evaluate(get_dataloader(test, args.batch_size))
    peak = None
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
    elif args.device == "mps":  # no peak counter: what the driver holds now
        peak = torch.mps.driver_allocated_memory() / 2**30
    return {
        "auroc": scores["roc_auc"],
        "auprc": scores["pr_auc"],
        "seconds": sum(history["seconds"]),  # training, over all sessions
        "peak_gpu_gb": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--synthetic", action="store_true", help="no download needed")
    parser.add_argument("--root", type=Path, help="PTB-XL v1.0.3 directory")
    parser.add_argument("--target", default="MI", help="MI, HYP, STTC, CD or NORM")
    parser.add_argument("--n-dev", type=int, default=1000)
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=["main", "hparam", "transfer"],
        choices=["main", "hparam", "transfer"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    parser.add_argument("--device", default=device)
    parser.add_argument("--cache-dir", default=None, help="PyHealth cache directory")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("taskaug_ptbxl_results")
    )
    args = parser.parse_args()
    if args.synthetic:
        make_splits = synthetic_splits
        defaults = {"seeds": [0, 1], "epochs": 10, "warmup_epochs": 3}
        defaults["batch_size"] = 32
    elif args.root is not None:
        make_splits = PTBXLSplits(args.root, args.target, args.n_dev, args.cache_dir)
        defaults = {"seeds": [0, 1, 2, 3, 4], "epochs": 100, "warmup_epochs": 5}
        defaults["batch_size"] = 128
    else:
        parser.error("pass --root or --synthetic")
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    target = "synthetic" if args.synthetic else args.target
    # Runs already in results.jsonl are not repeated, so a job that was cut
    # short picks up where it stopped.
    done = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            if (row["target"], row["epochs"]) == (target, args.epochs):
                done[row["experiment"], row["setting"], row["seed"]] = row
    results = defaultdict(list)

    def record(experiment, setting, seed, datasets, policy, **kwargs) -> None:
        """Trains and scores one setting, unless results.jsonl has it."""
        row = done.get((experiment, setting, seed))
        status = "done earlier"
        if row is None:
            name = f"checkpoint_{experiment}_{setting}_{seed}.pt".replace(" ", "_")
            checkpoint = args.output_dir / name
            row = {
                "experiment": experiment,
                "setting": setting,
                "seed": seed,
                "target": target,
                "epochs": args.epochs,
                **run(datasets, policy, args, seed, checkpoint, **kwargs),
            }
            with results_path.open("a") as file:
                file.write(json.dumps(row) + "\n")
            checkpoint.unlink()
            status = f"{row['seconds']:.0f} s"
        results[experiment, setting].append(row)
        print(
            f"[{experiment}] {setting} seed {seed}: AUROC {row['auroc']:.4f} "
            f"AUPRC {row['auprc']:.4f} ({status})",
            flush=True,
        )

    schema = {
        "input_schema": {"signal": "tensor"},
        "output_schema": {"label": "binary"},
    }
    for seed in args.seeds:
        datasets = [
            create_sample_dataset(split, **schema) for split in make_splits(seed)
        ]
        if "main" in args.experiment:
            for arm, make_policy in FIXED_POLICIES.items():
                policy = make_policy().requires_grad_(False)
                record("main", arm, seed, datasets, policy)

        # The default TaskAug run is main's last arm, the K=2 lr=0.01 cell of
        # hparam, and the source of the policy that transfer reuses.
        learned = TaskAugPolicy()
        policy_path = args.output_dir / f"policy_seed{seed}.pt"
        if ("main", "taskaug", seed) in done and policy_path.exists():
            learned.load_state_dict(torch.load(policy_path, map_location="cpu"))
        else:
            done.pop(("main", "taskaug", seed), None)
        record("main", "taskaug", seed, datasets, learned)
        torch.save(learned.state_dict(), policy_path)

        if "hparam" in args.experiment:
            for stages in (1, 2, 3):
                for policy_lr in (1e-3, 1e-2, 1e-1):
                    if (stages, policy_lr) == (2, 1e-2):
                        continue
                    setting = f"K={stages} lr={policy_lr:g}"
                    policy = TaskAugPolicy(num_stages=stages)
                    record(
                        "hparam", setting, seed, datasets, policy, policy_lr=policy_lr
                    )

        if "transfer" in args.experiment:
            arms = [
                ("sparcnet noaug", FIXED_POLICIES["noaug"]()),
                ("sparcnet initaug", FIXED_POLICIES["initaug"]()),
                ("sparcnet transferred", learned),
            ]
            for arm, policy in arms:
                policy.requires_grad_(False)
                record("transfer", arm, seed, datasets, policy, backbone="sparcnet")

    print(f"\n{'experiment':10s} {'setting':22s} {'n':>2s}  AUROC            AUPRC")
    for (experiment, setting), rows in results.items():
        cells = []
        for metric in ("auroc", "auprc"):
            values = np.array([row[metric] for row in rows])
            error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            cells.append(f"{values.mean():.4f} ± {error:.4f}")
        print(f"{experiment:10s} {setting:22s} {len(rows):2d}  {'  '.join(cells)}")
    print(f"\nPer-run results: {results_path}")


if __name__ == "__main__":
    main()
