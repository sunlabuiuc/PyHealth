"""Compare EEP and dynamic survival analysis on MIMIC-III decompensation.

Authors: Sarah Husted, Allyson Moran (Georgia Tech CSE/OAN 6250 project).

Reproduces part of Yeche et al., "Dynamic Survival Analysis for Early Event
Prediction", CHIL 2024, https://proceedings.mlr.press/v248/yeche24a.html, on
the decompensation cohort of Harutyunyan et al. (2019). Three models share the
same encoder (a causal GRU over the hourly series, :class:`RNN_V2`), patient
split and training settings, and are scored on the same hourly binary label
(death within ``horizon`` hours):

``eep``
    The baseline: one binary risk per hour, trained with cross-entropy.
``dsa_prior``
    A hazard for each of the next ``horizon`` hours, trained with the survival
    loss truncated to the horizon, with the bias of the hazard head initialized
    to the average hazard of each hour (Section 3.2 of the paper).
``dsa_noprior``
    The same model with the default initialization: the ablation of the bias
    initialization.

Each model is trained once per seed on a fixed patient split, and the test
scores are reported as mean and 95% confidence interval over seeds, with the
paired differences between arms.

Run from the repository root, for example on the demo data to check the
pipeline (the demo is too small for meaningful scores)::

    python examples/mimic3_decompensation_dsa_ablation.py \
        --root /path/to/mimic-iii-demo --epochs 3 --seeds 0 1 2

On the full data use more seeds, e.g. ``--seeds 0 1 2 3 4 5 6 7 8 9``.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader
from pyhealth.models import RNN_V2, DynamicSurvivalRNN
from pyhealth.tasks import DecompensationMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "pyhealth" / "datasets" / "configs" / "mimic3_icu.yaml"
)
ARMS = ("eep", "dsa_prior", "dsa_noprior")
METRICS = ["roc_auc", "pr_auc"]

# Two-sided 95% critical values of Student's t, by degrees of freedom.
T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def summarize(values):
    """Mean and the half-width of the 95% confidence interval of ``values``.

    Uses Student's t with ``len(values) - 1`` degrees of freedom. The half-width
    is NaN for fewer than two values.
    """
    values = [float(v) for v in values]
    if len(values) < 2:
        return {"mean": mean(values), "ci95": float("nan"), "n": len(values)}
    df = len(values) - 1
    known = [d for d in T_95 if d <= df]
    t = T_95[max(known)] if df <= 30 else 1.96
    half = t * stdev(values) / math.sqrt(len(values))
    return {"mean": mean(values), "ci95": half, "n": len(values)}


def make_loader(data, args, train):
    return get_dataloader(
        data,
        batch_size=args.batch_size,
        shuffle=train,
        **({"seed": args.split_seed} if train else {}),
    )


def build_model(arm, args, binary_train, survival_train):
    """A fresh model of an arm (seed set by the caller)."""
    shared = {
        "embedding_dim": {"dynamic": [{"values": args.embedding_dim}]},
        "hidden_dim": args.hidden_dim,
        "rnn_type": args.rnn_type,
        "dropout": args.dropout,
        "bidirectional": False,
        "mask": "concat",
    }
    if arm == "eep":
        return RNN_V2(dataset=binary_train, **shared)
    prior = None
    if arm == "dsa_prior":
        prior = DynamicSurvivalRNN.prior_from_samples(survival_train, args.horizon)
    return DynamicSurvivalRNN(
        dataset=survival_train,
        horizon=args.horizon,
        hazard_prior=prior,
        **shared,
    )


def train_and_test(model, loaders, args, output_dir, name):
    """Trains with early stopping on the validation loss; scores the test set."""
    train_loader, val_loader, test_loader = loaders
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device=args.device,
        output_path=str(output_dir),
        exp_name=name,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
        max_grad_norm=5.0,
    )
    return {k: float(v) for k, v in trainer.evaluate(test_loader).items()}


def aggregate(runs, arms):
    """Mean and 95% CI of every metric per arm, and paired differences."""
    metrics = [m for m in METRICS if m in runs[arms[0]][0]]
    summary = {arm: {m: summarize([r[m] for r in runs[arm]]) for m in metrics}
               for arm in arms}
    pairs = {}
    for a, b in (("dsa_prior", "eep"), ("dsa_prior", "dsa_noprior")):
        if a in runs and b in runs:
            pairs[f"{a} - {b}"] = {
                m: summarize([x[m] - y[m] for x, y in zip(runs[a], runs[b])])
                for m in metrics
            }
    return summary, pairs


def print_table(summary, pairs):
    print("\nTest metrics, mean +- 95% CI over seeds")
    for arm, metrics in summary.items():
        cells = "  ".join(
            f"{m} {s['mean']:.4f} +- {s['ci95']:.4f}" for m, s in metrics.items()
        )
        print(f"  {arm:12s} {cells}  (n={next(iter(metrics.values()))['n']})")
    print("Paired differences, mean +- 95% CI")
    for pair, metrics in pairs.items():
        cells = "  ".join(
            f"{m} {s['mean']:+.4f} +- {s['ci95']:.4f}" for m, s in metrics.items()
        )
        print(f"  {pair:28s} {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MIMIC-III directory")
    parser.add_argument(
        "--config-path", default=str(DEFAULT_CONFIG), help="Dataset config yaml"
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/dsa"))
    parser.add_argument("--horizon", type=int, default=24, help="Whole hours")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--rnn-type", choices=["GRU", "LSTM", "RNN"], default="GRU")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None, help="E.g. cpu or cuda:0")
    parser.add_argument("--dev", action="store_true", help="Use a few patients")
    args = parser.parse_args()

    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    base_dataset = MIMIC3Dataset(
        root=args.root,
        tables=["chartevents", "labevents"],
        config_path=args.config_path,
        dev=args.dev,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    # The same patients go to each split in both tasks (same ratios and seed),
    # so every arm trains and is tested on the same stays.
    split = PatientSplit(ratios=tuple(args.ratios), seed=args.split_seed)
    binary = base_dataset.set_task(
        DecompensationMIMIC3(horizon=args.horizon),
        split=split,
        num_workers=args.num_workers,
    )
    survival = base_dataset.set_task(
        DecompensationMIMIC3(horizon=args.horizon, label_mode="survival"),
        split=split,
        num_workers=args.num_workers,
    )
    sizes = {n: len(d) for n, d in zip(("train", "validation", "test"), binary)}
    if [len(d) for d in binary] != [len(d) for d in survival]:
        raise RuntimeError("The binary and survival splits differ in size.")
    if not all(sizes.values()):
        raise ValueError(f"Empty split {sizes}; use more patients.")
    print(f"horizon {args.horizon} h, samples per split: {sizes}", flush=True)

    loaders = {
        "eep": [make_loader(d, args, i == 0) for i, d in enumerate(binary)],
        "dsa": [make_loader(d, args, i == 0) for i, d in enumerate(survival)],
    }
    runs = {arm: [] for arm in args.arms}
    for seed in args.seeds:
        for arm in args.arms:
            set_seed(seed)
            model = build_model(arm, args, binary[0], survival[0])
            scores = train_and_test(
                model,
                loaders["eep" if arm == "eep" else "dsa"],
                args,
                output_dir,
                f"{arm}_seed{seed}",
            )
            print(f"seed {seed} {arm}: {scores}", flush=True)
            runs[arm].append(scores)
            (output_dir / "runs.json").write_text(json.dumps(runs, indent=2))

    summary, pairs = aggregate(runs, args.arms)
    print_table(summary, pairs)
    result = {
        "horizon": args.horizon,
        "samples": sizes,
        "seeds": args.seeds,
        "runs": runs,
        "summary": summary,
        "paired_differences": pairs,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nLogs, checkpoints and results: {output_dir}")


if __name__ == "__main__":
    main()
