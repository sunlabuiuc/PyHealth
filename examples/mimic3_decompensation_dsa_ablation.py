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

Besides the time-step ROC-AUC and PR-AUC, the risks of the test stays go
through an alarm policy (a threshold with silencing) and are scored with the
event-level metrics of Section 4.1 of the paper (:mod:`pyhealth.metrics.event_alarm`):
the area under the alarm precision / event recall curve, the event recall at
alarm precisions of 60, 70 and 80%, and the distance of the first alarm to the
event. They show how early and how often a model warns, which time-step scores
do not. The per-step test risks of each run are saved as ``*_test_risks.npz``.

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
from itertools import pairwise
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader
from pyhealth.metrics import event_alarm_metrics
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
# Event-level metrics of the alarm policy (see pyhealth.metrics.event_alarm).
PRECISIONS = (0.6, 0.7, 0.8)
EVENT_METRICS = (
    ["alarm_event_auprc"]
    + [f"event_recall@{round(p * 100)}" for p in PRECISIONS]
    + [f"first_alarm_distance@{round(p * 100)}" for p in PRECISIONS]
)

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


def stay_labels(loader):
    """Per stay of a survival-mode loader: step times, time to event, event, label.

    Read in loader order, so the loader must not shuffle.
    """
    stays = []
    for batch in loader:
        tte, time, offsets = (np.asarray(t.cpu()) for t in batch["time_to_event"])
        event = np.asarray(batch["event"][0].cpu()).reshape(-1)
        label = np.asarray(batch["decompensation"][0].cpu()).reshape(-1)
        tte = tte.reshape(-1)
        for a, b in pairwise(offsets):
            stays.append(
                {
                    "times": time[a:b],
                    "time_to_event": tte[a:b],
                    "event": event[a:b],
                    "label": label[a:b],
                }
            )
    return stays


def event_scores(risks, stays, args):
    """Alarm metrics of per-stay risks, checked against the stays' labels."""
    if len(risks) != len(stays):
        raise RuntimeError(f"{len(risks)} predicted stays, {len(stays)} labelled.")
    risks = [np.asarray(r, dtype=float).reshape(-1) for r in risks]
    for r, s in zip(risks, stays):
        if len(r) != len(s["times"]):
            raise RuntimeError("A stay's predictions and labels differ in length.")
    return event_alarm_metrics(
        scores=risks,
        times=[s["times"] for s in stays],
        time_to_event=[s["time_to_event"] for s in stays],
        event=[s["event"] for s in stays],
        horizon=args.horizon,
        silence=args.silence_hours,
        precisions=PRECISIONS,
    )


def train_and_test(model, loaders, args, output_dir, name, stays):
    """Trains with early stopping on the validation loss; scores the test set.

    Besides the time-step metrics, the risks of the test stays are scored with
    the alarm policy, and saved next to the logs.
    """
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
    scores = {k: float(v) for k, v in trainer.evaluate(test_loader).items()}
    y_true, risks, _ = trainer.inference(test_loader)
    for truth, s in zip(y_true, stays):
        if not np.array_equal(np.asarray(truth).reshape(-1), s["label"]):
            raise RuntimeError("Predicted stays are not in the order of the labels.")
    scores.update(event_scores(risks, stays, args))
    np.savez_compressed(
        output_dir / f"{name}_test_risks.npz",
        **{f"stay{i}": np.asarray(r).reshape(-1) for i, r in enumerate(risks)},
    )
    return scores


def aggregate(runs, arms):
    """Mean and 95% CI of every metric per arm, and paired differences."""
    # a metric is reported if every run has it (first-alarm distances are NaN
    # where no precision level is reached)
    metrics = [
        m
        for m in METRICS + EVENT_METRICS
        if all(m in r and not math.isnan(r[m]) for arm in arms for r in runs[arm])
    ]
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
    arms = list(summary)
    metrics = list(summary[arms[0]])
    n = summary[arms[0]][metrics[0]]["n"]
    print(f"\nTest metrics, mean +- 95% CI over {n} seeds")
    for m in metrics:
        print(f"  {m}")
        for arm in arms:
            s = summary[arm][m]
            print(f"    {arm:12s} {s['mean']:.4f} +- {s['ci95']:.4f}")
    print("Paired differences, mean +- 95% CI")
    for pair, per_metric in pairs.items():
        print(f"  {pair}")
        for m, s in per_metric.items():
            print(f"    {m:26s} {s['mean']:+.4f} +- {s['ci95']:.4f}")


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
    parser.add_argument(
        "--silence-hours",
        type=float,
        default=1.0,
        help="Silence after an alarm; the paper uses 1 h on MIMIC-III",
    )
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
    stays = stay_labels(loaders["dsa"][2])
    n_events = sum(bool(np.any(s["event"])) for s in stays)
    print(f"test stays: {len(stays)}, with an event: {n_events}", flush=True)
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
                stays,
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
