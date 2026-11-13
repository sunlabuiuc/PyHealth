"""SHy on MIMIC-III next-visit diagnosis prediction: pipeline and ablations.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: Leisheng Yu, Yanxiao Cai, Minxing Zhang, Xia Hu. "Self-Explaining
    Hypergraph Neural Networks for Diagnosis Prediction." CHIL 2025 (PMLR 287).
Paper link: https://arxiv.org/abs/2502.10689

Full pipeline: MIMIC3Dataset -> NextVisitDiagnosisPredictionMIMIC3 -> patient
split 80/10/10 -> SHy (or RETAIN) -> Trainer, selecting the checkpoint on the
validation split by Recall@20 -> test Recall@10/20 and nDCG@10/20.

Experimental setup
------------------
* Data: by default PyHealth's public synthetic MIMIC-III (no credentials;
  downloaded on first use), so the script runs anywhere. Pass ``--root`` with
  a MIMIC-III v1.4 folder for the real experiments, and ``--dev`` to use only
  the first 1000 patients.
* Task: one sample per patient with at least two admissions; input = all
  admissions but the last, label = ICD-9 codes of the last admission.
* Split: by patient, 80/10/10 (the ratio the paper states), seed
  ``--seed``; code and label vocabularies cover all patients, as in the
  paper. Unlike the authors' released code, which has no validation set,
  checkpoints are chosen on validation, never on test.
* Training: Adam, lr 1e-4, batch 128, up to ``--epochs`` (default 300; the
  authors allow 450) with early stopping after ``--patience`` epochs (default
  25) without a validation Recall@20 improvement. With 50 epochs both RETAIN
  and SHy are still improving, so shorter budgets understate both.
* Metrics: Recall@k and nDCG@k (k = 10, 20), as in the paper.

Ablations and extensions (none of them is in the paper)
-------------------------------------------------------
``--experiments`` selects any of:

* ``baseline``  RETAIN (already in PyHealth) vs SHy with the paper's settings.
* ``recovery``  false-negative recovery: ``add_ratio`` in {0, 0.1, 0.2, 0.4},
  plus a control that adds the same number of *random* links at 0.2.
* ``hierarchy`` hierarchical ICD-9 embeddings vs a flat embedding table.
* ``phenotypes`` number of phenotypes K in {3, 5, 8} (the paper only
  compares K = 1 and 5). Not run on MIMIC-III.
* ``omega``     alpha-loss weight 0.04 (the authors' README table) vs the
  ``baseline`` run's 0 (their released training command).
* ``hparams``   learning rate, hidden size and dropout grid for SHy. Not
  run on MIMIC-III.
* ``time_gaps`` extension: log(1 + days since previous visit) fed to the
  phenotype GRU.
* ``task``      task configuration with RETAIN: history length
  ``max_visits`` in {all, 3, 1} and 3-digit vs full codes.
* ``protocol``  reproduction check: RETAIN and SHy under the authors'
  protocol (90/10 patient split, no validation set, checkpoint chosen on
  the test split; their ``training.py`` monitors the test prediction loss,
  this script the test Recall@20). Only for comparison with the
  ``baseline`` rows; the test score is optimistic by construction.

``omega`` and ``protocol`` compare against the ``baseline`` rows. Every other
group retrains its control row (the ``baseline`` configuration) so it is
self-contained; the repeated runs double as a run-to-run noise estimate.

Results
-------
Each run prints a table and saves it to ``<output>/results.csv``. The
MIMIC-III v1.4 results (reproduction and ablations) are in
``examples/mimic3_next_visit_diagnosis_prediction_shy.md``.

Usage:
    python examples/mimic3_next_visit_diagnosis_prediction_shy.py --quick
    python examples/mimic3_next_visit_diagnosis_prediction_shy.py \\
        --root /path/to/mimic-iii/1.4 --experiments baseline recovery
"""

import argparse
import csv
import os
import time
from typing import Any

import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RETAIN, SHy
from pyhealth.tasks import NextVisitDiagnosisPredictionMIMIC3
from pyhealth.trainer import Trainer

SYNTHETIC_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
METRICS = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]

# Paper settings (README of the authors' code, MIMIC-III column), except that
# the alpha-loss weight follows their training command (0), see ``omega``.
SHY_PAPER = {
    "dim_per_level": 32,
    "num_levels": 4,
    "hgnn_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "emb_dim": 128,
    "num_phenotypes": 5,
    "add_ratio": 0.2,
    "temperature": 1.0,
    "num_cos_heads": 10,
    "hidden_dim": 128,
    "key_dim": 256,
    "num_attention_heads": 8,
    "dropout": 0.001,
    "loss_weights": (0.003, 0.00025, 0.0),  # README table: 0.04; command: 0
}
# Small settings for --quick runs on CPU.
SHY_QUICK = dict(
    SHY_PAPER,
    dim_per_level=8,
    hgnn_dim=32,
    num_heads=2,
    emb_dim=32,
    hidden_dim=32,
    key_dim=32,
    num_attention_heads=4,
)

# (name, model, model kwargs, task kwargs, learning rate). The task kwargs may
# carry the special key ``_protocol`` ("validation", default, or "authors").
Config = tuple[str, str, dict[str, Any], dict[str, Any], float]


def experiment_configs(name: str, shy: dict[str, Any], lr: float) -> list[Config]:
    """Lists the runs of one experiment.

    Args:
        name: Experiment name (see the module docstring).
        shy: Base SHy keyword arguments.
        lr: Base learning rate.

    Returns:
        The runs as ``(label, model, model_kwargs, task_kwargs, lr)``.
    """

    def variant(**changes: Any) -> dict[str, Any]:
        return dict(shy, **changes)

    if name == "baseline":
        return [
            ("RETAIN", "RETAIN", {}, {}, lr),
            ("SHy (paper settings)", "SHy", shy, {}, lr),
        ]
    if name == "recovery":
        runs = [
            (f"SHy add_ratio={p}", "SHy", variant(add_ratio=p), {}, lr)
            for p in (0.0, 0.1, 0.2, 0.4)
        ]
        runs.append(
            (
                "SHy random links 0.2",
                "SHy",
                variant(add_ratio=0.2, random_links=True),
                {},
                lr,
            )
        )
        return runs
    if name == "hierarchy":
        return [
            ("SHy hierarchical emb.", "SHy", shy, {}, lr),
            ("SHy flat emb.", "SHy", variant(use_hierarchy=False), {}, lr),
        ]
    if name == "phenotypes":
        return [
            (f"SHy K={k}", "SHy", variant(num_phenotypes=k), {}, lr) for k in (3, 5, 8)
        ]
    if name == "omega":
        # omega = 0 (the authors' released command) is the baseline run; the
        # README table's 0.04 is trained here for comparison.
        weights = shy["loss_weights"]
        return [
            (
                "SHy omega=0.04",
                "SHy",
                variant(loss_weights=weights[:2] + (0.04,)),
                {},
                lr,
            ),
        ]
    if name == "hparams":
        runs = [(f"SHy lr={x}", "SHy", shy, {}, x) for x in (1e-3, 1e-4)]
        runs += [
            (f"SHy hidden={h}", "SHy", variant(hidden_dim=h), {}, lr) for h in (64, 256)
        ]
        runs += [
            (f"SHy dropout={d}", "SHy", variant(dropout=d), {}, lr) for d in (0.0, 0.2)
        ]
        return runs
    if name == "time_gaps":
        gaps = {"include_visit_gaps": True}
        return [
            ("SHy no time gaps", "SHy", shy, gaps, lr),
            ("SHy + time gaps", "SHy", variant(use_time_gaps=True), gaps, lr),
        ]
    if name == "task":
        return [
            ("RETAIN all visits", "RETAIN", {}, {}, lr),
            ("RETAIN max_visits=3", "RETAIN", {}, {"max_visits": 3}, lr),
            ("RETAIN max_visits=1", "RETAIN", {}, {"max_visits": 1}, lr),
            ("RETAIN 3-digit codes", "RETAIN", {}, {"code_level": "3digit"}, lr),
        ]
    if name == "protocol":
        authors = {"_protocol": "authors"}
        return [
            ("RETAIN authors' protocol", "RETAIN", {}, authors, lr),
            ("SHy authors' protocol", "SHy", shy, authors, lr),
        ]
    raise ValueError(f"Unknown experiment {name!r}.")


def run(
    dataset: MIMIC3Dataset,
    config: Config,
    args: argparse.Namespace,
    cache: dict[str, Any],
) -> dict[str, Any]:
    """Trains one configuration and evaluates it on the test split.

    Args:
        dataset: The loaded MIMIC-III dataset.
        config: ``(label, model, model_kwargs, task_kwargs, lr)``.
        args: Command-line arguments.
        cache: Splits already built, keyed by task configuration.

    Returns:
        A result row with the label, the test metrics and the run time.
    """
    label, model_name, model_kwargs, task_kwargs, lr = config
    task_kwargs = dict(task_kwargs)
    protocol = task_kwargs.pop("_protocol", "validation")
    key = repr((protocol, sorted(task_kwargs.items())))
    if key not in cache:
        # Processors (code and label vocabularies) are fit on all samples so
        # every test label has an output unit, as in the paper (which keeps
        # all codes in the label space); the patient split comes after.
        samples = dataset.set_task(NextVisitDiagnosisPredictionMIMIC3(**task_kwargs))
        ratios = (0.9, 0.0, 0.1) if protocol == "authors" else (0.8, 0.1, 0.1)
        cache[key] = split_by_patient(samples, ratios, seed=args.seed)
    train, val, test = cache[key]
    if protocol == "authors":
        # The authors' code has no validation set and selects on the test
        # split; reproduced here only as a labelled comparison row.
        val = test

    torch.manual_seed(args.seed)
    if model_name == "RETAIN":
        model = RETAIN(train)
    else:
        model = SHy(train, **model_kwargs)

    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device=args.device,
        output_path=args.output,
        exp_name=label.replace(" ", "_"),
    )
    start = time.time()
    trainer.train(
        train_dataloader=get_dataloader(train, args.batch_size, shuffle=True),
        val_dataloader=get_dataloader(val, args.batch_size, shuffle=False),
        epochs=args.epochs,
        optimizer_params={"lr": lr},
        monitor="recall@20",
        monitor_criterion="max",
        patience=args.patience,
    )
    scores = trainer.evaluate(get_dataloader(test, args.batch_size, shuffle=False))
    row = {"configuration": label, **{m: scores[m] for m in METRICS}}
    row["minutes"] = (time.time() - start) / 60
    print(
        f"{label:<28}"
        + "".join(f"{row[m]:>11.4f}" for m in METRICS)
        + f"{row['minutes']:>9.1f}"
    )
    return row


def main() -> None:
    """Parses arguments, runs the selected experiments, saves the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=SYNTHETIC_ROOT, help="MIMIC-III folder")
    parser.add_argument("--dev", action="store_true", help="first 1000 patients")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline"],
        choices=[
            "baseline",
            "recovery",
            "hierarchy",
            "phenotypes",
            "omega",
            "hparams",
            "time_gaps",
            "task",
            "protocol",
            "all",
        ],
    )
    parser.add_argument("--quick", action="store_true", help="small model, 2 epochs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="./output/shy_mimic3")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    shy = SHY_PAPER
    if args.quick:
        shy, args.epochs, args.dev = SHY_QUICK, 2, True
    names = args.experiments
    if "all" in names:
        names = [
            "baseline",
            "recovery",
            "hierarchy",
            "phenotypes",
            "omega",
            "hparams",
            "time_gaps",
            "task",
            "protocol",
        ]

    dataset = MIMIC3Dataset(
        root=args.root,
        tables=["diagnoses_icd"],
        dev=args.dev,
        cache_dir=args.cache_dir,
    )
    print(
        f"{'configuration':<28}"
        + "".join(f"{m:>11}" for m in METRICS)
        + f"{'minutes':>9}"
    )
    cache: dict[str, Any] = {}
    rows = []
    for name in names:
        for config in experiment_configs(name, shy, args.lr):
            rows.append(dict(run(dataset, config, args, cache), experiment=name))

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "configuration", *METRICS, "minutes"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
