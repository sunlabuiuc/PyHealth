"""Reproduce the PTB-XL ECG benchmark of Nonaka & Seita (2021) in PyHealth.

Paper: *In-depth Benchmarking of Deep Neural Network Architectures for ECG
Diagnosis*, PMLR 149:414-439
(https://proceedings.mlr.press/v149/nonaka21a.html); reference code:
https://github.com/seitalab/dnn_ecg_comparison.

This driver trains the PyHealth port ``pyhealth.models.ResNet1D`` with the
benchmark's full protocol:

- one of the five official PTB-XL fold pairs (``PTBXL_PAPER_FOLD_SPLITS``),
  or all five with ``--split all``;
- per-lead normalization with mean/std fitted on the training folds only
  (``compute_ptbxl_lead_stats`` + ``standardize="train_stats"``);
- a random 2.5 s crop per record while training
  (``training_crop_len=1250`` at 500 Hz);
- sliding-window inference at validation/test: 2.5 s windows, 1.25 s
  stride, class logits aggregated by element-wise maximum
  (``eval_window_len`` / ``eval_window_stride``);
- for the 3-way multi-class task, cross-entropy weighted
  ``[1, n_target/n_normal, n_other/n_normal]`` from the train split
  (``compute_multiclass_class_weights``);
- optional loss extensions beyond the benchmark baseline, off by default:
  ``--pos-weight`` weights the multi-label BCE by each label's
  negatives/positives in the training split
  (``compute_multilabel_pos_weight``), and ``--label-smoothing`` smooths
  the multi-class cross-entropy targets;
- Adam with a fixed learning rate, model selection on the best validation
  score (macro ROC-AUC for multi-label, macro F1 for multi-class), early
  stopping, and up to ``--epochs`` 250 epochs, as in the paper;
- with more than one ``--depth``, the comparison the paper actually makes:
  every arm gets one run per split, the arms report the mean and standard
  deviation of those per-split scores, and each arm is tested against the
  baseline arm (ResNet-18 by default) with a two-sided Welch's t-test
  corrected across settings with Benjamini-Hochberg
  (``pyhealth.metrics.welch_relative_improvement_test``). Arms with no
  relative improvement are excluded from testing.

The paper's headline numbers for ResNet-18 on PTB-XL, for comparison with
``--split all``: macro ROC-AUC 0.914 ± 0.011 (all), 0.928 ± 0.007
(diagnostic), 0.932 ± 0.013 (sub-diagnostic), 0.933 ± 0.006
(super-diagnostic), 0.882 ± 0.011 (form), 0.921 ± 0.022 (rhythm), in its
Table 12; macro F1 0.862 ± 0.013 (AF), 0.620 ± 0.029 (I-AVB), 0.874 ±
0.023 (LBBB), 0.766 ± 0.039 (RBBB), in Tables 14-17. The ResNet-34 and
ResNet-50 rows sit in the same tables, and the paper's conclusion was that
no architecture improved significantly over ResNet-18.

Requires the PTB-XL v1.0.3 corpus (https://physionet.org/content/ptb-xl/)
and ``pip install 'pyhealth[ptbxl]'``. Examples:

    # one split, one architecture (the paper's ResNet-18 row)
    python examples/ecg/ptbxl/ptbxl_multilabel_classification_resnet1d.py \
        --root ~/.cache/pyhealth/datasets/ptbxl/1.0.3 \
        --split 1 --task multilabel --label-set diagnostic

    # the paper's five-split sweep, three depths, tested against ResNet-18
    python examples/ecg/ptbxl/ptbxl_multilabel_classification_resnet1d.py \
        --root ~/.cache/pyhealth/datasets/ptbxl/1.0.3 \
        --split all --task multilabel --label-set all --depth 18 34 50 \
        --result-json ptbxl_all_resnet1d.json

    # multi-class task, one of the paper's four target diagnoses
    python examples/ecg/ptbxl/ptbxl_multilabel_classification_resnet1d.py \
        --root ~/.cache/pyhealth/datasets/ptbxl/1.0.3 \
        --split all --task multiclass --dx AF --depth 18 50

    # no PhysioNet corpus? self-check the whole pipeline on synthetic data
    python examples/ecg/ptbxl/ptbxl_multilabel_classification_resnet1d.py \
        --synthetic-root /tmp/ptbxl-synthetic --split 1 --task multilabel \
        --epochs 1

``--synthetic-root`` writes a small pseudo PTB-XL corpus (deterministic
ramp waveforms, invented SCP labels, all five fold pairs populated) so
the script is runnable with synthetic data alone; every score from such a
run is meaningless by construction.

Note (documented deviations from the released benchmark code):

- Normalization is per lead with train-fold statistics, following the
  paper text (the released code pooled a scaler across leads).
- Losses use mean reduction (the original used ``reduction="sum"``);
  learning rates from its grid may need a small re-tune.
- ``Trainer`` validates every epoch; the original evaluated every 5
  epochs with patience 5 (evaluations), so the default patience here is
  25 epochs.
- Multi-class weights count the 3-way labels (target precedence), while
  the original summed raw NORM/target vectors; a record with both shifts
  the two counts slightly.
- ``--boot-samples`` adds a test-set percentile bootstrap for the headline
  metric. That interval comes from the released code, not from the paper's
  evaluation section (the paper compares arms across splits instead), so
  it is off by default here. When enabled: exactly ``--boot-samples``
  resamples are drawn and unusable ones are reported as skipped, while the
  original drew resamples until 100 were usable. The original's
  multi-class bootstrap reused the multi-label positive-class check on
  integer labels (a latent bug); here every resample must contain all
  classes. Resampling is per test sample, as in the original;
  ``pyhealth.metrics.bootstrap_metrics_ci`` also takes ``groups=`` for a
  patient-level cluster bootstrap.
- The original ``Monitor`` defines a test-threshold Fmax (CAFA-style)
  but never calls it in its train/evaluate scripts; it is not reproduced.

Deliberately not reproduced here: the lr/batch-size grid search and
SE-ResNet and other ablation architectures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    PTBXLDataset,
    compute_ptbxl_lead_stats,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.metrics import bootstrap_metrics_ci, welch_relative_improvement_test
from pyhealth.models import ResNet1D
from pyhealth.tasks import (
    PTBXLMulticlassClassification,
    PTBXLMultilabelClassification,
    compute_multiclass_class_weights,
    compute_multilabel_pos_weight,
)
from pyhealth.tasks.ptbxl_multiclass_classification import (
    PTBXL_PAPER_FOLD_SPLITS,
    PTBXL_PAPER_TARGET_DX,
)
from pyhealth.trainer import Trainer

SAMPLING_RATE = 500
ALL_FOLDS = tuple(range(1, 11))

# A pseudo PTB-XL corpus for running the pipeline without PhysioNet
# access (PTB-XL is CC-BY but the ~2 GB corpus needs a PhysioNet
# login). Three label patterns mirror the
# 3-way multi-class task: NORM-bearing (normal), AFIB-bearing
# (target for --dx AF), and neither (other); every pattern also
# carries rhythm codes and most carry a form code so each multilabel
# --label-set stays non-empty.
_SYNTHETIC_SCP_STATEMENTS = """\
,description,diagnostic,form,rhythm,diagnostic_class,diagnostic_subclass
NORM,normal ECG,1.0,,,NORM,NORM
IMI,inferior myocardial infarction,1.0,,,MI,IMI
STTC,non-specific ST-T changes,1.0,,,STTC,STTC
CD,conduction disturbance,1.0,,,CD,CD
SR,sinus rhythm,,,1.0,,
AFIB,atrial fibrillation,,,1.0,,
PAC,premature atrial contraction,,1.0,,,
"""

_SYNTHETIC_LABEL_PATTERNS = (
    {"NORM": 100.0, "SR": 50.0},
    {"IMI": 90.0, "AFIB": 80.0, "PAC": 40.0},
    {"STTC": 70.0, "CD": 60.0, "SR": 30.0, "PAC": 20.0},
)
_RECORDS_PER_PATTERN = 10

logger = logging.getLogger("ptbxl_ecg_benchmark")


def _write_synthetic_wfdb(
    record_base: Path, n_samples: int, fs: int, offset: int
) -> None:
    """Deterministic 12-lead WFDB record (scaled ramp; no real waveform)."""
    record_base.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((n_samples, 12), dtype=np.int16)
    for lead in range(12):
        data[:, lead] = np.arange(n_samples, dtype=np.int64) * 4 + 10 * lead + offset
    Path(str(record_base) + ".dat").write_bytes(data.tobytes())
    name = record_base.name
    lines = [f"{name} 12 {fs} {n_samples}"]
    for lead in range(12):
        lines.append(f"{name}.dat 16 1000.0(0)/uV 16 0 0 0 0 {lead}")
    Path(str(record_base) + ".hea").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_synthetic_corpus(root: Path) -> Path:
    """Write a deterministic pseudo PTB-XL corpus under ``root``.

    The corpus has the pieces :class:`pyhealth.datasets.PTBXLDataset`
    expects: ``scp_statements.csv`` / ``ptbxl_database.csv`` stubs (fake
    ids, ages and SCP-likelihood dicts) plus 30 synthesized 12-lead WFDB
    records, 10 s at both 100 and 500 Hz. Record ``i`` gets fold
    ``i % 10 + 1`` and pattern ``i // 10``, so every fold contains one
    record of each pattern: all five ``PTBXL_PAPER_FOLD_SPLITS`` pairs
    then have non-empty train/val/test sets with every task class
    present, and per-class metrics stay defined.

    Note:
        The waveforms are linear ramps and the labels are invented, so
        all scores are meaningless. This exists to check that the full
        pipeline (loading, task processing, training, sliding-window
        inference, bootstrap, significance test) runs without the
        ~2 GB PhysioNet download, not to reproduce paper numbers.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "scp_statements.csv").write_text(
        _SYNTHETIC_SCP_STATEMENTS, encoding="utf-8"
    )
    n_records = _RECORDS_PER_PATTERN * len(_SYNTHETIC_LABEL_PATTERNS)
    header = (
        "ecg_id,patient_id,age,sex,site,device,recording_date,scp_codes,"
        "strat_fold,filename_lr,filename_hr"
    )
    lines = [header]
    for i in range(n_records):
        codes = _SYNTHETIC_LABEL_PATTERNS[i // _RECORDS_PER_PATTERN]
        fold = i % 10 + 1
        ecg_id = i + 1
        lr = f"records100/{i // 100:05d}/{ecg_id:05d}_lr"
        hr = f"records500/{i // 100:05d}/{ecg_id:05d}_hr"
        lines.append(
            f"{ecg_id},{900000 + i % 12},{30 + (i * 7) % 50},{i % 2},{i % 3},"
            f"CS-12,1990-01-01 10:00:00,\"{codes}\",{fold},{lr},{hr}"
        )
    (root / "ptbxl_database.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    for i in range(n_records):
        _write_synthetic_wfdb(root / f"records100/{i // 100:05d}/{i + 1:05d}_lr",
                              n_samples=1000, fs=100, offset=i * 97)
        _write_synthetic_wfdb(root / f"records500/{i // 100:05d}/{i + 1:05d}_hr",
                              n_samples=5000, fs=500, offset=i * 97)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PTB-XL benchmark (Nonaka & Seita 2021) with ResNet1D"
    )
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument(
        "--root",
        help="PTB-XL v1.0.3 version directory (contains ptbxl_database.csv "
        "and scp_statements.csv)",
    )
    corpus.add_argument(
        "--synthetic-root",
        help="Generate a deterministic pseudo PTB-XL corpus here (synthetic "
        "ramp waveforms, invented labels) and use it as the data root; a "
        "pipeline self-check without the ~2 GB PhysioNet download, "
        "the scores are meaningless",
    )
    parser.add_argument(
        "--split",
        choices=[str(k) for k in sorted(PTBXL_PAPER_FOLD_SPLITS)] + ["all"],
        default="all",
        help="Benchmark split number ((val, test) folds "
        f"{PTBXL_PAPER_FOLD_SPLITS}); 'all' runs the paper's five trials "
        "(default), which is what the significance test needs",
    )
    parser.add_argument(
        "--task", choices=("multilabel", "multiclass"), default="multilabel"
    )
    parser.add_argument(
        "--label-set",
        choices=("all", "diagnostic", "subdiagnostic", "superdiagnostic",
                 "form", "rhythm"),
        default="diagnostic",
        help="Multi-label SCP statement set (--task multilabel)",
    )
    parser.add_argument(
        "--dx",
        choices=PTBXL_PAPER_TARGET_DX,
        default="AF",
        help="Target diagnosis for the 3-way task: the four the paper "
        "benchmarks. PTBXLMulticlassClassification also accepts PAC, PVC "
        "and STD, which the paper did not run on PTB-XL.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=(18, 34, 50),
        nargs="+",
        default=[18],
        help="ResNet1D depth(s); give several to compare architectures "
        "the way the paper does",
    )
    parser.add_argument(
        "--baseline-depth",
        type=int,
        choices=(18, 34, 50),
        default=18,
        help="Arm the Welch tests compare against (the paper uses ResNet-18)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=250,
        help="Epoch cap; the paper trains for at most 250 epochs with "
        "early stopping, which usually ends earlier",
    )
    parser.add_argument(
        "--patience", type=int, default=25,
        help="Epochs without validation improvement (the original used 5 "
        "evaluations spaced 5 epochs apart)",
    )
    parser.add_argument(
        "--crop-sec", type=float, default=2.5,
        help="Random training crop length in seconds",
    )
    parser.add_argument(
        "--window-sec", type=float, default=2.5,
        help="Evaluation window length in seconds (stride is half)",
    )
    parser.add_argument(
        "--pos-weight", action="store_true",
        help="Extension beyond the benchmark baseline: weight the "
        "multi-label BCE by each label's negatives/positives in the "
        "training split (--task multilabel)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Extension beyond the benchmark baseline: smooth the "
        "multi-class cross-entropy targets, in [0, 1) "
        "(--task multiclass)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the Welch tests and the "
        "Benjamini-Hochberg correction across settings",
    )
    parser.add_argument(
        "--boot-samples", type=int, default=0,
        help="Optional test-set percentile bootstrap of the headline "
        "metric, a protocol from the released code rather than the paper; "
        "0 (default) skips it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None,
        help="Default: cuda if available",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir", default=None,
        help="Enable Trainer logging/checkpoints under this directory",
    )
    parser.add_argument(
        "--result-json", default=None,
        help="Write per-run test metrics, and the Welch/BH comparison, to "
        "this JSON file",
    )
    return parser.parse_args()


def run_split(args: argparse.Namespace, split_number: int, depth: int) -> dict:
    """Train, select and evaluate one arm (depth) on one official split.

    Returns:
        dict with ``split``, ``depth``, ``label``, the test-fold metrics
        (``test``) and, when ``--boot-samples`` is set, the bootstrap
        intervals for the headline metric (``bootstrap``).
    """
    val_fold, test_fold = PTBXL_PAPER_FOLD_SPLITS[split_number]
    train_folds = tuple(
        f for f in ALL_FOLDS if f not in (val_fold, test_fold)
    )
    logger.info(
        "Split %d: train folds %s, val fold %d, test fold %d",
        split_number, list(train_folds), val_fold, test_fold,
    )

    root = Path(args.root).expanduser()
    dataset = PTBXLDataset(root=str(root), sampling_rate=SAMPLING_RATE)

    # Normalization statistics come from the training folds only, so they
    # must be computed before set_task materializes the samples.
    stats_path = compute_ptbxl_lead_stats(dataset, train_folds=train_folds)

    if args.task == "multilabel":
        task = PTBXLMultilabelClassification(
            scp_statements_path=root / "scp_statements.csv",
            label_set=args.label_set,
            standardize="train_stats",
            stats_path=stats_path,
        )
        monitor = "roc_auc_macro"
        metrics = ["roc_auc_macro", "f1_macro"]
        label = args.label_set
    else:
        task = PTBXLMulticlassClassification(
            target_dx=args.dx,
            standardize="train_stats",
            stats_path=stats_path,
        )
        monitor = "f1_macro"
        metrics = ["f1_macro", "roc_auc_macro_ovr"]
        label = args.dx

    samples = dataset.set_task(task)
    train_set, val_set, test_set = split_by_strat_fold(
        samples,
        train_folds=train_folds,
        val_folds=(val_fold,),
        test_folds=(test_fold,),
    )
    logger.info(
        "Samples: train=%d val=%d test=%d",
        len(train_set), len(val_set), len(test_set),
    )

    class_weight = None
    pos_weight = None
    if args.task == "multiclass":
        # One pass over the training split's labels. The benchmark takes
        # the weights from the train split only.
        class_weight = compute_multiclass_class_weights(train_set)
        logger.info("Multiclass class weights: %s", class_weight)
        if args.label_smoothing:
            logger.info(
                "Multiclass label smoothing: %.3f (baseline extension)",
                args.label_smoothing,
            )
    elif args.pos_weight:
        # Baseline extension, not the benchmark protocol: per-label
        # negatives/positives, counted on the training split only.
        pos_weight = compute_multilabel_pos_weight(train_set)
        logger.info(
            "Multilabel pos_weight (%d labels): min %.2f max %.2f",
            len(pos_weight), min(pos_weight), max(pos_weight),
        )

    model = ResNet1D(
        dataset=train_set,
        depth=depth,
        training_crop_len=int(args.crop_sec * SAMPLING_RATE),
        eval_window_len=int(args.window_sec * SAMPLING_RATE),
        eval_window_stride=int(args.window_sec * SAMPLING_RATE) // 2,
        class_weight=class_weight,
        pos_weight=pos_weight,
        label_smoothing=args.label_smoothing,
    )

    trainer = Trainer(
        model=model,
        metrics=metrics,
        device=args.device,
        enable_logging=args.output_dir is not None,
        output_path=args.output_dir,
        exp_name=f"split{split_number}_{args.task}_resnet{depth}",
    )

    train_loader = get_dataloader(
        train_set, batch_size=args.batch_size, shuffle=True,
        seed=args.seed, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = get_dataloader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": args.lr},
        monitor=monitor,
        monitor_criterion="max",
        patience=args.patience,
    )

    # Trainer restores the best-validation checkpoint; evaluate it on the
    # held-out test fold with the same sliding-window inference.
    test_scores = trainer.evaluate(test_loader)
    print(f"\nResNet{depth} | split {split_number} | {args.task} ({label})")
    for name, value in sorted(test_scores.items()):
        print(f"  {name}: {value:.4f}")

    # Optional, and from the released code rather than the paper: a
    # percentile bootstrap of the headline metric on the test fold.
    bootstrap = None
    if args.boot_samples > 0:
        y_true, y_prob, _ = trainer.inference(test_loader)
        bootstrap = bootstrap_metrics_ci(
            y_true,
            y_prob,
            [monitor],
            mode=args.task,
            n_boot=args.boot_samples,
            seed=args.seed,
            alpha=0.10,
        )
        for name, ci in bootstrap.items():
            print(
                f"  {name} bootstrap mean: {ci['mean']:.4f}  "
                f"(5%: {ci['lower']:.4f}, 95%: {ci['upper']:.4f}; "
                f"{ci['n_boot']} usable, {ci['n_skipped']} skipped)"
            )

    return {
        "split": split_number,
        "depth": depth,
        "label": label,
        "test": test_scores,
        "bootstrap": bootstrap,
    }


def collect_scores(runs: list[dict]) -> dict[str, dict[int, list[float]]]:
    """Group test metrics into ``{setting: {depth: [score per split]}}``.

    A setting is one label set (or target diagnosis) scored with one
    metric, e.g. ``diagnostic/roc_auc_macro``, which is how the paper's
    tables are laid out.
    """
    scores: dict[str, dict[int, list[float]]] = {}
    for run in runs:
        for metric, value in run["test"].items():
            setting = f"{run['label']}/{metric}"
            scores.setdefault(setting, {}).setdefault(run["depth"], []).append(value)
    return scores


def _fmt(value: float | None, spec: str = ".4f", dash: str = "   -  ") -> str:
    return format(value, spec) if value is not None else dash


def print_comparison(runs: list[dict], baseline_depth: int, alpha: float) -> list[dict]:
    """Print the paper's relative-improvement table with significance tests.

    One row per (setting, depth): the mean and sample standard deviation of
    the per-split scores, the relative improvement over the baseline arm,
    and the Welch p-value after Benjamini-Hochberg correction across every
    setting. Arms whose relative improvement is not positive are shown but
    excluded from testing.
    """
    scores = collect_scores(runs)
    rows = welch_relative_improvement_test(scores, baseline=baseline_depth, alpha=alpha)

    trials = max((len(v) for arm in scores.values() for v in arm.values()), default=0)
    print(
        f"\n=== {trials} trial(s) per arm, relative to ResNet-{baseline_depth} "
        f"(Welch t-test, Benjamini-Hochberg, alpha={alpha}) ==="
    )
    if not rows:
        print(f"  only ResNet-{baseline_depth} was run; pass more --depth values")
        return rows

    for setting in sorted({row["setting"] for row in rows}, key=str):
        print(f"\n  {setting}")
        print(
            f"    {'depth':>5}  {'mean':>7}  {'std':>7}  "
            f"{'rel. imp.%':>10}  {'p':>7}  {'q':>7}  significant"
        )
        for row in rows:
            if row["setting"] != setting:
                continue
            verdict = "yes" if row["significant"] else "no"
            if not row["tested"]:
                verdict = f"not tested ({row['reason']})"
            print(
                f"    {row['arm']:>5}  {_fmt(row['mean']):>7}  {_fmt(row['std']):>7}  "
                f"{_fmt(row['relative_improvement'], '.3f'):>10}  "
                f"{_fmt(row['p_value'], '.4f'):>7}  {_fmt(row['q_value'], '.4f'):>7}  "
                f"{verdict}"
            )
    if baseline_depth == 18:
        print(
            "\n  The paper reported no significant improvement over ResNet-18 "
            "for any architecture it tested."
        )
    return rows


def print_mean_summary(runs: list[dict]) -> None:
    """Print mean +/- sample std per (setting, arm) across the runs."""
    scores = collect_scores(runs)
    print(f"\n=== Mean +/- std over {len(runs)} run(s) ===")
    for setting, arms in sorted(scores.items()):
        for depth, values in sorted(arms.items()):
            std = np.std(values, ddof=1) if len(values) > 1 else float("nan")
            print(f"  ResNet-{depth} {setting}: {np.mean(values):.4f} +/- {std:.4f}")


def main() -> None:
    args = parse_args()
    if args.synthetic_root:
        args.root = str(
            build_synthetic_corpus(Path(args.synthetic_root).expanduser())
        )
        logger.info("Synthetic self-check corpus written to %s", args.root)
    if args.split == "all":
        splits = sorted(PTBXL_PAPER_FOLD_SPLITS)
    else:
        splits = [int(args.split)]
    depths = sorted(set(args.depth))
    if args.baseline_depth in depths:
        baseline_depth = args.baseline_depth
    else:
        baseline_depth = depths[0]
        logger.warning(
            "--baseline-depth %d was not run; comparing against ResNet-%d",
            args.baseline_depth, baseline_depth,
        )

    runs = []
    for depth in depths:
        for split_number in splits:
            torch.manual_seed(args.seed)
            runs.append(run_split(args, split_number, depth))

    comparison: list[dict] = []
    if len(depths) > 1:
        comparison = print_comparison(runs, baseline_depth, args.alpha)
    elif len(runs) > 1:
        print_mean_summary(runs)

    if args.result_json:
        label = args.label_set if args.task == "multilabel" else args.dx
        payload = {
            "task": args.task,
            "label": label,
            "splits": splits,
            "depths": depths,
            "baseline_depth": baseline_depth,
            "alpha": args.alpha,
            "pos_weight": args.pos_weight,
            "label_smoothing": args.label_smoothing,
            "runs": runs,
            "comparison": comparison,
        }
        Path(args.result_json).expanduser().write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info("Wrote per-split results and comparison to %s", args.result_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
