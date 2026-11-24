"""ProtoECGNet on PTB-XL: does the contrastive prototype loss help?

Reproduces one branch of ProtoECGNet (Sethi et al., MLHC 2025, Table 1;
https://arxiv.org/abs/2504.08713, code https://github.com/bbj-lab/protoecgnet)
over several seeds, and tests the paper's claim that the contrastive
prototype loss improves macro AUROC.

Pipeline (PTBXLDataset -> one branch of PTBXLMultilabelClassification ->
ProtoECGNet), with the recommended ``strat_fold`` split (1-8 train,
9 validation, 10 test). For each seed:

1. Train the ``feature_extractor`` stage once (backbone + plain linear head).
   Evaluated as-is, it is the black-box ResNet baseline (Table 1,
   "Black-box").
2. Starting from that same checkpoint, finish the staged protocol twice:
   ``joint`` -> prototype projection -> ``classifier``, once with the
   contrastive loss (``lam_cnrst=300``, the paper's default) and once without.
   Both variants share the starting weights, data order and hyperparameters,
   so the comparison is paired with an equal tuning budget. The paper tuned
   only the contrastive variant, with an Optuna search.

Each branch keeps only the ECGs with at least one of its labels, as in the
branch rows of Table 1. Scores use the logits, not ``y_prob``: after
projection the bias-free classifier can push logits to about -300, where a
float32 sigmoid is exactly 0 and every score ties.

Results are appended to ``<out-dir>/results.jsonl`` as each run finishes, and
test predictions go to ``<out-dir>/preds/``. Re-running with the same
``--out-dir`` skips finished runs. Runs from several machines can be merged
by copying their ``results.jsonl`` lines and ``preds/`` into one directory and
running ``--summarize-only``. ``<out-dir>/summary.md`` reports mean +/- std
over seeds, the paired gain per seed with a paired bootstrap 95% CI on fold 10
(test-sampling uncertainty only), and the paper's numbers.

Usage:
    # Smoke test on synthetic ECGs (no download, about a minute on CPU):
    python examples/ecg/ptbxl/protoecgnet_contrastive_sweep.py --synthetic \
        --out-dir runs/smoke

    # Rhythm branch (16 labels), 6 seeds, on PTB-XL 1.0.3 (open access:
    # https://physionet.org/content/ptb-xl/1.0.3/):
    python examples/ecg/ptbxl/protoecgnet_contrastive_sweep.py \
        --root /path/to/ptb-xl/1.0.3 --branch rhythm --seeds 0,1,2,3,4,5 \
        --out-dir runs/rhythm

Note:
    Results on ``--synthetic`` data only show that the pipeline runs.
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from pyhealth.datasets import PTBXLDataset, get_dataloader
from pyhealth.datasets.splitter import split_by_strat_fold
from pyhealth.models import PTBXL_BRANCH_KWARGS, PTBXL_BRANCH_LABELS, ProtoECGNet
from pyhealth.tasks import PTBXLMultilabelClassification
from pyhealth.trainer import Trainer

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SYNTHETIC_CODES = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANTS = ("blackbox", "no_contrastive", "contrastive")

#: Published fold-10 macro AUROC (Sethi et al. 2025, Table 1).
PAPER_MACRO_AUROC = {
    "rhythm": {"blackbox": 0.9403, "no_contrastive": 0.8903, "contrastive": 0.9064},
    "morphology": {"blackbox": 0.8872, "no_contrastive": 0.8533, "contrastive": 0.9051},
    "global": {"blackbox": 0.8649, "no_contrastive": 0.8362, "contrastive": 0.8667},
}


class PTBXLBranchClassification(PTBXLMultilabelClassification):
    """PTB-XL 71-label task restricted to the labels of one paper branch.

    ECGs that carry none of the branch's labels are dropped, matching the
    branch-specific rows of the paper's Table 1.

    Args:
        branch_labels: SCP codes to keep.
        **kwargs: Forwarded to :class:`PTBXLMultilabelClassification`
            (``label_set`` is forced to ``"all"``).
    """

    task_name: str = "PTBXLBranchClassification"

    def __init__(self, branch_labels: list[str], **kwargs: Any) -> None:
        kwargs["label_set"] = "all"
        super().__init__(**kwargs)
        # Part of vars(task), so each branch gets its own processed cache.
        self.branch_labels = sorted(branch_labels)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        keep = set(self.branch_labels)
        samples = []
        for sample in super().__call__(patient):
            labels = [label for label in sample["labels"] if label in keep]
            if labels:
                sample["labels"] = labels
                samples.append(sample)
        return samples


def build_synthetic_ptbxl(root: str, num_patients: int = 60, seed: int = 0) -> None:
    """Writes a small learnable PTB-XL-like folder for the smoke test."""
    rng = np.random.default_rng(seed)
    t = np.arange(1000) / 100.0
    rows, ecg_id = [], 0
    for patient in range(num_patients):
        for _ in range(1 + patient % 2):
            ecg_id += 1
            labels = list(
                rng.choice(
                    len(SYNTHETIC_CODES), size=1 + rng.integers(0, 2), replace=False
                )
            )
            signal = 0.3 * rng.normal(size=(1000, 12))
            for k in labels:
                signal[:, k] += np.sin(2 * np.pi * (k + 1) * t)
            rel = f"records100/00000/{ecg_id:05d}_lr"
            os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
            np.round(signal * 1000).astype("<i2").tofile(
                os.path.join(root, rel + ".dat")
            )
            header = [f"{os.path.basename(rel)} 12 100 1000"] + [
                f"{os.path.basename(rel)}.dat 16 1000/mV 16 0 0 0 0 {lead}"
                for lead in LEADS
            ]
            with open(os.path.join(root, rel + ".hea"), "w") as handle:
                handle.write("\n".join(header) + "\n")
            rows.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": float(patient),
                    "age": 60.0,
                    "sex": 1,
                    "site": 0,
                    "device": "CS-12",
                    "recording_date": f"199{patient % 10}-01-01 00:00:00",
                    "scp_codes": str({SYNTHETIC_CODES[k]: 100.0 for k in labels}),
                    "strat_fold": 1 + patient % 10,
                    "filename_lr": rel,
                    "filename_hr": rel.replace("records100", "records500"),
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)
    pd.DataFrame(
        {
            "diagnostic": 1,
            "form": 0,
            "rhythm": 0,
            "diagnostic_class": SYNTHETIC_CODES,
            "diagnostic_subclass": SYNTHETIC_CODES,
        },
        index=pd.Index(SYNTHETIC_CODES, name="scp_code"),
    ).to_csv(os.path.join(root, "scp_statements.csv"))


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fast_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-label AUROC via the rank-sum identity; NaN where a class is missing."""
    n = y_true.shape[0]
    pos = y_true.sum(axis=0)
    out = np.full(y_true.shape[1], np.nan)
    for j in np.flatnonzero((pos > 0) & (pos < n)):
        ranks = rankdata(y_prob[:, j])
        rank_sum = ranks[y_true[:, j] == 1].sum()
        out[j] = (rank_sum - pos[j] * (pos[j] + 1) / 2) / (pos[j] * (n - pos[j]))
    return out


def auroc_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Macro and prevalence-weighted AUROC over labels with both classes."""
    per_label = fast_auroc(y_true, y_prob)
    valid = ~np.isnan(per_label)
    if not valid.any():
        return {"macro": float("nan"), "weighted": float("nan"), "n_labels": 0}
    if (~valid).any():
        warnings.warn(f"{int((~valid).sum())} label(s) skipped (single class).")
    support = y_true.sum(axis=0)[valid]
    return {
        "macro": float(per_label[valid].mean()),
        "weighted": float(np.average(per_label[valid], weights=support)),
        "n_labels": int(valid.sum()),
    }


def load_results(out_dir: str) -> list[dict[str, Any]]:
    """Reads every finished run from ``results.jsonl``."""
    path = os.path.join(out_dir, "results.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_result(out_dir: str, row: dict[str, Any]) -> None:
    """Appends one finished run to ``results.jsonl``."""
    with open(os.path.join(out_dir, "results.jsonl"), "a", encoding="utf-8") as h:
        h.write(json.dumps(row) + "\n")


def pred_path(out_dir: str, branch: str, seed: int, variant: str) -> str:
    """Location of the saved fold-10 predictions for one run."""
    return os.path.join(out_dir, "preds", f"{branch}_seed{seed}_{variant}.npz")


def build_model(samples, train, args: argparse.Namespace) -> ProtoECGNet:
    """Builds a single-branch ProtoECGNet with the paper's branch settings."""
    labels = torch.cat(
        [b["labels"] for b in get_dataloader(train, batch_size=256, shuffle=False)]
    )
    weights, cooc = ProtoECGNet.compute_label_statistics(labels)
    label_groups = branch_kwargs = None
    if args.branch != "none":
        label_groups = {args.branch: PTBXL_BRANCH_LABELS[args.branch]}
        kwargs = dict(PTBXL_BRANCH_KWARGS[args.branch])
        if args.prototypes_per_label:
            kwargs["prototypes_per_label"] = args.prototypes_per_label
        branch_kwargs = {args.branch: kwargs}
    return ProtoECGNet(
        samples,
        label_groups=label_groups,
        branch_kwargs=branch_kwargs,
        fusion="average",
        proto_dim=args.proto_dim,
        prototypes_per_label=args.prototypes_per_label or 3,
        base_width=args.width,
        lam_cnrst=300.0,
        class_weights=weights,
        label_cooccurrence=cooc,
    )


def run_seed(samples, splits, args: argparse.Namespace, seed: int) -> None:
    """Runs the black-box, no-contrastive and contrastive variants for a seed."""
    done = {(r["branch"], r["seed"], r["variant"]) for r in load_results(args.out_dir)}
    todo = [v for v in VARIANTS if (args.branch, seed, v) not in done]
    if not todo:
        print(f"[seed {seed}] all variants done, skipping")
        return

    train, val, test = splits
    seed_everything(seed)
    model = build_model(samples, train, args)
    trainer = Trainer(model=model, metrics=[], device=args.device, enable_logging=False)
    val_loader = get_dataloader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test, batch_size=args.batch_size, shuffle=False)
    projection_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=False)

    def stage(name: str, epochs: int) -> None:
        model.set_stage(name)
        trainer.train(
            # A fresh loader per stage so its shuffle order follows the seed.
            train_dataloader=get_dataloader(
                train, batch_size=args.batch_size, shuffle=True
            ),
            val_dataloader=val_loader,
            epochs=epochs,
            monitor="loss",
            monitor_criterion="min",
            optimizer_params={"lr": args.lr},
        )

    def scores(loader) -> tuple:
        # Score with logits, not y_prob: after projection the bias-free
        # classifier can push logits to about -300, where float32 sigmoid
        # underflows to exactly 0. Every score then ties and AUROC reads 0.5,
        # even though the logits still rank the ECGs. AUROC is rank-based, so
        # scoring on logits gives the same value whenever sigmoid doesn't tie.
        y_true, _, loss, extra = trainer.inference(loader, additional_outputs=["logit"])
        return y_true, extra["logit"], loss

    def evaluate(variant: str, started: float) -> None:
        val_true, val_logit, _ = scores(val_loader)
        y_true, y_logit, loss = scores(test_loader)
        # Stored as y_prob for the bootstrap; any monotone score works there.
        np.savez_compressed(
            pred_path(args.out_dir, args.branch, seed, variant),
            y_true=y_true,
            y_prob=y_logit,
        )
        test_auc = auroc_summary(y_true, y_logit)
        row = {
            "branch": args.branch,
            "seed": seed,
            "variant": variant,
            "val_macro_auroc": round(auroc_summary(val_true, val_logit)["macro"], 4),
            "test_macro_auroc": round(test_auc["macro"], 4),
            "test_weighted_auroc": round(test_auc["weighted"], 4),
            "test_labels_scored": test_auc["n_labels"],
            "test_loss": round(float(loss), 4),
            "epochs": args.epochs,
            "classifier_epochs": args.classifier_epochs,
            "minutes": round((time.time() - started) / 60, 1),
            "device": args.device,
        }
        append_result(args.out_dir, row)
        print(json.dumps(row))

    ckpt = os.path.join(
        args.out_dir, "checkpoints", f"{args.branch}_seed{seed}_feature_extractor.pt"
    )
    started = time.time()
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[seed {seed}] reusing {ckpt}")
    else:
        print(f"[seed {seed}] stage feature_extractor ({args.epochs} epochs)")
        stage("feature_extractor", args.epochs)
        torch.save(model.state_dict(), ckpt)
    if "blackbox" in todo:
        model.set_stage("feature_extractor")
        evaluate("blackbox", started)

    for variant in ("no_contrastive", "contrastive"):
        if variant not in todo:
            continue
        started = time.time()
        # Same weights and same RNG stream for both variants: a paired design.
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        seed_everything(seed + 1000)
        model.lam_cnrst = 300.0 if variant == "contrastive" else 0.0
        for r in range(args.rounds):
            print(f"[seed {seed}] {variant}: joint round {r + 1}/{args.rounds}")
            stage("joint", args.epochs)
            model.project_prototypes(projection_loader)
        print(f"[seed {seed}] {variant}: classifier")
        stage("classifier", args.classifier_epochs)
        evaluate(variant, started)


def paired_bootstrap(a: dict, b: dict, n_boot: int, seed: int = 0) -> tuple:
    """95% CI of macro AUROC(a) - macro AUROC(b) by resampling test ECGs."""
    rng = np.random.default_rng(seed)
    y = a["y_true"]
    n = y.shape[0]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        da = fast_auroc(y[idx], a["y_prob"][idx])
        db = fast_auroc(y[idx], b["y_prob"][idx])
        both = ~np.isnan(da) & ~np.isnan(db)
        if both.any():
            diffs.append(da[both].mean() - db[both].mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fmt(value: float | None) -> str:
    """Formats an optional AUROC, using an en dash when it is missing."""
    return "–" if value is None else f"{value:.4f}"


def summarize(args: argparse.Namespace) -> str:
    """Writes ``summary.md`` comparing variants, seeds and the paper."""
    rows = [r for r in load_results(args.out_dir) if r["branch"] == args.branch]
    if not rows:
        return "No finished runs yet."
    df = pd.DataFrame(rows).drop_duplicates(["seed", "variant"], keep="last")
    paper = PAPER_MACRO_AUROC.get(args.branch, {})
    lines = [
        f"# ProtoECGNet sweep: branch `{args.branch}`",
        "",
        (
            "Fold-10 macro AUROC, mean ± std over seeds "
            "(std is 0 when there is only one seed)."
        ),
        "",
    ]
    if args.synthetic:
        lines += [
            (
                "> **Smoke test on synthetic ECGs** (about 12 test samples, 3 epochs): "
                "this only shows the pipeline runs. The numbers are meaningless."
            ),
            "",
        ]
    if not paper:
        lines += ["No paper reference for this setting, so paper columns show –.", ""]
    lines += [
        "| Variant | Seeds | Macro AUROC | Weighted AUROC | Paper macro |",
        "|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        sub = df[df.variant == variant]
        if sub.empty:
            continue
        lines.append(
            f"| {variant} | {len(sub)} "
            f"| {sub.test_macro_auroc.mean():.4f} ± {sub.test_macro_auroc.std(ddof=0):.4f} "
            f"| {sub.test_weighted_auroc.mean():.4f} ± {sub.test_weighted_auroc.std(ddof=0):.4f} "
            f"| {fmt(paper.get(variant))} |"
        )

    pivot = df.pivot(index="seed", columns="variant", values="test_macro_auroc")
    if {"contrastive", "no_contrastive"} <= set(pivot.columns):
        paired = pivot.dropna(subset=["contrastive", "no_contrastive"])
        lines += [
            "",
            "## Paired gain from the contrastive loss",
            "",
            "| Seed | No contrastive | Contrastive | Gain | Bootstrap 95% CI |",
            "|---|---|---|---|---|",
        ]
        gains = []
        for seed, row in paired.iterrows():
            gain = row.contrastive - row.no_contrastive
            gains.append(gain)
            ci = "n/a"
            pa = pred_path(args.out_dir, args.branch, seed, "contrastive")
            pb = pred_path(args.out_dir, args.branch, seed, "no_contrastive")
            if args.bootstrap and os.path.exists(pa) and os.path.exists(pb):
                lo, hi = paired_bootstrap(np.load(pa), np.load(pb), args.bootstrap)
                ci = f"[{lo:+.4f}, {hi:+.4f}]"
            lines.append(
                f"| {seed} | {row.no_contrastive:.4f} | {row.contrastive:.4f} "
                f"| {gain:+.4f} | {ci} |"
            )
        gains = np.array(gains)
        paper_gain = (paper["contrastive"] - paper["no_contrastive"]) if paper else None
        lines += [
            "",
            f"Mean gain {gains.mean():+.4f} ± {gains.std(ddof=0):.4f} (seed std, "
            f"n={len(gains)})"
            + (
                f"; paper reports {paper_gain:+.4f} from a single run with "
                "unequal tuning."
                if paper
                else "."
            ),
            "",
            (
                "The per-seed CIs cover test-set sampling only. Seed-to-seed "
                "spread is the training-variability estimate."
            ),
        ]
    if "blackbox" in pivot.columns and "contrastive" in pivot.columns:
        gap = (pivot.blackbox - pivot.contrastive).dropna()
        lines += [
            "",
            "## Black-box vs ProtoECGNet",
            "",
            f"Black-box − contrastive: {gap.mean():+.4f} ± {gap.std(ddof=0):.4f}"
            + (
                f" (paper: {paper['blackbox'] - paper['contrastive']:+.4f})."
                if paper
                else "."
            ),
        ]
    # Hand-rolled table: DataFrame.to_markdown needs the optional tabulate.
    lines += [
        "",
        "## All runs",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "---|" * len(df.columns),
    ]
    lines += [
        "| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as h:
        h.write(text)
    return text


def main() -> None:
    """Parses arguments, runs the sweep and writes the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="PTB-XL 1.0.3 directory")
    parser.add_argument(
        "--synthetic", action="store_true", help="tiny fake dataset for a smoke test"
    )
    parser.add_argument("--dev", action="store_true", help="small PTB-XL subset")
    parser.add_argument(
        "--branch",
        default="rhythm",
        choices=["rhythm", "morphology", "global", "none"],
        help="paper branch to train ('none' = superdiagnostic)",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epochs of the feature_extractor and joint "
        "stages (default 20, or 3 with --synthetic)",
    )
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=None,
        help="epochs of the last-layer stage after projection "
        "(default 15). Projection drives the bias-free "
        "classifier's logits to about -100 to -300, and it "
        "needs many cheap epochs to recover",
    )
    parser.add_argument(
        "--rounds", type=int, default=1, help="joint + projection rounds"
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--proto-dim", type=int, default=128)
    parser.add_argument(
        "--prototypes-per-label",
        type=int,
        default=0,
        help="0 keeps the paper's count for the branch",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--cache-dir", help="persistent processed-data cache")
    parser.add_argument("--out-dir", default="runs/protoecgnet_sweep")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="paired bootstrap resamples (0 disables)",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        args.branch, args.width, args.proto_dim = "none", 8, 16
        args.epochs = args.epochs or 3
        args.bootstrap = min(args.bootstrap, 100)
    elif not args.root and not args.summarize_only:
        parser.error("pass --root /path/to/ptb-xl/1.0.3 (or --synthetic)")
    args.epochs = args.epochs or 20
    args.classifier_epochs = args.classifier_epochs or (3 if args.synthetic else 15)
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    for sub in ("preds", "checkpoints"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as h:
        json.dump(vars(args), h, indent=2)

    if not args.summarize_only:
        scratch = tempfile.mkdtemp()
        try:
            if args.synthetic:
                args.root = os.path.join(scratch, "ptbxl")
                build_synthetic_ptbxl(args.root)
            dataset = PTBXLDataset(
                root=args.root,
                cache_dir=args.cache_dir or os.path.join(scratch, "cache"),
                dev=args.dev,
            )
            task_kwargs = {"scp_statements_path": dataset.scp_statements_path}
            if args.branch == "none":
                task = PTBXLMultilabelClassification(
                    label_set="superdiagnostic", **task_kwargs
                )
            else:
                task = PTBXLBranchClassification(
                    PTBXL_BRANCH_LABELS[args.branch], **task_kwargs
                )
            samples = dataset.set_task(task)
            splits = split_by_strat_fold(samples)
            print(
                f"branch={args.branch} device={args.device} train={len(splits[0])} "
                f"val={len(splits[1])} test={len(splits[2])}"
            )
            for seed in (int(s) for s in args.seeds.split(",")):
                run_seed(samples, splits, args, seed)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(summarize(args))


if __name__ == "__main__":
    sys.exit(main())
