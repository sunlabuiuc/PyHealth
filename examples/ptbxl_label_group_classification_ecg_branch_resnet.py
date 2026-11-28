"""ProtoECGNet vs. a fused black-box ResNet under different label groupings.

Author: Hanho Ryu (NetID: hanhor2, hanhor2@illinois.edu)

Paper: ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning (Sethi et al., MLHC 2025).
Paper link: https://arxiv.org/abs/2504.08713

Full pipeline::

    PTBXLDataset -> PTBXLLabelGroupClassification (71 SCP labels, 0.5 Hz
    high-pass, per-lead z-score) -> strat_fold split (1-8 / 9 / 10)
    -> ProtoECGNet or ECGBranchResNet -> macro AUROC on fold 10

The paper reports that matching prototype types to physician-defined label
groups beats a single prototype type, and that interpretability costs no
accuracy against black-box models. Two confounds remain in its evaluation:
the fused prototype model is the only model with a trained fusion head, and
the label grouping is never compared with an alternative. This script runs
three ablations that the paper does not contain:

* ``fusion`` (fusion-head control): the black-box ``ECGBranchResNet`` with the
  physician grouping and no fusion (the paper's "aggregated branch
  black-boxes"), logit fusion, and feature fusion (the black-box analogue of
  ProtoECGNet's fusion head) warm-started from the branch heads or randomly
  initialized like ProtoECGNet's, next to ProtoECGNet with and without
  learned fusion, and a single full-label black-box ResNet (the paper's 1D
  baseline).
* ``grouping``: physician vs. data-driven (spectral clustering of the
  training-set Jaccard label co-occurrence) vs. size-matched random label
  groups, for both models. Each new group reuses the branch architecture of
  the physician group it overlaps most (``match_label_groups``), so only the
  label partition changes.
* ``label_efficiency``: both models trained on 10/25/50/100% of the
  training *patients*.

Both models get the same backbone-epoch budget (ProtoECGNet: feature extractor
+ joint stage; black box: twice as many branch epochs), the same early
stopping, class weights, optimizer and seeds. Fusion heads of both models
start from their trained branch heads (a random start is kept as a control).
ProtoECGNet's classifier stage, which follows prototype projection, gets
``--classifier-epochs`` (15) because projection moves the prototypes onto
non-negative feature windows and the last layer starts far from a good fit.
Scores use logits (ProtoECGNet logits can underflow a float32 sigmoid after
projection). Results are
appended to ``<out-dir>/results.jsonl`` as runs finish (re-running skips
finished runs) and ``<out-dir>/summary.md`` reports mean +/- std over seeds.

Usage::

    # Smoke test on synthetic ECGs (CPU, about two minutes):
    python examples/ptbxl_label_group_classification_ecg_branch_resnet.py \\
        --synthetic --out-dir runs/synthetic

    # PTB-XL 1.0.3 (open access, https://physionet.org/content/ptb-xl/1.0.3/)
    # on one T4 GPU; run experiments separately to fit session limits:
    python examples/ptbxl_label_group_classification_ecg_branch_resnet.py \\
        --root /data/ptb-xl/1.0.3 --experiments fusion --seeds 0,1,2 \\
        --out-dir runs/ptbxl

Findings (PTB-XL 1.0.3, one T4 GPU, ``--epochs 10 --head-epochs 5``, test
macro AUROC on fold 10; mean +/- std over seeds 0 and 1 unless noted):

======================================  ===============  ======
Configuration                           Ours             Paper
======================================  ===============  ======
Black box, single 1D ResNet             0.878 +/- 0.008  0.9086
Black-box branches, no fusion           0.868 +/- 0.006  0.9052
Black-box branches, logit fusion        0.880 +/- 0.012  --
Black-box branches, feature fusion      0.871 +/- 0.007  --
Feature fusion, random init             0.876 +/- 0.011  --
ProtoECGNet, branches averaged          0.659 +/- 0.014  0.9120
ProtoECGNet, learned fusion             0.710 +/- 0.025  0.9213
======================================  ===============  ======

* Black boxes land 0.03-0.04 below the paper (no hyperparameter search, at
  most 20 backbone epochs). A trained fusion head changes them by -0.01 to
  +0.03 depending on the seed, i.e. within seed-to-seed noise.
* ProtoECGNet (PyHealth implementation, paper's staged protocol at this
  budget) does not reproduce: it stays about 0.16 below the black boxes. Its
  learned fusion helps in both seeds (+0.06, +0.04). After prototype
  projection the classifier-stage loss starts near 25,000; a longer
  classifier stage and a warm-started fusion head did not close the gap.
  The paper trains each stage for up to 200 epochs at lr 1e-4 after a
  1000-trial search.
* Grouping (seed 0 only): black box physician 0.864 / spectral 0.887 /
  random 0.865; ProtoECGNet 0.735 / 0.719 / 0.752. No evidence that the
  physician grouping beats data-driven or random groups.
* Label efficiency (seed 0; 10/25/50/100% of training patients): black box
  0.717 / 0.781 / 0.813 / 0.864, ProtoECGNet 0.646 / 0.648 / 0.731 / 0.735.
  The black box is ahead at every fraction.

Results on ``--synthetic`` data only show that the pipeline runs; they say
nothing about real ECGs.
"""

import argparse
import json
import math
import os
import random
import tempfile
import time
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import PTBXLDataset, get_dataloader
from pyhealth.models import PTBXL_BRANCH_KWARGS, ECGBranchResNet, ProtoECGNet
from pyhealth.tasks import PTBXLLabelGroupClassification
from pyhealth.tasks.ptbxl_label_group_classification import (
    PHYSICIAN_LABEL_GROUPS,
    label_cooccurrence,
    match_label_groups,
    random_label_groups,
    spectral_label_groups,
)
from pyhealth.trainer import Trainer

EXPERIMENTS = ("fusion", "grouping", "label_efficiency")

#: Reference macro AUROCs from the paper (Sethi et al., 2025, Table 1 and
#: Sec. 4) for the configurations that this script reproduces.
PAPER_AUROC = {
    ("blackbox", "single", "none"): 0.9086,
    ("blackbox", "physician", "none"): 0.9052,
    ("proto", "physician", "average"): 0.9120,
    ("proto", "physician", "learned"): 0.9213,
}

#: Synthetic SCP codes: a few labels from each physician group.
SYNTHETIC_CODES = {
    "rhythm": ["SR", "AFIB", "STACH", "SBRAD"],
    "morphology": ["IMI", "ASMI", "LVH", "IRBBB", "NDT", "PVC"],
    "global": ["NORM", "EL"],
}

#: Synthetic "conditions" that emit correlated codes, so that the label
#: co-occurrence matrix has structure for spectral clustering to find.
SYNTHETIC_CONDITIONS = [
    ["NORM", "SR"],
    ["AFIB", "NDT"],
    ["IMI", "ASMI", "SR"],
    ["LVH", "STACH"],
    ["IRBBB", "SBRAD", "EL"],
    ["PVC", "SR"],
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_synthetic_ptbxl(root: str, num_patients: int = 80, seed: int = 0) -> None:
    """Writes a small PTB-XL 1.0.3 lookalike whose labels are learnable.

    Each ECG draws one or two synthetic conditions; every emitted SCP code
    ``k`` adds a sine wave of ``k + 1`` Hz to lead ``k % 12`` on top of noise.

    Args:
        root: Output directory.
        num_patients: Number of synthetic patients (1-2 ECGs each).
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    codes = [c for group in SYNTHETIC_CODES.values() for c in group]
    t = np.arange(1000) / 100.0
    rows, ecg_id = [], 0
    for patient in range(num_patients):
        for _ in range(1 + patient % 2):
            ecg_id += 1
            picks = rng.choice(len(SYNTHETIC_CONDITIONS), 1 + rng.integers(0, 2))
            present = sorted({c for p in picks for c in SYNTHETIC_CONDITIONS[p]})
            signal = 0.3 * rng.normal(size=(1000, 12))
            for code in present:
                k = codes.index(code)
                signal[:, k % 12] += np.sin(2 * np.pi * (k + 1) * t)
            rel = f"records100/00000/{ecg_id:05d}_lr"
            os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
            np.round(signal * 1000).astype("<i2").tofile(
                os.path.join(root, rel + ".dat")
            )
            name = os.path.basename(rel)
            header = [f"{name} 12 100 1000"] + [
                f"{name}.dat 16 1000/mV 16 0 0 0 0 {lead}" for lead in range(12)
            ]
            with open(os.path.join(root, rel + ".hea"), "w") as handle:
                handle.write("\n".join(header) + "\n")
            rows.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": float(patient),
                    "age": 60.0,
                    "sex": patient % 2,
                    "site": 0,
                    "device": "CS-12",
                    "recording_date": f"199{patient % 10}-01-01 00:00:00",
                    "scp_codes": str({c: 100.0 for c in present}),
                    "strat_fold": 1 + patient % 10,
                    "filename_lr": rel,
                    "filename_hr": rel.replace("records100", "records500"),
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)
    kinds = {c: g for g, members in SYNTHETIC_CODES.items() for c in members}
    pd.DataFrame(
        {
            "diagnostic": [1.0 if kinds[c] != "rhythm" else np.nan for c in codes],
            "form": np.nan,
            "rhythm": [1.0 if kinds[c] == "rhythm" else np.nan for c in codes],
            "diagnostic_class": "",
            "diagnostic_subclass": "",
        },
        index=pd.Index(codes, name="scp_code"),
    ).to_csv(os.path.join(root, "scp_statements.csv"))


def scan_samples(samples) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reads labels, folds and patient ids of every sample in one pass.

    Args:
        samples: The task's SampleDataset.

    Returns:
        ``(labels, folds, patient_ids)``: a multi-hot ``(N, num_labels)``
        array in vocabulary order, the ``strat_fold`` of each sample and the
        patient id of each sample.
    """
    labels, folds, patients = [], [], []
    for batch in get_dataloader(samples, batch_size=512, shuffle=False):
        labels.append(batch["labels"].numpy())
        folds.extend(int(f) for f in batch["strat_fold"])
        patients.extend(str(p) for p in batch["patient_id"])
    return np.concatenate(labels), np.asarray(folds), patients


def subsample_patients(
    index: np.ndarray, patients: list[str], fraction: float, seed: int
) -> np.ndarray:
    """Keeps the samples of a random ``fraction`` of the patients in ``index``.

    Args:
        index: Sample indices to subsample (the training split).
        patients: Patient id of every sample in the dataset.
        fraction: Fraction of patients to keep, in ``(0, 1]``.
        seed: Seed of the patient draw.

    Returns:
        The kept sample indices, sorted.
    """
    if fraction >= 1.0:
        return index
    ids = sorted({patients[i] for i in index})
    keep_count = max(1, round(fraction * len(ids)))
    keep = set(np.random.default_rng(seed).choice(ids, keep_count, replace=False))
    return np.asarray([i for i in index if patients[i] in keep])


# --------------------------------------------------------------------------
# Groupings and models
# --------------------------------------------------------------------------
def build_groupings(
    label_names: list[str], train_labels: np.ndarray, seed: int
) -> dict[str, dict[str, list[str]]]:
    """Builds the label groupings compared in the ablations.

    All data-driven statistics use the training split only.

    Args:
        label_names: Label vocabulary in model order.
        train_labels: Multi-hot training labels ``(N, num_labels)``.
        seed: Seed of the random grouping.

    Returns:
        ``{"physician", "spectral", "random", "single"}`` groupings over the
        labels present in the vocabulary.
    """
    present = set(label_names)
    physician = {
        name: [c for c in members if c in present]
        for name, members in PHYSICIAN_LABEL_GROUPS.items()
    }
    physician = {name: members for name, members in physician.items() if members}
    label_lists = [
        [label_names[j] for j in np.flatnonzero(row)] for row in train_labels
    ]
    cooc = label_cooccurrence(label_lists, label_names)
    return {
        "physician": physician,
        "spectral": spectral_label_groups(
            cooc, label_names, num_groups=len(physician), seed=0
        ),
        "random": random_label_groups(
            label_names, sizes=[len(m) for m in physician.values()], seed=seed
        ),
        "single": {"all": list(label_names)},
    }


def branch_settings(
    groups: dict[str, list[str]],
    physician: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, dict]:
    """Per-branch ProtoECGNet settings borrowed from the matched paper branch.

    Args:
        groups: Grouping to build branches for.
        physician: The physician grouping (reference for the architecture).
        args: Command-line arguments.

    Returns:
        ``{branch: PrototypeBranch kwargs}`` following ``PTBXL_BRANCH_KWARGS``.
    """
    match = match_label_groups(groups, physician)
    settings = {}
    for name, ref in match.items():
        kwargs = dict(PTBXL_BRANCH_KWARGS[ref])
        if args.protos_per_label:
            kwargs["prototypes_per_label"] = args.protos_per_label
        if args.backbone_2d == "resnet1d":
            kwargs["backbone"] = "resnet1d"
        settings[name] = kwargs
    return settings


def build_model(
    samples,
    spec: dict,
    groups: dict[str, list[str]],
    physician: dict[str, list[str]],
    class_weights: torch.Tensor,
    cooccurrence: torch.Tensor,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Instantiates ProtoECGNet or ECGBranchResNet for one run.

    Args:
        samples: The full SampleDataset (fixes the label vocabulary).
        spec: Run specification (``model``, ``fusion`` ...).
        groups: Label grouping of the run.
        physician: Physician grouping (architecture reference).
        class_weights: Positive-class weights from the training split.
        cooccurrence: Jaccard label co-occurrence from the training split.
        args: Command-line arguments.

    Returns:
        The model.
    """
    settings = branch_settings(groups, physician, args)
    if spec["model"] == "proto":
        return ProtoECGNet(
            samples,
            label_groups=groups,
            branch_kwargs=settings,
            fusion=spec["fusion"],
            proto_dim=args.proto_dim,
            base_width=args.width,
            lam_l1=args.lam_l1,
            class_weights=class_weights,
            label_cooccurrence=cooccurrence,
        )
    fusion, _, init = spec["fusion"].partition("_")
    backbones = {n: {"backbone": s["backbone"]} for n, s in settings.items()}
    if spec["grouping"] == "single":
        # The paper's full-label black-box baseline is a 1D ResNet-18.
        backbones = {n: {"backbone": "resnet1d"} for n in groups}
    return ECGBranchResNet(
        samples,
        label_groups=groups,
        branch_kwargs=backbones,
        fusion=fusion,
        fusion_init=init or "branches",
        dropout=args.dropout,
        base_width=args.width,
        lam_l1=args.lam_l1,
        class_weights=class_weights,
    )


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
def run_specs(args: argparse.Namespace) -> Iterator[dict]:
    """Yields the (deduplicated) run specifications of the chosen experiments.

    Args:
        args: Command-line arguments.

    Yields:
        Dicts with ``model``, ``grouping``, ``fusion``, ``fraction`` and
        ``seed``.
    """
    specs = []
    for seed in args.seeds:
        if "fusion" in args.experiments:
            specs.append(("blackbox", "single", "none", 1.0, seed))
            for fusion in ("none", "logits", "features", "features_random"):
                specs.append(("blackbox", "physician", fusion, 1.0, seed))
            for fusion in ("average", "learned"):
                specs.append(("proto", "physician", fusion, 1.0, seed))
        if "grouping" in args.experiments:
            for grouping in ("physician", "spectral", "random"):
                specs.append(("blackbox", grouping, "features", 1.0, seed))
                specs.append(("proto", grouping, "learned", 1.0, seed))
        if "label_efficiency" in args.experiments:
            for fraction in args.fractions:
                specs.append(("blackbox", "physician", "features", fraction, seed))
                specs.append(("proto", "physician", "learned", fraction, seed))
    seen = set()
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            yield dict(zip(("model", "grouping", "fusion", "fraction", "seed"), spec))


def protocol(spec: dict) -> int:
    """Training-protocol version of a run specification.

    Version 2 (current) gives ProtoECGNet a longer classifier stage and a
    learned fusion head warm-started from the branch classifiers, and trains
    the ``single`` baseline with a 1D ResNet. Runs recorded under an older
    version are re-run and left out of the summary.

    Args:
        spec: Run specification.

    Returns:
        The protocol version.
    """
    return 2 if spec["model"] == "proto" or spec["grouping"] == "single" else 1


def run_key(spec: dict) -> str:
    """Unique string id of a run specification (with its protocol version)."""
    key = "{model}|{grouping}|{fusion}|{fraction}|{seed}".format(**spec)
    return key if protocol(spec) == 1 else f"{key}|p{protocol(spec)}"


@torch.no_grad()
def init_proto_fusion(model: ProtoECGNet) -> None:
    """Copies ProtoECGNet's branch classifiers into its fusion head.

    The fused model then starts exactly at the branch-wise ("average")
    prediction, as ``ECGBranchResNet(fusion_init="branches")`` does, instead of
    from a random layer over hundreds of large similarity scores.

    Note:
        This helper writes directly into ProtoECGNet internals
        (``branch_names``, ``branches[name].classifier``, ``branch_index``,
        ``num_prototypes`` and ``fusion_classifier``) and assumes the fusion
        input is the branch similarities concatenated in ``branch_names``
        order. It checks that layout and fails loudly if a future ProtoECGNet
        changes it.

    Args:
        model: A ProtoECGNet with ``fusion="learned"``.

    Raises:
        ValueError: If the model has no fusion head or its fusion input does
            not match the concatenated branch prototypes.
    """
    fusion = getattr(model, "fusion_classifier", None)
    if fusion is None:
        raise ValueError("init_proto_fusion requires ProtoECGNet(fusion='learned')")
    widths = [model.branches[name].num_prototypes for name in model.branch_names]
    if fusion.weight.shape != (model.num_labels, sum(widths)):
        raise ValueError(
            "unexpected ProtoECGNet fusion layout: fusion weight "
            f"{tuple(fusion.weight.shape)}, branch prototypes {widths}"
        )
    weight = torch.zeros_like(model.fusion_classifier.weight)
    offset = 0
    for name in model.branch_names:
        branch = model.branches[name]
        index = model.branch_index(name)
        width = branch.num_prototypes
        weight[index, offset : offset + width] = branch.classifier.weight
        offset += width
    model.fusion_classifier.weight.copy_(weight)
    model.fusion_classifier.bias.zero_()


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_auroc(y_true: np.ndarray, score: np.ndarray, columns=None) -> float:
    """Macro AUROC over labels with at least one positive and one negative.

    Args:
        y_true: Multi-hot labels ``(N, L)``.
        score: Scores ``(N, L)`` (logits).
        columns: Optional label indices to restrict to.

    Returns:
        The macro AUROC, or NaN if no label can be scored.
    """
    from sklearn.metrics import roc_auc_score

    columns = range(y_true.shape[1]) if columns is None else columns
    values = [
        roc_auc_score(y_true[:, j], score[:, j])
        for j in columns
        if 0 < y_true[:, j].sum() < len(y_true)
    ]
    return float(np.mean(values)) if values else float("nan")


def train_and_evaluate(
    samples, spec: dict, data: dict, args: argparse.Namespace
) -> dict:
    """Trains one model with the paper's staged protocol and scores fold 10.

    Args:
        samples: The full SampleDataset.
        spec: Run specification from :func:`run_specs`.
        data: Precomputed labels, splits and patient ids.
        args: Command-line arguments.

    Returns:
        A result row (JSON-serializable).
    """
    started = time.time()
    seed_everything(spec["seed"])
    labels, names = data["labels"], data["label_names"]
    train_index = subsample_patients(
        data["train"], data["patients"], spec["fraction"], spec["seed"]
    )
    groupings = build_groupings(names, labels[train_index], spec["seed"])
    groups = groupings[spec["grouping"]]
    weights, cooc = ProtoECGNet.compute_label_statistics(
        torch.from_numpy(labels[train_index]).float()
    )
    model = build_model(
        samples, spec, groups, groupings["physician"], weights, cooc, args
    )
    train = samples.subset(train_index.tolist())
    val = samples.subset(data["val"].tolist())
    test = samples.subset(data["test"].tolist())
    val_loader = get_dataloader(val, batch_size=args.batch_size, shuffle=False)
    trainer = Trainer(model=model, metrics=[], device=args.device, enable_logging=False)

    def stage(name: str, epochs: int) -> None:
        model.set_stage(name)
        trainer.train(
            train_dataloader=get_dataloader(
                train, batch_size=args.batch_size, shuffle=True
            ),
            val_dataloader=val_loader,
            epochs=epochs,
            optimizer_params={"lr": args.lr},
            weight_decay=args.weight_decay,
            monitor="loss",
            monitor_criterion="min",
            patience=args.patience,
        )

    if spec["model"] == "proto":
        # Equal backbone budget: feature extractor + joint = 2 x epochs.
        stage("feature_extractor", args.epochs)
        stage("joint", args.epochs)
        model.project_prototypes(
            get_dataloader(train, batch_size=args.batch_size, shuffle=False)
        )
        # After projection the prototypes are real (non-negative) feature
        # windows, every similarity is large and the -0.5 initial weights give
        # logits in the thousands; the last layer needs many more steps than
        # a fusion head to recover (paper: up to 200 epochs at lr 1e-4).
        stage("classifier", args.classifier_epochs)
        if spec["fusion"] == "learned":
            if args.proto_fusion_init == "classifier":
                init_proto_fusion(model)
            stage("fusion", args.head_epochs)
    else:
        stage("branches" if spec["fusion"] != "none" else "all", 2 * args.epochs)
        if spec["fusion"] != "none":
            stage("fusion", args.head_epochs)

    def scores(loader) -> tuple[np.ndarray, np.ndarray]:
        y_true, _, _, extra = trainer.inference(loader, additional_outputs=["logit"])
        return np.asarray(y_true), np.asarray(extra["logit"])

    y_val, s_val = scores(val_loader)
    y_test, s_test = scores(
        get_dataloader(test, batch_size=args.batch_size, shuffle=False)
    )
    index_of = {name: i for i, name in enumerate(names)}
    row = {
        **spec,
        "key": run_key(spec),
        "protocol": protocol(spec),
        "val_macro_auroc": round(macro_auroc(y_val, s_val), 4),
        "test_macro_auroc": round(macro_auroc(y_test, s_test), 4),
        "train_samples": len(train_index),
        "groups": {name: len(members) for name, members in groups.items()},
        "minutes": round((time.time() - started) / 60, 2),
        "device": str(trainer.device),
    }
    for group, members in groupings["physician"].items():
        cols = [index_of[m] for m in members]
        row[f"test_auroc_{group}"] = round(macro_auroc(y_test, s_test, cols), 4)
    return row


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def load_results(out_dir: str) -> list[dict]:
    """Reads ``results.jsonl`` (empty list if missing)."""
    path = os.path.join(out_dir, "results.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_std(values: list[float]) -> str:
    """Formats ``mean +/- std`` (population std; n shown separately)."""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def summarize(rows: list[dict], args: argparse.Namespace) -> str:
    """Builds the markdown summary of all finished runs.

    Args:
        rows: Result rows.
        args: Command-line arguments.

    Returns:
        Markdown text with one table per experiment.
    """
    rows = [r for r in rows if r.get("key") == run_key(r)]
    frame = pd.DataFrame(rows)
    lines = [
        "# ProtoECGNet vs. ECGBranchResNet on PTB-XL (71 labels)",
        "",
        (
            f"Data: {'synthetic' if args.synthetic else args.root}. "
            "Test = strat_fold 10; macro AUROC from logits; mean +/- std over seeds."
        ),
        "",
    ]
    if frame.empty:
        return "\n".join(lines + ["No finished runs."])
    group_cols = [c for c in frame.columns if c.startswith("test_auroc_")]

    def table(title: str, keys: list[tuple], by: list[str]) -> None:
        part = frame[
            frame.apply(
                lambda r: (
                    (r["model"], r["grouping"], r["fusion"], r["fraction"]) in keys
                ),
                axis=1,
            )
        ]
        found = {
            tuple(r)
            for r in part[["model", "grouping", "fusion", "fraction"]].itertuples(
                index=False
            )
        }
        if len(found) < 2:  # nothing to compare yet
            return
        lines.extend([f"## {title}", ""])
        head = by + ["n", "test macro AUROC", *group_cols, "paper"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + " --- |" * len(head))
        for key in keys:
            sub = part[
                (part["model"] == key[0])
                & (part["grouping"] == key[1])
                & (part["fusion"] == key[2])
                & (part["fraction"] == key[3])
            ]
            if sub.empty:
                continue
            paper = PAPER_AUROC.get(key[:3]) if key[3] == 1.0 else None
            cells = [str(sub.iloc[0][c]) for c in by]
            cells += [str(len(sub)), mean_std(sub["test_macro_auroc"].tolist())]
            cells += [mean_std(sub[c].tolist()) for c in group_cols]
            cells.append(f"{paper:.4f}" if paper else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    table(
        "Fusion-head control",
        [
            ("blackbox", "single", "none", 1.0),
            ("blackbox", "physician", "none", 1.0),
            ("blackbox", "physician", "logits", 1.0),
            ("blackbox", "physician", "features", 1.0),
            ("blackbox", "physician", "features_random", 1.0),
            ("proto", "physician", "average", 1.0),
            ("proto", "physician", "learned", 1.0),
        ],
        ["model", "grouping", "fusion"],
    )
    table(
        "Label grouping",
        [
            (m, g, f, 1.0)
            for g in ("physician", "spectral", "random")
            for m, f in (("blackbox", "features"), ("proto", "learned"))
        ],
        ["model", "grouping", "fusion"],
    )
    table(
        "Label efficiency",
        [
            (m, "physician", f, fr)
            for fr in sorted(frame["fraction"].unique())
            for m, f in (("blackbox", "features"), ("proto", "learned"))
        ],
        ["model", "fraction"],
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and fills in synthetic-mode defaults."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="PTB-XL 1.0.3 directory")
    parser.add_argument("--synthetic", action="store_true", help="fake ECGs")
    parser.add_argument("--dev", action="store_true", help="PTB-XL dev subset")
    parser.add_argument("--out-dir", default="runs/protoecgnet_groups")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--experiments", default=",".join(EXPERIMENTS), help="comma list"
    )
    parser.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2")
    parser.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--epochs", type=int, default=None, help="backbone epochs")
    parser.add_argument(
        "--head-epochs", type=int, default=None, help="fusion-stage epochs"
    )
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=None,
        help="ProtoECGNet classifier-stage epochs (after projection)",
    )
    parser.add_argument(
        "--proto-fusion-init",
        choices=["classifier", "random"],
        default="classifier",
        help="ProtoECGNet fusion head start: branch classifiers or random",
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lam-l1", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=None, help="1D base width")
    parser.add_argument("--proto-dim", type=int, default=None)
    parser.add_argument(
        "--protos-per-label", type=int, default=None, help="0 = paper (5/18/7)"
    )
    parser.add_argument(
        "--backbone-2d",
        choices=["resnet2d", "resnet1d"],
        default=None,
        help="resnet1d replaces the paper's 2D branches (cheaper)",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic and not args.root and not args.summarize_only:
        parser.error("pass --root or --synthetic")
    # Synthetic data is tiny: shrink the models so the run takes minutes on
    # a CPU. Real data defaults follow the paper (ResNet-18, 512-d prototypes,
    # 5/18/7 prototypes per label, 2D morphology/global branches).
    tiny = args.synthetic
    defaults = {
        "epochs": 3 if tiny else 15,
        "head_epochs": 2 if tiny else 5,
        "classifier_epochs": 2 if tiny else 15,
        "patience": 3 if tiny else 5,
        "width": 8 if tiny else 64,
        "proto_dim": 16 if tiny else 512,
        "protos_per_label": 2 if tiny else 0,
        "backbone_2d": "resnet1d" if tiny else "resnet2d",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.experiments = [e for e in args.experiments.split(",") if e]
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiments {sorted(unknown)}; use {EXPERIMENTS}")
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.fractions = [float(f) for f in args.fractions.split(",")]
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> None:
    """Runs every pending configuration and writes the summary."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.md")
    if args.summarize_only:
        with open(summary_path, "w") as handle:
            handle.write(summarize(load_results(args.out_dir), args))
        print(f"wrote {summary_path}")
        return

    if args.synthetic:
        args.root = tempfile.mkdtemp(prefix="ptbxl_synthetic_")
        build_synthetic_ptbxl(args.root)
    dataset_kwargs = {"root": args.root, "dev": args.dev}
    if args.cache_dir:
        dataset_kwargs["cache_dir"] = args.cache_dir
    dataset = PTBXLDataset(**dataset_kwargs)
    task = PTBXLLabelGroupClassification(dataset.scp_statements_path)
    samples = dataset.set_task(task, num_workers=args.num_workers)

    labels, folds, patients = scan_samples(samples)
    vocab = samples.output_processors["labels"].label_vocab
    data = {
        "labels": labels,
        "label_names": [n for n, _ in sorted(vocab.items(), key=lambda i: i[1])],
        "patients": patients,
        "train": np.flatnonzero(folds <= 8),
        "val": np.flatnonzero(folds == 9),
        "test": np.flatnonzero(folds == 10),
    }
    print(
        f"{len(samples)} ECGs, {len(data['label_names'])} labels, "
        f"train/val/test = {len(data['train'])}/{len(data['val'])}"
        f"/{len(data['test'])}"
    )

    done = {row["key"] for row in load_results(args.out_dir)}
    for spec in run_specs(args):
        if run_key(spec) in done:
            continue
        print(f"run {run_key(spec)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            row = train_and_evaluate(samples, spec, data, args)
        with open(os.path.join(args.out_dir, "results.jsonl"), "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row))
        with open(summary_path, "w") as handle:
            handle.write(summarize(load_results(args.out_dir), args))
    print(summarize(load_results(args.out_dir), args))


if __name__ == "__main__":
    main()
