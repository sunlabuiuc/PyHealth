"""ProtoECGNet on PTB-XL: full pipeline and ablation study.

Pipeline (PTBXLDataset -> PTBXLMultilabelClassification -> ProtoECGNet):

1. Load PTB-XL and build the multi-label task (12 leads, 100 Hz).
2. Split by the recommended ``strat_fold`` (1-8 train, 9 validation, 10 test).
3. Train in the reference stages: ``feature_extractor`` -> (``joint`` ->
   prototype projection) x ``--rounds`` -> ``classifier`` (-> ``fusion``).
4. Report test macro AUROC (the paper's primary metric).

``--paper-groups`` uses the paper's three branches on all 71 labels: a 1D
ResNet with global prototypes for the 16 rhythm labels, and 2D ResNet-18
branches for the 52 morphology labels (partial prototypes) and the 3 global
labels, joined by a learned fusion layer.

Ablation (``--ablation``) varies settings that the paper does not sweep:
(a) the contrastive prototype loss on/off, (b) cosine vs L2 similarity,
(c) the number of prototypes per label. Each variant uses identical data and
seeds; the table is saved as markdown.

Usage:
    # Quick check on synthetic ECGs (no download needed, runs on CPU):
    python examples/ptbxl_multilabel_classification_protoecgnet.py --synthetic

    # Real data (open access; only the 100 Hz records, ptbxl_database.csv and
    # scp_statements.csv are needed):
    #   wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
    python examples/ptbxl_multilabel_classification_protoecgnet.py \
        --root physionet.org/files/ptb-xl/1.0.3 --label-set superdiagnostic \
        --epochs 20 --width 64 --ablation

    # Paper setting (all 71 labels, three branches, learned fusion):
    python examples/ptbxl_multilabel_classification_protoecgnet.py \
        --root physionet.org/files/ptb-xl/1.0.3 --label-set all --paper-groups

Note:
    Results on ``--synthetic`` data only demonstrate that the pipeline runs;
    they say nothing about performance on real ECGs. For small compute budgets
    (e.g. a free Colab T4) use ``--dev`` and fewer epochs, and state the
    compute limit when comparing with the paper.
"""

import argparse
import itertools
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import PTBXLDataset, get_dataloader
from pyhealth.models import PTBXL_BRANCH_KWARGS, PTBXL_BRANCH_LABELS, ProtoECGNet
from pyhealth.tasks import PTBXLMultilabelClassification
from pyhealth.trainer import Trainer

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SYNTHETIC_CODES = ["NORM", "MI", "STTC", "CD", "HYP"]


def build_synthetic_ptbxl(root: str, num_patients: int = 60, seed: int = 0) -> None:
    """Writes a small PTB-XL-like dataset whose labels are learnable.

    Label ``k`` adds a sine wave of frequency ``k + 1`` Hz to lead ``k``, so a
    classifier can recover the labels from the signal.

    Args:
        root: Output directory.
        num_patients: Number of synthetic patients (1-2 ECGs each).
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1000) / 100.0
    rows, ecg_id = [], 0
    for patient in range(num_patients):
        for _ in range(1 + patient % 2):
            ecg_id += 1
            labels = list(
                rng.choice(len(SYNTHETIC_CODES), size=1 + rng.integers(0, 2),
                           replace=False)
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
            codes = {SYNTHETIC_CODES[k]: 100.0 for k in labels}
            rows.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": float(patient),
                    "age": 60.0,
                    "sex": 1,
                    "scp_codes": str(codes),
                    "strat_fold": 1 + patient % 10,
                    "filename_lr": rel,
                    "filename_hr": rel.replace("records100", "records500"),
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)
    statements = pd.DataFrame(
        {
            "diagnostic": 1,
            "form": 0,
            "rhythm": 0,
            "diagnostic_class": SYNTHETIC_CODES,
            "diagnostic_subclass": SYNTHETIC_CODES,
        },
        index=pd.Index(SYNTHETIC_CODES, name="scp_code"),
    )
    statements.to_csv(os.path.join(root, "scp_statements.csv"))


def split_by_fold(sample_dataset, root: str):
    """Splits a PTB-XL SampleDataset by the recommended ``strat_fold``.

    Args:
        sample_dataset: The dataset returned by ``set_task``.
        root: PTB-XL root containing ``ptbxl-metadata-pyhealth.csv``.

    Returns:
        ``(train, val, test)`` datasets (folds 1-8, 9 and 10).
    """
    meta = pd.read_csv(os.path.join(root, "ptbxl-metadata-pyhealth.csv"))
    fold_of = dict(zip(meta["patient_id"].astype(str), meta["strat_fold"]))
    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for patient, indices in sample_dataset.patient_to_index.items():
        fold = fold_of[patient]
        key = "val" if fold == 9 else "test" if fold == 10 else "train"
        buckets[key].extend(indices)
    return tuple(sample_dataset.subset(buckets[k]) for k in ("train", "val", "test"))


def collect_labels(dataset) -> torch.Tensor:
    """Collects the full multi-hot label matrix in one pass over the dataloader.

    ``compute_label_statistics`` requires a single tensor rather than a
    dataloader, so we accumulate all batches here before calling it.
    """
    loader = get_dataloader(dataset, batch_size=256, shuffle=False)
    return torch.cat([batch["labels"] for batch in loader])


def to_markdown_table(rows: list[dict[str, object]]) -> str:
    """Formats a list of dicts as a markdown table (no extra dependencies)."""
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def run_pipeline(
    samples,
    splits,
    args: argparse.Namespace,
    use_contrastive: bool = True,
    similarity: str = "cosine",
    prototypes_per_label: int = 0,
) -> dict[str, float]:
    """Trains ProtoECGNet in stages and evaluates it on the test fold.

    Args:
        samples: The full SampleDataset (defines the label vocabulary).
        splits: ``(train, val, test)`` datasets.
        args: Parsed command-line arguments.
        use_contrastive: Whether to use the contrastive prototype loss.
        similarity: ``"cosine"`` or ``"l2"``.
        prototypes_per_label: Prototypes per label (0 uses the CLI value).

    Returns:
        A dict with the test ``roc_auc_macro`` and ``loss``.
    """
    torch.manual_seed(args.seed)
    train, val, test = splits
    loaders = [
        get_dataloader(d, batch_size=args.batch_size, shuffle=(i == 0))
        for i, d in enumerate((train, val, test))
    ]
    projection_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=False)
    weights, cooc = ProtoECGNet.compute_label_statistics(collect_labels(train))
    branch_kwargs = None
    if args.paper_groups:
        # The paper uses 5 / 18 / 7 prototypes per label for the three
        # branches (heavy for a free GPU). Keep them only with
        # --paper-prototypes and when no ablation overrides the count.
        keep = args.paper_prototypes and not prototypes_per_label
        branch_kwargs = {
            name: {k: v for k, v in kwargs.items()
                   if keep or k != "prototypes_per_label"}
            for name, kwargs in PTBXL_BRANCH_KWARGS.items()
        }
    model = ProtoECGNet(
        samples,
        label_groups=PTBXL_BRANCH_LABELS if args.paper_groups else None,
        branch_kwargs=branch_kwargs,
        fusion="learned" if args.paper_groups else "average",
        proto_dim=args.proto_dim,
        prototypes_per_label=prototypes_per_label or args.prototypes_per_label,
        similarity=similarity,
        topk=args.topk,
        base_width=args.width,
        lam_cnrst=300.0 if use_contrastive else 0.0,
        lam_l1=args.lam_l1,
        class_weights=weights,
        label_cooccurrence=cooc,
    )
    trainer = Trainer(
        model=model,
        metrics=["roc_auc_macro"],
        device=args.device,
        enable_logging=False,
    )

    def stage(name: str, epochs: int) -> None:
        model.set_stage(name)
        trainer.train(
            train_dataloader=loaders[0],
            val_dataloader=loaders[1],
            epochs=epochs,
            monitor="roc_auc_macro",
            optimizer_params={"lr": args.lr},
        )

    stage("feature_extractor", args.epochs)
    for _ in range(args.rounds):  # the paper repeats joint training + projection
        stage("joint", args.epochs)
        model.project_prototypes(projection_loader)
    stage("classifier", max(1, args.epochs // 2))
    if args.paper_groups:
        stage("fusion", max(1, args.epochs // 2))
    return trainer.evaluate(loaders[2])


def main() -> None:
    """Parses arguments, runs the pipeline and optionally the ablation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=str, help="PTB-XL 1.0.3 directory")
    parser.add_argument("--synthetic", action="store_true", help="use fake ECGs")
    parser.add_argument("--dev", action="store_true", help="load a small subset")
    parser.add_argument("--label-set", default="superdiagnostic")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--proto-dim", type=int, default=None)
    parser.add_argument("--prototypes-per-label", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1, help="top-k window pooling")
    parser.add_argument("--rounds", type=int, default=1, help="joint+projection rounds")
    parser.add_argument("--lam-l1", type=float, default=0.0, help="L1 of final layers")
    parser.add_argument("--paper-prototypes", action="store_true",
                        help="with --paper-groups, use the paper's 5/18/7 "
                             "prototypes per label instead of "
                             "--prototypes-per-label")
    parser.add_argument("--paper-groups", action="store_true",
                        help="3 branches / 71 labels as in the paper (real data only)")
    parser.add_argument("--no-contrastive", action="store_true",
                        help="train the main run without the contrastive "
                             "prototype loss (for H2-style comparisons)")
    parser.add_argument("--remove-baseline", action="store_true",
                        help="0.5 Hz high-pass filter (as in the paper)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--output", default="protoecgnet_ablation.md")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    scratch = tempfile.mkdtemp()
    if args.synthetic:
        args.root = os.path.join(scratch, "ptbxl")
        os.makedirs(args.root)
        build_synthetic_ptbxl(args.root)
        args.epochs = args.epochs or 3
        args.width = args.width or 8
        args.proto_dim = args.proto_dim or 16
        args.label_set = "superdiagnostic"
    elif not args.root:
        parser.error("pass --root /path/to/ptb-xl/1.0.3 or use --synthetic")
    if args.paper_groups and (args.synthetic or args.label_set != "all"):
        parser.error("--paper-groups needs real PTB-XL data and --label-set all")
    args.epochs = args.epochs or 20
    args.width = args.width or 64
    args.proto_dim = args.proto_dim or 128

    dataset = PTBXLDataset(
        root=args.root, cache_dir=os.path.join(scratch, "cache"), dev=args.dev
    )
    task = PTBXLMultilabelClassification(
        label_set=args.label_set, remove_baseline=args.remove_baseline
    )
    samples = dataset.set_task(task)
    splits = split_by_fold(samples, args.root)
    print(f"samples: train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")

    baseline = run_pipeline(
        samples, splits, args, use_contrastive=not args.no_contrastive
    )
    name = "no-contrastive" if args.no_contrastive else "baseline"
    print(f"{name} test macro AUROC: {baseline['roc_auc_macro']:.4f}")

    if args.ablation:
        rows = []
        for contrastive, similarity, per_label in itertools.product(
            [True, False], ["cosine", "l2"], [1, 3]
        ):
            result = run_pipeline(
                samples, splits, args, contrastive, similarity, per_label
            )
            rows.append(
                {
                    "contrastive": contrastive,
                    "similarity": similarity,
                    "prototypes_per_label": per_label,
                    "test_roc_auc_macro": round(result["roc_auc_macro"], 4),
                }
            )
            print(rows[-1])
        table = to_markdown_table(rows)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print(table)


if __name__ == "__main__":
    main()
