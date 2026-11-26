"""Compare supervised CLIPStress with contrastive pretraining and fine-tuning.

Contributor: Siqi Zhu (siqizhu4).
Paper: https://proceedings.mlr.press/v287/yang25a.html

From the PyHealth repository root, run the offline demonstration::

    python examples/lifesnaps_stress_clip_stress.py --output-dir runs/demo

This reproduces the setting of the paper's Table 1 (LifeSnaps): random
splitting with generalized models, 5-fold cross-validation and 4 random seeds,
comparing the supervised multimodal network with contrastive pretraining
followed by end-to-end fine-tuning. Use --fold-data-dir with --paper-budget,
which applies Section 3.2.3: 300 supervised epochs; 300 pretraining and
150 fine-tuning epochs; learning rate 0.001; batch size 512; a 3-layer, 4-head
Transformer; the printed cosine contrastive objective (Eq. 1); every
training-fold window, with its label removed, for pretraining; and seeds 0-3.

The default paper_random protocol is the public code's StratifiedKFold over all
windows (shuffled, random_state 42, stratified by stressed/relaxed/unlabeled),
so a participant may appear in both training and test folds. The fold archives
contain fold-fitted quantile-normalized inputs and frozen BERT embeddings of
label-free prompts. reference_user (participant-disjoint, Table 2's
generalized non-random split) and validated_user remain available.

The paper tunes hyperparameters for each fold and seed without specifying the
search space; this example instead uses one fixed configuration and the final
checkpoint, so no epoch or setting is chosen using test data. The contrastive
temperature is not stated in the paper; --temperature defaults to 0.07, the
public code's value. --workers and --worker-index shard independent jobs.
Synthetic-mode metrics only demonstrate execution. No metric target is required.
"""

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

if __package__:
    from .clip_stress import build_parser, prepare_datasets, run, synthetic_data
    from .clip_stress_experiment_utils import file_hash, source_hashes, write_json
else:
    from clip_stress import build_parser, prepare_datasets, run, synthetic_data
    from clip_stress_experiment_utils import file_hash, source_hashes, write_json


def subject_folds(patient_ids, folds, seed, protocol="validated_user"):
    """Rotate validation/test subjects with disjoint training groups."""
    patients = np.unique(np.asarray(patient_ids).astype(str))
    if not 3 <= folds <= len(patients):
        raise ValueError("Need at least three folds and one subject per fold.")
    if protocol == "validated_user":
        np.random.default_rng(seed).shuffle(patients)
    groups = np.array_split(patients, folds)
    return [
        {
            "test": groups[i].tolist(),
            "validation": groups[(i + 1) % folds].tolist()
            if protocol == "validated_user"
            else [],
            "train": np.concatenate(
                [
                    groups[j]
                    for j in range(folds)
                    if j != i and (protocol == "reference_user" or j != (i + 1) % folds)
                ]
            ).tolist(),
        }
        for i in range(folds)
    ]


def random_folds(labels, folds, seed=42):
    """Paper random splitting: the public code's stratified row-level folds.

    Stratifies on -1 (unlabeled), 0 and 1 over all windows in archive order,
    exactly as ``StratifiedKFold(5, shuffle=True, random_state=42)`` in the
    authors' LifeSnaps preprocessing. There is no validation fold.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [
        {"train": train, "validation": np.empty(0, dtype=np.int64), "test": test}
        for train, test in splitter.split(np.zeros((len(labels), 1)), labels)
    ]


def experiment_folds(data, args):
    """Return per-fold splits and the keyword that passes them to ``run``."""
    if args.protocol == "paper_random":
        return random_folds(data["label"], args.folds, args.split_seed), "sample_splits"
    splits = subject_folds(
        data["patient_id"], args.folds, args.split_seed, args.protocol
    )
    return splits, "subject_splits"


def experiment_seeds(args):
    """Training seeds of the matrix; the data split does not depend on them."""
    return list(args.seeds) if args.seeds else [args.seed]


def experiment_jobs(args):
    """Enumerate the complete matrix before sharding, independently of scores."""
    # Put the expensive pretraining jobs first so that round-robin workers
    # receive equal shares of both recipes.
    recipes = [(f"tau-{args.temperature:g}", args.temperature), ("supervised", None)]
    return [
        (seed, fold, name, tau)
        for name, tau in recipes
        for seed in experiment_seeds(args)
        for fold in range(args.folds)
    ]


def job_path(args, seed, fold, name):
    """Result file of one fold/seed/recipe run."""
    return args.output_dir / f"fold-{fold + 1}-seed-{seed}-{name}.json"


def experiment_identity(args):
    """Hash scientific settings and inputs; exclude device/worker placement."""
    ignored = {
        "worker_index",
        "workers",
        "device",
        "threads",
        "output",
        "output_dir",
        "artifacts_dir",
        "max_seconds",
        "summarize_only",
    }
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ignored
    }
    identity = {
        "settings": settings,
        "source_sha256": source_hashes(),
        "data_sha256": (
            {
                str(p.name): file_hash(p)
                for p in sorted(args.fold_data_dir.glob("fold-*.npz"))
            }
            if args.fold_data_dir
            else file_hash(args.data)
            if args.data
            else "synthetic"
        ),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return digest, identity


def summarize(args):
    """Require a complete matrix and report mean and SD over all fold/seed runs."""
    digest, identity = experiment_identity(args)
    records = []
    for seed, fold, name, _ in experiment_jobs(args):
        path = job_path(args, seed, fold, name)
        record = json.loads(path.read_text())
        if record["experiment_id"] != digest or not record.get("complete"):
            raise ValueError(f"Incompatible or incomplete result: {path}")
        records.append(record)
    summary = {}
    for name in sorted({record["recipe"] for record in records}):
        rows = [
            next(iter(r["results"].values())) for r in records if r["recipe"] == name
        ]
        summary[name] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows], ddof=1)),
                "runs": len(rows),
            }
            for metric in rows[0]
        }
    paired = {}
    for name in summary:
        if name == "supervised":
            continue
        differences = []
        for seed in experiment_seeds(args):
            for fold in range(1, args.folds + 1):
                values = {
                    r["recipe"]: next(iter(r["results"].values()))["roc_auc"]
                    for r in records
                    if r["fold"] == fold and r["seed"] == seed
                }
                differences.append(values[name] - values["supervised"])
        paired[name] = {
            "roc_auc_differences": differences,
            "mean": float(np.mean(differences)),
        }
    report = {
        "complete": True,
        "experiment_id": digest,
        "identity": identity,
        "runs": len(records),
        "seeds": experiment_seeds(args),
        "summary": summary,
        "paired_against_supervised": paired,
        "interpretation": (
            "Mean and sample SD over all fold/seed runs, as in the paper's tables; "
            "runs share data, so the SD is descriptive, not a confidence interval."
        ),
    }
    write_json(args.output_dir / "summary.json", report)
    return report


def execute(args):
    """Preflight all splits, then execute or resume this worker's fixed jobs."""
    if args.downstream_stage != "finetune":
        raise ValueError("This fixed matrix requires --downstream-stage finetune.")
    if not 0 <= args.worker_index < args.workers or args.max_seconds <= 0:
        raise ValueError("Invalid worker assignment or time limit.")
    if args.fold_data_dir:
        path = args.fold_data_dir / "fold-1.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
    elif args.data:
        with np.load(args.data, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
    else:
        data = synthetic_data(args.split_seed)
    splits, split_keyword = experiment_folds(data, args)
    if args.protocol != "validated_user" and args.checkpoint_selection != "final":
        raise ValueError(
            "Train/test folds have no validation data; use final checkpoints."
        )
    if args.protocol != "validated_user" and args.data and not args.fold_data_dir:
        raise ValueError(
            "Real reference experiments require --fold-data-dir with fold-specific text."
        )
    if args.fold_data_dir:
        for fold in range(args.folds):
            path = args.fold_data_dir / f"fold-{fold + 1}.npz"
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if (
                    metadata.get("fold_index") != fold
                    or metadata.get("protocol") != args.protocol
                ):
                    raise ValueError(f"Incompatible fold cache: {path}")
                expected = (
                    metadata.get("test_rows") == splits[fold]["test"].tolist()
                    if args.protocol == "paper_random"
                    else metadata.get("subjects") == splits[fold]
                )
                if not expected:
                    raise ValueError(f"Fold cache split mismatch: {path}")
        if args.normalization != "none":
            raise ValueError(
                "Fold archives are already normalized; use --normalization none."
            )
    for split in splits:
        prepare_datasets(
            data,
            args.seed,
            args.label_fraction,
            normalization=args.normalization,
            pretrain_sampling=args.pretrain_sampling,
            **{split_keyword: split},
        )
    digest, identity = experiment_identity(args)
    start = time.monotonic()
    for index, (seed, fold, name, tau) in enumerate(experiment_jobs(args)):
        if index % args.workers != args.worker_index:
            continue
        path = job_path(args, seed, fold, name)
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["experiment_id"] != digest or not previous.get("complete"):
                raise ValueError(f"Refusing to reuse incompatible result: {path}")
            continue
        if time.monotonic() - start >= args.max_seconds:
            raise TimeoutError("Worker budget exhausted; completed runs are saved.")
        settings = copy.copy(args)
        if args.fold_data_dir:
            settings.data = args.fold_data_dir / f"fold-{fold + 1}.npz"
            with np.load(settings.data, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
        settings.ablation = "supervised" if tau is None else "multimodal"
        settings.seed = seed
        settings.artifacts_dir = args.output_dir / path.stem
        print(f"START {path.stem}", flush=True)
        result = run(settings, data=data, **{split_keyword: splits[fold]})
        result.update(
            complete=True,
            fold=fold + 1,
            recipe=name,
            experiment_id=digest,
            identity=identity,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        write_json(path, result)
        print(f"COMPLETE {path.stem}", flush=True)


def parser():
    """Return a CLI with fast offline defaults and explicit real-data budgets."""
    result = build_parser()
    result.description = __doc__
    result.set_defaults(
        seed=0, downstream_stage="finetune", checkpoint_selection="final"
    )
    result.add_argument("--fold-data-dir", type=Path)
    result.add_argument(
        "--protocol",
        choices=["paper_random", "reference_user", "validated_user"],
        default="paper_random",
    )
    result.add_argument(
        "--paper-budget",
        action="store_true",
        help="Paper Section 3.2.3: 5 folds, seeds 0-3, 300 pretrain / 150 "
        "fine-tune / 300 supervised epochs, batch 512, lr 0.001, cosine objective",
    )
    result.add_argument("--seeds", type=int, nargs="+")
    result.add_argument("--folds", type=int, default=3)
    result.add_argument("--split-seed", type=int, default=42)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--worker-index", type=int, default=0)
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--max-seconds", type=float, default=7200)
    result.add_argument("--summarize-only", action="store_true")
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    if options.paper_budget:
        options.folds = 5
        options.pretrain_epochs, options.epochs, options.supervised_epochs = (
            300,
            150,
            300,
        )
        options.batch_size, options.hidden_dim, options.projection_dim = 512, 64, 128
        options.num_layers, options.num_heads = 3, 4
        options.learning_rate, options.dropout = 0.001, 0.1
        options.seeds = [0, 1, 2, 3]
        options.objective = "paper_cosine"
        options.pretrain_sampling = "all"
    if options.fold_data_dir:
        options.normalization = "none"
    if options.summarize_only:
        print(json.dumps(summarize(options), indent=2))
    else:
        execute(options)
        if options.workers == 1:
            summarize(options)
