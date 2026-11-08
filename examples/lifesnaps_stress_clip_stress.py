"""Compare supervised CLIPStress and three contrastive temperatures.

Contributor: Siqi Zhu (siqizhu4).
Paper: https://proceedings.mlr.press/v287/yang25a.html

From the PyHealth repository root, run the offline demonstration::

    python examples/lifesnaps_stress_clip_stress.py --output-dir runs/demo

For reference-aligned LifeSnaps use --fold-data-dir, --paper-budget and --seed 0.
The fold archives contain quantile-normalized inputs and fold-specific frozen
BERT text embeddings. Reference subject folds use four training groups and one
held-out group, with no subject shuffling. Fixed final checkpoints avoid choosing
epochs using the test set. --workers and --worker-index shard independent jobs.

The ablation varies contrastive temperature (0.03, 0.07, 0.2). The reference-soft
objective is the default; paper_cosine is an explicitly named alternative.
Single-seed results are not the paper's four-seed, tuned performance estimates.
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


def experiment_jobs(args):
    """Enumerate the complete matrix before sharding, independently of scores."""
    recipes = [("supervised", None)] + [
        (f"tau-{temperature:g}", temperature) for temperature in (0.03, 0.07, 0.2)
    ]
    # Put all 15 expensive pretraining jobs first so two workers receive 8/7,
    # rather than 5/10 when alternating through four recipes within each fold.
    recipes = recipes[1:] + recipes[:1]
    return [(fold, name, tau) for name, tau in recipes for fold in range(args.folds)]


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
    """Require a complete matrix and report descriptive fold variation."""
    digest, identity = experiment_identity(args)
    records = []
    for fold, name, _ in experiment_jobs(args):
        path = args.output_dir / f"fold-{fold + 1}-{name}.json"
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
                "folds": len(rows),
            }
            for metric in rows[0]
        }
    paired = {}
    for name in summary:
        if name == "supervised":
            continue
        differences = []
        for fold in range(1, args.folds + 1):
            values = {
                r["recipe"]: next(iter(r["results"].values()))["roc_auc"]
                for r in records
                if r["fold"] == fold
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
        "seed": args.seed,
        "summary": summary,
        "paired_against_supervised": paired,
        "interpretation": "Single seed; SD describes folds, not seed robustness or CI.",
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
    splits = subject_folds(
        data["patient_id"], args.folds, args.split_seed, args.protocol
    )
    if args.protocol == "reference_user" and args.checkpoint_selection != "final":
        raise ValueError(
            "Reference four-fold training uses final checkpoints, without test selection."
        )
    if args.protocol == "reference_user" and args.data and not args.fold_data_dir:
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
                if (
                    metadata.get("subjects") != splits[fold]
                    or metadata.get("implementation") != "reference-aligned-v2"
                ):
                    raise ValueError(
                        f"Fold cache subjects or implementation mismatch: {path}"
                    )
        if args.normalization != "none":
            raise ValueError(
                "Fold archives are already normalized; use --normalization none."
            )
    for split in splits:
        prepare_datasets(
            data,
            args.seed,
            args.label_fraction,
            subject_splits=split,
            normalization=args.normalization,
            pretrain_sampling=args.pretrain_sampling,
        )
    digest, identity = experiment_identity(args)
    start = time.monotonic()
    for index, (fold, name, tau) in enumerate(experiment_jobs(args)):
        if index % args.workers != args.worker_index:
            continue
        path = args.output_dir / f"fold-{fold + 1}-{name}.json"
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
        settings.temperature = 0.07 if tau is None else tau
        settings.artifacts_dir = args.output_dir / path.stem
        print(f"START {path.stem}", flush=True)
        result = run(settings, data=data, subject_splits=splits[fold])
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
        choices=["reference_user", "validated_user"],
        default="reference_user",
    )
    result.add_argument(
        "--paper-budget",
        action="store_true",
        help="300 pretrain / 150 downstream / 300 supervised epochs, batch 512",
    )
    result.add_argument("--folds", type=int, default=3)
    result.add_argument("--split-seed", type=int, default=7)
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
        options.pretrain_sampling = "reference_bootstrap"
    if options.fold_data_dir:
        options.normalization = "none"
    if options.summarize_only:
        print(json.dumps(summarize(options), indent=2))
    else:
        execute(options)
        if options.workers == 1:
            summarize(options)
