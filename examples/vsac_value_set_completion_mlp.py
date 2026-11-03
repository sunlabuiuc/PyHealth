"""VSAC value set completion with PyHealth's MLP: reproduction and ablations.

Author: Bryan Chance (NetID: bchance2)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616

Pipeline: VSACDataset -> build_retrieval_index (SapBERT titles and display
names) -> ValueSetCompletion -> MLP with class-balanced BCE -> threshold tuned
for F1 on validation -> test metrics. Metrics follow the authors'
model_training/evaluate.py: pair-level AUROC, average precision, and F1, and
value-set-level macro precision, recall, and F1, where recall counts true
codes that retrieval never surfaced.

Ablations (none are in the paper), each a ValueSetCompletion argument:

- ``uncapped``: ``max_codes_per_set=None``. VSAC's ``$expand`` returns at most
  1,000 codes per call and the authors' download script never pages, so the
  benchmark labels are truncated for the largest value sets. This compares
  capped labels with complete ones.
- ``k5``, ``k20``, ``k40``: retrieved value sets per target (paper: K=10).
- ``no_score``, ``no_code_system``: drop the retrieval-similarity score or the
  code-system one-hot from the inputs.

Usage:

    # Runs end to end on generated value sets with a stub encoder (seconds).
    python examples/vsac_value_set_completion_mlp.py --synthetic

    # Real VSAC download (expansions/ + split_manifest.jsonl), on a GPU.
    python examples/vsac_value_set_completion_mlp.py --root vsac_corpus \\
        --configs main --epochs 30
    python examples/vsac_value_set_completion_mlp.py --root vsac_corpus \\
        --configs main uncapped k5 k20 k40 no_score no_code_system \\
        --sample-frac 0.2

``--sample-frac`` keeps a fixed fraction of target value sets (hashed by OID,
so every configuration sees the same sets); retrieval still searches the full
corpus. Results go to ``--output`` as ``results.json``, ``results.md``, and
one ``per_set_<config>.csv`` of value-set metrics per configuration.

Setup: VSAC expansions at the paper's pinned versions with the authors'
release splits (11,793 value sets), SapBERT index over 17,933 source sets,
MLP with hidden_dim 256 and 2 layers, AdamW (lr 3e-4, weight decay 1e-4),
batch 2,048, early stopping on validation PR-AUC. One NVIDIA A10G.

Findings, reproduction (full test split, 3,737 value sets, patience 5):

    Paper MLP:        pair AUROC 0.799, value-set F1 0.250
    PyHealth MLP:     pair AUROC 0.722, value-set F1 0.186

Pair counts match the paper's Table 1 within 3%, so the gap is the model:
PyHealth's MLP has a tower per input and no BatchNorm or dropout, and
validation PR-AUC peaks after 1-2 epochs. lr 1e-4 gives the same result.

Findings, ablations (20% of target value sets, about 750 test sets,
patience 2, one seed each; reruns of main and uncapped moved value-set F1
by 0.002):

    Config          Pair AUROC  Pair AP  VS F1  Retrieval recall
    main            0.749       0.304    0.182  0.541
    uncapped        0.775       0.325    0.194  0.553
    k5              0.720       0.363    0.199  0.476
    k20             0.764       0.248    0.198  0.588
    k40             0.827       0.245    0.192  0.616
    no_score        0.560       0.124    0.071  0.541
    no_code_system  0.748       0.308    0.193  0.541

- The retrieval score carries the classifier; without it F1 falls to 0.071.
- Larger K raises retrieval recall and AUROC (more easy negatives), but
  value-set F1 stays at 0.18-0.20.
- The code-system one-hot adds nothing measurable.
- Size trend under complete labels (paper section 5.3.2: classifiers improve
  with set size). Value-set F1 still rises with complete set size (Spearman
  rho 0.19 uncapped vs 0.17 capped, p < 1e-5). It rises from 0.12 at 3-5
  codes to 0.28 at 51-200, then plateaus. The trend holds within
  Condition/Clinical, Condition/Diagnosis, and Lab/Observation, so set type
  doesn't explain it, but Medication shows none (rho -0.1, n=80). On the 21
  test sets over 1,000 codes, capped labels understate F1 by 0.058 (paired
  bootstrap 95% CI 0.002-0.138).
"""

import argparse
import json
import random
import shutil
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from pyhealth.datasets import VSACDataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.tasks import ValueSetCompletion, build_retrieval_index
from pyhealth.tasks.value_set_completion import FEATURES, sapbert_encoder
from pyhealth.trainer import Trainer

CONFIGS: dict[str, dict] = {
    "main": {},
    "uncapped": {"max_codes_per_set": None},
    "k5": {"k": 5},
    "k20": {"k": 20},
    "k40": {"k": 40},
    "no_score": {"features": [f for f in FEATURES if f != "score"]},
    "no_code_system": {"features": [f for f in FEATURES if f != "code_system"]},
}
SOURCE_SPLITS = ("train", "val", "test", "corpus")


class SampledValueSetCompletion(ValueSetCompletion):
    """Keeps a fixed, OID-hashed fraction of target value sets."""

    def __init__(self, index_dir: str, sample_frac: float = 1.0, **kwargs: Any) -> None:
        super().__init__(index_dir, **kwargs)
        self.sample_frac = sample_frac

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.sample_frac >= 1.0:
            return df
        cutoff = int(self.sample_frac * 2**32)
        oids = df.select("patient_id").unique().collect()["patient_id"].to_list()
        kept = [o for o in oids if zlib.crc32(o.encode()) < cutoff]
        return df.filter(pl.col("patient_id").is_in(kept))


def stub_encoder(texts: Sequence[str]) -> np.ndarray:
    """Bag-of-words hashing encoder for the synthetic demo."""
    vecs = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, zlib.crc32(word.encode()) % 64] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def write_synthetic(root: Path, n_topics: int = 8, per_topic: int = 6) -> None:
    """Writes value sets in topic families that share most of their codes."""
    rng = random.Random(0)
    systems = ["http://snomed.info/sct", "http://hl7.org/fhir/sid/icd-10-cm"]
    splits = ["train", "train", "train", "val", "test", "corpus"]
    exp_dir = root / "expansions"
    exp_dir.mkdir(parents=True)
    manifest = []
    for t in range(n_topics):
        pool = [f"T{t}C{c}" for c in range(30)]
        for v in range(per_topic):
            oid = f"9.{t}.{v}-20240101"
            members = rng.sample(pool, rng.randint(8, 20))
            resource = {
                "id": oid,
                "title": f"topic{t} condition variant{v}",
                "version": "20240101",
                "expansion": {
                    "total": len(members),
                    "contains": [
                        {"system": systems[t % 2], "code": c, "display": f"{c} term"}
                        for c in members
                    ],
                },
            }
            (exp_dir / f"{oid}.json").write_text(json.dumps(resource))
            manifest.append({"oid": oid, "split": splits[v]})
    (root / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximizing F1 on validation, as in the authors' scripts."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[min(int(np.argmax(f1)), len(thresholds) - 1)])


def value_set_metrics(
    samples: Sequence[dict], y_prob: np.ndarray, threshold: float
) -> tuple[dict, pd.DataFrame]:
    """Macro precision/recall/F1 over value sets, recall over all true codes.

    Returns the macro averages and one row per value set.
    """
    predicted, hits, retrieved, n_true = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
        {},
    )
    for sample, prob in zip(samples, y_prob):
        oid, code = sample["patient_id"], sample["code"]
        n_true[oid] = int(sample["n_true_codes"])
        is_true = int(sample["label"]) == 1
        if is_true:
            retrieved[oid].add(code)
        if prob >= threshold:
            predicted[oid].add(code)
            if is_true:
                hits[oid].add(code)
    rows = []
    for oid, total in n_true.items():
        p = len(hits[oid]) / len(predicted[oid]) if predicted[oid] else 0.0
        r = len(hits[oid]) / total
        rows.append(
            {
                "oid": oid,
                "n_true_codes": total,
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "retrieval_recall": len(retrieved[oid]) / total,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "n_value_sets": len(frame),
        "macro_precision": float(frame["precision"].mean()),
        "macro_recall": float(frame["recall"].mean()),
        "macro_f1": float(frame["f1"].mean()),
        "retrieval_recall": float(frame["retrieval_recall"].mean()),
    }, frame


def run_config(
    dataset: VSACDataset,
    index_dir: str,
    name: str,
    args: argparse.Namespace,
) -> dict:
    """Trains and evaluates one configuration; returns its metrics."""
    torch.manual_seed(42)
    task = SampledValueSetCompletion(
        index_dir, sample_frac=args.sample_frac, **CONFIGS[name]
    )
    train, val, test = dataset.set_task(
        task, num_workers=args.num_workers, split=dataset.manifest_split()
    )
    model = MLP(
        dataset=train,
        embedding_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    model.set_pos_weight("balanced", train)
    trainer = Trainer(
        model,
        metrics=["roc_auc", "pr_auc"],
        device=args.device,
        output_path=str(Path(args.output) / "runs"),
        exp_name=name,
    )
    trainer.train(
        train_dataloader=get_dataloader(train, args.batch_size, shuffle=True, seed=42),
        val_dataloader=get_dataloader(val, args.batch_size),
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr, "weight_decay": 1e-4},
        monitor="pr_auc",
        patience=args.patience,
    )
    _, val_prob, _ = trainer.inference(get_dataloader(val, args.batch_size))[:3]
    val_true = np.array([int(s["label"]) for s in val])
    threshold = best_threshold(val_true, np.ravel(val_prob))
    test_true, test_prob, _ = trainer.inference(get_dataloader(test, args.batch_size))[
        :3
    ]
    test_true, test_prob = np.ravel(test_true), np.ravel(test_prob)
    summary, per_set = value_set_metrics(test, test_prob, threshold)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    per_set.to_csv(output / f"per_set_{name}.csv", index=False)
    result = {
        "config": name,
        "task_args": {k: v for k, v in vars(task).items() if k != "input_schema"},
        "n_pairs": {"train": len(train), "val": len(val), "test": len(test)},
        "threshold": threshold,
        "pair_auroc": float(roc_auc_score(test_true, test_prob)),
        "pair_ap": float(average_precision_score(test_true, test_prob)),
        "pair_f1": float(f1_score(test_true, test_prob >= threshold)),
        **summary,
    }
    if args.cleanup:
        shutil.rmtree(Path(dataset.cache_dir) / "tasks", ignore_errors=True)
    return result


def write_report(results: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str))
    rows = [
        (
            "| Config | Test pairs | Pair AUROC | Pair AP | Pair F1 | VS precision "
            "| VS recall | VS F1 | Retrieval recall |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['config']} | {r['n_pairs']['test']:,} | {r['pair_auroc']:.4f} "
        f"| {r['pair_ap']:.4f} | {r['pair_f1']:.4f} | {r['macro_precision']:.4f} "
        f"| {r['macro_recall']:.4f} | {r['macro_f1']:.4f} "
        f"| {r['retrieval_recall']:.4f} |"
        for r in results
    ]
    (output / "results.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="VSAC download (expansions/ + manifest)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--index-dir", help="Reuse or write the retrieval index")
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument("--output", default="vsac_results")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cleanup", action="store_true", help="Drop sample caches")
    args = parser.parse_args()
    unknown = set(args.configs) - set(CONFIGS)
    if unknown or not (args.root or args.synthetic):
        parser.error(f"need --root or --synthetic; known configs: {list(CONFIGS)}")

    work = Path(tempfile.mkdtemp()) if args.synthetic else None
    if args.synthetic:
        write_synthetic(work / "vsac")
        args.root = str(work / "vsac")
        args.cache_dir = str(work / "cache")
        args.index_dir = args.index_dir or str(work / "index")
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 256)
    dataset = VSACDataset(root=args.root, cache_dir=args.cache_dir)
    index_dir = args.index_dir or str(Path(args.output) / "index")
    if not (Path(index_dir) / "meta.json").exists():
        build_retrieval_index(
            dataset,
            stub_encoder if args.synthetic else sapbert_encoder(device=args.device),
            index_dir,
            max_k=40,
            source_splits=SOURCE_SPLITS,
        )
    results = []
    for name in args.configs:
        results.append(run_config(dataset, index_dir, name, args))
        write_report(results, Path(args.output))
    if work is not None:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
