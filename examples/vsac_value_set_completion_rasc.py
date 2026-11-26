"""VSAC value set completion with the RASC cross-encoder: reproduction and ablations.

Author: Wenchang Qi (NetID: qi14)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616

Pipeline: VSACDataset -> build_retrieval_index (SapBERT titles) ->
ValueSetPairClassification -> RASCCrossEncoder (SapBERT sentence-pair
classifier, class-balanced BCE) -> threshold tuned for F1 on validation ->
test metrics. Metrics follow examples/vsac_value_set_completion_mlp.py: pair
AUROC and average precision, and value-set-level macro precision, recall and
F1, where recall counts true codes that retrieval never surfaced. Each run
also reports the retrieval-only baseline (predict every candidate).

Paper targets (Tables 2 and 3): cross-encoder pair AUROC 0.852, AP 0.374,
value-set F1 0.298; retrieval-only value-set F1 0.136.

Configurations. ``main`` is the paper's model; the rest are not in the paper:

- ``main``: title and display name only, max_length 128, lr 2e-5.
- ``released``: the authors' fine-tuned weights, pinned to the commit of
  their Hugging Face repository that was evaluated here, without training.
  Separates a data mismatch from a training mismatch.
- ``fused``: adds the retrieval score and code-system one-hot to the logit
  (late fusion). The paper's MLP sees both; its cross-encoder sees neither.
- ``fused_score``: adds the retrieval score alone.
- ``len32``, ``len64``: token budget per pair. Titles and display names are
  short, so a smaller budget may cost nothing and trains faster.
- ``lr5e-5``, ``lr1e-5``: learning rate.
- ``hybrid``: ``main`` on candidates from SapBERT + BM25 title retrieval
  fused by reciprocal rank, at the same K=10. The paper names retrieval
  coverage as the binding constraint; this tests whether lexical matching
  raises retrieval recall and whether F1 follows.

Usage:

    # Runs end to end on generated value sets with a tiny random BERT built
    # on the fly and a stub retrieval encoder (CPU, a minute or two).
    python examples/vsac_value_set_completion_rasc.py --synthetic

    # Real VSAC download (expansions/ + split_manifest.jsonl), on a GPU.
    python examples/vsac_value_set_completion_rasc.py --root vsac_corpus \\
        --configs main released --mixed-precision
    python examples/vsac_value_set_completion_rasc.py --root vsac_corpus \\
        --configs main fused fused_score len32 len64 lr5e-5 lr1e-5 hybrid \\
        --sample-frac 0.2 --mixed-precision

``--sample-frac`` keeps a fixed fraction of target value sets (hashed by OID,
so every configuration sees the same sets); retrieval still searches the full
corpus. Results go to ``--output`` as ``results.json``, ``results.md``, and
one ``per_set_<config>.csv`` of value-set metrics per configuration.
``--cleanup`` frees disk between configurations by deleting the dataset's
whole ``tasks`` cache directory, so it also removes samples cached by other
scripts on the same dataset cache.

Differences from the authors' training script: batch size 64 replaces 16 with
4 accumulation steps; ``--mixed-precision`` runs the encoder in bfloat16 where
the authors use fp16; the authors' linear warmup (6%) and decay run inside
the optimizer (``WarmupAdamW``), since PyHealth's Trainer takes no scheduler;
early stopping watches validation PR-AUC instead of validation loss.

Training pairs are permuted in memory (``shuffled_dataloader``).
``get_dataloader(shuffle=True)`` shuffles a patient-level split only within
each value set's run of samples, so a batch of 64 held 1.3 value sets on
average. With that loader the full run stayed at validation AUROC 0.500
for three epochs, and adding warmup did not help on a 3% subset; with the
permutation it reaches the paper's numbers. The MLP example trains through the same stock loader.

Findings, real VSAC. 11,799 of the release manifest's 11,803 value sets were
still downloadable (train 6,632, validation 1,425, test 3,742), first 1,000
codes each as in the paper; SapBERT index over all of them, K=10; 4.00M
training, 0.92M validation and 2.25M test pairs. One NVIDIA A100, about
3 hours for ``main``. Every row is a single run.

Reproduction, full test split (3,736 value sets with candidates):

    Model                        Pair AUROC  Pair AP  VS P   VS R   VS F1
    Paper cross-encoder          0.852       0.374    -      -      0.298
    Authors' weights (released)  0.850       0.353    0.256  0.393  0.260
    Trained here (main)          0.850       0.386    0.305  0.342  0.274
    Retrieval only               -           -        -      0.555  0.115

- Pair-level ranking reproduces: both the authors' weights and a model
  trained with PyHealth's Trainer land within 0.002 AUROC of the paper.
- Value-set F1 is 0.02-0.04 below the paper's 0.298, for the authors' own
  weights too, so the gap is in the evaluation setup rather than in
  training: the threshold is tuned here on validation pairs pooled over
  value sets, and value sets whose true codes were never retrieved count
  with their full recall loss. Which of the paper's choices differ was not
  tested. Retrieval recall (0.555) matches the paper's 0.553; retrieval-only
  F1 is 0.115 against 0.136.

Ablations, 20% of the value sets of every split (755 test value sets,
0.81M training pairs), same recipe:

    Config       Test pairs  Pair AUROC  Pair AP  VS F1  Retrieval recall
    main         474,857     0.731       0.258    0.183  0.538
    fused        474,857     0.702       0.243    0.180  0.538
    fused_score  474,857     0.728       0.263    0.186  0.538
    len32        474,857     0.721       0.248    0.184  0.538
    len64        474,857     0.712       0.232    0.179  0.538
    lr5e-5       474,857     0.659       0.206    0.176  0.538
    lr1e-5       474,857     0.731       0.272    0.181  0.538
    hybrid       434,775     0.698       0.250    0.195  0.522

- Data volume matters most: ``main`` goes from AUROC 0.731 on 20% of the
  training value sets to 0.850 on all of them (0.574 on 3%, where training
  loss fell to 0.22 while validation loss rose).
- Run-to-run noise is about 0.01-0.02 AUROC: ``len64`` is below both
  ``len32`` and ``main``, which no mechanism explains. Differences of that
  size are not findings.
- ``lr5e-5`` is the one clear effect (-0.07 AUROC). The paper's 2e-5 and
  half of it are indistinguishable.
- Late fusion does not help. ``fused_score`` equals ``main`` within noise
  and ``fused`` is lower: the cross-encoder gains nothing from the two
  signals the paper's MLP relies on.
- A token budget of 32 costs at most 0.01 AUROC, so pairs can be truncated
  to a quarter of the paper's length.
- ``hybrid`` has the highest value-set F1 (0.195 against 0.183) on 8% fewer
  candidates, but lower AUROC and slightly lower retrieval recall (0.522
  against 0.538): BM25 displaces dense neighbors that held true codes. Its
  retrieval-only F1 is higher (0.109 against 0.102), so the pools are
  cleaner, not larger. One run each; the F1 gain is within what noise could
  produce.

Findings, synthetic demo (``--synthetic``: 144 generated value sets, a
random 2-layer BERT, K=8, 12 epochs, learning rates scaled by 50, constant
rate, seed 42, CPU). It shows that the pipeline runs and every configuration
trains; the numbers say nothing about VSAC:

    Config       Pair AUROC  Pair AP  VS F1  Retrieval-only F1
    main         0.958       0.869    0.907  0.407
    fused        1.000       1.000    1.000  0.407
    fused_score  1.000       1.000    1.000  0.407
    len32        0.958       0.869    0.907  0.407
    len64        0.958       0.869    0.907  0.407
    lr5e-5       0.643       0.336    0.503  0.407
    lr1e-5       0.861       0.702    0.634  0.407
    hybrid       0.967       0.953    0.861  0.472

- Fusion is perfect here by construction: a generated value set's true codes
  come only from same-title sets, whose retrieval score is exactly 1.
- Token budget changes nothing because the generated pairs are under 16
  tokens.
- Hybrid retrieval changes the pools (240 test pairs instead of 296), so its
  row is a different candidate set, not only a different model.
"""

import argparse
import json
import random
import shutil
import sys
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
    precision_recall_curve,
    roc_auc_score,
)

from pyhealth.datasets import VSACDataset, get_dataloader
from pyhealth.models import RASCCrossEncoder
from pyhealth.models.rasc import RELEASED_REVISION, RELEASED_WEIGHTS, SAPBERT
from pyhealth.tasks import (
    ValueSetPairClassification,
    build_hybrid_retrieval_index,
    build_retrieval_index,
)
from pyhealth.tasks.value_set_completion import sapbert_encoder
from pyhealth.trainer import Trainer

# Per configuration: ``features`` (task inputs fused beyond the two texts),
# ``model`` (RASCCrossEncoder arguments), ``lr``, ``retriever``, and
# ``train`` (False evaluates the loaded weights as they are).
CONFIGS: dict[str, dict] = {
    "main": {},
    "released": {
        "model": {"model_name": RELEASED_WEIGHTS, "revision": RELEASED_REVISION},
        "train": False,
    },
    "fused": {"features": ["score", "code_system"]},
    "fused_score": {"features": ["score"]},
    "len32": {"model": {"max_length": 32}},
    "len64": {"model": {"max_length": 64}},
    "lr5e-5": {"lr": 5e-5},
    "lr1e-5": {"lr": 1e-5},
    "hybrid": {"retriever": "hybrid"},
}
SOURCE_SPLITS = ("train", "val", "test", "corpus")
TOPICS = ["diabetes", "hypertension", "asthma", "anemia", "migraine", "nephritis"]
KINDS = ["acute", "chronic", "severe", "mild", "juvenile", "gestational"]


class WarmupAdamW(torch.optim.AdamW):
    """AdamW with the authors' schedule: linear warmup, then linear decay to 0.

    PyHealth's Trainer takes an optimizer class but no scheduler, so the
    schedule lives in the optimizer and advances once per ``step``.

    Args:
        params: Parameters or parameter groups.
        lr: Peak learning rate.
        total_steps: Optimizer steps in the whole run.
        warmup_frac: Fraction of ``total_steps`` spent warming up from 0.
        **kwargs: Passed to ``torch.optim.AdamW``.
    """

    def __init__(self, params, lr, total_steps, warmup_frac=0.06, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.peak_lr = lr
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = int(self.total_steps * warmup_frac)
        self.steps_done = 0

    def step(self, closure=None):
        done = self.steps_done
        if done < self.warmup_steps:
            scale = (done + 1) / self.warmup_steps
        else:
            left = self.total_steps - done
            scale = max(left / max(self.total_steps - self.warmup_steps, 1), 0.0)
        for group in self.param_groups:
            group["lr"] = self.peak_lr * scale
        self.steps_done += 1
        return super().step(closure)


def shuffled_dataloader(dataset, batch_size: int, seed: int):
    """A training loader that permutes all samples of the split every epoch.

    ``get_dataloader(shuffle=True)`` shuffles a patient-level split only
    within each value set's run of samples: on a RASC subset a batch of 64
    held 1.3 value sets on average. A value set has one title and often no
    positive candidate at all, so the cross-encoder saw long runs of
    one-title, all-negative batches and collapsed to a constant score (pair
    AUROC 0.50). Indexing the disk-backed split at random is slow (2.8 s per
    batch on the full cache), so the split is read once, in order, and
    permuted in memory: about 2 kB per pair, 8 GB for the full training split.

    Args:
        dataset: The training split.
        batch_size: Pairs per batch.
        seed: Seed of the permutations.

    Returns:
        A DataLoader over the split's samples with PyHealth's collation.
    """
    rows = [
        {k: sys.intern(v) if isinstance(v, str) else v for k, v in sample.items()}
        for sample in dataset
    ]
    return torch.utils.data.DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=get_dataloader(dataset, batch_size).collate_fn,
    )


class SampledPairClassification(ValueSetPairClassification):
    """Keeps a fixed, OID-hashed fraction of target value sets.

    Same sampling as in examples/vsac_value_set_completion_mlp.py, so the two
    examples evaluate the same value sets at a given fraction.

    Args:
        index_dir: A retrieval index directory.
        sample_frac: Fraction of value sets kept, by CRC32 of the OID, so every
            configuration sees the same sets.
        **kwargs: Passed to ``ValueSetPairClassification``.
    """

    def __init__(self, index_dir: str, sample_frac: float = 1.0, **kwargs: Any) -> None:
        super().__init__(index_dir, **kwargs)
        self.sample_frac = sample_frac

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps the events of the sampled value sets.

        Args:
            df: The dataset's global event frame.

        Returns:
            The frame restricted to sampled OIDs.
        """
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


def write_synthetic(root: Path) -> None:
    """Writes value sets whose membership can be read off the two texts.

    Each topic has four codes per kind, displayed as ``"<kind> <topic>
    finding <n>"``. A value set is titled ``"<topic> <kind>"`` and holds the
    topic's codes of that kind, so a code belongs when its kind is the
    title's. Retrieval brings in the same topic's other kinds as negatives.

    Args:
        root: Output directory.
    """
    rng = random.Random(0)
    splits = ["train"] * 12 + ["val"] * 3 + ["test"] * 3 + ["corpus"] * 6
    rng.shuffle(splits)
    exp_dir = root / "expansions"
    exp_dir.mkdir(parents=True)
    manifest = []
    for t, topic in enumerate(TOPICS):
        for v, split in enumerate(splits):
            oid = f"9.{t}.{v}-20240101"
            kind = KINDS[v % len(KINDS)]
            members = [
                {
                    "system": "http://snomed.info/sct",
                    "code": f"T{t}K{KINDS.index(kind)}N{n}",
                    "display": f"{kind} {topic} finding {n}",
                }
                for n in range(4)
            ]
            resource = {
                "resourceType": "ValueSet",
                "id": oid,
                "title": f"{topic} {kind}",
                "version": "20240101",
                "expansion": {"total": len(members), "contains": members},
            }
            (exp_dir / f"{oid}.json").write_text(json.dumps(resource))
            manifest.append({"oid": oid, "split": split})
    (root / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )


def write_tiny_bert(path: Path) -> str:
    """Saves a small random BERT pair classifier for the synthetic demo.

    Args:
        path: Output directory.

    Returns:
        ``path`` as a string, to pass as ``model_name``.
    """
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    path.mkdir(parents=True)
    words = [*TOPICS, *KINDS, "finding", *map(str, range(10))]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(str(path / "vocab.txt")).save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        num_labels=1,
    )
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(path)
    return str(path)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximizing F1 on validation, as in the authors' scripts.

    Args:
        y_true: Validation labels.
        y_prob: Validation probabilities.

    Returns:
        The probability cutoff with the highest F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[min(int(np.argmax(f1)), len(thresholds) - 1)])


def value_set_metrics(
    samples: Sequence[dict], y_prob: np.ndarray, threshold: float
) -> tuple[dict, pd.DataFrame]:
    """Macro precision/recall/F1 over value sets, recall over all true codes.

    Args:
        samples: Test samples, each with ``patient_id``, ``code``, ``label``,
            and ``n_true_codes``.
        y_prob: Predicted probability per sample.
        threshold: Probability cutoff for predicting a code in the set.

    Returns:
        The macro averages, and one row per value set (``oid``,
        ``n_true_codes``, ``precision``, ``recall``, ``f1``,
        ``retrieval_recall``).
    """
    predicted, hits, retrieved = defaultdict(set), defaultdict(set), defaultdict(set)
    n_true = {}
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


def get_index(dataset: VSACDataset, retriever: str, args: argparse.Namespace) -> str:
    """Returns the retrieval index for a retriever, building it if missing.

    Args:
        dataset: The VSAC dataset.
        retriever: ``"dense"`` or ``"hybrid"``.
        args: Parsed command-line arguments.

    Returns:
        The index directory.
    """
    index_dir = Path(args.index_dir) / retriever
    if not (index_dir / "meta.json").exists():
        encoder = (
            stub_encoder if args.synthetic else sapbert_encoder(device=args.device)
        )
        build = (
            build_hybrid_retrieval_index
            if retriever == "hybrid"
            else build_retrieval_index
        )
        build(
            dataset, encoder, str(index_dir), max_k=args.k, source_splits=SOURCE_SPLITS
        )
    return str(index_dir)


def run_config(
    dataset: VSACDataset, name: str, args: argparse.Namespace
) -> dict[str, Any]:
    """Trains and evaluates one configuration.

    Also writes ``per_set_<name>.csv`` with each test value set's metrics.

    Args:
        dataset: The VSAC dataset.
        name: Key of ``CONFIGS``.
        args: Parsed command-line arguments.

    Returns:
        The configuration, pair counts, the tuned threshold, test metrics,
        and the retrieval-only baseline on the same candidates.
    """
    config = CONFIGS[name]
    torch.manual_seed(42)
    task = SampledPairClassification(
        get_index(dataset, config.get("retriever", "dense"), args),
        sample_frac=args.sample_frac,
        k=args.k,
        fusion_features=config.get("features", []),
    )
    train, val, test = dataset.set_task(
        task, num_workers=args.num_workers, split=dataset.manifest_split()
    )
    model = RASCCrossEncoder(
        train,
        **{
            "model_name": args.model_name,
            "gradient_checkpointing": args.gradient_checkpointing,
            "mixed_precision": args.mixed_precision,
            **config.get("model", {}),
        },
    )
    trainer = Trainer(
        model,
        metrics=["roc_auc", "pr_auc"],
        device=args.device,
        output_path=str(Path(args.output) / "runs"),
        exp_name=name,
    )
    eval_batch = args.batch_size * 4
    if config.get("train", True) and args.epochs > 0:
        train_loader = shuffled_dataloader(train, args.batch_size, seed=42)
        optimizer_params = {"lr": config.get("lr", args.lr) * args.lr_scale}
        if not args.synthetic:
            # The small random encoder of the demo needs its full step size
            # throughout; the schedule is for fine-tuning pretrained SapBERT.
            optimizer_params["total_steps"] = len(train_loader) * args.epochs
            optimizer_params["warmup_frac"] = args.warmup_frac
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=get_dataloader(val, eval_batch),
            epochs=args.epochs,
            optimizer_class=torch.optim.AdamW if args.synthetic else WarmupAdamW,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=1.0,
            monitor="pr_auc",
            patience=args.patience,
        )
    val_true, val_prob, _ = trainer.inference(get_dataloader(val, eval_batch))[:3]
    threshold = best_threshold(np.ravel(val_true), np.ravel(val_prob))
    test_true, test_prob, _ = trainer.inference(get_dataloader(test, eval_batch))[:3]
    test_true, test_prob = np.ravel(test_true), np.ravel(test_prob)
    summary, per_set = value_set_metrics(test, test_prob, threshold)
    baseline, _ = value_set_metrics(test, np.ones_like(test_prob), 0.5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    per_set.to_csv(output / f"per_set_{name}.csv", index=False)
    result = {
        "config": name,
        "settings": {
            "lr": args.lr,
            "lr_scale": args.lr_scale,
            "max_length": model.max_length,
            "warmup_frac": args.warmup_frac,
            "mixed_precision": args.mixed_precision,
            **config,
        },
        "n_pairs": {"train": len(train), "val": len(val), "test": len(test)},
        "threshold": threshold,
        "pair_auroc": float(roc_auc_score(test_true, test_prob)),
        "pair_ap": float(average_precision_score(test_true, test_prob)),
        **summary,
        "retrieval_only_f1": baseline["macro_f1"],
    }
    if args.cleanup:
        shutil.rmtree(Path(dataset.cache_dir) / "tasks", ignore_errors=True)
    return result


def write_report(results: list[dict[str, Any]], output: Path) -> None:
    """Writes ``results.json`` and a markdown table, and prints the table.

    Args:
        results: One ``run_config`` result per configuration.
        output: Output directory.
    """
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str))
    rows = [
        (
            "| Config | Test pairs | Pair AUROC | Pair AP | VS precision | VS recall "
            "| VS F1 | Retrieval recall | Retrieval-only F1 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['config']} | {r['n_pairs']['test']:,} | {r['pair_auroc']:.4f} "
        f"| {r['pair_ap']:.4f} | {r['macro_precision']:.4f} "
        f"| {r['macro_recall']:.4f} | {r['macro_f1']:.4f} "
        f"| {r['retrieval_recall']:.4f} | {r['retrieval_only_f1']:.4f} |"
        for r in results
    ]
    (output / "results.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main() -> None:
    """Parses arguments and runs each configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="VSAC download (expansions/ + manifest)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--index-dir", help="Reuse or write the retrieval indexes")
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument("--output", default="vsac_rasc_results")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--model-name", default=SAPBERT)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--warmup-frac",
        type=float,
        default=0.06,
        help="Share of steps with linear warmup; decay to 0 follows (0: decay only)",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the dataset's whole task cache after each configuration, "
        "including samples cached by other scripts",
    )
    args = parser.parse_args()
    if not (args.root or args.synthetic):
        parser.error("need --root or --synthetic")
    # The released weights need a download, so the offline demo skips them.
    default = [c for c in CONFIGS if not (args.synthetic and c == "released")]
    args.configs = args.configs or default
    unknown = set(args.configs) - set(CONFIGS)
    if unknown:
        parser.error(f"unknown configs {sorted(unknown)}; known: {list(CONFIGS)}")

    work = Path(tempfile.mkdtemp()) if args.synthetic else None
    if args.synthetic:
        write_synthetic(work / "vsac")
        args.root = str(work / "vsac")
        args.cache_dir = str(work / "cache")
        args.index_dir = args.index_dir or str(work / "index")
        args.model_name = write_tiny_bert(work / "bert")
        # Past a title's 3 same-kind twins, so pools hold other kinds too.
        args.k = 8
        args.epochs, args.patience = 12, 12
        args.batch_size = 32
    # A small random encoder needs a far larger step than pretrained SapBERT.
    args.lr_scale = 50.0 if args.synthetic else 1.0
    dataset = VSACDataset(root=args.root, cache_dir=args.cache_dir)
    args.index_dir = args.index_dir or str(Path(args.output) / "index")
    results = []
    for name in args.configs:
        results.append(run_config(dataset, name, args))
        write_report(results, Path(args.output))
    if work is not None:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
