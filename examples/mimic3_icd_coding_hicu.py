"""HiCu on ICD-9 coding: baseline vs HiCu vs a shuffled hierarchy (our ablation).

Authors: Huyen Nguyen (hnguyen654), Aaron Tran (atran403)
Course: CSE 6250 Big Data for Health, Georgia Tech, Fall 2026

Paper: Ren et al., "HiCu: Leveraging Hierarchy for Curriculum Learning in
Automated ICD Coding", MLHC 2022. https://proceedings.mlr.press/v182/ren22a.html

It trains three configurations on the same split and prints a results table:

* ``baseline``: MultiResCNN alone (no curriculum, BCE loss).
* ``hicu``: HiCu-A + asymmetric loss, the paper's best MultiResCNN setup.
* ``shuffled_hierarchy``: **our ablation**: ``hicu`` on an ICD tree with the
  same shape whose ancestor paths are shuffled between codes. If HiCu's gain comes from
  the clinical grouping of codes, the shuffled hierarchy should lose it.

Usage (works the same on a laptop or on Colab; a GPU is used if present)::

    # synthetic notes, no data needed, a few minutes
    python examples/mimic3_icd_coding_hicu.py

    # MIMIC-III, top-50 codes (quick check on a few patients, then full run)
    python examples/mimic3_icd_coding_hicu.py --root /path/to/mimic-iii --dev
    python examples/mimic3_icd_coding_hicu.py --root /path/to/mimic-iii

Results on synthetic data (seed 0, in %):

==================  =======  =======  =======  =======  =====
config              AUC-mac  AUC-mic  F1-mac   F1-mic   P@5
==================  =======  =======  =======  =======  =====
baseline            65.0     86.9      6.7     58.9     25.4
hicu                80.5     92.9      8.6     58.7     27.2
shuffled_hierarchy  75.0     89.5      8.1     60.8     26.0
==================  =======  =======  =======  =======  =====

HiCu lifts macro AUC from 65.0 to 80.5 and is best on AUC and P@5. The
shuffled hierarchy (same tree shape, codes under the wrong groups) keeps
only part of that gain (75.0), so part of it comes from the clinical
grouping of the codes and part from training in stages. On MIMIC-III the
shuffled hierarchy loses most of the gain. The synthetic notes are built so
the ICD tree carries signal: this shows the method works, not the size of
the effect on MIMIC-III. The results repeat across runs (within about 0.2).
Differences from the paper: a patient split instead of the CAML split, and
random (not word2vec) word vectors.
"""

import argparse
import logging
import random
import string
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.models import HiCu
from pyhealth.models.hicu import icd9_code_path
from pyhealth.trainer import Trainer

CONFIGS = {
    "baseline": {"use_curriculum": False, "loss": "bce"},
    "hicu": {"hyperbolic": "add", "loss": "asl"},
    "shuffled_hierarchy": {"hyperbolic": "add", "loss": "asl"},
}

# small model for synthetic notes; the paper's settings for MIMIC-III
SYNTHETIC = {
    "level_epochs": [2, 2, 2, 2, 25],
    "patience": 5,
    "lr": 1e-3,
    "batch_size": 16,
    "model": {"embedding_dim": 64, "filter_sizes": (3, 5, 9), "num_filter_maps": 32},
}
MIMIC = {
    "level_epochs": [2, 3, 5, 10, 100],
    "patience": 10,
    "lr": 5e-5,
    "batch_size": 8,
    "model": {"embedding_dim": 100, "filter_sizes": (3, 5, 9, 15, 19, 25)},
}


def synthetic_samples(n_notes: int = 2000, seed: int = 1234) -> list[dict]:
    """Fake notes whose words follow the ICD-9 tree of their codes.

    Each note mentions two words of each code's chapter, two of its 3-digit
    category and (80% of the time) one of the code itself, among 40 noise
    words. Code frequencies are long-tailed, so many codes are rare.

    Args:
        n_notes: Number of notes (one patient each).
        seed: Random seed.

    Returns:
        Samples with ``patient_id``, ``text`` and ``icd_codes``.
    """
    rng = random.Random(seed)
    categories = "401 250 414 428 427 584 518 038 276 285 599 496 486 272 244 530"
    categories += " 287 311 V45 V58 36 38 39 96"
    codes = []
    for cat in categories.split():
        for s in rng.sample(range(10), rng.randint(2, 7)):
            two = rng.random() < 0.3 and not cat.startswith("V")
            codes.append(f"{cat}.{s}{rng.randint(0, 9)}" if two else f"{cat}.{s}")
    rng.shuffle(codes)
    weights = [1 / (rank + 1) ** 1.1 for rank in range(len(codes))]

    def word() -> str:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(7))

    node_words: dict[str, list[str]] = {}

    def words(node: str, k: int) -> list[str]:
        node_words.setdefault(node, [word() for _ in range(3)])
        return rng.sample(node_words[node], k)

    noise = [word() for _ in range(500)]
    samples = []
    for i in range(n_notes):
        # sorted: a set's order changes from run to run, which would change the notes
        note_codes = sorted(
            set(rng.choices(codes, weights, k=rng.choice([1, 1, 2, 2, 3, 4])))
        )
        tokens = rng.choices(noise, k=40)
        for code in note_codes:
            path = icd9_code_path(code)
            tokens += words(path[0], 2) + words(path[2], 2)
            if rng.random() < 0.8:
                tokens += words(code, 1)
        rng.shuffle(tokens)
        samples.append(
            {
                "patient_id": f"p{i}",
                "text": " ".join(tokens),
                "icd_codes": note_codes,
            }
        )
    return samples


def dot_icd9(code: str, is_procedure: bool) -> str:
    """Adds the dot MIMIC-III leaves out: ``4019`` -> ``401.9``, ``3893`` -> ``38.93``.

    Args:
        code: The code as MIMIC-III stores it.
        is_procedure: Whether it comes from PROCEDURES_ICD.

    Returns:
        The dotted code.
    """
    code = str(code).strip()
    size = 2 if is_procedure else (4 if code.startswith("E") else 3)
    return code if len(code) <= size else f"{code[:size]}.{code[size:]}"


def mimic3_samples(root: str, dev: bool, top_k: int = 50) -> list[dict]:
    """One sample per admission: its discharge summary and top-``top_k`` codes.

    Args:
        root: Folder with the MIMIC-III ``.csv.gz`` files.
        dev: Load a few patients only (quick check).
        top_k: Keep the most frequent codes (the paper's top-50 setting).

    Returns:
        Samples with ``patient_id``, ``text`` and ``icd_codes``.
    """
    from pyhealth.datasets import MIMIC3Dataset

    dataset = MIMIC3Dataset(
        root=root, tables=["diagnoses_icd", "procedures_icd", "noteevents"], dev=dev
    )
    samples = []
    for patient in dataset.iter_patients():
        for adm in patient.get_events(event_type="admissions"):
            hadm = [("hadm_id", "==", adm.hadm_id)]
            notes = [
                n.text
                for n in patient.get_events("noteevents", filters=hadm)
                if str(n.category).strip().lower() == "discharge summary"
            ]
            codes = {
                dot_icd9(e.icd9_code, False)
                for e in patient.get_events("diagnoses_icd", filters=hadm)
                if e.icd9_code
            } | {
                dot_icd9(e.icd9_code, True)
                for e in patient.get_events("procedures_icd", filters=hadm)
                if e.icd9_code
            }
            if notes and codes:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "text": " ".join(notes),
                        "icd_codes": sorted(codes),
                    }
                )
    if not samples:
        raise ValueError("No discharge summaries found (is NOTEEVENTS empty?)")
    counts = Counter(c for s in samples for c in s["icd_codes"])
    keep = {c for c, _ in counts.most_common(top_k)}
    for s in samples:
        s["icd_codes"] = [c for c in s["icd_codes"] if c in keep]
    return [s for s in samples if s["icd_codes"]]


def shuffled_hierarchy(labels: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Our ablation: each code keeps itself as the leaf but gets another
    code's ancestors, so the tree has the same shape but no clinical meaning."""
    paths = [icd9_code_path(c) for c in labels]
    random.Random(seed).shuffle(paths)
    return {label: path[:-1] + [label] for label, path in zip(labels, paths)}


def scores(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """The paper's metrics. Macro F1 is the F1 of macro precision and macro
    recall, as in the authors' code."""
    cols = [j for j in range(y_true.shape[1]) if 0 < y_true[:, j].sum() < len(y_true)]
    y_pred = (y_prob >= 0.5).astype(int)
    hit = (y_pred * y_true).sum(axis=0)
    p_mac = np.mean(hit / (y_pred.sum(axis=0) + 1e-10))
    r_mac = np.mean(hit / (y_true.sum(axis=0) + 1e-10))
    p_mic = hit.sum() / (y_pred.sum() + 1e-10)
    r_mic = hit.sum() / (y_true.sum() + 1e-10)
    top5 = np.argsort(-y_prob, axis=1)[:, :5]
    return {
        "auc_macro": np.mean([roc_auc_score(y_true[:, j], y_prob[:, j]) for j in cols]),
        "auc_micro": roc_auc_score(y_true.ravel(), y_prob.ravel()),
        "f1_macro": 2 * p_mac * r_mac / (p_mac + r_mac + 1e-10),
        "f1_micro": 2 * p_mic * r_mic / (p_mic + r_mic + 1e-10),
        "p@5": np.take_along_axis(y_true, top5, axis=1).sum(axis=1).mean() / 5,
    }


def run(name: str, train_ds, val_ds, test_ds, settings: dict, device: str) -> dict:
    """Trains one configuration level by level (top of the tree -> codes),
    with early stopping at the codes level, and tests it."""
    torch.manual_seed(0)
    options = dict(CONFIGS[name])
    if name == "shuffled_hierarchy":
        options["code_paths"] = shuffled_hierarchy(
            list(train_ds.output_processors["icd_codes"].label_vocab)
        )
    model = HiCu(train_ds, **options, **settings["model"])
    trainer = Trainer(
        model=model, metrics=["roc_auc_micro"], device=device, enable_logging=False
    )
    bs = settings["batch_size"]
    train_loader = get_dataloader(train_ds, batch_size=bs, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=bs)
    epochs = settings["level_epochs"][-model.num_levels :]  # baseline: last only
    for level, n_epochs in enumerate(epochs):
        model.set_level(level)
        last = level == model.num_levels - 1
        trainer.train(
            train_loader,
            val_loader if last else None,
            epochs=n_epochs,
            optimizer_params={"lr": settings["lr"]},
            monitor="roc_auc_micro" if last else None,
            patience=settings["patience"] if last else None,
        )
    y_true, y_prob, _ = trainer.inference(get_dataloader(test_ds, batch_size=bs))
    return scores(y_true, y_prob)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HiCu: baseline vs HiCu vs shuffled hierarchy"
    )
    parser.add_argument("--root", help="MIMIC-III folder; leave out for synthetic data")
    parser.add_argument("--dev", action="store_true", help="MIMIC-III: a few patients")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    args = parser.parse_args()

    logging.getLogger("pyhealth").setLevel(logging.WARNING)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = mimic3_samples(args.root, args.dev) if args.root else synthetic_samples()
    settings = MIMIC if args.root else SYNTHETIC
    dataset = create_sample_dataset(
        samples,
        input_schema={"text": "text"},
        output_schema={"icd_codes": "multilabel"},
        dataset_name="hicu_icd9",
    )
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.7, 0.1, 0.2], seed=0)
    print(
        f"{len(samples)} notes, split {len(train_ds)}/{len(val_ds)}/{len(test_ds)}, "
        f"device {device}"
    )

    rows = []
    for name in args.configs:
        result = run(name, train_ds, val_ds, test_ds, settings, device)
        rows.append(
            f"| {name} | "
            + " | ".join(f"{100 * v:.1f}" for v in result.values())
            + " |"
        )
        print(rows[-1], flush=True)
    print("\n| config | AUC macro | AUC micro | F1 macro | F1 micro | P@5 |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
