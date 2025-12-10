"""
Example: Radiology sentence-level anomaly detection with PyHealth

This example shows how to use PyHealth with a custom text dataset
to reproduce the *sentence-level* part of a radiology anomaly
detection pipeline inspired by:

    Kim et al., "Integrating ChatGPT into Secure Hospital Networks:
    A Case Study on Improving Radiology Report Analysis" (MLHC 2024).

We assume you have already run a cloud LLM (e.g., GPT-5.1, Claude
Sonnet) on chest X-ray reports (e.g., from MIMIC-CXR) and produced
sentence-level labels:

    - normal
    - abnormal
    - uncertain

Each CSV row is treated as one *sentence sample*. We then:

  1. Wrap these samples into a `SampleDataset`.
  2. Use PyHealth's `TextProcessor` (via the "text" schema) and
     `MultiClassLabelProcessor` (via the "multiclass" schema).
  3. Train a `pyhealth.models.Transformer` classifier for 3-way
     normal / abnormal / uncertain prediction.
  4. Evaluate sentence-level multiclass metrics.
  5. Aggregate sentence predictions to *document-level* anomaly
     labels (abnormal vs normal) using max over sentence abnormal
     probabilities and compute binary metrics.

This example is intentionally minimal and self-contained so that
users can easily adapt it to their own radiology or clinical text
datasets.

Run from the PyHealth repo root (after `pip install -e .`):

    python examples/radiology_sentence_kd_example.py \\
        --csv_path /path/to/sentence_labels.csv \\
        --output_dir ./radiology_kd_runs \\
        --epochs 3 \\
        --batch_size 16

Expected CSV columns:

    - doc_id         : report identifier (string or int)
    - sent_id        : sentence index within the report
    - sentence_text  : free-text radiology sentence
    - label          : either {0,1,2} or {"normal","abnormal","uncertain"}

The CSV itself should NOT contain PHI. For MIMIC-CXR, please follow
the PhysioNet DUA and any local IRB/approval rules.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import SampleDataset
from pyhealth.datasets.splitter import split_by_sample
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer
from pyhealth.metrics.multiclass import multiclass_metrics_fn
from pyhealth.metrics.binary import binary_metrics_fn


# ---------------------------------------------------------------------
# Data loading and SampleDataset construction
# ---------------------------------------------------------------------


def load_sentence_dataset(csv_path: str) -> SampleDataset:
    """
    Load a sentence-level radiology KD dataset from a CSV file and wrap
    it into a PyHealth SampleDataset.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file with at least the following columns:
            - doc_id
            - sent_id
            - sentence_text
            - label   (0/1/2 or "normal"/"abnormal"/"uncertain")

    Returns
    -------
    dataset : SampleDataset
        A SampleDataset with:
            - input_schema  = {"sentence_text": "text"}
            - output_schema = {"label": "multiclass"}

        Each sample dict has keys:
            - patient_id  (here: doc_id)
            - visit_id    (doc_id + "_" + sent_id)
            - sentence_text
            - label       (integer in {0,1,2})
    """
    df = pd.read_csv(csv_path)

    required_cols = {"doc_id", "sent_id", "sentence_text", "label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Normalize labels to integers {0, 1, 2}
    if df["label"].dtype == "object":
        label_map = {"normal": 0, "abnormal": 1, "uncertain": 2}
        df["label"] = df["label"].map(label_map)
        if df["label"].isna().any():
            bad = df[df["label"].isna()]
            raise ValueError(
                "Found unknown label values in 'label' column. "
                "Expected one of: normal / abnormal / uncertain. "
                f"Offending rows:\n{bad.head()}"
            )

    # Build list[dict] of samples compatible with SampleDataset
    samples = []
    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        sent_id = str(row["sent_id"])

        samples.append(
            {
                # PyHealth expects patient_id / visit_id
                "patient_id": doc_id,
                "visit_id": f"{doc_id}_{sent_id}",
                "sentence_text": str(row["sentence_text"]),
                # Ensure Python int
                "label": int(row["label"]),
            }
        )

    # Define schemas using string keys so PyHealth picks TextProcessor
    # and MultiClassLabelProcessor automatically.
    input_schema = {"sentence_text": "text"}
    output_schema = {"label": "multiclass"}

    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="RadiologySentenceKD",
        task_name="SentenceAnomalyClassification",
    )

    # Fit processors (tokenizer + label encoder) on the samples
    dataset.build()
    return dataset


# ---------------------------------------------------------------------
# Document-level aggregation (sentence -> report)
# ---------------------------------------------------------------------


def aggregate_to_document(
    y_true_all,
    y_prob_all,
    patient_ids,
    abnormal_class_index: int = 1,
):
    """
    Aggregate sentence-level multiclass predictions to document-level
    binary anomaly labels using max over sentence abnormal probability.

    Parameters
    ----------
    y_true_all : array-like, shape (N_sentences,)
        Sentence-level ground-truth labels (0=normal, 1=abnormal, 2=uncertain).

    y_prob_all : array-like, shape (N_sentences, num_classes)
        Sentence-level predicted class probabilities.

    patient_ids : list[str]
        Patient IDs from PyHealth's Trainer.inference(..., return_patient_ids=True).
        In this example, each patient_id corresponds to one doc_id.

    abnormal_class_index : int, default=1
        Index of the "abnormal" class in y_true / y_prob vectors.

    Returns
    -------
    doc_ids : list[str]
        Unique document IDs (patient_ids) in a stable order.

    y_true_doc : np.ndarray, shape (N_docs,)
        Binary ground-truth anomaly labels per document:
            1 if any sentence is labeled abnormal, else 0.

    y_prob_doc : np.ndarray, shape (N_docs,)
        Document-level abnormal probabilities, computed as:
            max_j p_abnormal(sentence_j in doc).
    """
    y_true_all = np.asarray(y_true_all)
    y_prob_all = np.asarray(y_prob_all)

    doc_label = {}
    doc_probs = defaultdict(list)

    for label, probs, pid in zip(y_true_all, y_prob_all, patient_ids):
        # Sentence-level abnormal -> document considered abnormal
        is_abnormal = int(label == abnormal_class_index)

        if pid not in doc_label:
            doc_label[pid] = is_abnormal
        else:
            # If any sentence is abnormal, doc becomes abnormal
            doc_label[pid] = max(doc_label[pid], is_abnormal)

        doc_probs[pid].append(float(probs[abnormal_class_index]))

    doc_ids = sorted(doc_label.keys())
    y_true_doc = np.array([doc_label[pid] for pid in doc_ids], dtype=int)
    y_prob_doc = np.array([max(doc_probs[pid]) for pid in doc_ids], dtype=float)
    return doc_ids, y_true_doc, y_prob_doc


# ---------------------------------------------------------------------
# Main training / evaluation script
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sentence-level radiology anomaly detection example using PyHealth. "
            "Assumes a CSV with LLM-distilled sentence labels."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV with columns: doc_id, sent_id, sentence_text, label.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./radiology_kd_runs",
        help="Directory to store trainer logs / checkpoints.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="radiology_sentence_kd",
        help="Experiment name used by Trainer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training / evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Basic reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load user-provided sentence-level dataset
    print(f"[INFO] Loading sentence dataset from {args.csv_path}")
    dataset = load_sentence_dataset(args.csv_path)
    print(f"[INFO] Loaded {len(dataset)} sentence samples.")

    # 2. Split by sample into train / val / test
    print("[INFO] Splitting dataset by sample with ratios 0.6 / 0.2 / 0.2")
    train_ds, val_ds, test_ds = split_by_sample(
        dataset, ratios=[0.6, 0.2, 0.2], seed=args.seed
    )

    from pyhealth.datasets.utils import get_dataloader

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Build a text Transformer model over sentences
    print("[INFO] Building Transformer model for multiclass sentence classification")
    model = Transformer(
        dataset=dataset,
        feature_keys=["sentence_text"],
        label_key="label",
        mode="multiclass",  # 3-way: normal / abnormal / uncertain
    )

    # 4. Train with PyHealth Trainer
    print("[INFO] Starting training")
    trainer = Trainer(
        model=model,
        output_path=args.output_dir,
        exp_name=args.exp_name,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        # Optionally set monitor to a specific metric name once metrics are configured.
        # monitor="f1_macro",
    )

    # 5. Sentence-level evaluation (multiclass metrics)
    print("[INFO] Running sentence-level evaluation on test set")
    # Returns: y_true_all, y_prob_all, loss_mean, patient_ids
    y_true_all, y_prob_all, test_loss, patient_ids = trainer.inference(
        test_loader, return_patient_ids=True
    )

    sent_metrics = multiclass_metrics_fn(
        y_true_all,
        y_prob_all,
        metrics=[
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "roc_auc_macro_ovr",
        ],
    )
    print("\n=== Sentence-level multiclass metrics ===")
    for k, v in sent_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"test_loss: {test_loss:.4f}")

    # 6. Document-level aggregation: sentence -> report
    print("\n[INFO] Aggregating to document-level anomaly labels")
    doc_ids, y_true_doc, y_prob_doc = aggregate_to_document(
        y_true_all=y_true_all,
        y_prob_all=y_prob_all,
        patient_ids=patient_ids,
        abnormal_class_index=1,  # label==1 is "abnormal"
    )

    doc_metrics = binary_metrics_fn(
        y_true_doc,
        y_prob_doc,
        metrics=["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "f1"],
    )

    print("\n=== Document-level binary anomaly metrics (max over sentences) ===")
    for k, v in doc_metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\n[INFO] Evaluated {len(doc_ids)} unique documents.")


if __name__ == "__main__":
    main()
