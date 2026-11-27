"""Query-focused EHR summarization reproduction example.

Based on McInerney et al. (2020), MLHC.

Demonstrates:
- PyHealth MIMIC3Dataset and QueryFocusedEHRTask
- Patient-disjoint train/validation/test splits
- Query-conditioned neural diagnosis prediction
- TF-IDF cosine-similarity baseline
- AUROC and average precision
- History-length ablations with matched patient/visit/query cohorts
- Optional in-memory frozen ClinicalBERT embedding caching
- Full Dataset -> Task -> Model test using generated synthetic MIMIC-III tables
- Configurable ICD-9 diagnosis groups
- Top-k extractive sentence ranking

Synthetic mode verifies execution only; it is not a
clinical reproduction or a performance benchmark.

Examples:
    # Fast smoke test: sample dictionaries -> TinyEncoder model
    python examples/mimic3_query_focused_ehr_attention.py --synthetic

    # Full pipeline smoke test: generated MIMIC-III CSV -> Dataset -> Task -> TinyEncoder
    python examples/mimic3_query_focused_ehr_attention.py --synthetic-mimic

    # Full pipeline + actual ClinicalBERT (requires model download, GPU recommended)
    python examples/mimic3_query_focused_ehr_attention.py --synthetic-mimic --use-clinicalbert --history-windows 30 --epochs 1

    # Real MIMIC-III (requires credentialed local data)
    python examples/mimic3_query_focused_ehr_attention.py --root /path/to/mimic-iii

    python examples/mimic3_query_focused_ehr_attention.py --synthetic --fine-tune --learning-rate 0.001 --epochs 20 --history-windows 30 90 365

Requires scikit-learn, PyTorch, transformers and PyHealth.
"""

import argparse
import csv
import gzip
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models.query_focused_ehr import QueryFocusedEHRModel
from pyhealth.tasks.query_focused_ehr import QueryFocusedEHRTask

DEFAULT_DIAGNOSIS_GROUPS = {
    "pulmonary_embolism": {
        "description": "Pulmonary embolism",
        "codes": ["41511", "41512", "41519"],
    },
    "pneumonia": {
        "description": "Pneumonia",
        "codes": ["486"],
    },
}


def load_diagnosis_groups(path):
    """Read and validate a diagnosis-group JSON configuration."""
    groups = (
        DEFAULT_DIAGNOSIS_GROUPS
        if path is None
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Diagnosis groups must be a nonempty JSON object.")
    for group_id, config in groups.items():
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("Diagnosis group identifiers must be nonempty strings.")
        if not isinstance(config, dict):
            raise TypeError(f"Diagnosis group {group_id!r} must be an object.")
        if not isinstance(config.get("description"), str) or not config["description"].strip():
            raise ValueError(f"Diagnosis group {group_id!r} requires a description.")
        codes = config.get("codes")
        if not isinstance(codes, list) or not codes or any(
            not isinstance(code, str) or not code.strip() for code in codes
        ):
            raise ValueError(f"Diagnosis group {group_id!r} requires a nonempty list of codes.")
    return groups


def normalize_icd(code):
    return str(code).strip().upper().replace(".", "")


def sample_key(sample):
    """Stable longitudinal example identity across history windows."""
    return (
        str(sample["patient_id"]),
        str(sample["visit_id"]),
        str(sample["diagnosis_code"]),
    )


def class_counts(samples):
    return {
        "positive": sum(label_value(item) == 1 for item in samples),
        "negative": sum(label_value(item) == 0 for item in samples),
    }

# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def label_value(sample):
    """Convert PyHealth binary labels to Python integers."""
    label = sample["label"]
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)

# ---------------------------------------------------------
# 2. Synthetic smoke-test encoder
# ---------------------------------------------------------

class HashTokenizer:
    """Small tokenizer used only for synthetic smoke tests."""

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ):
        sequences = []
        for text in texts:
            words = str(text).lower().split()
            ids = [
                1 + sum(word.encode("utf-8")) % 255
                for word in words[:max_length]
            ]
            sequences.append(ids or [1])

        width = max(map(len, sequences))

        return {
            "input_ids": torch.tensor([
                seq + [0] * (width - len(seq))
                for seq in sequences
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                [1] * len(seq) + [0] * (width - len(seq))
                for seq in sequences
            ], dtype=torch.long),
        }


class TinyEncoder(nn.Module):
    """Small trainable encoder; NOT pretrained ClinicalBERT."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(256, hidden_size, padding_idx=0)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        vectors = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (vectors * mask).sum(1) / mask.sum(1).clamp(min=1)
        contextual = self.projection(pooled)

        output = vectors.clone()
        output[:, 0, :] = contextual
        return SimpleNamespace(last_hidden_state=output)


def synthetic_samples(history_days, diagnosis_groups=None):
    """Generate deterministic synthetic examples for smoke testing.

    Default query vocabulary gives each patient one positive and
    one negative; custom groups may yield other label balances.
    Sentences are intentionally artificial.
    """
    samples = []

    diagnosis_groups = diagnosis_groups or DEFAULT_DIAGNOSIS_GROUPS

    for i in range(40):
        is_embolism = i % 2 == 0

        recent = (
            "Patient reports chest pain and shortness of breath."
            if is_embolism
            else "Patient has cough and fever."
        )

        medium = (
            "History of deep vein thrombosis."
            if is_embolism
            else "History of respiratory infection."
        )

        older = "Previous hospitalization and medication review."

        sentences = [recent]

        if history_days >= 90:
            sentences.insert(0, medium)

        if history_days >= 365:
            sentences.insert(0, older)

        observed_codes = {"41519"} if is_embolism else {"486"}
        for group_id, group in diagnosis_groups.items():
            positive = bool(
                observed_codes.intersection(normalize_icd(c) for c in group["codes"])
            )
            samples.append({
                "patient_id": f"synthetic_{i:03d}",
                "visit_id": f"visit_{i:03d}",
                "sentences": list(sentences),
                "query": group["description"],
                "diagnosis_code": group_id,
                "label": int(positive),
            })

    return samples


# ---------------------------------------------------------
# 3. Synthetic MIMIC-III CSV fixtures (full pipeline smoke test)
# ---------------------------------------------------------

def write_synthetic_mimic_tables(root, num_patients=40):
    """Write a small, de-identified, *artificial* MIMIC-III-style dataset.

    The five CSV.gz tables match PyHealth's existing MIMIC3Dataset loader.
    Each patient has three earlier note-bearing admissions and one target
    admission with an ICD-9 diagnosis. Note timestamps are 18, 51 and
    ~219 days before the target, enabling 30/90/365-day ablations.

    This checks loader and task integration; it is not clinical evidence.
    All files are generated locally and need no PhysioNet credentials.
    """
    root.mkdir(parents=True, exist_ok=True)

    columns = {
        "PATIENTS": [
            "SUBJECT_ID", "GENDER", "DOB", "DOD", "DOD_HOSP",
            "DOD_SSN", "EXPIRE_FLAG",
        ],
        "ADMISSIONS": [
            "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME",
            "ADMISSION_TYPE", "ADMISSION_LOCATION", "INSURANCE",
            "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY",
            "EDREGTIME", "EDOUTTIME", "DEATHTIME", "DIAGNOSIS",
            "DISCHARGE_LOCATION", "HOSPITAL_EXPIRE_FLAG",
        ],
        "ICUSTAYS": [
            "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME",
            "OUTTIME", "FIRST_CAREUNIT", "LAST_CAREUNIT", "DBSOURCE",
        ],
        "NOTEEVENTS": [
            "ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE",
            "CHARTTIME", "STORETIME", "CATEGORY", "DESCRIPTION", "TEXT",
        ],
        "DIAGNOSES_ICD": [
            "ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE",
        ],
    }
    rows = {key: [] for key in columns}
    admission_dates = [
        ("2020-05-25 08:00:00", "2020-05-29 12:00:00",
         "2020-05-27 09:00:00", "2020-05-27 10:00:00"),
        ("2020-11-09 08:00:00", "2020-11-13 12:00:00",
         "2020-11-11 09:00:00", "2020-11-11 10:00:00"),
        ("2020-12-11 08:00:00", "2020-12-15 12:00:00",
         "2020-12-13 09:00:00", "2020-12-13 10:00:00"),
        ("2021-01-01 08:00:00", "2021-01-05 12:00:00", None, None),
    ]
    note_id = 1
    for idx in range(1, num_patients + 1):
        patient_id = str(idx)
        embolism = idx % 2 == 0
        code = "41519" if embolism else "486"
        rows["PATIENTS"].append({
            "SUBJECT_ID": patient_id,
            "GENDER": "M" if embolism else "F",
            "DOB": "1970-01-01 00:00:00",
            "EXPIRE_FLAG": "0",
        })
        history = (
            ["Past anticoagulation medication review.",
             "History of deep vein thrombosis.",
             "Patient reports chest pain and shortness of breath."]
            if embolism else
            ["Past respiratory medication review.",
             "History of respiratory infection.",
             "Patient has cough and fever."]
        )
        for visit_idx, (admit, discharge, chart, store) in enumerate(
            admission_dates
        ):
            visit_id = str(100000 + idx * 10 + visit_idx)
            rows["ADMISSIONS"].append({
                "SUBJECT_ID": patient_id,
                "HADM_ID": visit_id,
                "ADMITTIME": admit,
                "DISCHTIME": discharge,
                "ADMISSION_TYPE": "EMERGENCY",
                "DIAGNOSIS": "Chest pain" if embolism else "Cough",
                "HOSPITAL_EXPIRE_FLAG": "0",
            })
            if chart is not None:
                rows["NOTEEVENTS"].append({
                    "ROW_ID": str(note_id),
                    "SUBJECT_ID": patient_id,
                    "HADM_ID": visit_id,
                    "CHARTDATE": chart[:10],
                    "CHARTTIME": chart,
                    "STORETIME": store,
                    "CATEGORY": "Physician",
                    "DESCRIPTION": "Progress note",
                    "TEXT": history[visit_idx],
                })
                note_id += 1
            else:
                # Only target admissions receive the future diagnosis labels.
                rows["DIAGNOSES_ICD"].append({
                    "ROW_ID": str(idx),
                    "SUBJECT_ID": patient_id,
                    "HADM_ID": visit_id,
                    "SEQ_NUM": "1",
                    "ICD9_CODE": code,
                })
        # PyHealth expects this table even when it is not used by our task.
        rows["ICUSTAYS"].append({
            "SUBJECT_ID": patient_id,
            "HADM_ID": str(100000 + idx * 10 + 3),
            "ICUSTAY_ID": str(200000 + idx),
            "INTIME": "2021-01-01 10:00:00",
            "OUTTIME": "2021-01-03 10:00:00",
            "FIRST_CAREUNIT": "MICU",
            "LAST_CAREUNIT": "MICU",
            "DBSOURCE": "metavision",
        })

    for table, fieldnames in columns.items():
        with gzip.open(
            root / f"{table}.csv.gz", "wt", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[table])

    print(
        f"Created artificial MIMIC-III tables: "
        f"{num_patients} patients, {len(rows['ADMISSIONS'])} admissions, "
        f"{len(rows['NOTEEVENTS'])} notes, "
        f"{len(rows['DIAGNOSES_ICD'])} diagnosis rows."
    )
    return root


# ---------------------------------------------------------
# 4. Load clinical or synthetic MIMIC-III samples
# ---------------------------------------------------------

def load_mimic_samples(root, history_days, diagnosis_groups):
    """Build query-focused examples through PyHealth."""

    dataset = MIMIC3Dataset(
        root=str(root),
        tables=["noteevents", "diagnoses_icd"],
        num_workers=1,
    )

    task = QueryFocusedEHRTask(
        diagnosis_groups=diagnosis_groups,
        history_days=history_days,
        max_sentences=64,
        note_categories=[
            "Radiology",
            "Physician",
            "Nursing",
            "Discharge summary",
        ],
        negative_sampling="never_observed",
    )

    processed = dataset.set_task(task, num_workers=1)

    # Materialize PyHealth samples before closing the dataset.
    # Patient selection is performed across ALL windows in main(),
    # not independently here, to make the ablation comparable.
    try:
        samples = [processed[i] for i in range(len(processed))]
    finally:
        processed.close()
    return samples


# ---------------------------------------------------------
# 4. Patient-level splitting
# ---------------------------------------------------------

def patient_split_ids(patient_ids, seed):
    """Create one reproducible patient-disjoint partition."""
    patient_ids = sorted(set(patient_ids))
    if len(patient_ids) < 3:
        raise ValueError("At least three eligible patients are needed for splitting.")
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return (
        set(patient_ids[:n_train]),
        set(patient_ids[n_train:n_train + n_val]),
        set(patient_ids[n_train + n_val:]),
    )


def split_by_patient(samples, seed, split_ids=None):
    """Keep each patient's examples entirely within one split."""
    if split_ids is None:
        split_ids = patient_split_ids(
            (str(sample["patient_id"]) for sample in samples), seed
        )
    train_ids, val_ids, test_ids = split_ids
    if not (train_ids.isdisjoint(val_ids) and
            train_ids.isdisjoint(test_ids) and
            val_ids.isdisjoint(test_ids)):
        raise ValueError("Patient split identifiers overlap.")
    return tuple(
        [sample for sample in samples if str(sample["patient_id"]) in ids]
        for ids in split_ids
    )


def build_matched_windows(args, diagnosis_groups):
    """Select common patient/visit/query examples in EVERY history window.

    For a strict history-length ablation, only historical text may vary.
    Sample labels, patient IDs, admissions, and query IDs are identical.
    This can reduce the eligible cohort substantially for short windows.
    """
    windows = {}
    for days in args.history_windows:
        if args.synthetic:
            samples = synthetic_samples(days, diagnosis_groups)
        else:
            samples = load_mimic_samples(
                root=args.root, history_days=days,
                diagnosis_groups=diagnosis_groups,
            )
        keys = [sample_key(sample) for sample in samples]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate patient/visit/query key in {days}-day window.")
        windows[days] = {sample_key(s): s for s in samples}

    common_keys = set.intersection(*(set(w) for w in windows.values()))
    if not common_keys:
        raise ValueError(
            "No identical patient/visit/query examples exist across all "
            "history windows. Consider a larger cohort or longer minimum window."
        )

    # Select eligible patients ONCE, consistently for all windows.
    common_patient_ids = sorted({key[0] for key in common_keys})
    rng = random.Random(args.seed)
    rng.shuffle(common_patient_ids)
    selected_ids = set(common_patient_ids[:args.max_patients])
    common_keys = sorted(key for key in common_keys if key[0] in selected_ids)
    if len(selected_ids) < 3:
        raise ValueError("Fewer than three shared eligible patients across windows.")

    # Explicitly verify that labels do not depend on the window size.
    for key in common_keys:
        labels = {label_value(windows[days][key]) for days in windows}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent diagnosis labels across windows for {key}.")

    split_ids = patient_split_ids(selected_ids, args.seed)
    matched = {days: [windows[days][key] for key in common_keys] for days in windows}
    return matched, split_ids



# ---------------------------------------------------------
# 5. Evaluation metrics
# ---------------------------------------------------------

def evaluate_binary(y_true, y_score):
    """Return AUROC and average precision when defined."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) == 0:
        return {"auroc": None, "average_precision": None}

    # AUROC cannot be computed for a single-label-class set.
    auroc = (
        float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2
        else None
    )

    ap = (
        float(average_precision_score(y_true, y_score))
        if np.any(y_true == 1)
        else None
    )

    return {
        "auroc": auroc,
        "average_precision": ap,
    }


# ---------------------------------------------------------
# 6. TF-IDF baseline
# ---------------------------------------------------------

def evaluate_tfidf(train, test):
    """Train-only TF-IDF vocabulary; cosine retrieval baseline.

    For each diagnosis query, take the maximum cosine
    similarity with any historical sentence as its score.
    """

    training_texts = [
        text
        for sample in train
        for text in sample["sentences"]
    ]
    training_texts += [
        sample["query"] for sample in train
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
    )
    vectorizer.fit(training_texts)

    y_true, y_score = [], []

    for sample in test:
        sentence_vectors = vectorizer.transform(
            sample["sentences"]
        )
        query_vector = vectorizer.transform(
            [sample["query"]]
        )

        similarities = cosine_similarity(
            sentence_vectors, query_vector
        ).ravel()

        y_true.append(label_value(sample))
        y_score.append(float(similarities.max()))

    return evaluate_binary(y_true, y_score)


# ---------------------------------------------------------
# 7. Neural model construction and batching
# ---------------------------------------------------------

def make_model(args, device):
    # Synthetic data can also exercise the *real* pretrained text encoder.
    use_tiny_encoder = (
        (args.synthetic or args.synthetic_mimic) and not args.use_clinicalbert
    )
    if use_tiny_encoder:
        model = QueryFocusedEHRModel(
            dataset=None,
            hidden_dim=32,
            freeze_encoder=not args.fine_tune,
            encoder=TinyEncoder(hidden_size=32),
            tokenizer=HashTokenizer(),
            max_token_length=64,
            encoder_batch_size=args.encoder_batch_size,
            cache_embeddings=args.cache_embeddings and not args.fine_tune,
            cache_max_size=args.cache_max_size,
        )
    else:
        model = QueryFocusedEHRModel(
            dataset=None,
            pretrained_model_name=args.encoder,
            hidden_dim=64,
            freeze_encoder=not args.fine_tune,
            max_token_length=128,
            encoder_batch_size=args.encoder_batch_size,
            cache_embeddings=args.cache_embeddings and not args.fine_tune,
            cache_max_size=args.cache_max_size,
        )

    return model.to(device)


def batches(samples, batch_size, shuffle=False, seed=42):
    indices = list(range(len(samples)))

    if shuffle:
        random.Random(seed).shuffle(indices)

    for start in range(0, len(indices), batch_size):
        subset = [
            samples[i]
            for i in indices[start:start + batch_size]
        ]

        yield {
            "sentences": [s["sentences"] for s in subset],
            "query": [s["query"] for s in subset],
            "label": torch.tensor(
                [label_value(s) for s in subset],
                dtype=torch.float32,
            ),
        }


@torch.no_grad()
def evaluate_model(model, samples, batch_size):
    """Evaluate neural diagnosis probabilities."""

    model.eval()
    labels, probabilities = [], []

    for batch in batches(samples, batch_size):
        outputs = model(**batch)

        labels.extend(
            batch["label"].cpu().tolist()
        )
        probabilities.extend(
            outputs["y_prob"].reshape(-1).cpu().tolist()
        )

    return evaluate_binary(labels, probabilities)


# ---------------------------------------------------------
# 8. Neural training
# ---------------------------------------------------------

def train_model(args, train, val, device):
    model = make_model(args, device)

    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.learning_rate,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in batches(
            train,
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        for batch in batches(val, args.batch_size):
            with torch.no_grad():
                output = model(**batch)
            val_losses.append(float(output["loss"].item()))

        mean_train = float(np.mean(losses))
        mean_val = float(np.mean(val_losses))

        print(
            f"  Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={mean_train:.4f} "
            f"val_loss={mean_val:.4f}"
        )

        if mean_val < best_val:
            best_val = mean_val
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------
# 9. Sentence ranking
# ---------------------------------------------------------

def show_sentence_ranking(model, test):
    """Inspect ranked sentences for one held-out example."""

    if not test:
        return

    example = test[0]

    ranking = model.extract_top_k(
        sentences=[example["sentences"]],
        query=[example["query"]],
        k=5,
    )[0]

    print("\nExample extractive summary")
    print("Query:", example["query"])
    print("True label:", label_value(example))

    for rank, result in enumerate(ranking, start=1):
        print(
            f"  {rank}. [{result['score']:.3f}] "
            f"{result['sentence']}"
        )


# ---------------------------------------------------------
# 10. Main experiment
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query-focused EHR reproduction example."
    )

    parser.add_argument(
        "--root", type=Path,
        help="Directory containing credentialed MIMIC-III CSV tables.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Fast artificial-sample smoke test (skips Dataset and Task).",
    )
    parser.add_argument(
        "--synthetic-mimic", action="store_true",
        help=(
            "Generate artificial MIMIC-III tables, then run "
            "MIMIC3Dataset -> QueryFocusedEHRTask -> Model -> evaluation."
        ),
    )
    parser.add_argument(
        "--synthetic-patients", type=int, default=40,
        help="Number of artificial patients for --synthetic-mimic (default: 40).",
    )
    parser.add_argument(
        "--use-clinicalbert", action="store_true",
        help="Use real pretrained ClinicalBERT with synthetic inputs instead of TinyEncoder.",
    )
    parser.add_argument(
        "--encoder",
        default="emilyalsentzer/Bio_ClinicalBERT",
    )
    parser.add_argument(
        "--fine-tune", action="store_true",
        help="Update the transformer/encoder weights; disables embedding cache.",
    )
    parser.add_argument(
        "--no-cache", dest="cache_embeddings", action="store_false",
        help="Disable the frozen encoder's in-memory embedding cache.",
    )
    parser.set_defaults(cache_embeddings=True)
    parser.add_argument("--cache-max-size", type=int, default=50000)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument(
        "--diagnosis-groups", type=Path,
        help="JSON file mapping query group IDs to description and ICD-9 codes.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
    )
    parser.add_argument(
        "--max-patients", type=int, default=1000,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--history-windows",
        nargs="+", type=int,
        default=[30, 90, 365],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("query_focused_ehr_results.json"),
    )

    args = parser.parse_args()

    if args.synthetic and args.synthetic_mimic:
        parser.error("Select either --synthetic or --synthetic-mimic, not both.")
    if (args.synthetic or args.synthetic_mimic) and args.root is not None:
        parser.error("Do not combine a synthetic mode with --root.")
    if not (args.synthetic or args.synthetic_mimic) and args.root is None:
        parser.error(
            "Provide --root for real MIMIC-III, "
            "--synthetic-mimic for the full artificial-data pipeline, "
            "or --synthetic for a fast smoke test."
        )

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.max_patients < 3
        or args.synthetic_patients < 3
        or args.encoder_batch_size < 1
        or args.cache_max_size < 1
        or any(d <= 0 for d in args.history_windows)
        or len(set(args.history_windows)) != len(args.history_windows)
    ):
        parser.error("Invalid numeric experiment argument.")

    set_seed(args.seed)
    diagnosis_groups = load_diagnosis_groups(args.diagnosis_groups)

    # When fine-tuning, text embeddings depend on changing model weights.
    if args.fine_tune:
        args.cache_embeddings = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if args.synthetic_mimic:
        print("Experiment: ARTIFICIAL MIMIC-III FULL PIPELINE")
        # Keep the generated CSV files alive for the entire experiment.
        synthetic_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        args.root = write_synthetic_mimic_tables(
            Path(synthetic_tmp.name) / "mimic3", args.synthetic_patients
        )
    else:
        synthetic_tmp = None
        print(
            "Experiment:",
            "SYNTHETIC SMOKE TEST"
            if args.synthetic else "MIMIC-III",
        )

    results = {
        "dataset": (
            "synthetic-direct" if args.synthetic else
            "synthetic-mimic" if args.synthetic_mimic else "MIMIC-III"
        ),
        "seed": args.seed,
        "encoder": (
            "TinyEncoder"
            if (args.synthetic or args.synthetic_mimic) and not args.use_clinicalbert
            else args.encoder
        ),
        "fine_tune": args.fine_tune,
        "cache_embeddings": args.cache_embeddings,
        "cache_max_size": args.cache_max_size if args.cache_embeddings else None,
        "encoder_batch_size": args.encoder_batch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_patients": args.max_patients,
        "negative_sampling": "synthetic_balanced" if args.synthetic else "never_observed",
        "diagnosis_groups": diagnosis_groups,
        "history_windows": {},
    }

    print("Building matched patient/visit/query cohorts across windows...")
    matched_windows, split_ids = build_matched_windows(args, diagnosis_groups)
    results["common_patient_counts"] = {
        "train": len(split_ids[0]),
        "validation": len(split_ids[1]),
        "test": len(split_ids[2]),
    }

    for days in args.history_windows:
        print(f"\n{'=' * 60}")
        print(f"History window: {days} days")
        print(f"{'=' * 60}")

        samples = matched_windows[days]
        train, val, test = split_by_patient(
            samples, args.seed, split_ids=split_ids
        )

        if not train or not val or not test:
            raise ValueError(
                "Empty data split. Increase cohort size."
            )

        print(
            f"Samples: train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

        tfidf_metrics = evaluate_tfidf(train, test)

        print("\nTraining query-focused attention model")
        model = train_model(
            args, train, val, device
        )

        neural_metrics = evaluate_model(
            model, test, args.batch_size
        )

        print("\nTF-IDF:", tfidf_metrics)
        print("Neural:", neural_metrics)

        show_sentence_ranking(model, test)

        cache_info = (
            model.embedding_cache_info()
            if args.cache_embeddings else None
        )
        if cache_info is not None:
            print("Embedding cache:", cache_info)

        results["history_windows"][str(days)] = {
            "sample_counts": {
                "train": len(train),
                "validation": len(val),
                "test": len(test),
            },
            "class_counts": {
                "train": class_counts(train),
                "validation": class_counts(val),
                "test": class_counts(test),
            },
            "tfidf": tfidf_metrics,
            "neural": neural_metrics,
            "embedding_cache": cache_info,
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    args.output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", args.output)
    if args.synthetic or args.synthetic_mimic:
        print(
            "Synthetic metrics are smoke-test results, "
            "not clinical reproduction findings."
        )
    if synthetic_tmp is not None:
        synthetic_tmp.cleanup()


if __name__ == "__main__":
    main()
