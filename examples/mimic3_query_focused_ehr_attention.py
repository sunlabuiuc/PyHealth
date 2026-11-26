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
- Synthetic augmentation of the public MIMIC-III demo at runtime
- Configurable ICD-9 diagnosis groups, note counts, history offsets and seed
- Top-k extractive sentence ranking

Data modes:
  --synthetic         Download the public MIMIC-III demo (or pass --demo-zip),
                      augment it with synthetic historical admissions and notes
                      in temporary storage, and run Dataset -> Task -> Model.
                      Uses real Bio_ClinicalBERT by default.
  --root PATH         Load an existing local MIMIC-III dataset.

The synthetic augmentation is NOT a clinical benchmark. All note text added
in --synthetic mode is artificial and is not patient-linked clinical evidence.
No generated datasets or public-demo files are committed to this repository.

Examples:
    # Main --synthetic mode: reproduce the demo mimic data + synthetic data augmentation
    # with the standard experiment args (no synthetic overrides required):
    python examples/mimic3_query_focused_ehr_attention.py --synthetic --history-windows 30 90 365 --epochs 10 --batch-size 4 --learning-rate 2e-5 --max-patients 100 --seed 42 --output hybrid_enhanced_results.json

    # Customize synthetic augmentation independently from evaluation windows:
    python examples/mimic3_query_focused_ehr_attention.py --synthetic --synthetic-notes-per-original-admission 3 --synthetic-history-offsets-days 200 80 15 --synthetic-notes-per-historical-admission 4 --synthetic-augmentation-seed 123 --history-windows 30 90 365 --epochs 10

    # Use an already-downloaded public demo ZIP instead of downloading again:
    python examples/mimic3_query_focused_ehr_attention.py --synthetic --demo-zip /path/to/mimiciii-demo-1.4.zip --epochs 10

    # Real MIMIC-III (requires authorized access and local files):
    python examples/mimic3_query_focused_ehr_attention.py --root /path/to/mimic-iii

Default --synthetic augmentation configuration (based on Colab):
  --synthetic-notes-per-original-admission 5
      Five synthetic notes per ORIGINAL admission: 129 x 5 = 645 notes.
  --synthetic-history-offsets-days 230 65 20
      Three synthetic historical admissions per patient (100 x 3 = 300).
      Each new admission lasts two days.
  --synthetic-notes-per-historical-admission 5
      Five synthetic notes per HISTORICAL admission: 300 x 5 = 1,500.
  --synthetic-augmentation-seed 42
      Both independent synthetic note-sampling stages use seed 42.

The published public demo contains 100 patients and 129 admissions, yielding
429 admissions and 2,145 synthetic notes with the defaults. Diagnosis records
remain unchanged. Values depend on the source demo release. Note counts are
per ADMISSION, not per patient. --history-windows controls lookback at model
sample construction; it does not control how many admissions are generated.

The example does not guarantee reproducing the original Colab AUROC/AP because
package versions, data splits, training and stochasticity can affect results.
Requires pandas, scikit-learn, PyTorch, transformers and PyHealth.
"""

import argparse
import json
import random
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

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
# 2. Synthetic augmentation of public MIMIC-III demo (full pipeline)
# ---------------------------------------------------------

MIMIC_III_DEMO_URL = "https://physionet.org/content/mimiciii-demo/get-zip/1.4/"
HYBRID_TABLES = ("PATIENTS", "ADMISSIONS", "ICUSTAYS", "DIAGNOSES_ICD", "NOTEEVENTS")
HYBRID_NOTE_COLUMNS = (
    "row_id", "subject_id", "hadm_id", "chartdate", "charttime",
    "storetime", "category", "description", "cgid", "iserror", "text",
)
# 26 generic synthetic-note templates used in the Colab augmentation.
HYBRID_NOTE_TEMPLATES = [
    "Patient reports intermittent fatigue and reduced exercise tolerance.",
    "Vital signs are stable during routine clinical observation.",
    "Patient reports occasional shortness of breath with exertion.",
    "Medication history and adherence were reviewed.",
    "Patient reports intermittent chest discomfort.",
    "Respiratory examination and oxygen saturation were assessed.",
    "Patient reports occasional productive cough.",
    "Patient reports mild discomfort in the lower extremities.",
    "Previous outpatient laboratory findings were reviewed.",
    "Patient reports intermittent fever and malaise.",
    "Nursing assessment documented stable respiratory status.",
    "Patient reports difficulty sleeping during hospitalization.",
    "Clinical team reviewed previous medical history.",
    "Patient reports occasional dizziness while standing.",
    "Fluid balance and nutritional intake were assessed.",
    "Patient reports generalized weakness and reduced mobility.",
    "Cardiovascular examination was performed.",
    "Patient reports intermittent abdominal discomfort.",
    "Routine blood tests were obtained for clinical monitoring.",
    "Patient received counseling about medication management.",
    "Physical examination findings were discussed with the care team.",
    "Patient reports occasional nausea and reduced appetite.",
    "Mobility assessment and fall-risk precautions were reviewed.",
    "Clinical progress was reassessed during morning rounds.",
    "Patient reports mild headache without additional symptoms.",
    "Continued monitoring and follow-up evaluation were recommended.",
]


def write_synthetic_augmented_demo(
    root,
    demo_zip=None,
    notes_per_original_admission=5,
    history_offsets_days=(230, 65, 20),
    notes_per_historical_admission=5,
    seed=42,
):
    """Augment the public demo with synthetic notes, as in Colab Hybrid_Enhanced.

    Defaults match the notebook: five synthetic notes per ORIGINAL
    admission (129 * 5 = 645 notes), plus three synthetic historical
    admissions per patient (offsets 230, 65 and 20 days). Each of those
    admissions gets five additional notes (100 * 3 * 5 = 1,500 notes).
    Thus the 100-patient demo yields 429 admissions and 2,145 notes.

    These counts are dataset-specific; the number of patients and original
    admissions comes from the public demo, not function arguments.
    ``notes_per_original_admission`` and
    ``notes_per_historical_admission`` are *per admission*, not per patient.

    Override these defaults with the corresponding ``--synthetic-*`` arguments.
    ``history_offsets_days`` determines synthetic admission dates, while the
    independent ``--history-windows`` flag filters model input histories.
    The two stages each use a separate random generator seeded with ``seed``.
    A local ``demo_zip`` can be provided to avoid downloading. The generated
    files are written beneath ``root``; main() supplies a temporary folder
    outside the git repository and cleans it up after training.

    All clinical note text is synthetic and unrelated to the patients'
    actual diagnoses. This reproduces the data-construction procedure,
    not guaranteed bitwise-identical model scores or clinical validity.
    """
    if notes_per_original_admission < 1 or notes_per_historical_admission < 1:
        raise ValueError("Notes per admission must be positive.")
    if not history_offsets_days or any(days <= 2 for days in history_offsets_days):
        raise ValueError("History offsets must contain days greater than two.")
    if len(set(history_offsets_days)) != len(history_offsets_days):
        raise ValueError("History offsets must be unique to avoid duplicate admission dates.")
    if max(notes_per_original_admission, notes_per_historical_admission) > len(HYBRID_NOTE_TEMPLATES):
        raise ValueError("Requested note count exceeds distinct templates.")
    root.mkdir(parents=True, exist_ok=True)
    if demo_zip is None:
        demo_zip = root / "mimiciii-demo-1.4.zip"
        print(f"Downloading public MIMIC-III demo to {demo_zip}")
        urllib.request.urlretrieve(MIMIC_III_DEMO_URL, demo_zip)

    tables = {}
    with ZipFile(demo_zip) as archive:
        members = {
            Path(member).name.upper(): member
            for member in archive.namelist()
            if member.lower().endswith(".csv")
        }
        for name in HYBRID_TABLES:
            key = f"{name}.CSV"
            if key not in members:
                raise ValueError(f"Demo ZIP is missing {key}")
            with archive.open(members[key]) as stream:
                table = pd.read_csv(stream, low_memory=False)
            table.columns = table.columns.str.strip().str.lower()
            tables[name] = table

    patients = tables["PATIENTS"]
    admissions = tables["ADMISSIONS"].copy()
    original_notes = tables["NOTEEVENTS"]
    if len(original_notes):
        raise ValueError(
            "This reproduction expects the public demo's empty NOTEEVENTS; "
            "refusing to overwrite authentic notes."
        )
    if not len(patients) or not len(admissions):
        raise ValueError("Demo patients/admissions are empty.")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    # Colab cell 13: five distinct random note templates per ORIGINAL
    # admission, chronological timestamps within the admission.
    rng = random.Random(seed)
    initial_notes = []
    valid_patients = set(patients["subject_id"])
    for admission in admissions.itertuples(index=False):
        patient_id, hadm_id = admission.subject_id, admission.hadm_id
        admit_time, discharge_time = admission.admittime, admission.dischtime
        if patient_id not in valid_patients:
            raise ValueError(f"Unknown patient: {patient_id}")
        if pd.isna(admit_time) or pd.isna(discharge_time) or discharge_time <= admit_time:
            continue
        for j, sentence in enumerate(
            rng.sample(HYBRID_NOTE_TEMPLATES, k=notes_per_original_admission)
        ):
            timestamp = admit_time + (discharge_time - admit_time) * (
                (j + 1) / (notes_per_original_admission + 1)
            )
            initial_notes.append({
                "row_id": len(initial_notes) + 1,
                "subject_id": patient_id,
                "hadm_id": hadm_id,
                "chartdate": timestamp.strftime("%Y-%m-%d"),
                "charttime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "storetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "category": "Physician",
                "description": "Synthetic progress note",
                "cgid": None,
                "iserror": None,
                "text": "[SYNTHETIC] " + sentence,
            })
    note_df = pd.DataFrame(initial_notes, columns=HYBRID_NOTE_COLUMNS)

    # Colab cell 16: independently reseed the second random generator, then
    # add historical admissions for each patient with notes in each stay.
    rng = random.Random(seed)
    next_hadm_id = int(pd.to_numeric(admissions["hadm_id"]).max()) + 1
    next_note_id = int(pd.to_numeric(note_df["row_id"]).max()) + 1
    new_admissions = []
    new_notes = []
    for patient_id, group in admissions.groupby("subject_id"):
        first_admission = group["admittime"].min()
        if pd.isna(first_admission):
            continue
        for days_before in history_offsets_days:
            hadm_id = next_hadm_id
            next_hadm_id += 1
            admit_time = first_admission - pd.Timedelta(days=days_before)
            discharge_time = admit_time + pd.Timedelta(days=2)
            new_admission = {column: None for column in admissions.columns}
            new_admission.update({
                "subject_id": patient_id,
                "hadm_id": hadm_id,
                "admittime": admit_time,
                "dischtime": discharge_time,
            })
            if "admission_type" in new_admission:
                new_admission["admission_type"] = "ELECTIVE"
            if "diagnosis" in new_admission:
                new_admission["diagnosis"] = "Synthetic historical admission"
            if "hospital_expire_flag" in new_admission:
                new_admission["hospital_expire_flag"] = 0
            new_admissions.append(new_admission)
            for j, sentence in enumerate(
                rng.sample(HYBRID_NOTE_TEMPLATES, k=notes_per_historical_admission)
            ):
                # Preserve original 6, 14, 22, 30, 38-hour timestamps
                # for the default five-note configuration.
                timestamp = admit_time + pd.Timedelta(
                    hours=(6 + j * 8)
                    if notes_per_historical_admission == 5
                    else 48 * (j + 1) / (notes_per_historical_admission + 1)
                )
                new_notes.append({
                    "row_id": next_note_id,
                    "subject_id": patient_id,
                    "hadm_id": hadm_id,
                    "chartdate": timestamp.strftime("%Y-%m-%d"),
                    "charttime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "storetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "category": "Physician",
                    "description": "Synthetic historical progress note",
                    "cgid": None,
                    "iserror": None,
                    "text": "[SYNTHETIC] " + sentence,
                })
                next_note_id += 1

    enhanced_admissions = pd.concat(
        [admissions, pd.DataFrame(new_admissions, columns=admissions.columns)],
        ignore_index=True,
    )
    if "row_id" in enhanced_admissions.columns:
        original_row_ids = pd.to_numeric(admissions["row_id"], errors="coerce").dropna()
        next_row_id = int(original_row_ids.max()) + 1 if len(original_row_ids) else 1
        enhanced_admissions.loc[len(admissions):, "row_id"] = range(
            next_row_id, next_row_id + len(new_admissions)
        )
    enhanced_notes = pd.concat(
        [note_df, pd.DataFrame(new_notes, columns=HYBRID_NOTE_COLUMNS)],
        ignore_index=True,
    )

    # Guard against accidental patient / visit mismatches.
    valid_pairs = set(zip(enhanced_admissions["subject_id"], enhanced_admissions["hadm_id"]))
    if not set(enhanced_admissions["subject_id"]).issubset(valid_patients):
        raise ValueError("Augmented admissions contain unknown patient IDs")
    if not enhanced_admissions["hadm_id"].is_unique:
        raise ValueError("Duplicate admission ID in augmented data")
    if not set(zip(enhanced_notes["subject_id"], enhanced_notes["hadm_id"])).issubset(valid_pairs):
        raise ValueError("Augmented notes refer to an unknown patient/admission")

    tables["ADMISSIONS"] = enhanced_admissions
    tables["NOTEEVENTS"] = enhanced_notes
    for name in HYBRID_TABLES:
        tables[name].to_csv(root / f"{name}.csv.gz", index=False, compression="gzip")

    print(
        "Created runtime public-demo synthetic augmentation: "
        f"{len(patients)} patients, {len(enhanced_admissions)} admissions, "
        f"{len(enhanced_notes)} synthetic notes, "
        f"{len(tables['DIAGNOSES_ICD'])} unchanged diagnosis rows."
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
    """Construct the ClinicalBERT model for either supported data mode."""
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
        help=(
            "Augment the public MIMIC-III demo with synthetic notes and "
            "historical admissions in temporary storage; run full PyHealth "
            "pipeline using Bio_ClinicalBERT by default."
        ),
    )
    parser.add_argument(
        "--demo-zip", type=Path,
        help="Optional local MIMIC-III demo v1.4 ZIP for --synthetic.",
    )
    parser.add_argument(
        "--synthetic-notes-per-original-admission", type=int, default=5,
        metavar="N",
        help=(
            "Synthetic notes per original MIMIC-III demo admission "
            "(default: 5; 129 x 5 = 645 with the 100-patient demo)."
        ),
    )
    parser.add_argument(
        "--synthetic-history-offsets-days", nargs="+", type=int,
        default=[230, 65, 20], metavar="DAYS",
        help=(
            "Days before each patient's first original admission at which "
            "to create synthetic historical admissions (default: 230 65 20). "
            "Each new admission lasts two days. Not the same as --history-windows."
        ),
    )
    parser.add_argument(
        "--synthetic-notes-per-historical-admission", type=int, default=5,
        metavar="N",
        help="Synthetic notes per generated historical admission (default: 5).",
    )
    parser.add_argument(
        "--synthetic-augmentation-seed", type=int, default=42,
        metavar="SEED",
        help=(
            "Random seed for both independent note-template sampling stages "
            "(default: 42); separate from --seed used for splitting/training."
        ),
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

    if args.synthetic and args.root is not None:
        parser.error("Choose exactly one data source: --synthetic or --root.")
    if args.demo_zip is not None and not args.synthetic:
        parser.error("--demo-zip requires --synthetic.")
    if not args.synthetic and args.root is None:
        parser.error("Provide either --synthetic or --root PATH.")

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.max_patients < 3
        or args.encoder_batch_size < 1
        or args.cache_max_size < 1
        or any(d <= 0 for d in args.history_windows)
        or len(set(args.history_windows)) != len(args.history_windows)
    ):
        parser.error("Invalid numeric experiment argument.")

    if (
        args.synthetic_notes_per_original_admission < 1
        or args.synthetic_notes_per_historical_admission < 1
        or max(
            args.synthetic_notes_per_original_admission,
            args.synthetic_notes_per_historical_admission,
        ) > len(HYBRID_NOTE_TEMPLATES)
    ):
        parser.error(
            f"Synthetic notes per admission must be between 1 and "
            f"{len(HYBRID_NOTE_TEMPLATES)} (sampled without replacement)."
        )
    if (
        any(days <= 2 for days in args.synthetic_history_offsets_days)
        or len(set(args.synthetic_history_offsets_days))
        != len(args.synthetic_history_offsets_days)
    ):
        parser.error("Synthetic history offsets must be unique and greater than 2 days.")

    set_seed(args.seed)
    diagnosis_groups = load_diagnosis_groups(args.diagnosis_groups)

    # When fine-tuning, text embeddings depend on changing model weights.
    if args.fine_tune:
        args.cache_embeddings = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if args.synthetic:
        print("Experiment: PUBLIC MIMIC-III DEMO + SYNTHETIC AUGMENTATION")
        # Temporary CSV files persist for the entire experiment, then are deleted.
        synthetic_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        args.root = write_synthetic_augmented_demo(
            Path(synthetic_tmp.name) / "mimic3",
            demo_zip=args.demo_zip,
            notes_per_original_admission=args.synthetic_notes_per_original_admission,
            history_offsets_days=tuple(args.synthetic_history_offsets_days),
            notes_per_historical_admission=args.synthetic_notes_per_historical_admission,
            seed=args.synthetic_augmentation_seed,
        )
    else:
        synthetic_tmp = None
        print("Experiment: LOCAL MIMIC-III FILES")

    results = {
        "dataset": "public-demo-synthetic-augmentation" if args.synthetic else "MIMIC-III",
        "seed": args.seed,
        "encoder": args.encoder,
        "fine_tune": args.fine_tune,
        "cache_embeddings": args.cache_embeddings,
        "cache_max_size": args.cache_max_size if args.cache_embeddings else None,
        "encoder_batch_size": args.encoder_batch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_patients": args.max_patients,
        "negative_sampling": "never_observed",
        "synthetic_augmentation": (
            {
                "notes_per_original_admission": args.synthetic_notes_per_original_admission,
                "history_offsets_days": args.synthetic_history_offsets_days,
                "notes_per_historical_admission": args.synthetic_notes_per_historical_admission,
                "augmentation_seed": args.synthetic_augmentation_seed,
                "demo_zip": str(args.demo_zip) if args.demo_zip else None,
                "synthetic_notes": True,
            }
            if args.synthetic else None
        ),
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
    if args.synthetic:
        print(
            "Synthetic-note evaluations are not clinical performance "
            "benchmarks; results may not match the original Colab scores."
        )
    if synthetic_tmp is not None:
        synthetic_tmp.cleanup()


if __name__ == "__main__":
    main()
