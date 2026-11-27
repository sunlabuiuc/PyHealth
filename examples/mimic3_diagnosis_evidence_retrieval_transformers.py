
"""Diagnosis evidence retrieval using an existing PyHealth model.

This example demonstrates how historical note deduplication affects
diagnosis-specific evidence retrieval.

Three task configurations are compared:
    1. No deduplication.
    2. Exact text deduplication.
    3. Exact text and report-version deduplication.

The experiment uses synthetic patients and manually assigned evidence
annotations. No protected MIMIC-III data or model training is required.

Experimental Results
--------------------
Evaluated three synthetic patient-diagnosis instances using PyHealth's
existing TransformersModel with pretrained Bio_ClinicalBERT embeddings,
mean pooling, cosine similarity ranking, and a retrieval budget of
three notes.

Macro-averaged results:

| Deduplication policy | Avg. notes | Precision@3 | Unique Recall@3 | Redundancy@3 |
|---|---:|---:|---:|---:|
| None | 9.0 | 0.889 | 0.556 | 0.333 |
| Exact | 7.0 | 0.778 | 0.667 | 0.111 |
| Report versions | 6.0 | 0.778 | 0.778 | 0.000 |

Report-version deduplication reduced the historical note count by
33.3%, improved unique evidence recall by 40% relative to the
non-deduplicated baseline, and eliminated redundant relevant evidence
from the top-three results.

Conventional precision decreased because repeated relevant notes
were previously counted as separate relevant retrievals. This
highlights the importance of measuring distinct evidence coverage
alongside standard retrieval precision.

Limitations:
    The experiment uses three deliberately constructed synthetic
    patients with manually specified relevance annotations. Results
    demonstrate task behavior under controlled conditions and should
    not be generalized to real clinical populations. The pretrained
    encoder is used without fine-tuning or classification-head
    training.

Reference:
    Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges.
    https://arxiv.org/abs/2309.04550
"""

from datetime import datetime

import polars as pl
import torch
import torch.nn.functional as F

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import TransformersModel
from pyhealth.tasks import DiagnosisEvidenceRetrievalMIMIC3


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TOP_K = 3
POLICIES = ("none", "exact", "report_versions")


def build_patient(
    patient_id: str,
    diagnosis: str,
    evidence: dict[str, str],
) -> tuple[Patient, dict, dict]:
    """Construct a synthetic patient and independent evidence annotations."""

    rows = [
        {
            "patient_id": patient_id,
            "event_type": "admissions",
            "timestamp": datetime(2100, 1, 2, 8),
            "admissions/hadm_id": "20",
            "admissions/admission_type": "EMERGENCY",
            "admissions/edregtime": "2100-01-02 08:00:00",
        }
    ]

    annotations = {}

    def add_note(
        note_id: str,
        text: str,
        evidence_id: str | None = None,
        category: str = "Nursing",
        description: str = "Synthetic note",
        charttime: str = "2100-01-01 09:00:00",
        storetime: str = "2100-01-01 10:00:00",
    ) -> None:
        """Add a note event and its external relevance annotation."""

        rows.append(
            {
                "patient_id": patient_id,
                "event_type": "noteevents",
                "timestamp": datetime.fromisoformat(charttime),
                "noteevents/row_id": note_id,
                "noteevents/hadm_id": "20",
                "noteevents/chartdate": charttime[:10],
                "noteevents/charttime_missing": False,
                "noteevents/storetime": storetime,
                "noteevents/iserror": None,
                "noteevents/text": text,
                "noteevents/category": category,
                "noteevents/description": description,
            }
        )

        annotations[note_id] = evidence_id

    findings = list(evidence.items())

    # A relevant note and two exact copies of the same information.
    add_note("1", findings[0][1], findings[0][0])
    add_note("2", findings[0][1], findings[0][0])
    add_note("3", findings[0][1], findings[0][0])

    # Two report versions describe the same clinical finding.
    add_note(
        "4",
        f"Clip #123 PFI REPORT: {findings[1][1]}",
        findings[1][0],
        category="Radiology",
        description="Prior imaging",
    )

    add_note(
        "5",
        f"Clip #123 FINAL REPORT: {findings[1][1]}",
        findings[1][0],
        category="Radiology",
        description="Prior imaging",
        storetime="2100-01-01 11:00:00",
    )

    # A third distinct piece of relevant clinical evidence.
    add_note("6", findings[2][1], findings[2][0])

    # Irrelevant historical notes.
    add_note("7", "Routine dental cleaning.")
    add_note("8", "Patient requests assistance with transportation.")
    add_note("9", "History of an uncomplicated finger laceration.")

    # A relevant note that was not available at imaging time.
    # The task should exclude it under strict availability.
    add_note(
        "10",
        findings[0][1],
        findings[0][0],
        storetime="2100-01-03 10:00:00",
    )

    # The reviewed index radiology report is not retrieval evidence.
    add_note(
        "100",
        f"FINAL REPORT: Findings consistent with {diagnosis}.",
        category="Radiology",
        description="Index imaging",
        charttime="2100-01-02 12:00:00",
        storetime="2100-01-02 12:00:00",
    )

    patient = Patient(
        patient_id=patient_id,
        data_source=pl.DataFrame(rows, infer_schema_length=None),
    )

    query = {
        "patient_id": patient_id,
        "index_note_id": "100",
        "diagnosis": diagnosis,
    }

    return patient, query, annotations


def build_cohort() -> list:
    """Create three synthetic patient-diagnosis instances."""

    return [
        build_patient(
            "1",
            "stroke",
            {
                "hypertension": "Patient has chronic hypertension.",
                "atherosclerosis": "Carotid atherosclerosis is present.",
                "atrial_fibrillation": "History of atrial fibrillation.",
            },
        ),
        build_patient(
            "2",
            "pulmonary embolism",
            {
                "immobility": "Patient has prolonged immobilization.",
                "dvt": "Imaging demonstrates deep venous thrombosis.",
                "surgery": "Patient recently underwent major surgery.",
            },
        ),
        build_patient(
            "3",
            "lung cancer",
            {
                "smoking": "Patient has a long history of tobacco smoking.",
                "fibrosis": "Imaging demonstrates pulmonary fibrosis.",
                "asbestos": "History of occupational asbestos exposure.",
            },
        ),
    ]


def load_model(cohort: list) -> TransformersModel:
    """Initialize the existing PyHealth pretrained text encoder."""

    # TransformersModel requires one input feature and one label.
    # Use independently annotated synthetic historical notes.
    samples = []

    for patient, _, annotations in cohort:
        for event in patient.get_events(event_type="noteevents"):
            note_id = event.attr_dict["row_id"]

            if note_id in {"10", "100"}:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "text": event.attr_dict["text"],
                    "label": int(annotations[note_id] is not None),
                }
            )

    dataset = create_sample_dataset(
        samples,
        input_schema={"text": "text"},
        output_schema={"label": "binary"},
        in_memory=True,
    )

    model = TransformersModel(
        dataset=dataset,
        model_name=MODEL_NAME,
        pooling="mean",
        dropout=0.0,
        max_length=128,
    )

    model.eval()
    return model


@torch.inference_mode()
def rank_notes(
    model: TransformersModel,
    sample: dict,
    embedding_cache: dict,
) -> list[str]:
    """Rank historical notes by cosine similarity to the diagnosis."""

    query = f"Risk factors for {sample['diagnosis']}."
    texts = [query] + sample["historical_notes"]

    # Reuse embeddings across deduplication configurations.
    missing = list(
        dict.fromkeys(
            text for text in texts if text not in embedding_cache
        )
    )

    if missing:
        embeddings = model.encode(missing)

        for text, embedding in zip(missing, embeddings):
            embedding_cache[text] = embedding.cpu()

    query_embedding = embedding_cache[query]

    note_embeddings = torch.stack(
        [embedding_cache[text] for text in texts[1:]]
    )

    scores = F.cosine_similarity(
        query_embedding.unsqueeze(0),
        note_embeddings,
        dim=-1,
    )

    # Stable ordering gives reproducible handling of tied scores.
    ranking = torch.argsort(
        scores,
        descending=True,
        stable=True,
    ).tolist()

    return [sample["note_ids"][i] for i in ranking]


def evaluate(
    ranked_ids: list[str],
    annotations: dict,
    k: int = TOP_K,
) -> dict[str, float]:
    """Evaluate retrieval against independent synthetic evidence labels."""

    retrieved = ranked_ids[:k]

    relevant = [
        annotations[note_id]
        for note_id in retrieved
        if annotations[note_id] is not None
    ]

    gold_evidence = {
        evidence_id
        for note_id, evidence_id in annotations.items()
        if evidence_id is not None and note_id not in {"10", "100"}
    }

    unique_relevant = set(relevant)

    return {
        "precision": len(relevant) / k,
        "unique_recall": len(unique_relevant) / len(gold_evidence),
        "redundancy": (len(relevant) - len(unique_relevant)) / k,
    }


def main() -> None:
    """Run and compare the three task configurations."""

    torch.manual_seed(42)

    cohort = build_cohort()
    model = load_model(cohort)

    reviewed_queries = [query for _, query, _ in cohort]
    embedding_cache = {}
    results = []

    for policy in POLICIES:
        task = DiagnosisEvidenceRetrievalMIMIC3(
            reviewed_queries=reviewed_queries,
            min_prior_notes=1,
            strict_availability=True,
            unknown_availability="exclude",
            deduplication=policy,
            evidence_type="risk",
        )

        patient_results = []

        print(f"\nConfiguration: {policy}")

        for patient, _, annotations in cohort:
            samples = task(patient)

            assert len(samples) == 1
            sample = samples[0]

            # Ensure unavailable notes and the index report are excluded.
            assert "10" not in sample["note_ids"]
            assert "100" not in sample["note_ids"]

            ranked_ids = rank_notes(
                model,
                sample,
                embedding_cache,
            )

            metrics = evaluate(ranked_ids, annotations)

            metrics["notes"] = len(sample["note_ids"])
            patient_results.append(metrics)

            audit = sample["history_audit"]

            print(
                f"  Patient {patient.patient_id}: "
                f"{sample['diagnosis']}"
            )
            print(f"    Notes retained: {sample['note_ids']}")
            print(f"    Top-{TOP_K}: {ranked_ids[:TOP_K]}")
            print(
                "    Exact duplicates removed:",
                audit["exact_duplicates_removed"],
            )
            print(
                "    Report versions removed:",
                audit["report_versions_removed"],
            )
            print(
                f"    Precision: {metrics['precision']:.3f}, "
                f"Unique recall: {metrics['unique_recall']:.3f}, "
                f"Redundancy: {metrics['redundancy']:.3f}"
            )

        # Macro-average metrics across patient-diagnosis instances.
        results.append(
            {
                "policy": policy,
                **{
                    key: sum(row[key] for row in patient_results)
                    / len(patient_results)
                    for key in (
                        "notes",
                        "precision",
                        "unique_recall",
                        "redundancy",
                    )
                },
            }
        )

    print("\nABLATION RESULTS")
    print("-" * 76)
    print(
        f"{'Policy':<20}"
        f"{'Avg notes':>10}"
        f"{'P@3':>12}"
        f"{'Recall@3':>14}"
        f"{'Redundancy':>14}"
    )
    print("-" * 76)

    for row in results:
        print(
            f"{row['policy']:<20}"
            f"{row['notes']:>10.1f}"
            f"{row['precision']:>12.3f}"
            f"{row['unique_recall']:>14.3f}"
            f"{row['redundancy']:>14.3f}"
        )

    print("-" * 76)
    print(f"Patients evaluated: {len(cohort)}")
    print(f"Unique texts encoded: {len(embedding_cache)}")
    print("Evaluation uses synthetic evidence annotations only.")


if __name__ == "__main__":
    main()
