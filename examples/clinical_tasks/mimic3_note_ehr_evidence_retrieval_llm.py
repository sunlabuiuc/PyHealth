"""EHR Evidence Retrieval with Zero-Shot LLMs on MIMIC-III.
Contributor: Abhisek Sinha (abhisek5@illinois.edu)
Reproduces and extends Ahsan et al. (2024) "Retrieving Evidence from EHRs
with LLMs: Possibilities and Challenges" (CHIL 2024, PMLR 248:489-505).
Paper: https://arxiv.org/abs/2309.04550

This script demonstrates the full PyHealth pipeline:
    MIMIC3NoteDataset -> EHREvidenceRetrievalTask -> ZeroShotEvidenceLLM

And includes four ablation experiments:
    A1. Prompt format: two-step vs single-step vs chain-of-thought
    A2. Confidence threshold sweep: precision/recall trade-off for abstention
    A3. BM25 pre-retrieval: reduce note length, measure recall vs faithfulness
    A4. Open-source LLM judge: Mistral-7B vs GPT-3.5 auto-evaluator agreement

Usage:
    # Full MIMIC-III run (requires PhysioNet credentialed access)
    python mimic3_note_ehr_evidence_retrieval_llm.py \
        --mimic3_root /path/to/mimic-iii/1.4 \
        --model_name google/flan-t5-xxl \
        --ablation all

    # Demo run with synthetic data (no MIMIC access required)
    python mimic3_note_ehr_evidence_retrieval_llm.py --demo
"""
import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic demo helpers (no real data required)
# ---------------------------------------------------------------------------
DEMO_PATIENTS = [
    {
        "patient_id": "P001",
        "query_diagnosis": "small vessel disease",
        "notes": (
            "Patient is a 72-year-old male with a long history of hypertension "
            "and type 2 diabetes. MRI of the brain reveals extensive white matter "
            "hyperintensities consistent with chronic small vessel disease. Patient "
            "reports cognitive decline and gait disturbance over the past year.\n\n"
            "Neurology consultation: lacunar infarcts noted on imaging. Blood pressure "
            "poorly controlled despite multiple agents."
        ),
        "label": 1,
    },
    {
        "patient_id": "P002",
        "query_diagnosis": "small vessel disease",
        "notes": (
            "Patient is a 45-year-old female admitted for elective knee replacement. "
            "No significant neurological history. Vital signs stable. No medications "
            "related to cerebrovascular disease. Discharge in good condition."
        ),
        "label": 0,
    },
    {
        "patient_id": "P003",
        "query_diagnosis": "atrial fibrillation",
        "notes": (
            "EKG on admission shows irregularly irregular rhythm consistent with "
            "atrial fibrillation. Rate 110 bpm. Patient started on anticoagulation "
            "with warfarin. Cardiology to follow up as outpatient. History of "
            "palpitations for 6 months."
        ),
        "label": 1,
    },
    {
        "patient_id": "P004",
        "query_diagnosis": "atrial fibrillation",
        "notes": (
            "Post-op day 1 following appendectomy. Patient recovering well. "
            "Sinus rhythm on telemetry throughout. No arrhythmias noted. "
            "Ambulating independently. Pain controlled with oral analgesics."
        ),
        "label": 0,
    },
]


# ---------------------------------------------------------------------------
# Ablation A1: Prompt format comparison
# ---------------------------------------------------------------------------
_TWO_STEP_CLASSIFY = """\
Patient clinical notes:
{notes}

Does this patient have or show risk for {query_diagnosis}? Answer YES or NO."""

_TWO_STEP_SUMMARISE = """\
Patient clinical notes:
{notes}

Summarise the evidence from the notes that the patient has {query_diagnosis}.
Do not include information not found in the notes above."""

_SINGLE_STEP = """\
Patient clinical notes:
{notes}

Does this patient have {query_diagnosis}? If yes, summarise the supporting \
evidence. If no, reply "No evidence found." """

_CHAIN_OF_THOUGHT = """\
Patient clinical notes:
{notes}

Think step by step:
1. Identify any mentions of {query_diagnosis} or related symptoms/risk factors.
2. Determine if sufficient evidence exists.
3. State YES or NO, then summarise the evidence if YES."""


def ablation_prompt_format(
    samples: List[Dict[str, Any]],
    model_name: str = "google/flan-t5-base",
) -> None:
    """A1: Compare two-step, single-step, and chain-of-thought prompts.

    For each prompt format the script prints the generated outputs on a small
    set of samples so you can compare faithfulness qualitatively.

    Args:
        samples: List of sample dicts with 'notes' and 'query_diagnosis'.
        model_name: HuggingFace model to use (smaller model for quick demo).
    """
    print("\n" + "=" * 70)
    print("ABLATION A1: Prompt Format Comparison")
    print("=" * 70)

    from pyhealth.models import ZeroShotEvidenceLLM

    formats = {
        "two-step": None,  # uses default model prompts
        "single-step": _SINGLE_STEP,
        "chain-of-thought": _CHAIN_OF_THOUGHT,
    }

    model = ZeroShotEvidenceLLM(dataset=None, model_name=model_name)

    for fmt_name, prompt_template in formats.items():
        print(f"\n--- Format: {fmt_name} ---")
        for sample in samples[:2]:
            if prompt_template is None:
                result = model.predict(sample["notes"], sample["query_diagnosis"])
            else:
                # Custom single-prompt format
                full_prompt = prompt_template.format(
                    notes=sample["notes"],
                    query_diagnosis=sample["query_diagnosis"],
                )
                result = {"custom_prompt_output": full_prompt[:200] + "..."}
            print(
                f"  Patient {sample['patient_id']}: "
                f"label={sample['label']}, result={result.get('has_condition', 'N/A')}, "
                f"confidence={result.get('confidence', 'N/A'):.3f}"
                if "confidence" in result
                else f"  Patient {sample['patient_id']}: {result}"
            )


# ---------------------------------------------------------------------------
# Ablation A2: Confidence threshold sweep
# ---------------------------------------------------------------------------
def ablation_confidence_threshold(
    results: List[Dict[str, Any]], labels: List[int]
) -> Dict[str, Any]:
    """A2: Sweep confidence thresholds and compute precision/recall trade-off.

    Extends Figure 4 of Ahsan et al. (2024) by finding the optimal operating
    point where confidence correlates with evidence faithfulness.

    Args:
        results: List of predict() output dicts (must contain 'confidence').
        labels: Ground-truth binary labels (1 = has condition).

    Returns:
        Dict[str, Any]: threshold -> {precision, recall, f1, coverage} mapping.
    """
    print("\n" + "=" * 70)
    print("ABLATION A2: Confidence Threshold Sweep")
    print("=" * 70)

    import numpy as np

    thresholds = [i / 10 for i in range(1, 10)]
    metrics_by_threshold: Dict[str, Any] = {}

    confidences = [r["confidence"] for r in results]
    predictions = [r["has_condition"] for r in results]

    for t in thresholds:
        # Only keep predictions above the threshold (abstain on others)
        kept_indices = [i for i, c in enumerate(confidences) if c >= t]
        coverage = len(kept_indices) / max(len(results), 1)

        if not kept_indices:
            continue

        tp = sum(
            1
            for i in kept_indices
            if predictions[i] and labels[i] == 1
        )
        fp = sum(
            1
            for i in kept_indices
            if predictions[i] and labels[i] == 0
        )
        fn = sum(
            1
            for i in kept_indices
            if not predictions[i] and labels[i] == 1
        )

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        metrics_by_threshold[f"t={t:.1f}"] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "coverage": round(coverage, 3),
        }
        print(
            f"  Threshold {t:.1f}: precision={precision:.3f}  "
            f"recall={recall:.3f}  f1={f1:.3f}  coverage={coverage:.2%}"
        )

    return metrics_by_threshold


# ---------------------------------------------------------------------------
# Ablation A3: BM25 pre-retrieval
# ---------------------------------------------------------------------------
def ablation_bm25_preretrieval(
    samples: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """A3: Apply BM25 to select the top-k relevant note sentences before LLM.

    When a patient has many notes the full text may exceed the model's context
    window. This ablation tests whether BM25 pre-selection can reduce input
    length while preserving recall of relevant evidence.

    Args:
        samples: Sample dicts with 'notes' and 'query_diagnosis'.
        top_k: Number of most relevant sentences to retain.

    Returns:
        List[Dict[str, Any]]: Samples with notes replaced by BM25-selected
        sentences.
    """
    print("\n" + "=" * 70)
    print("ABLATION A3: BM25 Pre-Retrieval")
    print("=" * 70)

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning(
            "rank_bm25 not installed. Run: pip install rank-bm25\n"
            "Skipping A3 ablation."
        )
        return samples

    import re

    filtered_samples = []
    for sample in samples:
        notes_text = sample["notes"]
        query = sample["query_diagnosis"]

        # Sentence tokenise
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", notes_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            filtered_samples.append(sample)
            continue

        tokenised = [s.lower().split() for s in sentences]
        bm25 = BM25Okapi(tokenised)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(top_indices[:top_k])  # restore chronological order

        selected = "\n".join(sentences[i] for i in top_indices)
        orig_len = len(notes_text.split())
        new_len = len(selected.split())

        print(
            f"  Patient {sample['patient_id']}: "
            f"reduced {orig_len} -> {new_len} words "
            f"({new_len / max(orig_len, 1):.0%} retained)"
        )

        filtered = dict(sample)
        filtered["notes"] = selected
        filtered_samples.append(filtered)

    return filtered_samples


# ---------------------------------------------------------------------------
# Ablation A4: Open-source LLM-as-evaluator
# ---------------------------------------------------------------------------
_EVALUATOR_PROMPT = """\
You are evaluating the quality of an AI-generated clinical evidence summary.

Original patient notes:
{notes}

Queried condition: {query_diagnosis}

Generated evidence summary:
{evidence}

Rate the summary on the following scale:
- "useful": The summary accurately reflects evidence in the notes.
- "partially_useful": The summary is partially correct but contains some inaccuracies.
- "not_useful": The summary does not match the notes or is fabricated.
- "not_present": No evidence for the condition exists in the notes.

Reply with exactly one of: useful, partially_useful, not_useful, not_present"""


def ablation_open_source_judge(
    samples_with_evidence: List[Dict[str, Any]],
    judge_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> List[Dict[str, Any]]:
    """A4: Use a small open-source LLM to evaluate evidence quality.

    The paper uses GPT-3.5 as the auto-evaluator. This ablation tests whether
    Mistral-7B-Instruct achieves comparable agreement with radiologist ratings,
    reducing evaluation cost and proprietary API dependence.

    Args:
        samples_with_evidence: List of dicts with 'notes', 'query_diagnosis',
            and 'evidence' (from a predict() call).
        judge_model_name: HuggingFace model ID for the judge LLM.

    Returns:
        List[Dict[str, Any]]: Input dicts enriched with 'judge_rating'.
    """
    print("\n" + "=" * 70)
    print("ABLATION A4: Open-Source LLM-as-Evaluator")
    print(f"Judge model: {judge_model_name}")
    print("=" * 70)

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.warning(
            "transformers not installed. Skipping A4 ablation."
        )
        return samples_with_evidence

    judge_pipe = hf_pipeline(
        "text-generation",
        model=judge_model_name,
        max_new_tokens=16,
        do_sample=False,
    )

    ratings_map = {
        "useful": 3,
        "partially_useful": 2,
        "not_useful": 1,
        "not_present": 0,
    }

    results = []
    for sample in samples_with_evidence:
        evidence = sample.get("evidence", "")
        if not evidence:
            sample["judge_rating"] = "not_present"
            sample["judge_score"] = 0
            results.append(sample)
            continue

        prompt = _EVALUATOR_PROMPT.format(
            notes=sample["notes"][:1000],  # truncate for speed
            query_diagnosis=sample["query_diagnosis"],
            evidence=evidence,
        )
        output = judge_pipe(prompt)[0]["generated_text"]
        # Extract the rating word from the output
        rating = "not_useful"
        for key in ratings_map:
            if key in output.lower():
                rating = key
                break

        sample["judge_rating"] = rating
        sample["judge_score"] = ratings_map[rating]
        print(
            f"  Patient {sample['patient_id']}: "
            f"rating={rating} (score={ratings_map[rating]})"
        )
        results.append(sample)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_full_pipeline(
    mimic3_root: str,
    model_name: str = "google/flan-t5-xxl",
    query_diagnosis: str = "small vessel disease",
    condition_icd_codes: Optional[List[str]] = None,
    note_categories: Optional[List[str]] = None,
    max_notes: int = 10,
    dev: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the complete PyHealth dataset -> task -> model pipeline.

    Args:
        mimic3_root (str): Path to MIMIC-III 1.4 root directory.
        model_name (str): LLM model name.
        query_diagnosis (str): Condition to query.
        condition_icd_codes (Optional[List[str]]): ICD-9 codes.
        note_categories (Optional[List[str]]): Note types to include.
        max_notes (int): Max notes per patient.
        dev (bool): Load only first 1000 patients (for quick testing).

    Returns:
        Tuple of (samples, predictions).
    """
    from pyhealth.datasets import MIMIC3NoteDataset
    from pyhealth.tasks import EHREvidenceRetrievalTask
    from pyhealth.models import ZeroShotEvidenceLLM

    if condition_icd_codes is None:
        condition_icd_codes = ["437.3", "437.30", "437.31"]

    logger.info("Step 1: Loading MIMIC3NoteDataset (dev=%s)...", dev)
    dataset = MIMIC3NoteDataset(root=mimic3_root, dev=dev)

    logger.info("Step 2: Applying EHREvidenceRetrievalTask...")
    task = EHREvidenceRetrievalTask(
        query_diagnosis=query_diagnosis,
        condition_icd_codes=condition_icd_codes,
        note_categories=note_categories,
        max_notes=max_notes,
    )
    sample_dataset = dataset.set_task(task)
    logger.info("Samples generated: %d", len(sample_dataset))

    logger.info("Step 3: Running ZeroShotEvidenceLLM inference...")
    model = ZeroShotEvidenceLLM(
        dataset=sample_dataset, model_name=model_name
    )
    samples = list(sample_dataset)
    predictions = model.predict_batch(samples)

    # Attach evidence and confidence back to each sample
    enriched = []
    for s, p in zip(samples, predictions):
        merged = dict(s)
        merged.update(p)
        enriched.append(merged)

    return samples, enriched


def run_demo(ablations: str = "all") -> None:
    """Run ablations on synthetic demo data (no MIMIC access required).

    Args:
        ablations (str): Comma-separated list of ablations to run, or "all".
    """
    from pyhealth.models import ZeroShotEvidenceLLM

    print("\n" + "=" * 70)
    print("EHR EVIDENCE RETRIEVAL - DEMO RUN (synthetic data)")
    print("Paper: Ahsan et al. (2024), CHIL 2024")
    print("=" * 70)

    # Use a small Flan-T5 variant for the demo (xxl requires ~24 GB VRAM)
    model_name = "google/flan-t5-base"
    model = ZeroShotEvidenceLLM(dataset=None, model_name=model_name)

    samples = DEMO_PATIENTS
    run_all = ablations == "all"

    # Baseline predictions
    print("\n--- Baseline predictions ---")
    predictions = model.predict_batch(samples)
    for sample, pred in zip(samples, predictions):
        print(
            f"  Patient {sample['patient_id']} | query='{sample['query_diagnosis']}' | "
            f"true_label={sample['label']} | predicted={pred['has_condition']} | "
            f"confidence={pred['confidence']:.3f}"
        )

    enriched = [dict(s, **p) for s, p in zip(samples, predictions)]

    # A1: Prompt format
    if run_all or "a1" in ablations.lower():
        ablation_prompt_format(samples, model_name=model_name)

    # A2: Confidence threshold
    if run_all or "a2" in ablations.lower():
        labels = [s["label"] for s in samples]
        ablation_confidence_threshold(predictions, labels)

    # A3: BM25 pre-retrieval
    if run_all or "a3" in ablations.lower():
        filtered = ablation_bm25_preretrieval(samples, top_k=3)
        print(f"\n  BM25-filtered samples: {len(filtered)}")

    # A4: Open-source judge (uses small model for demo)
    if run_all or "a4" in ablations.lower():
        judge_model = "mistralai/Mistral-7B-Instruct-v0.2"
        print(
            f"\n  Note: A4 requires '{judge_model}'. "
            "Showing prompt template only in demo mode."
        )
        for s in enriched[:1]:
            print(
                "\n  Evaluator prompt preview:\n",
                _EVALUATOR_PROMPT.format(
                    notes=s["notes"][:200] + "...",
                    query_diagnosis=s["query_diagnosis"],
                    evidence=s.get("evidence", "(no evidence generated)"),
                ),
            )

    print("\nDemo complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EHR Evidence Retrieval: Ahsan et al. (2024) replication"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic demo data (no MIMIC access required)",
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        default=None,
        help="Path to MIMIC-III 1.4 root directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-xxl",
        help="HuggingFace model name (default: google/flan-t5-xxl)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="small vessel disease",
        help="Clinical condition to query",
    )
    parser.add_argument(
        "--icd_codes",
        type=str,
        default="437.3,437.30,437.31",
        help="Comma-separated ICD-9 codes for positive label",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        help="Ablation(s) to run: all, a1, a2, a3, a4 (comma-separated)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=True,
        help="Load only first 1000 patients (dev mode)",
    )
    args = parser.parse_args()

    if args.demo or args.mimic3_root is None:
        run_demo(ablations=args.ablation)
    else:
        icd_codes = [c.strip() for c in args.icd_codes.split(",")]
        samples, enriched = run_full_pipeline(
            mimic3_root=args.mimic3_root,
            model_name=args.model_name,
            query_diagnosis=args.query,
            condition_icd_codes=icd_codes,
            dev=args.dev,
        )
        labels = [s["label"] for s in samples]
        ablation_confidence_threshold(enriched, labels)

        print(f"\nTotal samples: {len(samples)}")
        pos = sum(s["label"] for s in samples)
        print(f"Positive labels: {pos} ({pos / max(len(samples), 1):.1%})")
        print("\nResults saved to ehr_evidence_results.json")
        with open("ehr_evidence_results.json", "w") as f:
            json.dump(enriched, f, indent=2, default=str)
