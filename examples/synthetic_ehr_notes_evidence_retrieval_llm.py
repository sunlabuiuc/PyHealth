"""Example and ablation for LLM-based EHR evidence retrieval.

Paper:
    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Paper link:
    https://proceedings.mlr.press/v248/ahsan24a.html

This script exercises the full pipeline end-to-end on the bundled
:class:`SyntheticEHRNotesDataset`. It then runs the sequential-vs
single-prompt ablation described in section 6 of the proposal, as well
as the LLM-vs-IR-baseline comparison. No network access is required
because both the LLM backend and the IR encoder default to offline
stubs; swap them in-place to evaluate against real models.

Usage:
    python examples/synthetic_ehr_notes_evidence_retrieval_llm.py

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import tempfile
from typing import Dict, Iterable, List

from pyhealth.datasets import SyntheticEHRNotesDataset
from pyhealth.models import (
    CBERTLiteRetriever,
    LLMEvidenceRetriever,
    LLMRetrieverConfig,
    StubLLMBackend,
)
from pyhealth.tasks import EvidenceRetrievalMIMIC3


def _binary_metrics(
    predictions: Iterable[int], labels: Iterable[int]
) -> Dict[str, float]:
    """Return accuracy, precision, recall, and false-positive count.

    Args:
        predictions: Iterable of predicted binary decisions.
        labels: Iterable of ground-truth binary labels.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``, ``fp``.
    """
    preds = list(predictions)
    targets = list(labels)
    assert len(preds) == len(targets)
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 0)
    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fp": float(fp),
    }


def _samples_as_lists(sample_dataset) -> Dict[str, List]:
    """Materialize a ``SampleDataset`` into parallel python lists."""
    out = {
        "note_text": [],
        "condition": [],
        "note_id": [],
        "is_positive": [],
    }
    for sample in sample_dataset:
        out["note_text"].append(sample["note_text"])
        out["condition"].append(sample["condition"])
        out["note_id"].append(sample.get("note_id", ""))
        out["is_positive"].append(int(sample["is_positive"]))
    return out


def run_pipeline() -> None:
    """Load the dataset, apply the task, run both retrievers, report."""
    with tempfile.TemporaryDirectory() as data_root, \
            tempfile.TemporaryDirectory() as cache:
        print("=" * 70)
        print("Step 1: load the synthetic EHR notes dataset")
        print("=" * 70)
        dataset = SyntheticEHRNotesDataset(root=data_root, cache_dir=cache)
        print(f"  patients: {len(dataset.unique_patient_ids)}")
        print(f"  conditions: {dataset.conditions}")

        print()
        print("=" * 70)
        print("Step 2: apply the evidence-retrieval task")
        print("=" * 70)
        task = EvidenceRetrievalMIMIC3()
        samples = dataset.set_task(task)
        try:
            data = _samples_as_lists(samples)
        finally:
            samples.close()
        print(f"  samples generated: {len(data['note_text'])}")
        print(f"  positive labels: {sum(data['is_positive'])}")
        print(
            f"  negative labels: {len(data['is_positive']) - sum(data['is_positive'])}"
        )

        print()
        print("=" * 70)
        print("Step 3: ablation — sequential vs single-prompt LLM retriever")
        print("=" * 70)
        sequential = LLMEvidenceRetriever(
            backend=StubLLMBackend(),
            config=LLMRetrieverConfig(prompt_style="sequential"),
        )
        single = LLMEvidenceRetriever(
            backend=StubLLMBackend(),
            config=LLMRetrieverConfig(prompt_style="single"),
        )

        seq_snips = sequential.retrieve_evidence_batch(
            data["note_text"], data["condition"], data["note_id"]
        )
        single_snips = single.retrieve_evidence_batch(
            data["note_text"], data["condition"], data["note_id"]
        )
        seq_preds = [1 if s.decision == "yes" else 0 for s in seq_snips]
        single_preds = [1 if s.decision == "yes" else 0 for s in single_snips]

        seq_metrics = _binary_metrics(seq_preds, data["is_positive"])
        single_metrics = _binary_metrics(single_preds, data["is_positive"])

        _print_metrics("sequential prompting", seq_metrics)
        _print_metrics("single-prompt ablation", single_metrics)

        seq_generated = sum(1 for s in seq_snips if s.is_generated)
        single_generated = sum(1 for s in single_snips if s.is_generated)
        print(
            f"  sequential explanations generated: {seq_generated}/{len(seq_snips)}"
        )
        print(
            f"  single-prompt explanations generated: "
            f"{single_generated}/{len(single_snips)}"
        )

        print()
        print("=" * 70)
        print("Step 4: LLM vs CBERT-lite IR baseline — note-level decision")
        print("=" * 70)
        baseline = CBERTLiteRetriever(top_k=2)
        baseline_outputs = baseline(
            note_text=data["note_text"],
            condition=data["condition"],
            note_id=data["note_id"],
        )
        baseline_probs = baseline_outputs["y_prob"].squeeze(-1).tolist()
        baseline_preds = [1 if p >= 0.5 else 0 for p in baseline_probs]
        baseline_metrics = _binary_metrics(baseline_preds, data["is_positive"])
        _print_metrics("CBERT-lite IR baseline", baseline_metrics)

        print()
        print("=" * 70)
        print("Step 5: ablation — note-length budget sweep (PyHealth-specific)")
        print("=" * 70)
        # This axis is *not* in the paper: it quantifies how aggressively
        # PyHealth users can truncate notes at ingest time without losing
        # evidence-retrieval accuracy. It is directly useful for anyone
        # deploying the retriever against long MIMIC-III discharge summaries
        # with a token budget.
        for budget in (80, 160, 320, 4000):
            trimmed = LLMEvidenceRetriever(
                backend=StubLLMBackend(),
                config=LLMRetrieverConfig(
                    prompt_style="sequential", max_note_chars=budget
                ),
            )
            snips = trimmed.retrieve_evidence_batch(
                data["note_text"], data["condition"], data["note_id"]
            )
            preds = [1 if s.decision == "yes" else 0 for s in snips]
            metrics = _binary_metrics(preds, data["is_positive"])
            _print_metrics(f"max_note_chars={budget}", metrics)

        print()
        print("=" * 70)
        print("Step 6: qualitative examples")
        print("=" * 70)
        for snippet in seq_snips[:3]:
            print(f"  note={snippet.note_id} condition={snippet.condition}")
            print(f"    decision: {snippet.decision}")
            print(f"    role: {snippet.role}")
            print(f"    explanation: {snippet.explanation}")
            print(f"    source sentence: {snippet.source_sentence}")
            print()


def _print_metrics(label: str, metrics: Dict[str, float]) -> None:
    """Helper — pretty-print a metrics dict."""
    print(
        f"  {label:30s}  acc={metrics['accuracy']:.3f}  "
        f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
        f"fp={int(metrics['fp'])}"
    )


if __name__ == "__main__":
    run_pipeline()
