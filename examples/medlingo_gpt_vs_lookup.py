
"""
GPT vs Lookup baseline evaluation for MedLingo-style clinical abbreviation interpretation.

This script compares:
1. AbbreviationLookupModel baseline
2. GPT-based abbreviation expansion

Evaluation conditions:
- abbreviation only
- lowercase abbreviation
- punctuation noise
- short clinical context

IMPORTANT:
This script is intended to run only on cleaned, derived benchmark samples
(e.g., test-resources/medlingo_samples.json), not on raw MIMIC notes.

This script requires an OpenAI API key and is an optional secondary modern LLM evaluation conducted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from pyhealth.datasets.medlingo import MedLingoDataset
from pyhealth.models.abbreviation_lookup import AbbreviationLookupModel
from pyhealth.tasks.clinical_abbreviation import ClinicalAbbreviationTask


def normalize_label(text: str) -> str:
    """
    Normalize text for exact-match scoring.

    This removes common punctuation and markdown-like formatting so GPT outputs
    such as '**Shortness of breath**.' still score correctly.
    """
    text = text.strip().lower()
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"[^\w\s/]+", "", text)
    return " ".join(text.split())


def score_prediction(pred: str, gold: str) -> int:
    """Return 1 if prediction matches gold after normalization, else 0."""
    return int(normalize_label(pred) == normalize_label(gold))


def build_prompt(input_text: str, use_context: bool) -> str:
    """
    Build a constrained prompt for GPT evaluation.

    This version explicitly tells the model to identify and expand
    the abbreviation, improving performance under contextual input.
    """
    if use_context:
        return (
            f"Sentence: {input_text}\n"
            "What does the abbreviation in this sentence stand for?\n"
            "Return only the expansion in plain text. "
            "No explanation, no markdown, no punctuation."
        )

    return (
        "Expand the following clinical abbreviation into its medical meaning.\n"
        f"Abbreviation: {input_text}\n"
        "Return only the expansion in plain text. "
        "No explanation, no markdown, no punctuation."
    )


def accuracy_lookup(
    samples: list[dict[str, Any]],
    model: AbbreviationLookupModel,
    task: ClinicalAbbreviationTask,
) -> float:
    """Evaluate lookup baseline accuracy."""
    correct = 0
    total = 0

    for sample in samples:
        processed = task(sample)
        pred = model.predict(processed["input"])
        gold = processed["label"]

        correct += score_prediction(pred, gold)
        total += 1

    return correct / total if total > 0 else 0.0


def query_gpt(
    client: OpenAI,
    prompt: str,
    model_name: str = "gpt-4.1-mini",
) -> str:
    """
    Query GPT and return stripped text output.

    If a request fails, return 'unknown' so the evaluation can continue.
    """
    try:
        response = client.responses.create(
            model=model_name,
            input=prompt,
        )
        return response.output_text.strip()
    except Exception as exc:
        print(f"GPT query failed: {exc}")
        return "unknown"


def accuracy_gpt(
    samples: list[dict[str, Any]],
    task: ClinicalAbbreviationTask,
    client: OpenAI,
    model_name: str = "gpt-4.1-mini",
    max_samples: int | None = None,
) -> float:
    """Evaluate GPT accuracy on the benchmark."""
    correct = 0
    total = 0

    eval_samples = samples[:max_samples] if max_samples is not None else samples

    for sample in eval_samples:
        processed = task(sample)
        prompt = build_prompt(
            input_text=processed["input"],
            use_context=task.use_context,
        )
        pred = query_gpt(client, prompt, model_name=model_name)
        gold = processed["label"]

        correct += score_prediction(pred, gold)
        total += 1

    return correct / total if total > 0 else 0.0


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping GPT example:a OPENAI_API_KEY not found.")
        return

    client = OpenAI(api_key=api_key)

    dataset = MedLingoDataset(root="test-resources")
    records = dataset.process()

    samples = []
    for record in records:
        for s in record["medlingo"]:
            samples.append(s)

    # Baseline model
    lookup_model = AbbreviationLookupModel(normalize=True)
    lookup_model.fit(samples)

    # Tasks
    base_task = ClinicalAbbreviationTask(use_context=False)
    context_task = ClinicalAbbreviationTask(use_context=True)

    lowercase_samples = [{**s, "abbr": s["abbr"].lower()} for s in samples]
    noisy_samples = [{**s, "abbr": s["abbr"] + "!!!"} for s in samples]

    # Keep GPT run small at first for cost/control
    max_gpt_samples = min(10, len(samples))
    print(f"Using {max_gpt_samples} samples for GPT evaluation.\n")

    results = {
        "lookup_base_abbreviation_only": accuracy_lookup(
            samples, lookup_model, base_task
        ),
        "lookup_lowercase_abbreviation": accuracy_lookup(
            lowercase_samples, lookup_model, base_task
        ),
        "lookup_short_clinical_context": accuracy_lookup(
            samples, lookup_model, context_task
        ),
        "lookup_punctuation_noise": accuracy_lookup(
            noisy_samples, lookup_model, base_task
        ),
        "gpt_base_abbreviation_only": accuracy_gpt(
            samples, base_task, client, max_samples=max_gpt_samples
        ),
        "gpt_lowercase_abbreviation": accuracy_gpt(
            lowercase_samples, base_task, client, max_samples=max_gpt_samples
        ),
        "gpt_short_clinical_context": accuracy_gpt(
            samples, context_task, client, max_samples=max_gpt_samples
        ),
        "gpt_punctuation_noise": accuracy_gpt(
            noisy_samples, base_task, client, max_samples=max_gpt_samples
        ),
    }

    print("=== GPT vs Lookup Results ===")
    print(f"{'Condition':30} {'Lookup':>8} {'GPT':>8}")
    print("-" * 50)
    print(
        f"{'Abbreviation only':30} "
        f"{results['lookup_base_abbreviation_only']:.3f} "
        f"{results['gpt_base_abbreviation_only']:.3f}"
    )
    print(
        f"{'Lowercase abbreviation':30} "
        f"{results['lookup_lowercase_abbreviation']:.3f} "
        f"{results['gpt_lowercase_abbreviation']:.3f}"
    )
    print(
        f"{'Short clinical context':30} "
        f"{results['lookup_short_clinical_context']:.3f} "
        f"{results['gpt_short_clinical_context']:.3f}"
    )
    print(
        f"{'Punctuation noise':30} "
        f"{results['lookup_punctuation_noise']:.3f} "
        f"{results['gpt_punctuation_noise']:.3f}"
    )


if __name__ == "__main__":
    main()