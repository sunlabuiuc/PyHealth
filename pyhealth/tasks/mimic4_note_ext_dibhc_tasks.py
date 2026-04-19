# -*- coding: utf-8 -*-
"""Tasks for MIMIC-IV-Note patient summary generation and hallucination detection.

This module implements two PyHealth tasks derived from:

    Hegselmann et al. "A Data-Centric Approach To Generate Faithful and
    High Quality Patient Summaries with Large Language Models."
    Conference on Health, Inference, and Learning (CHIL) 2024.
    https://arxiv.org/abs/2402.15422

Both tasks operate on the MIMIC-IV-Note-Ext-DI-BHC dataset, where the
Brief Hospital Course (BHC) serves as input context and the Discharge
Instructions (DI) serve as the target patient summary.

Tasks defined here:

1. class:BHCSummarizationTask - Section 4.1 of the paper.
   Given a BHC, generate a patient-friendly discharge summary.
   Paper results: LED-large achieves ROUGE-1 43.82, GPT-4 0-shot 38.26.

2. class:HallucinationDetectionTask - Section 4.7 of the paper.
   Given a (BHC, summary) pair, predict whether the summary contains
   hallucinations (unsupported facts). Paper results: GPT-4 achieves
   F1 ~20% at span level; MedCat baseline achieves ~10% F1.

Example:
    >>> from pyhealth.datasets import MIMIC4NoteExtDIBHCDataset
    >>> from pyhealth.tasks import BHCSummarizationTask
    >>> from pyhealth.tasks import HallucinationDetectionTask
    >>> dataset = MIMIC4NoteExtDIBHCDataset(root="/path/to/data/")
    >>> summ_samples = dataset.set_task(BHCSummarizationTask())
    >>> halluc_samples = dataset.set_task(HallucinationDetectionTask())
    >>> print(summ_samples[0]["context"][:60])
    >>> print(halluc_samples[0]["label"])
"""

from typing import Any, Dict, List

from pyhealth.tasks import BaseTask


class BHCSummarizationTask(BaseTask):
    """Patient summary generation from Brief Hospital Course text.

    Given a Brief Hospital Course (BHC) as input context, generates a
    patient-friendly Discharge Instructions (DI) summary. This implements
    the summarization task from Section 4.1 of Hegselmann et al. (CHIL
    2024), where BHC is used as the context and DI as the target summary.

    The paper demonstrates that fine-tuned LED-large achieves ROUGE-1 of
    43.82 on this task, while GPT-4 0-shot achieves 38.26. We reproduce
    this using BART-large as a free open-source alternative.

    Input schema:
        - context (str): Brief Hospital Course text — the clinical
          notes written by medical staff summarizing the hospital stay.

    Output schema:
        - summary (str): Target patient-facing discharge instructions
          written in plain language for patient comprehension.

    Note:
        This task produces raw text targets for seq2seq generation models.
        Evaluate generated summaries using ROUGE and BERTScore metrics.
        Average context length is ~552 words; average summary ~113 words
        per paper Table 6.

    Example:
        >>> from pyhealth.datasets import MIMIC4NoteExtDIBHCDataset
        >>> from pyhealth.tasks import BHCSummarizationTask
        >>> dataset = MIMIC4NoteExtDIBHCDataset(root="/path/to/data/")
        >>> samples = dataset.set_task(BHCSummarizationTask())
        >>> print(samples[0]["context"][:80])
        Brief Hospital Course: Patient presented with chest pain...
        >>> print(samples[0]["summary"][:80])
        You were admitted to the hospital for chest pain...
    """

    task_name: str = "BHCSummarizationMIMIC4Note"

    input_schema: Dict[str, str] = {
        "context": "str",
    }

    output_schema: Dict[str, str] = {
        "summary": "str",
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into summarization samples.

        Iterates over all visits and discharge note events for the patient,
        extracting (BHC context, DI summary) pairs. Each valid pair becomes
        one training sample for sequence-to-sequence generation.

        Args:
            patient (Any): A PyHealth Patient object. Each visit must
                contain discharge note events with brief_hospital_course
                and summary attributes populated by the dataset class.

        Returns:
            List[Dict[str, Any]]: List of sample dicts, one per valid
            discharge note event. Each sample contains:

            - patient_id (str): Patient identifier.
            - visit_id (str): Visit (admission) identifier.
            - context (str): Brief Hospital Course text - model input.
            - summary (str): Discharge instructions - generation target.

        Note:
            Samples with empty context or summary are silently skipped.
            This mirrors the paper's preprocessing pipeline which removes
            records with missing BHC or summaries shorter than 350 chars.

        Example:
            >>> task = BHCSummarizationTask()
            >>> samples = task(patient)
            >>> len(samples)
            1
            >>> "context" in samples[0]
            True
        """
        samples: List[Dict[str, Any]] = []

        for visit in patient.visits.values():
            for event in visit.get_event_list(source_table="discharge"):
                context: str = getattr(
                    event, "brief_hospital_course", ""
                ).strip()
                summary: str = getattr(event, "summary", "").strip()

                if not context or not summary:
                    continue

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit.visit_id,
                        "context": context,
                        "summary": summary,
                    }
                )

        return samples


class HallucinationDetectionTask(BaseTask):
    """Binary hallucination detection for clinical patient summaries.

    Given a Brief Hospital Course (BHC) as grounding context and a
    patient-facing Discharge Instructions (DI) summary, predicts whether
    the summary contains at least one hallucination - a fact not supported
    by the BHC. This implements the automatic hallucination detection task
    from Section 4.7 of Hegselmann et al. (CHIL 2024).

    The paper evaluates hallucination detection at the span level using
    expert-annotated datasets (Hallucinations-MIMIC-DI and
    Hallucinations-Generated-DI, each with 100 examples). We extend this
    to binary document-level classification as a novel contribution.

    Input schema:
        - context (str): Brief Hospital Course - the only ground truth
          source for determining whether summary facts are supported.
        - summary (str): Patient-facing discharge instructions to
          evaluate for hallucinations.

    Output schema:
        - label (binary): 1 if the summary contains at least one
          hallucination span per expert annotation, 0 if faithful.
          Defaults to -1 when no expert annotation is available.

    Args:
        default_label (int): Label assigned when no expert annotation is
            available. Default is -1.

    Example:
        >>> from pyhealth.datasets import MIMIC4NoteExtDIBHCDataset
        >>> from pyhealth.tasks import HallucinationDetectionTask
        >>> dataset = MIMIC4NoteExtDIBHCDataset(root="/path/to/data/")
        >>> samples = dataset.set_task(HallucinationDetectionTask())
        >>> print(samples[0]["label"])
        -1
    """

    task_name: str = "HallucinationDetectionMIMIC4Note"

    input_schema: Dict[str, str] = {
        "context": "str",
        "summary": "str",
    }

    output_schema: Dict[str, str] = {
        "label": "binary",
    }

    def __init__(self, default_label: int = -1) -> None:
        self.default_label = default_label

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into hallucination detection samples.

        Iterates over all visits and discharge note events, extracting
        (BHC context, DI summary) pairs with binary hallucination labels
        where expert annotations are available.

        Args:
            patient (Any): A PyHealth Patient object. Each visit must
                contain discharge note events with brief_hospital_course
                and summary attributes. Optionally, events may include
                a has_hallucination attribute (0 or 1) from expert
                annotation via the PhysioNet ann-pt-summ dataset.

        Returns:
            List[Dict[str, Any]]: List of sample dicts, one per valid
            discharge note event. Each sample contains:

            - patient_id (str): Patient identifier.
            - visit_id (str): Visit identifier.
            - context (str): Brief Hospital Course text.
            - summary (str): Patient discharge instructions.
            - label (int): 1 hallucination present, 0 faithful,
              -1 annotation unavailable.

        Note:
            Samples with empty context or summary are silently skipped.

        Example:
            >>> task = HallucinationDetectionTask()
            >>> samples = task(patient)
            >>> samples[0]["label"] in (-1, 0, 1)
            True
        """
        samples: List[Dict[str, Any]] = []

        for visit in patient.visits.values():
            for event in visit.get_event_list(source_table="discharge"):
                context: str = getattr(
                    event, "brief_hospital_course", ""
                ).strip()
                summary: str = getattr(event, "summary", "").strip()

                if not context or not summary:
                    continue

                label: int = int(
                    getattr(event, "has_hallucination", self.default_label)
                )

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit.visit_id,
                        "context": context,
                        "summary": summary,
                        "label": label,
                    }
                )

        return samples
