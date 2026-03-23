"""Multitask Prompted Fine-tuning (MPF) clinical prediction on FHIR timelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from pyhealth.datasets.mimic4_fhir import (
    ConceptVocab,
    FHIRPatient,
    build_cehr_sequences,
    ensure_special_tokens,
    infer_mortality_label,
)

from .base_task import BaseTask


def _pad_int(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq + [pad] * (max_len - len(seq))


def _pad_float(seq: List[float], max_len: int, pad: float = 0.0) -> List[float]:
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq + [pad] * (max_len - len(seq))


class MPFClinicalPredictionTask(BaseTask):
    """Binary mortality prediction from FHIR CEHR sequences with optional MPF tokens.

    The task consumes :class:`~pyhealth.datasets.mimic4_fhir.FHIRPatient` rows produced
    by :class:`~pyhealth.datasets.MIMIC4FHIRDataset` (not ``pyhealth.data.Patient``).
    For disk loads, assign ``task.vocab`` from the dataset before
    :meth:`~pyhealth.datasets.MIMIC4FHIRDataset.gather_samples`; :meth:`set_task`
    performs that wiring automatically.

    Attributes:
        max_len: Truncated sequence length (must be >= 2 for boundary tokens).
        use_mpf: If True, use ``<mor>`` / ``<reg>`` specials; else ``<cls>`` / ``<reg>``.
        vocab: Shared concept vocabulary (usually the dataset's vocab).
    """

    task_name: str = "MPFClinicalPredictionFHIR"
    input_schema: Dict[str, Any] = {
        "concept_ids": ("tensor", {"dtype": torch.long}),
        "token_type_ids": ("tensor", {"dtype": torch.long}),
        "time_stamps": "tensor",
        "ages": "tensor",
        "visit_orders": ("tensor", {"dtype": torch.long}),
        "visit_segments": ("tensor", {"dtype": torch.long}),
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, max_len: int = 512, use_mpf: bool = True) -> None:
        if max_len < 2:
            raise ValueError("max_len must be >= 2 for MPF boundary tokens")
        self.max_len = max_len
        self.use_mpf = use_mpf
        self.vocab: Optional[ConceptVocab] = None
        self._specials: Optional[Dict[str, int]] = None

    def _ensure_vocab(self) -> ConceptVocab:
        if self.vocab is None:
            self.vocab = ConceptVocab()
        if self._specials is None:
            self._specials = ensure_special_tokens(self.vocab)
        return self.vocab

    def __call__(self, patient: FHIRPatient) -> List[Dict[str, Any]]:
        """Build one labeled sample dict per patient (empty list if no tokens).

        Args:
            patient: Grouped FHIR resources for a single logical patient id.

        Returns:
            A one-element list with ``concept_ids``, tensor-ready feature lists, and
            ``label`` (0/1), or ``[]`` if the CEHR sequence is empty after parsing.
        """
        vocab = self._ensure_vocab()
        (
            concept_ids,
            token_types,
            time_stamps,
            ages,
            visit_orders,
            visit_segments,
        ) = build_cehr_sequences(patient, vocab, self.max_len)

        if not concept_ids:
            return []

        ml = self.max_len
        concept_ids = _pad_int(concept_ids, ml, vocab.pad_id)
        token_types = _pad_int(token_types, ml, 0)
        time_stamps = _pad_float(time_stamps, ml, 0.0)
        ages = _pad_float(ages, ml, 0.0)
        visit_orders = _pad_int(visit_orders, ml, 0)
        visit_segments = _pad_int(visit_segments, ml, 0)

        assert self._specials is not None
        if self.use_mpf:
            concept_ids[0] = self._specials["<mor>"]
            concept_ids[-1] = self._specials["<reg>"]
        else:
            concept_ids[0] = self._specials["<cls>"]
            concept_ids[-1] = self._specials["<reg>"]

        label = infer_mortality_label(patient)
        return [
            {
                "patient_id": patient.patient_id,
                "visit_id": f"{patient.patient_id}-0",
                "concept_ids": concept_ids,
                "token_type_ids": token_types,
                "time_stamps": time_stamps,
                "ages": ages,
                "visit_orders": visit_orders,
                "visit_segments": visit_segments,
                "label": label,
            }
        ]
