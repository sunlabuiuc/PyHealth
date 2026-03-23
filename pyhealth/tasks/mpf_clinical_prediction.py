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


def _left_pad_int(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    if len(seq) > max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def _left_pad_float(seq: List[float], max_len: int, pad: float = 0.0) -> List[float]:
    if len(seq) > max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


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
        """Build one labeled sample dict per patient.

        Args:
            patient: Grouped FHIR resources for a single logical patient id.

        Returns:
            A one-element list with ``concept_ids``, tensor-ready feature lists, and
            ``label`` (0/1). Boundary tokens are always included; when
            ``max_len == 2`` the sequence is ``<mor>``/``<cls>`` and ``<reg>`` only.
        """
        vocab = self._ensure_vocab()
        clinical_cap = max(0, self.max_len - 2)
        (
            concept_ids,
            token_types,
            time_stamps,
            ages,
            visit_orders,
            visit_segments,
        ) = build_cehr_sequences(patient, vocab, clinical_cap)

        assert self._specials is not None
        mor_id = self._specials["<mor>"] if self.use_mpf else self._specials["<cls>"]
        reg_id = self._specials["<reg>"]
        z0 = 0
        zf = 0.0
        concept_ids = [mor_id] + concept_ids + [reg_id]
        token_types = [z0] + token_types + [z0]
        time_stamps = [zf] + time_stamps + [zf]
        ages = [zf] + ages + [zf]
        visit_orders = [z0] + visit_orders + [z0]
        visit_segments = [z0] + visit_segments + [z0]

        ml = self.max_len
        concept_ids = _left_pad_int(concept_ids, ml, vocab.pad_id)
        token_types = _left_pad_int(token_types, ml, 0)
        time_stamps = _left_pad_float(time_stamps, ml, 0.0)
        ages = _left_pad_float(ages, ml, 0.0)
        visit_orders = _left_pad_int(visit_orders, ml, 0)
        visit_segments = _left_pad_int(visit_segments, ml, 0)

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
