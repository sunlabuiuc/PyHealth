"""ReXPref-Prior report editing task for MIMIC-CXR.

Author: Rishi Gopalan
Paper: Direct Preference Optimization for Suppressing Hallucinated Prior
    Exams in Radiology Report Generation (Banerjee et al., 2024)
Paper link: https://arxiv.org/abs/2406.06496
Description: Join ReXPref-Prior sentence edits to PyHealth's supported
    MIMIC-CXR dataset and create sentence-level edit-prediction samples.

The task joins sentence-level ReXPref-Prior edits to
:class:`~pyhealth.datasets.MIMIC4CXRDataset` metadata events by ``study_id``.
It reconstructs the original (rejected) and edited (chosen) report and emits
one binary-classification sample per original sentence.  The label says
whether ReXPref-Prior rewrote or deleted that sentence; it does not assert
that the sentence contains a hallucination.

ReXPref-Prior and MIMIC-CXR are credentialed datasets.  They are supplied by
the caller and are never downloaded by this module.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import polars as pl

from ..data import Patient
from .base_task import BaseTask

ContextMode = Literal["sentence_only", "neighbor_context", "section_context"]

_CONTEXT_MODES = ("sentence_only", "neighbor_context", "section_context")
_RELEASE_SPLITS = ("train", "val", "test")
_SECTIONS = ("findings", "impression")
_REQUIRED_COLUMNS = {
    "study_id",
    "sentence_id",
    "new_sentence",
    "orig_sentence",
    "section",
}


def _normalize_study_id(value: Any) -> str:
    """Normalizes the numeric MIMIC-CXR study identifier."""
    study_id = str(value).strip()
    if len(study_id) > 1 and study_id[0].lower() == "s" and study_id[1:].isdigit():
        study_id = study_id[1:]
    if not study_id or not study_id.isdigit():
        raise ValueError("study_id must be a non-empty numeric identifier")
    return study_id


@dataclass(frozen=True)
class _SentenceEdit:
    study_id: str
    split: str
    section: str
    sentence_id: int
    original: str
    edited: str | None

    @property
    def edit_type(self) -> str:
        if self.edited is None:
            return "deleted"
        if self.edited == self.original:
            return "unchanged"
        return "rewritten"

    @property
    def needs_edit(self) -> int:
        return int(self.edit_type != "unchanged")


class _ReXPrefAdapter:
    """Validated, picklable index of local ReXPref-Prior CSV files."""

    def __init__(self, annotation_files: Mapping[str, str | Path]) -> None:
        if not annotation_files:
            raise ValueError("annotation_files must contain at least one split")

        unknown_splits = set(annotation_files) - set(_RELEASE_SPLITS)
        if unknown_splits:
            raise ValueError(
                "annotation_files keys must be train, val, or test; got "
                f"{sorted(unknown_splits)}"
            )

        paths: dict[str, Path] = {}
        sources = []
        for split in _RELEASE_SPLITS:
            if split not in annotation_files:
                continue
            path = Path(annotation_files[split]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"ReXPref-Prior file does not exist: {path}")
            stat = path.stat()
            paths[split] = path
            sources.append((split, str(path), stat.st_size, stat.st_mtime_ns))

        self.sources = tuple(sources)
        self.by_study = self._load(paths)

    def __repr__(self) -> str:
        return f"_ReXPrefAdapter(sources={self.sources!r})"

    @staticmethod
    def _load(paths: Mapping[str, Path]) -> dict[str, tuple[_SentenceEdit, ...]]:
        by_study: dict[str, list[_SentenceEdit]] = defaultdict(list)
        study_splits: dict[str, str] = {}
        seen_keys: set[tuple[str, str, int]] = set()

        for split in _RELEASE_SPLITS:
            path = paths.get(split)
            if path is None:
                continue
            with path.open(newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                columns = set(reader.fieldnames or [])
                missing = sorted(_REQUIRED_COLUMNS - columns)
                if missing:
                    raise ValueError(
                        f"{path.name} is missing required columns: {missing}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    try:
                        study_id = _normalize_study_id(row["study_id"])
                        section = row["section"].strip().lower()
                        if section not in _SECTIONS:
                            raise ValueError(
                                f"section must be one of {_SECTIONS}, got {section!r}"
                            )
                        sentence_id = int(row["sentence_id"].strip())
                        if sentence_id < 0:
                            raise ValueError("sentence_id must be non-negative")
                        original = row["orig_sentence"]
                        if not original.strip():
                            raise ValueError("orig_sentence must not be blank")
                        new_sentence = row["new_sentence"]
                        edited = None if not new_sentence.strip() else new_sentence
                    except (AttributeError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"Invalid ReXPref-Prior row {row_number} in {path.name}: "
                            f"{error}"
                        ) from error

                    previous_split = study_splits.setdefault(study_id, split)
                    if previous_split != split:
                        raise ValueError(
                            "A study_id appears in more than one annotation split"
                        )
                    key = (study_id, section, sentence_id)
                    if key in seen_keys:
                        raise ValueError(
                            "Duplicate (study_id, section, sentence_id) annotation"
                        )
                    seen_keys.add(key)
                    by_study[study_id].append(
                        _SentenceEdit(
                            study_id=study_id,
                            split=split,
                            section=section,
                            sentence_id=sentence_id,
                            original=original,
                            edited=edited,
                        )
                    )

        section_order = {section: index for index, section in enumerate(_SECTIONS)}
        return {
            study_id: tuple(
                sorted(
                    edits,
                    key=lambda edit: (
                        section_order[edit.section],
                        edit.sentence_id,
                    ),
                )
            )
            for study_id, edits in by_study.items()
        }


def _report(edits: tuple[_SentenceEdit, ...], *, chosen: bool) -> str:
    """Reconstructs a report with stable section and sentence ordering."""
    output = []
    for section in _SECTIONS:
        section_edits = [edit for edit in edits if edit.section == section]
        if not section_edits:
            continue
        sentences = [edit.edited if chosen else edit.original for edit in section_edits]
        sentences = [sentence for sentence in sentences if sentence is not None]
        body = "\n".join(sentences)
        output.append(f"{section.upper()}:" + (f"\n{body}" if body else ""))
    return "\n".join(output)


def _context_text(
    edits: tuple[_SentenceEdit, ...], index: int, mode: ContextMode
) -> str:
    """Formats one target sentence and its optional original-report context."""
    target = edits[index]
    output = [f"Target sentence:\n{target.original}"]
    if mode == "sentence_only":
        return output[0]

    section_edits = [edit for edit in edits if edit.section == target.section]
    section_index = section_edits.index(target)
    if mode == "neighbor_context":
        if section_index > 0:
            output.append(
                f"Previous sentence:\n{section_edits[section_index - 1].original}"
            )
        if section_index + 1 < len(section_edits):
            output.append(
                f"Next sentence:\n{section_edits[section_index + 1].original}"
            )
        return "\n\n".join(output)

    context = [
        (f"[TARGET] {edit.original} [/TARGET]" if edit is target else edit.original)
        for edit in section_edits
    ]
    output.append(f"{target.section.title()} context:\n" + "\n".join(context))
    return "\n\n".join(output)


class PriorExamPreferenceTask(BaseTask):
    """Predicts whether ReXPref-Prior edited a MIMIC-CXR report sentence.

    The task must be applied to a
    :class:`~pyhealth.datasets.MIMIC4CXRDataset` containing its ``metadata``
    table.  It joins local ReXPref-Prior annotations to those events by
    ``study_id``, reconstructs the original and edited report, and returns one
    sample per original sentence.

    Args:
        annotation_files: Mapping from release split (``train``, ``val`` or
            ``test``) to a local ReXPref-Prior CSV path.  Files must contain
            ``study_id``, ``section``, ``sentence_id``, ``orig_sentence`` and
            ``new_sentence``.  They are read locally and are not downloaded.
        context_mode: Text given to the model: the target sentence alone, with
            adjacent sentences in its section, or with its full section.

    Note:
        ``needs_edit`` means that ReXPref-Prior rewrote or deleted the sentence.
        It is not a label for whether a prior-exam statement was hallucinated.

    Examples:
        >>> from pyhealth.datasets import MIMIC4CXRDataset
        >>> from pyhealth.tasks import PriorExamPreferenceTask
        >>> dataset = MIMIC4CXRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-cxr-jpg/2.1.0",
        ...     tables=["metadata"],
        ... )
        >>> task = PriorExamPreferenceTask(  # doctest: +SKIP
        ...     {"train": "/path/to/train_rexprefprior.csv"},
        ...     context_mode="sentence_only",
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "PriorExamPreferenceTask"
    input_schema: ClassVar[dict[str, str]] = {"text": "text"}
    output_schema: ClassVar[dict[str, str]] = {"needs_edit": "binary"}

    def __init__(
        self,
        annotation_files: Mapping[str, str | Path],
        context_mode: ContextMode = "sentence_only",
    ) -> None:
        super().__init__()
        if context_mode not in _CONTEXT_MODES:
            raise ValueError(
                f"context_mode must be one of {_CONTEXT_MODES}, got {context_mode!r}"
            )
        self.context_mode = context_mode
        self._annotations = _ReXPrefAdapter(annotation_files)

    @property
    def annotation_files(self) -> dict[str, str]:
        """Returns the configured split-to-file mapping."""
        return {split: path for split, path, _, _ in self._annotations.sources}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps only annotated MIMIC-CXR metadata events."""
        columns = set(df.collect_schema().names())
        required = {"event_type", "metadata/study_id", "metadata/image_path"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                "PriorExamPreferenceTask requires the MIMIC-CXR metadata table; "
                f"missing columns: {missing}"
            )
        return df.filter(
            (pl.col("event_type") == "metadata")
            & pl.col("metadata/study_id").is_in(sorted(self._annotations.by_study))
        )

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        studies: dict[str, list[Any]] = defaultdict(list)
        for event in patient.get_events(event_type="metadata"):
            if "study_id" not in event or event.study_id is None:
                continue
            study_id = _normalize_study_id(event.study_id)
            if study_id in self._annotations.by_study:
                studies[study_id].append(event)

        matched_splits = {
            self._annotations.by_study[study_id][0].split for study_id in studies
        }
        if len(matched_splits) > 1:
            raise ValueError(
                "A patient has annotated studies in more than one release split"
            )

        samples = []
        for study_id in sorted(studies, key=int):
            edits = self._annotations.by_study[study_id]
            events = sorted(
                studies[study_id],
                key=lambda event: (
                    str(getattr(event, "dicom_id", "") or ""),
                    str(getattr(event, "image_path", "") or ""),
                ),
            )
            image_paths = list(
                dict.fromkeys(
                    str(event.image_path)
                    for event in events
                    if getattr(event, "image_path", None)
                )
            )
            dicom_ids = list(
                dict.fromkeys(
                    str(event.dicom_id)
                    for event in events
                    if getattr(event, "dicom_id", None)
                )
            )
            chosen_report = _report(edits, chosen=True)
            rejected_report = _report(edits, chosen=False)
            edit_counts = dict(Counter(edit.edit_type for edit in edits))

            for index, edit in enumerate(edits):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": (f"{study_id}:{edit.section}:{edit.sentence_id}"),
                        "study_id": study_id,
                        "section": edit.section,
                        "sentence_id": edit.sentence_id,
                        "text": _context_text(edits, index, self.context_mode),
                        "needs_edit": edit.needs_edit,
                        "edit_type": edit.edit_type,
                        "original_sentence": edit.original,
                        "edited_sentence": edit.edited,
                        "rejected_report": rejected_report,
                        "chosen_report": chosen_report,
                        "edit_counts": edit_counts,
                        "image_paths": image_paths,
                        "dicom_ids": dicom_ids,
                        "split": edit.split,
                    }
                )
        return samples
