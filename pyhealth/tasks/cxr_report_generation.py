"""
CXR report generation task (MIMIC-IV unified dataset w/ CXR + radiology notes).

Pairs chest X-ray images with a target text section (e.g., FINDINGS) extracted from
de-identified radiology reports.

Notes:
- Uses MIMIC-style [** ... **] tag normalization and section parsing adapted from the
  user's report_parser.py.
- Designed to work with MIMIC4Dataset unified mode (note_root + cxr_root).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import polars as pl

from .base_task import BaseTask


class _MIMICRe:
    """Tiny regex helper (adapted from report_parser.py)."""

    def __init__(self) -> None:
        self._cached: Dict[int, re.Pattern] = {}

    def get(self, pattern: str, flags: int = 0) -> re.Pattern:
        key = hash((pattern, flags))
        if key not in self._cached:
            self._cached[key] = re.compile(pattern, flags=flags)
        return self._cached[key]

    def sub(self, pattern: str, repl: str, string: str, flags: int = 0) -> str:
        return self.get(pattern, flags=flags).sub(repl, string)

    def rm(self, pattern: str, string: str, flags: int = 0) -> str:
        return self.sub(pattern, "", string, flags=flags)

    def get_id(self, tag: str, flags: int = 0) -> re.Pattern:
        return self.get(r"\[\*\*.*{:s}.*?\*\*\]".format(tag), flags=flags)

    def sub_id(self, tag: str, repl: str, string: str, flags: int = 0) -> str:
        return self.get_id(tag, flags=flags).sub(repl, string)


def _parse_report_text(report_text: str) -> Dict[str, str]:
    """
    Parse a de-identified MIMIC-style radiology report into sections.
    Returns a dict like {"findings": "...", "impression": "..."} when present.
    """
    mimic_re = _MIMICRe()
    report = (report_text or "").lower()

    # Normalize common [** ... **] tags (adapted from report_parser.py)
    report = mimic_re.sub_id(r"(?:location|address|university|country|state|unit number)", "LOC", report)
    report = mimic_re.sub_id(r"(?:year|month|day|date)", "DATE", report)
    report = mimic_re.sub_id(r"(?:hospital)", "HOSPITAL", report)
    report = mimic_re.sub_id(
        r"(?:identifier|serial number|medical record number|social security number|md number)", "ID", report
    )
    report = mimic_re.sub_id(r"(?:age)", "AGE", report)
    report = mimic_re.sub_id(r"(?:phone|pager number|contact info|provider number)", "PHONE", report)
    report = mimic_re.sub_id(r"(?:name|initial|dictator|attending)", "NAME", report)
    report = mimic_re.sub_id(r"(?:company)", "COMPANY", report)
    report = mimic_re.sub_id(r"(?:clip number)", "CLIP_NUM", report)

    report = mimic_re.sub(
        (
            r"\[\*\*(?:"
            r"\d{4}"
            r"|\d{0,2}[/-]\d{0,2}"
            r"|\d{0,2}[/-]\d{4}"
            r"|\d{0,2}[/-]\d{0,2}[/-]\d{4}"
            r"|\d{4}[/-]\d{0,2}[/-]\d{0,2}"
            r")\*\*\]"
        ),
        "DATE",
        report,
    )
    report = mimic_re.sub(r"\[\*\*.*\*\*\]", "OTHER", report)
    report = mimic_re.sub(r"(?:\d{1,2}:\d{2})", "TIME", report)

    report = mimic_re.rm(r"_{2,}", report, flags=re.MULTILINE)
    report = mimic_re.rm(r"the study and the report were reviewed by the staff radiologist\.", report)

    # Section split: lines like "FINDINGS:" "IMPRESSION:" etc.
    matches = list(mimic_re.get(r"^(?P<title>[ \w()]+):", flags=re.MULTILINE).finditer(report))
    parsed: Dict[str, str] = {}
    for (m, next_m) in zip(matches, matches[1:] + [None]):
        start = m.end()
        end = next_m.start() if next_m else None
        title = (m.group("title") or "").strip().lower()

        paragraph = report[start:end]
        paragraph = mimic_re.sub(r"\s{2,}", " ", paragraph).strip()
        if paragraph:
            parsed[title] = paragraph.replace("\n", "\\n")

    return parsed


def _get_view_position(x: Any) -> Optional[str]:
    for key in ("ViewPosition", "view_position", "viewPosition"):
        v = getattr(x, key, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _build_mimic_cxr_jpg_path(cxr_root: str, subject_id: str, study_id: str, dicom_id: str) -> str:
    """
    Construct a MIMIC-CXR-JPG path:
      <cxr_root>/files/pXX/p<subject_id>/s<study_id>/<dicom_id>.jpg
    where XX are the first two digits of subject_id.
    """
    sid = str(subject_id)
    prefix = sid[:2] if len(sid) >= 2 else sid[:1]
    return os.path.join(cxr_root, "files", f"p{prefix}", f"p{sid}", f"s{study_id}", f"{dicom_id}.jpg")


class CXRReportGenerationMIMIC4(BaseTask):
    """
    Task: chest X-ray report generation (image -> text).

    Output samples include:
      - patient_id, visit_id
      - image (path to .jpg)
      - report (selected section text, e.g. FINDINGS)

    This task expects:
      - x-ray metadata events in event_type="xrays_metadata"
      - radiology note events in event_type="radiology"
    consistent with other MIMIC4 tasks.
    """

    task_name: str = "CXRReportGenerationMIMIC4"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"report": "text"}

    def __init__(
        self,
        cxr_root: Optional[str] = None,
        report_section: str = "findings",
        view_positions: Optional[List[str]] = None,
        require_nonempty_report: bool = True,
    ) -> None:
        """
        Args:
            cxr_root: Root directory for MIMIC-CXR-JPG. If None, task will try to use
                      event.image_path if present; otherwise it will skip samples.
            report_section: Which section to extract ("findings", "impression", or "full").
            view_positions: Optional filter, e.g. ["AP"] to match Boag-style preprocessing.
            require_nonempty_report: Drop samples with empty extracted section.
        """
        self.cxr_root = cxr_root
        self.report_section = (report_section or "findings").strip().lower()
        self.view_positions = [v.strip() for v in (view_positions or []) if v and v.strip()]
        self.require_nonempty_report = require_nonempty_report

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Keep only what we need from the global event dataframe (best-effort).
        if "event_type" not in df.columns:
            return df
        return df.filter(pl.col("event_type").is_in(["admissions", "radiology", "xrays_metadata"]))

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        # Collect radiology notes across the patient; later we try to match by study_id if present.
        radiology_notes = patient.get_events(event_type="radiology")
        study_to_text: Dict[str, str] = {}

        for note in radiology_notes:
            raw = getattr(note, "radiology", None) or getattr(note, "text", None) or getattr(note, "note", None)
            if not isinstance(raw, str) or not raw.strip():
                continue

            parsed = _parse_report_text(raw)
            if self.report_section == "full":
                section_text = raw.strip()
            else:
                section_text = parsed.get(self.report_section, "")

            note_study_id = getattr(note, "study_id", None) or getattr(note, "study", None)
            if note_study_id is not None:
                study_to_text[str(note_study_id)] = section_text

        # Iterate visits; yield one sample per x-ray (best-effort pairing).
        for adm in admissions:
            visit_id = getattr(adm, "hadm_id", None) or getattr(adm, "visit_id", None) or "unknown_visit"

            xrays = patient.get_events(event_type="xrays_metadata")
            if not xrays:
                continue

            for x in xrays:
                study_id = getattr(x, "study_id", None) or getattr(x, "study", None)
                dicom_id = getattr(x, "dicom_id", None) or getattr(x, "dicom", None)

                if study_id is None or dicom_id is None:
                    continue

                vp = _get_view_position(x)
                if self.view_positions and (vp is None or vp not in self.view_positions):
                    continue

                # Resolve report text
                report_text = ""
                if str(study_id) in study_to_text:
                    report_text = study_to_text[str(study_id)]
                else:
                    # Fallback: concatenate all radiology notes if we can't match by study_id
                    if radiology_notes:
                        joined = "\n".join(
                            (getattr(n, "radiology", "") or getattr(n, "text", "") or "").strip()
                            for n in radiology_notes
                        ).strip()
                        parsed = _parse_report_text(joined)
                        report_text = joined if self.report_section == "full" else parsed.get(self.report_section, "")

                if self.require_nonempty_report and not report_text.strip():
                    continue

                # Resolve image path
                image_path = getattr(x, "image_path", None) or getattr(x, "path", None)
                if not image_path:
                    if not self.cxr_root:
                        continue
                    subject_id = getattr(patient, "patient_id", None) or getattr(patient, "subject_id", None)
                    if subject_id is None:
                        continue
                    image_path = _build_mimic_cxr_jpg_path(
                        self.cxr_root, str(subject_id), str(study_id), str(dicom_id)
                    )

                samples.append(
                    {
                        "patient_id": getattr(patient, "patient_id", None) or getattr(patient, "subject_id", None),
                        "visit_id": visit_id,
                        "study_id": str(study_id),
                        "dicom_id": str(dicom_id),
                        "view_position": vp,
                        "image": image_path,
                        "report": report_text,
                    }
                )

        return samples
