"""Dependency-free time and history rules, shared by cohort and PyHealth adapter."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime


def clean_id(value):
    """Normalize an identifier and handle missing values.

    Examples:
        >>> clean_id("105.0")
        '105'
        >>> clean_id(None)
        ''
    """
    text = "" if value is None else str(value).strip()
    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    return re.sub(r"\.0$", "", text)


def parse_time(value):
    """Parse a timestamp into a timezone-naive datetime.

    Examples:
        >>> parse_time("2150-01-01 09:00:00")
        datetime.datetime(2150, 1, 1, 9, 0)
        >>> parse_time("invalid") is None
        True
    """
    if value is None or str(value).strip().lower() in {
        "",
        "none",
        "nan",
        "nat",
        "<na>",
    }:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=None)
    except ValueError:
        return None


def eligible_history(
    notes, report, strict_availability=True, unknown_availability="retain"
):
    """Select historical notes available before the index report.

    Exact chart timestamps must precede the report timestamp.
    Date-only notes must be from a previous calendar day.
    Strict availability excludes notes stored at or after the report.
    Missing storage timestamps are retained or excluded according
    to the unknown availability policy.

    Notes without HADM_ID can be retained through patient linkage.

    Examples:
        >>> report = {
        ...     "subject_id": "1",
        ...     "report_id": "20",
        ...     "report_time": "2150-01-02 12:00:00",
        ... }
        >>> notes = [{
        ...     "subject_id": "1",
        ...     "row_id": "10",
        ...     "text": "History of hypertension.",
        ...     "charttime": "2150-01-01 09:00:00",
        ...     "storetime": "2150-01-01 10:00:00",
        ... }]
        >>> history = eligible_history(notes, report)
        >>> [note["row_id"] for note in history]
        ['10']
    """
    if unknown_availability not in {"retain", "exclude"}:
        raise ValueError("unknown_availability must be retain or exclude")
    target = parse_time(report["report_time"])
    if target is None:
        raise ValueError("Report requires an exact timestamp")
    result = {}
    for source in notes:
        n = dict(source)
        if clean_id(n.get("subject_id")) != clean_id(report["subject_id"]):
            continue
        if clean_id(n.get("row_id")) == clean_id(report["report_id"]):
            continue
        if (
            str(n.get("iserror", "")).strip() in {"1", "1.0"}
            or not isinstance(n.get("text"), str)
            or not n["text"].strip()
        ):
            continue
        exact, day, stored = (
            parse_time(n.get(k)) for k in ("charttime", "chartdate", "storetime")
        )
        if exact is not None:
            if exact >= target:
                continue
            when = exact
        else:
            if day is None or day.date() >= target.date():
                continue
            when = day
        if strict_availability and stored is not None and stored >= target:
            continue
        if stored is None and unknown_availability == "exclude":
            continue
        n.update(
            note_time=when.isoformat(),
            date_only=exact is None,
            availability_unknown=stored is None,
        )
        key = clean_id(n.get("row_id"))
        if not key:
            raise ValueError("Missing source note ROW_ID")
        if key in result and result[key] != n:
            raise ValueError("Conflicting duplicate note ROW_ID")
        result[key] = n
    return sorted(
        result.values(), key=lambda n: (n["note_time"], clean_id(n["row_id"]))
    )


def deduplicate_history(notes, policy="exact"):
    """Collapse exact copies and optionally known report versions.

    Similar progress notes remain distinct. Examination grouping
    requires an exact chart timestamp, description and explicit
    Clip identifier within one patient.

    Unknown-availability report versions are flagged and retained.

    Examples:
        >>> notes = [
        ...     {
        ...         "subject_id": "1",
        ...         "row_id": "10",
        ...         "text": "Hypertension",
        ...         "note_time": "2150-01-01",
        ...     },
        ...     {
        ...         "subject_id": "1",
        ...         "row_id": "11",
        ...         "text": "Hypertension",
        ...         "note_time": "2150-01-02",
        ...     },
        ... ]
        >>> retained, audit = deduplicate_history(notes)
        >>> len(retained)
        1
        >>> audit["exact_duplicates_removed"]
        1
    """
    if policy not in {"none", "exact", "report_versions"}:
        raise ValueError("Invalid history deduplication policy")
    notes = [dict(n) for n in notes]
    audit = {
        "input_notes": len(notes),
        "exact_duplicates_removed": 0,
        "report_versions_removed": 0,
        "report_version_groups": 0,
        "unresolved_report_version_groups": 0,
    }
    if policy != "none":
        groups = {}
        for n in notes:
            key = (clean_id(n["subject_id"]), re.sub(r"\s+", " ", n["text"]).strip())
            groups.setdefault(key, []).append(n)
        notes = []
        for group in groups.values():
            group.sort(
                key=lambda n: (
                    n.get("availability_unknown", True),
                    n["note_time"],
                    clean_id(n["row_id"]),
                )
            )
            kept = group[0]
            if len(group) > 1:
                kept["duplicate_note_ids"] = [clean_id(n["row_id"]) for n in group[1:]]
                audit["exact_duplicates_removed"] += len(group) - 1
            notes.append(kept)
    exams = {}
    for n in notes:
        if str(n.get("category", "")).strip().casefold() != "radiology":
            continue
        clip = re.search(
            r"Clip\s*#\s*(\[\*\*[^\r\n]+?\*\*\]|\d+)", n["text"], re.IGNORECASE
        )
        when = parse_time(n.get("charttime"))
        if clip and when:
            key = (
                clean_id(n["subject_id"]),
                when.isoformat(),
                str(n.get("description", "")).strip().casefold(),
                clip.group(1),
            )
            exams.setdefault(key, []).append(n)
    removed = set()
    for group in exams.values():
        if len(group) < 2:
            continue
        audit["report_version_groups"] += 1
        ids = sorted(clean_id(n["row_id"]) for n in group)
        for n in group:
            n["report_version_note_ids"] = ids
        finals = [
            n for n in group if re.search(r"\bFINAL REPORT\b", n["text"], re.IGNORECASE)
        ]
        prelims = [
            n
            for n in group
            if re.search(r"\bPFI REPORT\b", n["text"], re.IGNORECASE)
            and n not in finals
        ]
        known = all(not n.get("availability_unknown", True) for n in group)
        if (
            policy == "report_versions"
            and known
            and len(finals) == 1
            and len(prelims) == len(group) - 1
        ):
            final = finals[0]
            final["superseded_note_ids"] = [clean_id(n["row_id"]) for n in prelims]
            removed.update(final["superseded_note_ids"])
            audit["report_versions_removed"] += len(prelims)
        else:
            audit["unresolved_report_version_groups"] += 1
    notes = sorted(
        [n for n in notes if clean_id(n["row_id"]) not in removed],
        key=lambda n: (n["note_time"], clean_id(n["row_id"])),
    )
    audit["retained_notes"] = len(notes)
    return notes, audit


def choose_history(notes, max_notes=0, order="recent"):
    """Select a limited number of chronologically ordered notes.

    A max_notes value of zero retains all available notes.

    Examples:
        >>> notes = [
        ...     {"row_id": "1", "note_time": "2150-01-01"},
        ...     {"row_id": "2", "note_time": "2150-01-02"},
        ... ]
        >>> choose_history(notes, max_notes=1)[0]["row_id"]
        '2'
        >>> choose_history(notes, max_notes=1, order="earliest")[0]["row_id"]
        '1'
    """
    if max_notes < 0 or order not in {"recent", "earliest"}:
        raise ValueError("Invalid history selection")
    notes = sorted(notes, key=lambda n: (n["note_time"], clean_id(n["row_id"])))
    if not max_notes:
        return notes
    return notes[-max_notes:] if order == "recent" else notes[:max_notes]


def make_samples(
    reviews,
    notes,
    min_prior_notes=10,
    strict_availability=True,
    unknown_availability="retain",
    deduplication="none",
):
    """Construct the earliest eligible sample per patient and diagnosis.

    Apply temporal filtering and deduplication before checking the
    minimum historical note requirement. The resulting samples do
    not contain ground-truth relevance labels.

    Examples:
        >>> reviews = [{
        ...     "subject_id": "1",
        ...     "report_id": "20",
        ...     "report_time": "2150-01-02 12:00:00",
        ...     "diagnosis": "Stroke",
        ... }]
        >>> notes = [{
        ...     "subject_id": "1",
        ...     "row_id": "10",
        ...     "text": "History of hypertension.",
        ...     "charttime": "2150-01-01 09:00:00",
        ...     "storetime": "2150-01-01 10:00:00",
        ... }]
        >>> samples = make_samples(reviews, notes, min_prior_notes=1)
        >>> len(samples)
        1
        >>> samples[0]["diagnosis"]
        'stroke'
        >>> samples[0]["prior_note_count"]
        1
    """
    samples, seen = [], set()
    for r in sorted(
        reviews, key=lambda x: (x["report_time"], x["report_id"], x["diagnosis"])
    ):
        pair = (clean_id(r["subject_id"]), r["diagnosis"].strip().casefold())
        if pair in seen:
            continue
        history = eligible_history(notes, r, strict_availability, unknown_availability)
        excluded_unknown = (
            (
                len(eligible_history(notes, r, strict_availability, "retain"))
                - len(history)
            )
            if unknown_availability == "exclude"
            else 0
        )
        history, audit = deduplicate_history(history, deduplication)
        audit["unknown_availability_excluded"] = excluded_unknown
        audit["unknown_availability_retained"] = sum(
            n["availability_unknown"] for n in history
        )
        if len(history) < min_prior_notes:
            continue
        seen.add(pair)
        samples.append(
            {
                **r,
                "diagnosis": pair[1],
                "notes": history,
                "prior_note_count": len(history),
                "history_policy": {
                    "unknown_availability": unknown_availability,
                    "deduplication": deduplication,
                },
                "history_audit": audit,
            }
        )
    return samples


def digest(value):
    """Generate a deterministic SHA-256 hash of a JSON-serialized value.

    Examples:
        >>> len(digest("clinical note"))
        64
        >>> digest("note A") == digest("note A")
        True
        >>> digest("note A") == digest("note B")
        False
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
