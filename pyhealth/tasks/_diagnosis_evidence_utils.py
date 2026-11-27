"""Dependency-free time and history rules, shared by cohort and PyHealth adapter."""

from __future__ import annotations
from datetime import datetime
import re


def clean_id(value):
    text = "" if value is None else str(value).strip()
    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    return re.sub(r"\.0$", "", text)


def parse_time(value):
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


def eligible_history(notes, report, strict_availability=True, unknown_availability="retain"):
    """CHARTTIME < report; date-only notes must be from a previous calendar day.

    STORETIME after the report is excluded in the primary analysis. Missing
    STORETIME is retained and flagged: retrospective availability is then unknown.
    Notes without HADM_ID are retained through patient-level linkage.
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
    """Collapse exact whitespace-normalized copies; optionally known report versions.

    Similar progress notes remain distinct. Examination grouping requires an exact
    chart timestamp, description and explicit Clip identifier within one patient.
    Unknown-availability versions are flagged and retained, never silently replaced.
    """
    if policy not in {"none", "exact", "report_versions"}:
        raise ValueError("Invalid history deduplication policy")
    notes = [dict(n) for n in notes]
    audit = {"input_notes": len(notes), "exact_duplicates_removed": 0,
             "report_versions_removed": 0, "report_version_groups": 0,
             "unresolved_report_version_groups": 0}
    if policy != "none":
        groups = {}
        for n in notes:
            key = (clean_id(n["subject_id"]), re.sub(r"\s+", " ", n["text"]).strip())
            groups.setdefault(key, []).append(n)
        notes = []
        for group in groups.values():
            group.sort(key=lambda n: (n.get("availability_unknown", True),
                                      n["note_time"], clean_id(n["row_id"])))
            kept = group[0]
            if len(group) > 1:
                kept["duplicate_note_ids"] = [clean_id(n["row_id"]) for n in group[1:]]
                audit["exact_duplicates_removed"] += len(group) - 1
            notes.append(kept)
    exams = {}
    for n in notes:
        if str(n.get("category", "")).strip().casefold() != "radiology":
            continue
        clip = re.search(r"Clip\s*#\s*(\[\*\*[^\r\n]+?\*\*\]|\d+)", n["text"], re.I)
        when = parse_time(n.get("charttime"))
        if clip and when:
            key = (clean_id(n["subject_id"]), when.isoformat(),
                   str(n.get("description", "")).strip().casefold(), clip.group(1))
            exams.setdefault(key, []).append(n)
    removed = set()
    for group in exams.values():
        if len(group) < 2:
            continue
        audit["report_version_groups"] += 1
        ids = sorted(clean_id(n["row_id"]) for n in group)
        for n in group:
            n["report_version_note_ids"] = ids
        finals = [n for n in group if re.search(r"\bFINAL REPORT\b", n["text"], re.I)]
        prelims = [n for n in group if re.search(r"\bPFI REPORT\b", n["text"], re.I)
                   and n not in finals]
        known = all(not n.get("availability_unknown", True) for n in group)
        if policy == "report_versions" and known and len(finals) == 1 and len(prelims) == len(group) - 1:
            final = finals[0]
            final["superseded_note_ids"] = [clean_id(n["row_id"]) for n in prelims]
            removed.update(final["superseded_note_ids"])
            audit["report_versions_removed"] += len(prelims)
        else:
            audit["unresolved_report_version_groups"] += 1
    notes = sorted([n for n in notes if clean_id(n["row_id"]) not in removed],
                   key=lambda n: (n["note_time"], clean_id(n["row_id"])))
    audit["retained_notes"] = len(notes)
    return notes, audit


def choose_history(notes, max_notes=0, order="recent"):
    if max_notes < 0 or order not in {"recent", "earliest"}:
        raise ValueError("Invalid history selection")
    notes = sorted(notes, key=lambda n: (n["note_time"], clean_id(n["row_id"])))
    if not max_notes:
        return notes
    return notes[-max_notes:] if order == "recent" else notes[:max_notes]


def make_samples(reviews, notes, min_prior_notes=10, strict_availability=True,
                 unknown_availability="retain", deduplication="none"):
    """One earliest eligible sample per patient/canonical diagnosis; no gold labels."""
    samples, seen = [], set()
    for r in sorted(
        reviews, key=lambda x: (x["report_time"], x["report_id"], x["diagnosis"])
    ):
        pair = (clean_id(r["subject_id"]), r["diagnosis"].strip().casefold())
        if pair in seen:
            continue
        history = eligible_history(notes, r, strict_availability, unknown_availability)
        excluded_unknown = (len(eligible_history(notes, r, strict_availability, "retain"))
                            - len(history)) if unknown_availability == "exclude" else 0
        history, audit = deduplicate_history(history, deduplication)
        audit["unknown_availability_excluded"] = excluded_unknown
        audit["unknown_availability_retained"] = sum(n["availability_unknown"] for n in history)
        if len(history) < min_prior_notes:
            continue
        seen.add(pair)
        samples.append(
            {
                **r,
                "diagnosis": pair[1],
                "notes": history,
                "prior_note_count": len(history),
                "history_policy": {"unknown_availability": unknown_availability,
                                   "deduplication": deduplication},
                "history_audit": audit,
            }
        )
    return samples
