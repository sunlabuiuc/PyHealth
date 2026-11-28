"""Transparent event-level evaluation for early-event prediction examples.

This lightweight example utility deliberately has no dataset or local-file dependency.
Each record is a dict with stay, period_length, last_t, ever_event,
y_true (whether an alarm at that instant is timely), and the score column.
One event at most per stay is assumed. Time units must be consistent.

This implementation uses *time-based* alarm silence. With silence_steps=1,
an alarm at hour 5 suppresses another at hour 6 (next eligible hour 7).
That equals the source script for contiguous hourly observations, but differs
when observations are missing. It is NOT a drop-in replacement for previous
reported results without protocol equivalence checks.
"""

from collections import defaultdict
from math import isfinite
from statistics import mean, median


def prepare_records(records, score_col="score"):
    """Validate and group records by stay, ordered by observation time."""
    grouped = defaultdict(list)
    for record in records:
        for field in ("stay", "period_length", "last_t", "ever_event", "y_true", score_col):
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
        row = dict(record)
        for field in ("period_length", "last_t", score_col):
            row[field] = float(row[field])
            if not isfinite(row[field]):
                raise ValueError(f"Non-finite {field}")
        for field in ("ever_event", "y_true"):
            if row[field] not in (0, 1, False, True):
                raise ValueError(f"{field} must be 0 or 1")
            row[field] = bool(row[field])
        grouped[str(row["stay"])].append(row)
    if not grouped:
        raise ValueError("At least one observation is required")
    for stay, rows in grouped.items():
        rows.sort(key=lambda r: r["period_length"])
        if len({r["period_length"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate observation times in stay {stay}")
        if len({r["ever_event"] for r in rows}) != 1:
            raise ValueError(f"Inconsistent ever_event within stay {stay}")
        if any(r["y_true"] and not r["ever_event"] for r in rows):
            raise ValueError(f"Positive y_true in non-event stay {stay}")
    return dict(grouped)


def evaluate_threshold(grouped, threshold, *, score_col="score", silence_steps=1):
    """Evaluate a fixed alarm threshold; no threshold fitting occurs here.

    Precision is None if no alarms; recall is None if no event stays.
    False alarms count alarms with y_true=False, even on event stays.
    Lead time is measured at the *first timely alarm per event stay*.
    """
    if not isfinite(threshold) or silence_steps < 0:
        raise ValueError("Invalid threshold or silence_steps")
    total = true = detected = event_stays = 0
    leads, per_stay_alarms, per_stay_false = [], [], []
    for rows in grouped.values():
        event_stays += int(rows[0]["ever_event"])
        last_alarm_time = float("-inf")
        first_lead = None
        n_alarms = n_false = 0
        for row in rows:
            now = row["period_length"]
            if now <= last_alarm_time + silence_steps:
                continue
            if row[score_col] < threshold:
                continue
            last_alarm_time = now
            total += 1
            n_alarms += 1
            if row["y_true"]:
                true += 1
                if first_lead is None:
                    first_lead = row["last_t"] + 1.0 - now
            else:
                n_false += 1
        if first_lead is not None:
            detected += 1
            leads.append(float(first_lead))
        per_stay_alarms.append(n_alarms)
        per_stay_false.append(n_false)
    n_stays = len(grouped)
    false_alarms = total - true
    return {
        "threshold": float(threshold),
        "event_recall": detected / event_stays if event_stays else None,
        "alarm_precision": true / total if total else None,
        "total_alarms": total,
        "true_alarms": true,
        "false_alarms": false_alarms,
        "events": event_stays,
        "detected_events": detected,
        "stays": n_stays,
        "false_alarms_per_100_stays": 100.0 * false_alarms / n_stays,
        "mean_lead_steps": mean(leads) if leads else None,
        "median_lead_steps": median(leads) if leads else None,
        "mean_alarms_per_stay": mean(per_stay_alarms),
        "stays_ge2_false_alarms": sum(n >= 2 for n in per_stay_false) / n_stays,
    }


def select_validation_threshold(validation, thresholds, target_precision, *, score_col="score", silence_steps=1):
    """Choose max validation event recall subject to alarm precision constraint.

    Empty-alarm operating points are INELIGIBLE. Tie-break by precision,
    then threshold (deterministically). Never reads the test set.
    """
    if not 0 <= target_precision <= 1:
        raise ValueError("target_precision must be between zero and one")
    candidates = []
    for threshold in thresholds:
        metric = evaluate_threshold(validation, float(threshold), score_col=score_col, silence_steps=silence_steps)
        if metric["total_alarms"] and metric["alarm_precision"] >= target_precision:
            candidates.append(metric)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["event_recall"] if x["event_recall"] is not None else -1, x["alarm_precision"], x["threshold"]))


def evaluate_validation_selected_test(val_records, test_records, thresholds, target_precision, *, score_col="score", silence_steps=1):
    """Select threshold only on validation; apply unchanged to test."""
    val = prepare_records(val_records, score_col=score_col)
    test = prepare_records(test_records, score_col=score_col)
    chosen = select_validation_threshold(val, thresholds, target_precision, score_col=score_col, silence_steps=silence_steps)
    if chosen is None:
        return {"validation": None, "test": None}
    return {"validation": chosen, "test": evaluate_threshold(test, chosen["threshold"], score_col=score_col, silence_steps=silence_steps)}
