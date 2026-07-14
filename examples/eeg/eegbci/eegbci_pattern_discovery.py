from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyhealth.datasets import EEGBCIDataset
from pyhealth.tasks import EEGBCIPatternDiscovery


ANALYSIS_VERSION = "eegbci_pattern_moment_report_v1"
REPORT_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
STATE_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def scalar_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("Empty value in integer list")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError("Range start must be <= range end")
            items.extend(range(start, end + 1))
        else:
            items.append(int(part))
    return items


def sample_to_row(sample: dict) -> dict:
    bandpower = sample["bandpower"]
    model_label = scalar_value(sample["label"])
    eegbci_label = scalar_value(sample.get("eegbci_label", model_label))
    return {
        "patient_id": sample["patient_id"],
        "record_id": sample["record_id"],
        "subject_id": sample["subject_id"],
        "run": sample["run"],
        "run_type": sample["run_type"],
        "trial_id": sample["trial_id"],
        "event_code": sample["event_code"],
        "task_label": sample["task_label"],
        "label_family": sample["label_family"],
        "label": eegbci_label,
        "eegbci_label": eegbci_label,
        "model_label": model_label,
        "start_time": sample["start_time"],
        "end_time": sample["end_time"],
        "dominant_band": bandpower["dominant_band"],
        "alpha_beta_ratio": bandpower["alpha_beta_ratio"],
        "theta_beta_ratio": bandpower["theta_beta_ratio"],
        **{key: value for key, value in bandpower.items() if key.endswith("_power")},
        **{key: value for key, value in bandpower.items() if key.endswith("_relative")},
    }


def _mean_band_values(rows: list[dict]) -> dict:
    means = {}
    for band in REPORT_BANDS:
        key = f"{band}_relative"
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        if values:
            means[key] = sum(values) / len(values)
    return means


def build_rest_baselines(rows: list[dict]) -> dict:
    rest_rows = [row for row in rows if row.get("task_label") == "rest"]
    same_subject_run = {}
    same_subject_all_runs = {}

    subject_run_keys = sorted({(row["subject_id"], row["run"]) for row in rest_rows})
    for key in subject_run_keys:
        subject_id, run = key
        grouped = [
            row
            for row in rest_rows
            if row["subject_id"] == subject_id and row["run"] == run
        ]
        same_subject_run[key] = _mean_band_values(grouped)

    subject_keys = sorted({row["subject_id"] for row in rest_rows})
    for subject_id in subject_keys:
        grouped = [row for row in rest_rows if row["subject_id"] == subject_id]
        same_subject_all_runs[subject_id] = _mean_band_values(grouped)

    return {
        "same_subject_run": same_subject_run,
        "same_subject_all_runs": same_subject_all_runs,
        "global_rest": _mean_band_values(rest_rows) if rest_rows else None,
    }


def _baseline_for_row(row: dict, baselines: dict) -> tuple[str, dict | None]:
    subject_run_key = (row["subject_id"], row["run"])
    if subject_run_key in baselines["same_subject_run"]:
        return "same_subject_run", baselines["same_subject_run"][subject_run_key]
    if row["subject_id"] in baselines["same_subject_all_runs"]:
        return "same_subject_all_runs", baselines["same_subject_all_runs"][row["subject_id"]]
    if baselines["global_rest"]:
        return "global_rest", baselines["global_rest"]
    return "unavailable", None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def derive_state_hypothesis(row: dict) -> dict:
    delta_bands = {
        band: row.get(f"rest_{band}_relative_delta") for band in REPORT_BANDS
    }
    try:
        normalized_deltas = {
            band: float(value)
            for band, value in delta_bands.items()
            if value not in ("", None)
        }
    except (TypeError, ValueError):
        normalized_deltas = {}
    has_rest_deltas = len(normalized_deltas) == len(REPORT_BANDS) and all(
        math.isfinite(value) for value in normalized_deltas.values()
    )

    if has_rest_deltas:
        evidence_values = normalized_deltas
        scores = {
            "idle_alpha_profile": max(evidence_values["alpha"], 0.0)
            + max(-evidence_values["beta"], 0.0),
            "sensorimotor_engagement_profile": max(
                -evidence_values["alpha"], 0.0
            )
            + max(evidence_values["beta"], 0.0)
            + max(evidence_values["gamma"], 0.0),
            "slow_wave_dominant_pattern": max(evidence_values["delta"], 0.0)
            + max(evidence_values["theta"], 0.0),
            "possible_artifact_profile": max(
                evidence_values["gamma"] - 0.03, 0.0
            )
            * 2.0,
        }
        evidence_basis = "rest_normalized_delta"
    else:
        evidence_values = {
            band: float(row.get(f"{band}_relative", 0.0) or 0.0)
            for band in REPORT_BANDS
        }
        alpha_beta = float(row.get("alpha_beta_ratio", 0.0) or 0.0)
        theta_beta = float(row.get("theta_beta_ratio", 0.0) or 0.0)
        scores = {
            "idle_alpha_profile": _clip01(
                (evidence_values["alpha"] - 0.25)
                + min(alpha_beta / 8.0, 0.40)
            ),
            "sensorimotor_engagement_profile": _clip01(
                (evidence_values["beta"] - 0.20)
                + max(evidence_values["gamma"] - 0.12, 0.0)
                + max(0.0, 1.5 - alpha_beta) / 6.0
            ),
            "slow_wave_dominant_pattern": _clip01(
                (evidence_values["delta"] + evidence_values["theta"])
                - 0.45
                + min(theta_beta / 8.0, 0.20)
            ),
            "possible_artifact_profile": _clip01(
                (evidence_values["gamma"] - 0.22) * 2.0
                + max(evidence_values["delta"] - 0.50, 0.0)
            ),
        }
        evidence_basis = "absolute_band_profile"

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winning_score = ordered[0]
    margin = winning_score - ordered[1][1]

    if has_rest_deltas:
        if winning_score < 0.02 or margin < 0.01:
            state = "mixed_ambiguous_profile"
            confidence = "low"
        elif winning_score >= 0.15 and margin >= 0.08:
            state = winner
            confidence = "high"
        elif winning_score >= 0.06 and margin >= 0.025:
            state = winner
            confidence = "medium"
        else:
            state = winner
            confidence = "low"
        evidence_score = round(min(winning_score / 0.20, 1.0), 3)
    else:
        if winning_score < 0.20 or margin < 0.08:
            state = "mixed_ambiguous_profile"
            evidence_score = round(max(winning_score, 0.10), 3)
            confidence = "low"
        else:
            state = winner
            evidence_score = round(winning_score, 3)
            if winning_score >= 0.65 and margin >= 0.20:
                confidence = "high"
            elif winning_score >= 0.35 and margin >= 0.12:
                confidence = "medium"
            else:
                confidence = "low"

    return {
        "state_hypothesis": state,
        "state_confidence": confidence,
        "evidence_score": evidence_score,
        "evidence_summary": (
            f"basis={evidence_basis}; delta={evidence_values['delta']:.3f}; "
            f"theta={evidence_values['theta']:.3f}; "
            f"alpha={evidence_values['alpha']:.3f}; "
            f"beta={evidence_values['beta']:.3f}; "
            f"gamma={evidence_values['gamma']:.3f}; margin={margin:.3f}"
        ),
    }


def derive_task_state_relation(row: dict) -> dict:
    label_family = row.get("label_family", "")
    task_label = row.get("task_label", "")
    state = row.get("state_hypothesis", "")

    if state == "possible_artifact_profile":
        relation = "not_applicable"
        confidence = "medium"
        rationale = (
            "Artifact-like frequency evidence is flagged for inspection instead of "
            "task-label comparison."
        )
    elif state == "mixed_ambiguous_profile":
        relation = "ambiguous"
        confidence = "low"
        rationale = (
            "No frequency-profile state won clearly enough to compare strongly with "
            "the task label."
        )
    elif task_label == "rest" and state == "idle_alpha_profile":
        relation = "supports_label"
        confidence = "medium"
        rationale = "The idle-like alpha profile is consistent with a rest-labeled EEGBCI window."
    elif label_family == "motor_execution" and state == "sensorimotor_engagement_profile":
        relation = "supports_label"
        confidence = "medium"
        rationale = (
            "The motor-engaged frequency profile is consistent with an "
            "execution-labeled window."
        )
    elif label_family == "motor_imagery" and state == "sensorimotor_engagement_profile":
        relation = "adds_detail"
        confidence = "medium"
        rationale = (
            "The motor-engaged frequency profile adds signal detail to an "
            "imagery-labeled window."
        )
    elif label_family in {"motor_execution", "motor_imagery"} and state == "idle_alpha_profile":
        relation = "disagrees"
        confidence = "medium"
        rationale = "The idle-like alpha profile does not align with a motor-labeled EEGBCI window."
    elif state == "slow_wave_dominant_pattern":
        relation = "adds_detail"
        confidence = "low"
        rationale = "The slow-wave dominant pattern adds frequency detail but is not a direct task match."
    else:
        relation = "ambiguous"
        confidence = "low"
        rationale = (
            "The task label and frequency-profile state do not have a stronger "
            "deterministic mapping."
        )

    return {
        "task_state_relation": relation,
        "task_state_rationale": rationale,
        "task_state_confidence": confidence,
    }


def derive_quality_columns(row: dict) -> dict:
    flags = str(row.get("quality_flags", ""))
    state = row.get("state_hypothesis", "")
    confidence = row.get("state_confidence", row.get("confidence", ""))
    return {
        "is_low_confidence": confidence == "low",
        "is_possible_artifact": state == "possible_artifact_profile"
        or "artifact" in flags
        or "high_gamma" in flags,
        "is_mixed_or_ambiguous": state == "mixed_ambiguous_profile"
        or "ambiguous" in flags,
    }


def derive_moment_interpretation(row: dict) -> str:
    state = row.get("state_hypothesis", "missing")
    confidence = row.get("state_confidence", "missing")
    evidence = row.get("evidence_score", "")
    relation = row.get("task_state_relation", "missing")
    task = row.get("task_label", "missing")
    dominant = row.get("dominant_band", "missing")
    scope = row.get("rest_reference_scope", "missing")
    return (
        f"The segment is consistent with `{state}` based on a `{dominant}`-dominant "
        f"frequency profile ({confidence} confidence, evidence {evidence}). "
        f"The task label is `{task}`, the task/state relation is `{relation}`, "
        f"and the rest reference is `{scope}`."
    )


BASE_OUTPUT_COLUMNS = (
    "patient_id",
    "record_id",
    "subject_id",
    "run",
    "run_type",
    "trial_id",
    "event_code",
    "task_label",
    "label_family",
    "label",
    "eegbci_label",
    "model_label",
    "start_time",
    "end_time",
    "dominant_band",
    "alpha_beta_ratio",
    "theta_beta_ratio",
    "interpretation",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "gamma_relative",
)

MOMENT_REPORT_COLUMNS = (
    "analysis_version",
    "state_hypothesis",
    "state_confidence",
    "evidence_score",
    "evidence_summary",
    "rest_reference_scope",
    "rest_delta_relative_delta",
    "rest_theta_relative_delta",
    "rest_alpha_relative_delta",
    "rest_beta_relative_delta",
    "rest_gamma_relative_delta",
    "task_state_relation",
    "task_state_rationale",
    "task_state_confidence",
    "is_low_confidence",
    "is_possible_artifact",
    "is_mixed_or_ambiguous",
)

OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + MOMENT_REPORT_COLUMNS


def annotate_moment_rows(rows: list[dict], baselines: dict) -> list[dict]:
    annotated = []
    for row in rows:
        next_row = dict(row)
        scope, baseline = _baseline_for_row(next_row, baselines)
        next_row["analysis_version"] = ANALYSIS_VERSION
        next_row["rest_reference_scope"] = scope

        for band in REPORT_BANDS:
            source_key = f"{band}_relative"
            delta_key = f"rest_{band}_relative_delta"
            if baseline and source_key in baseline and next_row.get(source_key) not in ("", None):
                next_row[delta_key] = round(
                    float(next_row[source_key]) - float(baseline[source_key]), 6
                )
            else:
                next_row[delta_key] = ""

        next_row.update(derive_state_hypothesis(next_row))
        next_row.update(derive_task_state_relation(next_row))
        next_row["interpretation"] = derive_moment_interpretation(next_row)
        next_row.update(derive_quality_columns(next_row))
        annotated.append(next_row)
    return annotated


def _stable_row_key(row: dict) -> tuple:
    return (
        row.get("subject_id", 0),
        row.get("run", 0),
        float(row.get("start_time", 0.0) or 0.0),
    )


def _strongest_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("evidence_score", 0.0) or 0.0),
            -STATE_CONFIDENCE_RANK.get(row.get("state_confidence", "low"), 0),
            *_stable_row_key(row),
        ),
    )[0]


def select_representative_windows(rows: list[dict]) -> dict:
    definitions = {
        "strongest_idle_like": "idle_alpha_profile",
        "strongest_motor_engaged": "sensorimotor_engagement_profile",
        "strongest_slow_wave": "slow_wave_dominant_pattern",
        "strongest_artifact_like": "possible_artifact_profile",
    }
    cards = {}
    absent = []

    for card_name, state in definitions.items():
        candidate = _strongest_row(
            [row for row in rows if row.get("state_hypothesis") == state]
        )
        if candidate is None:
            absent.append(card_name)
        else:
            cards[card_name] = candidate

    ambiguous = [
        row for row in rows if row.get("state_hypothesis") == "mixed_ambiguous_profile"
    ]
    if ambiguous:
        cards["most_ambiguous"] = sorted(
            ambiguous,
            key=lambda row: (
                float(row.get("evidence_score", 0.0) or 0.0),
                -STATE_CONFIDENCE_RANK.get(row.get("state_confidence", "low"), 0),
                *_stable_row_key(row),
            ),
        )[0]
    else:
        absent.append("most_ambiguous")

    disagreement = _strongest_row(
        [row for row in rows if row.get("task_state_relation") == "disagrees"]
    )
    if disagreement is None:
        absent.append("strongest_task_state_disagreement")
    else:
        cards["strongest_task_state_disagreement"] = disagreement

    return {"cards": cards, "absent": absent}


def _format_count_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- {label}: {count}" for label, count in counter.most_common()]


def _format_card(row: dict) -> list[str]:
    bands = ", ".join(
        f"{band}={float(row.get(f'{band}_relative', 0.0) or 0.0):.3f}"
        for band in REPORT_BANDS
    )
    deltas = ", ".join(
        f"{band}={row.get(f'rest_{band}_relative_delta', '')}"
        for band in REPORT_BANDS
    )
    return [
        f"- Subject {row.get('subject_id')} run {row.get('run')} trial {row.get('trial_id')}",
        f"  - Task: {row.get('task_label')} from {row.get('start_time')}s to {row.get('end_time')}s",
        (
            f"  - State: {row.get('state_hypothesis')} "
            f"({row.get('state_confidence')}, evidence {row.get('evidence_score')})"
        ),
        f"  - Dominant band: {row.get('dominant_band')}; relative bands: {bands}",
        f"  - Rest deltas: {deltas}; scope: {row.get('rest_reference_scope')}",
        (
            f"  - Task relation: {row.get('task_state_relation')} "
            f"({row.get('task_state_confidence')})"
        ),
        (
            f"  - Flags: low_confidence={row.get('is_low_confidence')}, "
            f"possible_artifact={row.get('is_possible_artifact')}, "
            f"mixed_or_ambiguous={row.get('is_mixed_or_ambiguous')}"
        ),
        f"  - Rationale: {row.get('task_state_rationale')}",
    ]


def render_summary(rows: list[dict], config: dict) -> str:
    state_counts = Counter(row.get("state_hypothesis", "missing") for row in rows)
    task_counts = Counter(row.get("task_label", "missing") for row in rows)
    confidence_counts = Counter(row.get("state_confidence", "missing") for row in rows)
    relation_counts = Counter(row.get("task_state_relation", "missing") for row in rows)
    unavailable_rest = sum(
        row.get("rest_reference_scope") == "unavailable" for row in rows
    )
    low_confidence = sum(bool(row.get("is_low_confidence")) for row in rows)
    artifacts = sum(bool(row.get("is_possible_artifact")) for row in rows)
    ambiguous = sum(bool(row.get("is_mixed_or_ambiguous")) for row in rows)
    representatives = select_representative_windows(rows)

    executive = []
    if not rows:
        executive.append("No windows were produced for the requested configuration.")
    else:
        top_state, top_state_count = state_counts.most_common(1)[0]
        executive.append(
            f"Processed {len(rows)} windows. Most common state: `{top_state}` "
            f"({top_state_count}/{len(rows)})."
        )
        if low_confidence == len(rows):
            executive.append("Every window is low confidence.")
        if len(state_counts) == 1:
            executive.append(
                "Every window maps to the same state; broaden coverage or review thresholds."
            )
        if unavailable_rest == len(rows):
            executive.append("No rest baseline was available for the emitted rows.")
    if config.get("output_was_capped"):
        executive.append("Output was capped by `--max-windows`.")

    lines = [
        "# EEGBCI Pattern Discovery Moment Report",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Executive Result",
        "",
        *[f"- {item}" for item in executive],
        "",
        "## Run Configuration",
        "",
        f"- Subjects: {config.get('subjects')}",
        f"- Runs: {config.get('runs')}",
        f"- Max windows: {config.get('max_windows')}",
        f"- Baseline source rows: {config.get('baseline_row_count')}",
        "",
        "## Window Coverage",
        "",
        f"- Output windows: {len(rows)}",
        f"- Task labels: {dict(task_counts)}",
        "",
        "## Moment-State Summary",
        "",
        *_format_count_lines(state_counts),
        "",
        "## Task Label x State Matrix",
        "",
    ]

    matrix = Counter(
        (row.get("task_label", "missing"), row.get("state_hypothesis", "missing"))
        for row in rows
    )
    if matrix:
        for (task_label, state), count in sorted(matrix.items()):
            lines.append(f"- {task_label} x {state}: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Rest-Normalized Bandpower Summary",
            "",
            f"- Rows with unavailable rest baseline: {unavailable_rest}",
        ]
    )
    for band in REPORT_BANDS:
        key = f"rest_{band}_relative_delta"
        values = [float(row.get(key)) for row in rows if row.get(key) not in ("", None)]
        if values:
            lines.append(f"- {band}: mean delta {sum(values) / len(values):.3f}")
        else:
            lines.append(f"- {band}: unavailable")

    lines.extend(
        [
            "",
            "## Confidence and Quality Audit",
            "",
            f"- State confidence: {dict(confidence_counts)}",
            f"- Task-state relations: {dict(relation_counts)}",
            f"- Low-confidence rows: {low_confidence}",
            f"- Possible artifact rows: {artifacts}",
            f"- Mixed or ambiguous rows: {ambiguous}",
            "",
            "## Representative Windows",
            "",
        ]
    )
    if representatives["cards"]:
        for card_name, row in representatives["cards"].items():
            lines.append(f"### {card_name.replace('_', ' ').title()}")
            lines.extend(_format_card(row))
            lines.append("")
    else:
        lines.append("- None")
    if representatives["absent"]:
        lines.append(
            f"- Absent representative classes: {', '.join(representatives['absent'])}"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- These labels are signal-pattern summaries from short EEG windows. "
                "They are not clinical findings and should not be read as evidence "
                "of a subject's cognition."
            ),
        ]
    )
    if unavailable_rest:
        lines.append("- No rest baseline was available for at least one emitted row.")
    if config.get("output_was_capped"):
        lines.append(
            "- The output was capped, so the artifact may not represent all requested windows."
        )

    lines.extend(
        [
            "",
            "## Next Checks",
            "",
            "- Run with broader subjects/runs to verify that state diversity improves.",
            "- Inspect possible artifact rows before drawing conclusions from state counts.",
            "- Compare rest-normalized deltas against the raw relative band shares.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_summary(rows: list[dict], path: Path, config: dict) -> None:
    path.write_text(render_summary(rows, config), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/.cache/pyhealth/eegbci")
    parser.add_argument("--subjects", default="1,2,3")
    parser.add_argument("--runs", default="3-14")
    parser.add_argument("--output-dir", default="outputs/eegbci_pattern_discovery")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_subjects = parse_int_list(args.subjects)
    requested_runs = parse_int_list(args.runs)
    dataset = EEGBCIDataset(
        root=str(Path(args.root).expanduser()),
        subjects=requested_subjects,
        runs=requested_runs,
        download=args.download,
    )
    sample_dataset = dataset.set_task(EEGBCIPatternDiscovery(compute_stft=False))

    all_rows = [sample_to_row(sample) for sample in sample_dataset]
    baseline_row_count = sum(row.get("task_label") == "rest" for row in all_rows)
    baselines = build_rest_baselines(all_rows)
    annotated_rows = annotate_moment_rows(all_rows, baselines)
    output_rows = (
        annotated_rows[: args.max_windows]
        if args.max_windows is not None
        else annotated_rows
    )
    output_was_capped = (
        args.max_windows is not None and len(annotated_rows) > len(output_rows)
    )

    csv_path = output_dir / "eegbci_pattern_windows.csv"
    summary_path = output_dir / "eegbci_pattern_summary.md"
    pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)
    write_summary(
        output_rows,
        summary_path,
        {
            "subjects": getattr(dataset, "subjects", requested_subjects),
            "runs": getattr(dataset, "runs", requested_runs),
            "max_windows": args.max_windows,
            "baseline_row_count": baseline_row_count,
            "output_was_capped": output_was_capped,
        },
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
