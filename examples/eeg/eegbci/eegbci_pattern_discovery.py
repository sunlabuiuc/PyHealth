from __future__ import annotations

import argparse
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
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            items.extend(range(int(start), int(end) + 1))
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
        "brain_state_hypothesis": sample["brain_state_hypothesis"],
        "confidence": sample["confidence"],
        "quality_flags": sample["quality_flags"],
        "interpretation": sample["interpretation"],
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
    delta = float(row.get("delta_relative", 0.0) or 0.0)
    theta = float(row.get("theta_relative", 0.0) or 0.0)
    alpha = float(row.get("alpha_relative", 0.0) or 0.0)
    beta = float(row.get("beta_relative", 0.0) or 0.0)
    gamma = float(row.get("gamma_relative", 0.0) or 0.0)
    alpha_beta = float(row.get("alpha_beta_ratio", 0.0) or 0.0)
    theta_beta = float(row.get("theta_beta_ratio", 0.0) or 0.0)

    scores = {
        "idle_alpha_profile": _clip01((alpha - 0.25) + min(alpha_beta / 8.0, 0.40)),
        "sensorimotor_engagement_profile": _clip01(
            (beta - 0.20)
            + max(gamma - 0.12, 0.0)
            + max(0.0, 1.5 - alpha_beta) / 6.0
        ),
        "slow_wave_dominant_pattern": _clip01(
            (delta + theta) - 0.45 + min(theta_beta / 8.0, 0.20)
        ),
        "possible_artifact_profile": _clip01(
            (gamma - 0.22) * 2.0 + max(delta - 0.50, 0.0)
        ),
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winning_score = ordered[0]
    runner_up = ordered[1][1]
    margin = winning_score - runner_up

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
            f"delta={delta:.3f}; theta={theta:.3f}; alpha={alpha:.3f}; "
            f"beta={beta:.3f}; gamma={gamma:.3f}; alpha_beta={alpha_beta:.3f}; "
            f"margin={margin:.3f}"
        ),
    }


def write_summary(rows: list[dict], path: Path) -> None:
    task_counts = Counter(row["task_label"] for row in rows)
    hypothesis_counts = Counter(row["brain_state_hypothesis"] for row in rows)
    lines = [
        "# EEGBCI Pattern Discovery Summary",
        "",
        "Brain-state hypotheses are exploratory signal metadata, not clinical diagnoses.",
        "",
        f"Processed windows: {len(rows)}",
        "",
        "## Task Labels",
        "",
    ]
    for label, count in task_counts.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Brain-State Hypotheses", ""])
    for label, count in hypothesis_counts.most_common():
        lines.append(f"- {label}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    dataset = EEGBCIDataset(
        root=str(Path(args.root).expanduser()),
        subjects=parse_int_list(args.subjects),
        runs=parse_int_list(args.runs),
        download=args.download,
    )
    sample_dataset = dataset.set_task(EEGBCIPatternDiscovery(compute_stft=False))

    rows = []
    for idx, sample in enumerate(sample_dataset):
        if args.max_windows is not None and idx >= args.max_windows:
            break
        rows.append(sample_to_row(sample))

    csv_path = output_dir / "eegbci_pattern_windows.csv"
    summary_path = output_dir / "eegbci_pattern_summary.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_summary(rows, summary_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
