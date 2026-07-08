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
