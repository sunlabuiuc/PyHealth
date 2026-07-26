#!/usr/bin/env python3
"""Compute per-modality token-count distributions and availability for Table 2.

For each patient sample, computes a per-sample "token count" for every
modality (notes: actual tokenizer token count; labs/ICD: number of real
timesteps / codes), then reports the median and max of that distribution
plus availability — the % of samples that have any real (non-placeholder)
data for that modality.

Iterates patients directly via dataset.iter_patients() and calls the task
function on each one — never materializes all samples in memory, no litdata
write, no GPU needed.

Usage (on CC):
    python scripts/compute_token_stats_labs_only.py \
        --ehr-root /projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2 \
        --note-root /projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note \
        --cache-dir /u/rianatri/pyhealth_cache \
        --task notes_labs

    python /home/wp14/PyHealth/scripts/compute_token_stats_labs_only.py \
        --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \
        --task labs

    # Add --log-file to also save the printed stats to a file:
    python PyHealth/scripts/compute_token_stats_labs_only.py \
        --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \
        --task labs \
        --log-file logs/token_stats_labs.log
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm


class Tee:
    """Duplicates writes to multiple streams (e.g. stdout + a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ehr-root", required=True)
    p.add_argument(
        "--note-root",
        default=None,
        help="Required unless --task labs (labs-only task needs no notes).",
    )
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dev", action="store_true", help="Limit to 1000 patients")
    p.add_argument(
        "--task",
        type=str,
        choices=["notes_labs", "labs"],
        default="notes_labs",
        help=(
            "Task to profile. 'notes_labs' (default) uses admission-context "
            "text sections; 'labs' profiles the labs-only baseline (no notes, "
            "no note-root needed)."
        ),
    )
    p.add_argument(
        "--icd-codes",
        action="store_true",
        default=False,
        help="Include discharge-coded ICD codes when using --task notes_labs.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Path to also write stdout output to. If omitted, only prints to console.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.task != "labs" and not args.note_root:
        raise SystemExit("--note-root is required unless --task labs")

    if args.log_file:
        log_f = open(args.log_file, "w")
        sys.stdout = Tee(sys.stdout, log_f)
        sys.stderr = Tee(sys.stderr, log_f)
        print(f"Logging output to {args.log_file}")

    os.environ.setdefault("PYHEALTH_DISABLE_DASK_DISTRIBUTED", "1")

    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4, NotesLabsMIMIC4

    print("Building dataset (uses cache if available)...")

    if args.task == "labs":
        ehr_tables = ["labevents"]
        note_tables = []
        task = LabsMIMIC4(window_hours=24)
    else:  # notes_labs
        ehr_tables = (
            ["diagnoses_icd", "procedures_icd", "labevents"]
            if args.icd_codes
            else ["labevents"]
        )
        note_tables = ["discharge"]
        task = NotesLabsMIMIC4(window_hours=24, include_icd=args.icd_codes)

    kwargs = dict(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_tables=note_tables,
        dev=args.dev,
    )
    if args.note_root:
        kwargs["note_root"] = args.note_root
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    dataset = MIMIC4Dataset(**kwargs)

    MISSING_TEXT = ""
    has_notes = args.task == "notes_labs"
    has_icd = args.task == "notes_labs" and args.icd_codes

    tokenizer = None
    if has_notes:
        from transformers import AutoTokenizer

        tokenizer_model = task.input_schema["admission_note_times"][1][
            "tokenizer_model"
        ]
        print(f"Loading tokenizer: {tokenizer_model}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Per-sample token counts (one entry per patient sample)
    note_token_counts = []
    note_has_data = []
    lab_token_counts = []
    lab_has_data = []
    icd_token_counts = []
    icd_has_data = []

    n_patients = n_samples = 0

    print("Iterating patients and accumulating stats (no litdata write)...")
    for patient in tqdm(dataset.iter_patients(), total=len(dataset.unique_patient_ids)):
        n_patients += 1
        samples = task(patient)
        if not samples:
            continue
        for s in samples:
            n_samples += 1

            # ── notes ────────────────────────────────────────────
            if has_notes:
                note_texts, _ = s["admission_note_times"]
                real_texts = [t for t in note_texts if t != MISSING_TEXT]
                n_tokens = sum(
                    len(tokenizer.encode(t, add_special_tokens=False))
                    for t in real_texts
                )
                note_token_counts.append(n_tokens)
                note_has_data.append(bool(real_texts))

            # ── ICD codes ────────────────────────────────────────
            if has_icd:
                _, icd_visits = s["icd_codes"]
                real_visits = [v for v in icd_visits if v != [MISSING_TEXT]]
                n_codes = sum(len(v) for v in real_visits)
                icd_token_counts.append(n_codes)
                icd_has_data.append(bool(real_visits))

            # ── labs ─────────────────────────────────────────────
            _, lab_masks = s["labs_mask"]
            n_real_timesteps = sum(1 for mask_row in lab_masks if any(mask_row))
            lab_token_counts.append(n_real_timesteps)
            lab_has_data.append(n_real_timesteps > 0)

    def median_max(counts):
        if not counts:
            return float("nan"), float("nan")
        arr = np.array(counts)
        return float(np.median(arr)), float(np.max(arr))

    def availability_pct(has_data):
        if not has_data:
            return float("nan")
        return 100.0 * sum(has_data) / len(has_data)

    print("\n" + "=" * 60)
    print(f"TOKEN STATS — {task.task_name}")
    print("=" * 60)
    print(f"  Patients processed : {n_patients:>8,}")
    print(f"  Samples (patients) : {n_samples:>8,}")

    if has_notes:
        med, mx = median_max(note_token_counts)
        avail = availability_pct(note_has_data)
        print("\n── Notes (admission-context) ───────────────────────────────")
        print(f"  Median tokens : {med:>10.1f}")
        print(f"  Max tokens    : {mx:>10.0f}")
        print(f"  Availability  : {avail:>9.1f}%")

    if has_icd:
        med, mx = median_max(icd_token_counts)
        avail = availability_pct(icd_has_data)
        print("\n── ICD Codes ────────────────────────────────────────────────")
        print(f"  Median tokens : {med:>10.1f}")
        print(f"  Max tokens    : {mx:>10.0f}")
        print(f"  Availability  : {avail:>9.1f}%")

    med, mx = median_max(lab_token_counts)
    avail = availability_pct(lab_has_data)
    print("\n── Labs ─────────────────────────────────────────────────────")
    print(f"  Median tokens (timesteps) : {med:>10.1f}")
    print(f"  Max tokens                : {mx:>10.0f}")
    print(f"  Availability              : {avail:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
