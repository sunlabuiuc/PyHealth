#!/usr/bin/env python3
"""Compute per-modality token counts and missing-token rates for Table 2.

Iterates patients directly via dataset.iter_patients() and calls the task
function on each one — never materializes all samples in memory, no litdata
write, no GPU needed.

Usage (on CC):
    python scripts/compute_token_stats.py \
        --ehr-root /projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2 \
        --note-root /projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note \
        --cache-dir /u/rianatri/pyhealth_cache \
        --task notes_labs
"""
import argparse
import os

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ehr-root", required=True)
    p.add_argument("--note-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dev", action="store_true", help="Limit to 1000 patients")
    p.add_argument(
        "--task",
        type=str,
        choices=["clinical_notes_icd_labs", "notes_labs"],
        default="notes_labs",
        help=(
            "Task to profile. 'notes_labs' (default) uses admission-context "
            "text sections; 'clinical_notes_icd_labs' profiles the legacy task."
        ),
    )
    p.add_argument(
        "--icd-codes",
        action="store_true",
        default=False,
        help="Include discharge-coded ICD codes when using --task notes_labs.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("PYHEALTH_DISABLE_DASK_DISTRIBUTED", "1")

    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.tasks.multimodal_mimic4 import (
        ClinicalNotesICDLabsMIMIC4,
        NotesLabsMIMIC4,
    )

    print("Building dataset (uses cache if available)...")

    if args.task == "clinical_notes_icd_labs":
        ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]
        note_tables = ["discharge", "radiology"]
        task = ClinicalNotesICDLabsMIMIC4()
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
        note_root=args.note_root,
        ehr_tables=ehr_tables,
        note_tables=note_tables,
        dev=args.dev,
    )
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    dataset = MIMIC4Dataset(**kwargs)

    MISSING_TEXT = ""
    LAB_CATEGORIES = task.LAB_CATEGORY_NAMES  # 10 names

    # Counters
    n_patients = n_samples = 0
    note_total = note_missing = 0
    note_empty_extracted = 0  # notes present but section extraction returned nothing
    icd_total_visits = icd_missing_visits = 0
    icd_total_codes = 0
    lab_total_timesteps = lab_missing_timesteps = 0
    lab_per_cat_missing = np.zeros(len(LAB_CATEGORIES), dtype=np.int64)

    print("Iterating patients and accumulating stats (no litdata write)...")
    for patient in tqdm(dataset.iter_patients(), total=len(dataset.unique_patient_ids)):
        n_patients += 1
        samples = task(patient)
        if not samples:
            continue
        for s in samples:
            n_samples += 1

            # ── notes ────────────────────────────────────────────
            if args.task == "clinical_notes_icd_labs":
                disc_texts, _ = s["discharge_note_times"]
                note_total += len(disc_texts)
                note_missing += sum(1 for t in disc_texts if t == MISSING_TEXT)

                rad_texts, _ = s["radiology_note_times"]
                note_total += len(rad_texts)
                note_missing += sum(1 for t in rad_texts if t == MISSING_TEXT)
            else:
                note_texts, _ = s["admission_note_times"]
                note_total += len(note_texts)
                for t in note_texts:
                    if t == MISSING_TEXT:
                        note_missing += 1
                    elif len(t) <= 1024 and t == t[:1024]:
                        # Heuristic: if the note is exactly 1024 chars, it likely
                        # came from the fallback raw-note path (no sections found).
                        note_empty_extracted += 1

            # ── ICD codes ────────────────────────────────────────
            if "icd_codes" in s:
                _, icd_visits = s["icd_codes"]
                icd_total_visits += len(icd_visits)
                for visit_codes in icd_visits:
                    if visit_codes == [MISSING_TEXT]:
                        icd_missing_visits += 1
                    else:
                        icd_total_codes += len(visit_codes)

            # ── labs ─────────────────────────────────────────────
            _, lab_masks = s["labs_mask"]
            for mask_row in lab_masks:
                lab_total_timesteps += 1
                if not any(mask_row):
                    lab_missing_timesteps += 1
                for i, observed in enumerate(mask_row):
                    if not observed:
                        lab_per_cat_missing[i] += 1

    def pct(n, d):
        return 100.0 * n / d if d else float("nan")

    print("\n" + "=" * 60)
    print(f"TOKEN STATS — {task.task_name}")
    print("=" * 60)
    print(f"  Patients processed : {n_patients:>8,}")
    print(f"  Samples (patients) : {n_samples:>8,}")

    if args.task == "clinical_notes_icd_labs":
        print("\n── Notes (discharge + radiology) ───────────────────────────")
    else:
        print("\n── Admission-context notes ─────────────────────────────────")
    print(f"  Total note tokens      : {note_total:>8,}")
    print(f"  Missing (empty str)    : {note_missing:>8,}  ({pct(note_missing, note_total):.1f}%)")
    if args.task == "notes_labs":
        print(
            f"  Fallback (raw prefix)  : {note_empty_extracted:>8,}  "
            f"({pct(note_empty_extracted, note_total):.1f}%)"
        )
        print(
            f"  Effective coverage     : {note_total - note_missing:>8,}  "
            f"({pct(note_total - note_missing, note_total):.1f}%)"
        )

    if "icd_codes" in s:
        print("\n── ICD Codes ────────────────────────────────────────────────")
        print(f"  Total visit tokens  : {icd_total_visits:>8,}")
        print(
            f'  Missing visits [""] : {icd_missing_visits:>8,}  '
            f"({pct(icd_missing_visits, icd_total_visits):.1f}%)"
        )
        print(f"  Total ICD codes     : {icd_total_codes:>8,}  (in non-missing visits)")

    print("\n── Labs ─────────────────────────────────────────────────────")
    print(f"  Total timestep tokens : {lab_total_timesteps:>8,}")
    print(
        f"  Missing timesteps     : {lab_missing_timesteps:>8,}  "
        f"({pct(lab_missing_timesteps, lab_total_timesteps):.1f}%)"
    )
    print(f"\n  Per-category missingness (across all timesteps):")
    for i, cat in enumerate(LAB_CATEGORIES):
        n_miss = int(lab_per_cat_missing[i])
        print(
            f"    {cat:<15}: {n_miss:>8,} / {lab_total_timesteps:>8,}  "
            f"({pct(n_miss, lab_total_timesteps):.1f}% missing)"
        )

    print("\n── Summary ──────────────────────────────────────────────────")
    total_tokens = note_total + icd_total_visits + lab_total_timesteps
    total_missing = note_missing + icd_missing_visits + lab_missing_timesteps
    print(f"  All modalities total tokens  : {total_tokens:>8,}")
    print(
        f"  All modalities missing tokens: {total_missing:>8,}  "
        f"({pct(total_missing, total_tokens):.1f}%)"
    )
    print()


if __name__ == "__main__":
    main()
