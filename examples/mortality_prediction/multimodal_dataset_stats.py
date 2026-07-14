"""Standalone multimodal dataset statistics auditor.

Loads a MIMIC-IV multimodal task dataset and reports per-modality missingness
rates and token/element counts — no model, no GPU, no trainer required.

Works directly on the processed SampleDataset. Processed schema (single sample,
no batch dim):
  discharge_note_times / radiology_note_times:
    (input_ids, attn_mask, token_type_ids, time, type_tag)
    input_ids shape: (N_notes, 128)  attn_mask shape: (N_notes, 128)
    attn_mask.sum() gives real (non-padding) tokens seen by the model.
    Note: each note is independently truncated to 128 wordpieces — no chunking.
  icd_codes:
    (time, value)  value shape: (N_visits, vocab_size)  [multi-hot]
  labs_mask:
    (time, value)  value shape: (N_timesteps, 10)  [bool/float]
  cxr_image_times:
    (image, time, paths)  image shape: (N_images, 3, 224, 224)

Usage:
    python examples/mortality_prediction/multimodal_dataset_stats.py \\
        --dev --quick-test

    python examples/mortality_prediction/multimodal_dataset_stats.py \\
        --task ClinicalNotesICDLabsCXRMIMIC4 --output-csv /tmp/stats.csv
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List, Tuple

import numpy as np

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import (
    ClinicalNotesICDLabsCXRMIMIC4,
    ClinicalNotesICDLabsMIMIC4,
    ClinicalNotesMIMIC4,
    ICDLabsMIMIC4,
)

TASK_MAP = {
    "ClinicalNotesICDLabsCXRMIMIC4": ClinicalNotesICDLabsCXRMIMIC4,
    "ClinicalNotesICDLabsMIMIC4": ClinicalNotesICDLabsMIMIC4,
    "ClinicalNotesMIMIC4": ClinicalNotesMIMIC4,
    "ICDLabsMIMIC4": ICDLabsMIMIC4,
}

# bert-base-uncased encodes "[MISSING_TEXT]" to ~7 tokens with padding.
# Real notes are always longer. Used to detect the missingness sentinel.
_MISSING_NOTE_TOKEN_THRESHOLD = 15


def _note_stats(tup: tuple, note_max_len: int) -> Tuple[bool, int, int]:
    """Stats from a processed note tuple.

    Schema: (input_ids, attn_mask, token_type_ids, time, type_tag)
      input_ids shape: (N_notes, note_max_len)
      attn_mask shape: (N_notes, note_max_len)

    Returns (is_missing, n_notes, seen_tokens_total).
    seen_tokens = attn_mask.sum() — real non-padding tokens, already capped
    at note_max_len by the processor's truncation.
    """
    input_ids, attn_mask = tup[0], tup[1]
    n_notes = input_ids.shape[0]
    seen = int(attn_mask.sum().item())
    is_missing = (n_notes == 1) and (seen < _MISSING_NOTE_TOKEN_THRESHOLD)
    if is_missing:
        return True, 0, 0
    return False, n_notes, seen


def _icd_stats(tup: tuple) -> Tuple[bool, int, int]:
    """Stats from a processed icd_codes tuple.

    Schema: (time, value)  value shape: (N_visits, vocab_size) [multi-hot]

    Returns (is_missing, n_visits, n_code_activations).
    """
    value = tup[1]
    n_visits = value.shape[0]
    n_act = int(value.sum().item())
    return n_act == 0, n_visits, n_act


def _labs_stats(tup: tuple) -> Tuple[bool, int]:
    """Stats from a processed labs_mask tuple.

    Schema: (time, value)  value shape: (N_timesteps, 10) [bool/float]

    Returns (is_missing, n_observed).
    """
    value = tup[1]
    n_obs = int(value.sum().item())
    return n_obs == 0, n_obs


def _cxr_stats(tup: tuple, patch_count: int) -> Tuple[bool, int, int]:
    """Stats from a processed cxr_image_times tuple.

    Schema: (image, time, paths)  image shape: (N_images, 3, H, W)

    Returns (is_missing, n_images, cxr_tokens).
    A zero-valued image tensor indicates the missing-image sentinel.
    """
    image = tup[0]
    n_images = image.shape[0]
    if n_images == 0 or float(image.sum()) < 1e-8:
        return True, 0, 0
    return False, n_images, n_images * patch_count


def audit_sample(
    sample: Dict[str, Any],
    note_max_len: int,
    cxr_patch_count: int,
) -> Dict[str, Any]:
    """Audit one processed SampleDataset sample dict."""
    row: Dict[str, Any] = {}

    for note_key in ("discharge_note_times", "radiology_note_times"):
        if note_key not in sample:
            continue
        miss, n_notes, seen_tok = _note_stats(
            sample[note_key], note_max_len
        )
        row[f"{note_key}__missing"] = int(miss)
        row[f"{note_key}__n_notes"] = n_notes
        row[f"{note_key}__seen_tokens"] = seen_tok

    if "icd_codes" in sample:
        miss, n_visits, n_act = _icd_stats(sample["icd_codes"])
        row["icd_codes__missing"] = int(miss)
        row["icd_codes__n_visits"] = n_visits
        row["icd_codes__n_activations"] = n_act

    if "labs_mask" in sample:
        miss, n_obs = _labs_stats(sample["labs_mask"])
        row["labs__missing"] = int(miss)
        row["labs__n_obs"] = n_obs

    if "cxr_image_times" in sample:
        miss, n_img, cxr_tok = _cxr_stats(
            sample["cxr_image_times"], cxr_patch_count
        )
        row["cxr__missing"] = int(miss)
        row["cxr__n_images"] = n_img
        row["cxr__tokens"] = cxr_tok

    note_tok = sum(
        row.get(f"{k}__seen_tokens", 0)
        for k in ("discharge_note_times", "radiology_note_times")
    )
    row["total_tokens"] = (
        note_tok
        + row.get("icd_codes__n_activations", 0)
        + row.get("labs__n_obs", 0)
        + row.get("cxr__tokens", 0)
    )
    return row


def _stats(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "N/A"
    return (
        f"mean={arr.mean():.1f}  median={np.median(arr):.0f}"
        f"  p90={np.percentile(arr, 90):.0f}  max={arr.max():.0f}"
    )


def print_report(rows: List[Dict], args: argparse.Namespace) -> None:
    n = len(rows)
    print()
    print(f"Task: {args.task}   Samples: {n:,}   dev={args.dev}")
    print()

    def _arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if key in r], dtype=float)

    hdr = (
        f"{'Modality':<28} {'missing%':>10}"
        f"  {'mean':>8}  {'median':>8}  {'p90':>8}  {'max':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    modality_specs = []
    for note_key, label in [
        ("discharge_note_times", "discharge notes"),
        ("radiology_note_times", "radiology notes"),
    ]:
        miss = _arr(f"{note_key}__missing")
        if len(miss):
            modality_specs.append(
                (label, miss, _arr(f"{note_key}__n_notes"))
            )

    if rows and "icd_codes__missing" in rows[0]:
        modality_specs.append((
            "icd codes",
            _arr("icd_codes__missing"),
            _arr("icd_codes__n_activations"),
        ))
    if rows and "labs__missing" in rows[0]:
        modality_specs.append((
            "labs (observations)",
            _arr("labs__missing"),
            _arr("labs__n_obs"),
        ))
    if rows and "cxr__missing" in rows[0]:
        modality_specs.append((
            "cxr images",
            _arr("cxr__missing"),
            _arr("cxr__n_images"),
        ))

    for label, miss, counts in modality_specs:
        miss_pct = f"{miss.mean() * 100:.1f}%"
        print(
            f"{label:<28} {miss_pct:>10}  "
            f"{counts.mean():>8.1f}  {np.median(counts):>8.0f}  "
            f"{np.percentile(counts, 90):>8.0f}  {counts.max():>8.0f}"
        )

    print()
    print(f"Note seen-tokens (cap@{args.note_max_len}, from attn_mask):")
    for note_key, label in [
        ("discharge_note_times", "  discharge"),
        ("radiology_note_times", "  radiology"),
    ]:
        seen = _arr(f"{note_key}__seen_tokens")
        if len(seen) == 0:
            continue
        pct_full = (seen >= args.note_max_len).mean() * 100
        print(
            f"{label}: {_stats(seen)}"
            f"   % hitting cap={pct_full:.0f}%"
        )

    if rows and "cxr__tokens" in rows[0]:
        cxr_tok = _arr("cxr__tokens")
        print()
        print(f"CXR tokens (@{args.cxr_patch_count} patches/image):")
        print(f"  {_stats(cxr_tok)}")

    total = _arr("total_tokens")
    print()
    print("Aggregate tokens/sample:")
    print(f"  {_stats(total)}")
    print()


def run(args: argparse.Namespace) -> None:
    task_cls = TASK_MAP[args.task]
    needs_notes = args.task in (
        "ClinicalNotesMIMIC4",
        "ClinicalNotesICDLabsMIMIC4",
        "ClinicalNotesICDLabsCXRMIMIC4",
    )
    needs_cxr = args.task == "ClinicalNotesICDLabsCXRMIMIC4"
    needs_icd = args.task in (
        "ClinicalNotesICDLabsMIMIC4",
        "ICDLabsMIMIC4",
        "ClinicalNotesICDLabsCXRMIMIC4",
    )

    ehr_tables = (
        ["diagnoses_icd", "procedures_icd", "labevents"]
        if needs_icd
        else []
    )
    note_tables = ["discharge", "radiology"] if needs_notes else []
    cxr_tables = (
        ["metadata", "negbio", "chexpert", "split"] if needs_cxr else []
    )

    print("Loading MIMIC4Dataset ...")
    base_dataset = MIMIC4Dataset(
        ehr_root=args.ehr_root,
        note_root=args.note_root if needs_notes else None,
        cxr_root=args.cxr_root if needs_cxr else None,
        cxr_variant=args.cxr_variant,
        ehr_tables=ehr_tables,
        note_tables=note_tables,
        cxr_tables=cxr_tables,
        cache_dir=args.cache_dir,
        dev=args.dev,
        num_workers=args.num_workers,
    )

    task = task_cls(window_hours=args.observation_window_hours)
    print("Running set_task ...")
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)
    total = len(sample_dataset)
    print(f"Total samples: {total:,}")

    limit = (
        min(args.sample_limit, total)
        if args.sample_limit and args.sample_limit > 0
        else total
    )
    print(f"Auditing {limit:,} samples ...")

    rows: List[Dict] = []
    for i in range(limit):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {limit:,} ...")
        rows.append(
            audit_sample(
                sample_dataset[i],
                args.note_max_len,
                args.cxr_patch_count,
            )
        )

    if not rows:
        print("No samples. Check roots/tables/task combination.")
        return

    print_report(rows, args)

    if args.output_csv:
        all_keys = list(rows[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-sample CSV written to: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit modality missingness and token counts "
            "for MIMIC-IV multimodal tasks."
        )
    )
    parser.add_argument(
        "--ehr-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/mimiciv/2.2",
    )
    parser.add_argument(
        "--note-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/mimic-note",
    )
    parser.add_argument(
        "--cxr-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/MIMIC-CXR",
    )
    parser.add_argument(
        "--cxr-variant",
        type=str,
        default="sunlab",
        choices=["default", "sunlab"],
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/shared/eng/pyhealth"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ClinicalNotesICDLabsCXRMIMIC4",
        choices=list(TASK_MAP.keys()),
    )
    parser.add_argument("--observation-window-hours", type=int, default=24)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--note-max-len", type=int, default=128)
    parser.add_argument("--cxr-patch-count", type=int, default=196)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None)

    args = parser.parse_args()
    if args.quick_test:
        args.dev = True
        if args.sample_limit is None:
            args.sample_limit = 50
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
