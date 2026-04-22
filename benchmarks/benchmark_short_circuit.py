#!/usr/bin/env python3
"""
Benchmark: Short-circuit vs fetch-all-then-check in task sample processing.

Issue #843: Investigate whether short-circuit if statements speed up task
sample processing.

This script benchmarks the ACTUAL PyHealth task processing pipeline on real
MIMIC-III (and optionally MIMIC-IV) data. It compares:
  A) Current pattern  -- fetch all event types, THEN check if any are empty
  B) Proposed pattern -- fetch one, check, skip remaining if empty (short-circuit)

The benchmark measures wall-clock time for the task.__call__() hot path over
every patient in the dataset, which is where the optimization matters.

Prerequisites:
  - PyHealth installed in editable mode: pip install -e .
  - MIMIC-III data at a known path (or use the bundled demo data)
  - Optionally: MIMIC-IV data for additional benchmarks

Usage:
    # Quick run with bundled 100-patient demo data (no download needed)
    python benchmarks/benchmark_short_circuit.py

    # Full run on real MIMIC-III (46K+ patients)
    python benchmarks/benchmark_short_circuit.py --mimic3-root /path/to/mimic-iii/1.4

    # Full run on both MIMIC-III and MIMIC-IV
    python benchmarks/benchmark_short_circuit.py \\
        --mimic3-root /path/to/mimic-iii/1.4 \\
        --mimic4-root /path/to/mimic-iv/2.2

    # Control number of benchmark trials
    python benchmarks/benchmark_short_circuit.py \\
        --mimic3-root /path/to/mimic-iii/1.4 \\
        --trials 10
"""

import argparse
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure PyHealth is importable (handles running from repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC3Dataset

logger = logging.getLogger(__name__)


# ============================================================================
# TASK IMPLEMENTATIONS: Original vs Short-circuit
# ============================================================================
# We define both variants as plain functions that take a Patient object,
# matching the signature of task.__call__(). This lets us benchmark them
# head-to-head on the same patient data without modifying the installed code.
# ============================================================================


def mortality_original(patient: Any) -> List[Dict[str, Any]]:
    """CURRENT pattern: fetch all event types, then multiply-check."""
    samples = []
    visits = patient.get_events(event_type="admissions")
    if len(visits) <= 1:
        return []

    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]

        if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.hospital_expire_flag)

        # Fetch ALL three -- even if the first one is empty
        diagnoses = patient.get_events(
            event_type="diagnoses_icd",
            filters=[("hadm_id", "==", visit.hadm_id)],
        )
        procedures = patient.get_events(
            event_type="procedures_icd",
            filters=[("hadm_id", "==", visit.hadm_id)],
        )
        prescriptions = patient.get_events(
            event_type="prescriptions",
            filters=[("hadm_id", "==", visit.hadm_id)],
        )

        conditions = [event.icd9_code for event in diagnoses]
        procedures_list = [event.icd9_code for event in procedures]
        drugs = [event.ndc for event in prescriptions if event.ndc]

        # Multiply-check
        if len(conditions) * len(procedures_list) * len(drugs) == 0:
            continue

        samples.append(
            {
                "hadm_id": visit.hadm_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures_list,
                "drugs": drugs,
                "mortality": mortality_label,
            }
        )
    return samples


def mortality_short_circuit(patient: Any) -> List[Dict[str, Any]]:
    """PROPOSED pattern: fetch one, check, skip remaining if empty."""
    samples = []
    visits = patient.get_events(event_type="admissions")
    if len(visits) <= 1:
        return []

    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]

        if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.hospital_expire_flag)

        # Short-circuit: check after EACH fetch
        diagnoses = patient.get_events(
            event_type="diagnoses_icd",
            filters=[("hadm_id", "==", visit.hadm_id)],
        )
        conditions = [event.icd9_code for event in diagnoses]
        if not conditions:
            continue

        procedures = patient.get_events(
            event_type="procedures_icd",
            filters=[("hadm_id", "==", visit.hadm_id)],
        )
        procedures_list = [event.icd9_code for event in procedures]
        if not procedures_list:
            continue

        prescriptions = patient.get_events(
            event_type="prescriptions",
            filters=[("hadm_id", "==", visit.hadm_id)],
        )
        drugs = [event.ndc for event in prescriptions if event.ndc]
        if not drugs:
            continue

        samples.append(
            {
                "hadm_id": visit.hadm_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures_list,
                "drugs": drugs,
                "mortality": mortality_label,
            }
        )
    return samples


# ============================================================================
# BENCHMARK HARNESS
# ============================================================================


def load_patients(dataset, max_patients: int = None) -> List[Patient]:
    """Load all Patient objects from a PyHealth dataset.

    Uses dataset.iter_patients() which is the most efficient way to
    iterate — it partitions the LazyFrame and yields Patient objects
    without the O(n) membership check that get_patient() does.
    """
    patients = []
    for p in dataset.iter_patients():
        patients.append(p)
        if max_patients and len(patients) >= max_patients:
            break
    return patients


def verify_equivalence(
    patients: List[Patient],
    fn_a,
    fn_b,
    max_patients: int = 50,
) -> None:
    """Verify both functions produce identical outputs on a sample of patients."""
    for p in patients[:max_patients]:
        result_a = fn_a(p)
        result_b = fn_b(p)
        assert len(result_a) == len(result_b), (
            f"Patient {p.patient_id}: original produced {len(result_a)} samples, "
            f"short-circuit produced {len(result_b)} samples"
        )
        for sa, sb in zip(result_a, result_b):
            assert sa == sb, (
                f"Patient {p.patient_id}: sample mismatch\n"
                f"  Original:      {sa}\n"
                f"  Short-circuit: {sb}"
            )
    print(f"  OK  Equivalence verified on {min(max_patients, len(patients))} patients")


def count_skips(patients: List[Patient]) -> Dict[str, int]:
    """
    Count how often each event type is empty (would trigger a skip).
    This tells us the THEORETICAL benefit of short-circuiting.
    """
    stats = {
        "total_visits": 0,
        "empty_diagnoses": 0,
        "empty_procedures": 0,
        "empty_prescriptions": 0,
        "any_empty": 0,
    }

    for p in patients:
        visits = p.get_events(event_type="admissions")
        if len(visits) <= 1:
            continue
        for i in range(len(visits) - 1):
            visit = visits[i]
            stats["total_visits"] += 1

            diagnoses = p.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            conditions = [e.icd9_code for e in diagnoses]

            procedures = p.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            procs = [e.icd9_code for e in procedures]

            prescriptions = p.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            drugs = [e.ndc for e in prescriptions if e.ndc]

            d_empty = len(conditions) == 0
            p_empty = len(procs) == 0
            r_empty = len(drugs) == 0

            if d_empty:
                stats["empty_diagnoses"] += 1
            if p_empty:
                stats["empty_procedures"] += 1
            if r_empty:
                stats["empty_prescriptions"] += 1
            if d_empty or p_empty or r_empty:
                stats["any_empty"] += 1

    return stats


def run_benchmark(
    patients: List[Patient],
    fn_original,
    fn_short_circuit,
    n_trials: int = 5,
    label: str = "MortalityPrediction",
) -> Dict[str, Any]:
    """Run timed benchmark comparing two task functions."""

    original_times = []
    short_circuit_times = []

    for trial in range(n_trials):
        # --- Original ---
        t0 = time.perf_counter()
        total_original = 0
        for p in patients:
            total_original += len(fn_original(p))
        t1 = time.perf_counter()
        original_times.append(t1 - t0)

        # --- Short-circuit ---
        t0 = time.perf_counter()
        total_sc = 0
        for p in patients:
            total_sc += len(fn_short_circuit(p))
        t1 = time.perf_counter()
        short_circuit_times.append(t1 - t0)

    orig_mean = statistics.mean(original_times)
    orig_std = statistics.stdev(original_times) if n_trials > 1 else 0.0
    sc_mean = statistics.mean(short_circuit_times)
    sc_std = statistics.stdev(short_circuit_times) if n_trials > 1 else 0.0

    speedup = orig_mean / sc_mean if sc_mean > 0 else float("inf")
    pct_saved = (1 - sc_mean / orig_mean) * 100 if orig_mean > 0 else 0

    return {
        "label": label,
        "n_patients": len(patients),
        "total_samples": total_original,
        "original_mean": orig_mean,
        "original_std": orig_std,
        "short_circuit_mean": sc_mean,
        "short_circuit_std": sc_std,
        "speedup": speedup,
        "pct_saved": pct_saved,
        "original_times": original_times,
        "short_circuit_times": short_circuit_times,
    }


def print_result(result: Dict[str, Any]) -> None:
    """Pretty-print benchmark result."""
    print(f"\n{'=' * 70}")
    print(f"  {result['label']}")
    print(f"  {result['n_patients']} patients, {result['total_samples']} samples produced")
    print(f"{'=' * 70}")
    print(f"  Fetch-all-then-check (original): {result['original_mean']:.4f}s +/- {result['original_std']:.4f}s")
    print(f"  Short-circuit (proposed):        {result['short_circuit_mean']:.4f}s +/- {result['short_circuit_std']:.4f}s")
    print(f"  Speedup:                         {result['speedup']:.2f}x ({result['pct_saved']:.1f}% faster)")
    print(f"{'=' * 70}")


def print_skip_stats(stats: Dict[str, int]) -> None:
    """Print skip statistics."""
    total = stats["total_visits"]
    if total == 0:
        print("  No visits found.")
        return
    print(f"\n  Skip analysis ({total} total visits across all patients):")
    print(f"    Empty diagnoses:     {stats['empty_diagnoses']:>6d} ({stats['empty_diagnoses']/total*100:.1f}%)")
    print(f"    Empty procedures:    {stats['empty_procedures']:>6d} ({stats['empty_procedures']/total*100:.1f}%)")
    print(f"    Empty prescriptions: {stats['empty_prescriptions']:>6d} ({stats['empty_prescriptions']/total*100:.1f}%)")
    print(f"    Any field empty:     {stats['any_empty']:>6d} ({stats['any_empty']/total*100:.1f}%)")
    print(f"    ---")
    print(f"    Short-circuit can skip 1-2 get_events() calls")
    print(f"    for {stats['any_empty']/total*100:.1f}% of visits")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark short-circuit optimization in PyHealth task processing (Issue #843)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with bundled demo data (100 patients)
  python benchmarks/benchmark_short_circuit.py

  # Full MIMIC-III benchmark
  python benchmarks/benchmark_short_circuit.py --mimic3-root /path/to/mimic-iii/1.4

  # Full benchmark with more trials for statistical confidence
  python benchmarks/benchmark_short_circuit.py --mimic3-root /path/to/mimic-iii/1.4 --trials 10

  # Only run skip analysis (no timing, useful for understanding the data)
  python benchmarks/benchmark_short_circuit.py --mimic3-root /path/to/mimic-iii/1.4 --skip-analysis-only
""",
    )
    parser.add_argument(
        "--mimic3-root",
        type=str,
        default=None,
        help="Path to MIMIC-III v1.4 data directory (contains PATIENTS.csv.gz, etc.). "
        "If not provided, uses the bundled 100-patient demo dataset.",
    )
    parser.add_argument(
        "--mimic4-root",
        type=str,
        default=None,
        help="Path to MIMIC-IV v2.2 data directory (contains hosp/, icu/ subdirs). Optional.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of benchmark trials (default: 5). More trials = tighter confidence.",
    )
    parser.add_argument(
        "--skip-analysis-only",
        action="store_true",
        help="Only run skip analysis (count empty rates), no timing benchmark.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit the number of patients to process (useful for quick testing on full MIMIC).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PyHealth Issue #843: Short-circuit Optimization Benchmark")
    print("=" * 70)

    results = []

    # ------------------------------------------------------------------
    # MIMIC-III Benchmark
    # ------------------------------------------------------------------
    if args.mimic3_root:
        mimic3_path = args.mimic3_root
        print(f"\n  Loading MIMIC-III from: {mimic3_path}")
    else:
        # Use bundled demo data
        mimic3_path = str(REPO_ROOT / "test-resources" / "core" / "mimic3demo")
        print(f"\n  No --mimic3-root provided, using bundled demo data")
        print(f"  Path: {mimic3_path}")

    print(f"  Loading dataset...")
    t0 = time.perf_counter()
    dataset = MIMIC3Dataset(
        root=mimic3_path,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    )
    t1 = time.perf_counter()
    print(f"  Dataset loaded in {t1 - t0:.1f}s")

    print(f"  Loading patient objects...")
    t0 = time.perf_counter()
    patients = load_patients(dataset, max_patients=args.max_patients)
    t1 = time.perf_counter()
    suffix = f" (limited by --max-patients)" if args.max_patients else ""
    print(f"  Loaded {len(patients)} patients{suffix} in {t1 - t0:.1f}s")

    # --- Skip analysis ---
    print(f"\n  Running skip analysis...")
    skip_stats = count_skips(patients)
    print_skip_stats(skip_stats)

    if not args.skip_analysis_only:
        # --- Equivalence check ---
        print(f"\n  Verifying equivalence...")
        verify_equivalence(patients, mortality_original, mortality_short_circuit)

        # --- Timed benchmark ---
        print(f"\n  Running benchmark ({args.trials} trials)...")
        result = run_benchmark(
            patients,
            mortality_original,
            mortality_short_circuit,
            n_trials=args.trials,
            label=f"MortalityPredictionMIMIC3 ({len(patients)} patients)",
        )
        print_result(result)
        results.append(result)

    # ------------------------------------------------------------------
    # MIMIC-IV Benchmark (optional)
    # ------------------------------------------------------------------
    if args.mimic4_root:
        from pyhealth.datasets import MIMIC4EHRDataset

        print(f"\n  Loading MIMIC-IV from: {args.mimic4_root}")
        t0 = time.perf_counter()
        dataset4 = MIMIC4EHRDataset(
            root=args.mimic4_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        )
        t1 = time.perf_counter()
        print(f"  Dataset loaded in {t1 - t0:.1f}s")

        patients4 = load_patients(dataset4, max_patients=args.max_patients)
        print(f"  Loaded {len(patients4)} patients")

        # MIMIC-IV uses icd_code instead of icd9_code
        def mortality_original_m4(patient: Any) -> List[Dict[str, Any]]:
            samples = []
            demographics = patient.get_events(event_type="patients")
            if not demographics:
                return []
            admissions = patient.get_events(event_type="admissions")
            if len(admissions) <= 1:
                return []
            for i in range(len(admissions) - 1):
                admission = admissions[i]
                next_admission = admissions[i + 1]
                if next_admission.hospital_expire_flag not in [0, 1, "0", "1"]:
                    mortality_label = 0
                else:
                    mortality_label = int(next_admission.hospital_expire_flag)
                try:
                    dischtime = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
                except (ValueError, AttributeError):
                    continue
                diagnoses_icd = patient.get_events(
                    event_type="diagnoses_icd", start=admission.timestamp, end=dischtime
                )
                procedures_icd = patient.get_events(
                    event_type="procedures_icd", start=admission.timestamp, end=dischtime
                )
                prescriptions = patient.get_events(
                    event_type="prescriptions", start=admission.timestamp, end=dischtime
                )
                conditions = [str(getattr(e, "icd_code", "")).strip() for e in diagnoses_icd if getattr(e, "icd_code", None)]
                procedures_list = [str(getattr(e, "icd_code", "")).strip() for e in procedures_icd if getattr(e, "icd_code", None)]
                drugs = [str(getattr(e, "ndc", "")).strip() for e in prescriptions if getattr(e, "ndc", None)]
                if len(conditions) * len(procedures_list) * len(drugs) == 0:
                    continue
                samples.append({
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                })
            return samples

        def mortality_short_circuit_m4(patient: Any) -> List[Dict[str, Any]]:
            samples = []
            demographics = patient.get_events(event_type="patients")
            if not demographics:
                return []
            admissions = patient.get_events(event_type="admissions")
            if len(admissions) <= 1:
                return []
            for i in range(len(admissions) - 1):
                admission = admissions[i]
                next_admission = admissions[i + 1]
                if next_admission.hospital_expire_flag not in [0, 1, "0", "1"]:
                    mortality_label = 0
                else:
                    mortality_label = int(next_admission.hospital_expire_flag)
                try:
                    dischtime = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
                except (ValueError, AttributeError):
                    continue
                diagnoses_icd = patient.get_events(
                    event_type="diagnoses_icd", start=admission.timestamp, end=dischtime
                )
                conditions = [str(getattr(e, "icd_code", "")).strip() for e in diagnoses_icd if getattr(e, "icd_code", None)]
                if not conditions:
                    continue
                procedures_icd = patient.get_events(
                    event_type="procedures_icd", start=admission.timestamp, end=dischtime
                )
                procedures_list = [str(getattr(e, "icd_code", "")).strip() for e in procedures_icd if getattr(e, "icd_code", None)]
                if not procedures_list:
                    continue
                prescriptions = patient.get_events(
                    event_type="prescriptions", start=admission.timestamp, end=dischtime
                )
                drugs = [str(getattr(e, "ndc", "")).strip() for e in prescriptions if getattr(e, "ndc", None)]
                if not drugs:
                    continue
                samples.append({
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                })
            return samples

        if not args.skip_analysis_only:
            print(f"\n  Verifying equivalence (MIMIC-IV)...")
            verify_equivalence(patients4, mortality_original_m4, mortality_short_circuit_m4)

            print(f"\n  Running benchmark ({args.trials} trials)...")
            result4 = run_benchmark(
                patients4,
                mortality_original_m4,
                mortality_short_circuit_m4,
                n_trials=args.trials,
                label=f"MortalityPredictionMIMIC4 ({len(patients4)} patients)",
            )
            print_result(result4)
            results.append(result4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if results:
        print(f"\n\n{'=' * 70}")
        print(f"  SUMMARY")
        print(f"{'=' * 70}")
        for r in results:
            print(
                f"  {r['label']:<55s} "
                f"{r['speedup']:.2f}x ({r['pct_saved']:.1f}% faster)"
            )
        print(f"{'=' * 70}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()