"""
PyHealth Multimodal MIMIC-IV Demo: Patient-Level Mortality Prediction

This script demonstrates PyHealth's capability to load and process
multimodal medical data from MIMIC-IV at the PATIENT LEVEL, including:
- EHR codes (ICD-10 diagnoses and procedures)
- Clinical notes (discharge summaries, radiology reports)
- Lab events (time-series lab values)
- Chest X-ray images

Patient-Level Aggregation:
    - For patients who die: aggregates all data UP TO (excluding) death visit
    - For patients who survive: aggregates all data across all visits
    - Returns ONE sample per patient with complete medical history

It also benchmarks memory usage and processing time.

Data Sources:
    - EHR data: MIMIC-IV hosp module (patients, admissions, diagnoses, etc.)
    - Clinical notes: MIMIC-IV note module (discharge, radiology)
    - Chest X-rays: MIMIC-CXR (images and metadata)

Usage:
    python multimodal_mimic4_demo.py
    python multimodal_mimic4_demo.py --ehr-root /path/to/mimic-iv/hosp
    python multimodal_mimic4_demo.py --note-root /path/to/mimic-iv/note
    python multimodal_mimic4_demo.py --cxr-root /path/to/mimic-cxr
    python multimodal_mimic4_demo.py --save-labs ./output
    python multimodal_mimic4_demo.py --dev
"""

from __future__ import annotations

import argparse
import os
import shutil
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import psutil


# =============================================================================
# Utility Functions
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_directory_size(path: str | Path) -> int:
    """Calculate total size of a directory."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    try:
        for entry in p.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except FileNotFoundError:
                    pass
    except Exception as e:
        print(f"Error calculating size for {p}: {e}")
    return total


def ensure_empty_dir(path: str | Path) -> None:
    """Ensure directory exists and is empty."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def remove_dir(path: str | Path, retries: int = 3, delay: float = 1.0) -> None:
    """Remove a directory with retry logic."""
    p = Path(path)
    if not p.exists():
        return
    for attempt in range(retries):
        try:
            shutil.rmtree(p)
            return
        except OSError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"Warning: Failed to delete {p}: {e}")


def truncate_text(text: str, max_words: int = 100) -> str:
    """Truncate text to max_words with '...' suffix."""
    if not text:
        return "[No text available]"
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def print_section(title: str, width: int = 80) -> None:
    """Print a section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# Memory Tracking
# =============================================================================

class PeakMemoryTracker:
    """Tracks peak RSS for current process + children."""

    def __init__(self, poll_interval_s: float = 0.1) -> None:
        self._proc = psutil.Process(os.getpid())
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self._peak = 0

    def stop(self) -> None:
        self._stop.set()

    def peak_bytes(self) -> int:
        with self._lock:
            return self._peak

    def _total_rss_bytes(self) -> int:
        total = 0
        try:
            total += self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._total_rss_bytes()
            with self._lock:
                if rss > self._peak:
                    self._peak = rss
            time.sleep(self._poll_interval_s)


# =============================================================================
# MedCode Lookup
# =============================================================================

def lookup_icd_codes(
    codes: List[str],
    code_system: str = "ICD10CM",
    max_display: int = 10,
) -> List[Dict[str, str]]:
    """Look up ICD code names using PyHealth medcode.

    Args:
        codes: List of ICD codes
        code_system: Either "ICD10CM" or "ICD9CM"
        max_display: Maximum number of codes to look up

    Returns:
        List of dicts with code and name
    """
    try:
        from pyhealth.medcode import InnerMap

        icd_map = InnerMap.load(code_system)

        results = []
        for code in codes[:max_display]:
            try:
                name = icd_map.lookup(code)
                results.append({"code": code, "name": name})
            except (KeyError, Exception):
                try:
                    clean_code = code.replace(".", "")
                    name = icd_map.lookup(clean_code)
                    results.append({"code": code, "name": name})
                except (KeyError, Exception):
                    results.append({"code": code, "name": "[Unknown code]"})

        return results

    except ImportError:
        return [{"code": c, "name": "[medcode not available]"}
                for c in codes[:max_display]]
    except Exception as e:
        print(f"Warning: Could not load {code_system}: {e}")
        return [{"code": c, "name": f"[{code_system} unavailable]"}
                for c in codes[:max_display]]


# =============================================================================
# Display Functions
# =============================================================================

LAB_CATEGORIES = [
    "Sodium", "Potassium", "Chloride", "Bicarbonate", "Glucose",
    "Calcium", "Magnesium", "Anion Gap", "Osmolality", "Phosphate"
]


def display_lab_stats(labs_data: tuple) -> None:
    """Display lab event statistics."""
    lab_times, lab_values = labs_data

    print(f"  Total lab measurements: {len(lab_times)}")
    print(f"  Time span: {min(lab_times):.1f}h to {max(lab_times):.1f}h")

    print("\n  Lab Category Statistics:")
    print("  " + "-" * 56)
    print(f"  {'Category':<15} {'Count':>8} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 56)

    for idx, category in enumerate(LAB_CATEGORIES):
        values = [v[idx] for v in lab_values if v[idx] is not None]
        if values:
            arr = np.array(values)
            print(
                f"  {category:<15} {len(values):>8} "
                f"{np.mean(arr):>10.1f} {np.min(arr):>10.1f} {np.max(arr):>10.1f}"
            )
        else:
            print(f"  {category:<15} {'N/A':>8} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10}")


def save_labs_to_csv(
    labs_data: tuple,
    patient_id: str,
    output_path: str,
) -> bool:
    """Save lab events to a CSV file.

    Args:
        labs_data: Tuple of (times_list, values_list) from the sample
        patient_id: Patient identifier for the filename
        output_path: Directory to save the CSV file

    Returns:
        True if successful, False otherwise
    """
    try:
        import csv

        lab_times, lab_values = labs_data

        # Create output directory if needed
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"patient_{patient_id}_labs.csv"

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ["time_hours"] + LAB_CATEGORIES
            writer.writerow(header)

            # Write data rows
            for time_val, values in zip(lab_times, lab_values):
                row = [f"{time_val:.2f}"]
                for val in values:
                    if val is not None:
                        row.append(f"{val:.2f}")
                    else:
                        row.append("")
                writer.writerow(row)

        print(f"  ✓ Lab events saved to: {csv_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error saving labs to CSV: {e}")
        return False


def display_image(image_path: str, output_path: Optional[str] = None) -> bool:
    """Display chest X-ray image if available."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        if not os.path.exists(image_path):
            print(f"  [Image not found: {image_path}]")
            return False

        img = Image.open(image_path)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title("Chest X-Ray", fontsize=14)
        ax.axis("off")

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Image saved to: {output_path}")
        else:
            plt.show()

        plt.close()
        return True

    except ImportError:
        print("  [matplotlib/PIL not available for image display]")
        return False
    except Exception as e:
        print(f"  [Error displaying image: {e}]")
        return False


def showcase_sample(sample: Dict[str, Any], cxr_root: Optional[str] = None,
                    save_image: Optional[str] = None,
                    save_labs_dir: Optional[str] = None) -> None:
    """Display a patient-level multimodal sample with all its components.

    Args:
        sample: The patient sample dict containing aggregated multimodal data
        cxr_root: Root directory for CXR images
        save_image: Path to save X-ray image visualization
        save_labs_dir: Directory to save lab events CSV file
    """

    # =========================================================================
    # EHR Codes with MedCode Lookup
    # =========================================================================
    print_section("EHR Codes (ICD-10)")

    # Diagnoses
    conditions = sample.get("conditions", [])
    print_subsection(f"Diagnosis Codes ({len(conditions)} total)")

    if conditions:
        diagnosis_info = lookup_icd_codes(conditions, "ICD10CM", max_display=10)
        if all(info["name"] == "[Unknown code]" for info in diagnosis_info):
            diagnosis_info = lookup_icd_codes(conditions, "ICD9CM", max_display=10)

        for info in diagnosis_info:
            print(f"  • {info['code']}: {info['name']}")

        if len(conditions) > 10:
            print(f"  ... and {len(conditions) - 10} more codes")

    # Procedures
    procedures = sample.get("procedures", [])
    print_subsection(f"Procedure Codes ({len(procedures)} total)")

    if procedures:
        procedure_info = lookup_icd_codes(procedures, "ICD10PROC", max_display=5)
        if all(info["name"] in ["[Unknown code]", "[ICD10PROC unavailable]"]
               for info in procedure_info):
            procedure_info = lookup_icd_codes(procedures, "ICD9PROC", max_display=5)

        for info in procedure_info:
            print(f"  • {info['code']}: {info['name']}")

        if len(procedures) > 5:
            print(f"  ... and {len(procedures) - 5} more codes")

    # Drugs
    drugs = sample.get("drugs", [])
    print_subsection(f"Drug Prescriptions ({len(drugs)} total)")
    if drugs:
        for drug in drugs[:5]:
            print(f"  • {drug}")
        if len(drugs) > 5:
            print(f"  ... and {len(drugs) - 5} more drugs")

    # =========================================================================
    # Clinical Notes
    # =========================================================================
    print_section("Clinical Notes")

    # Radiology report (truncated to <100 words)
    print_subsection("Radiology Report Summary (<100 words)")
    radiology_text = sample.get("radiology", "")
    truncated_radiology = truncate_text(radiology_text, max_words=100)
    wrapped = textwrap.fill(truncated_radiology, width=70,
                            initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

    # Discharge summary (truncated)
    print_subsection("Discharge Summary Excerpt (<100 words)")
    discharge_text = sample.get("discharge", "")
    truncated_discharge = truncate_text(discharge_text, max_words=100)
    wrapped = textwrap.fill(truncated_discharge, width=70,
                            initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

    # =========================================================================
    # Lab Events (Aggregated Across All Visits)
    # =========================================================================
    print_section("Lab Events (Time-Series, Aggregated)")

    labs_data = sample.get("labs")
    if labs_data and isinstance(labs_data, tuple) and len(labs_data) == 2:
        display_lab_stats(labs_data)

        # Save to CSV if directory specified
        if save_labs_dir:
            patient_id = sample.get("patient_id", "unknown")
            save_labs_to_csv(labs_data, patient_id, save_labs_dir)
    else:
        print("  [No lab data available]")

    # =========================================================================
    # X-Ray Data
    # =========================================================================
    print_section("Chest X-Ray Data")

    # X-ray NegBio findings
    xray_findings = sample.get("xrays_negbio", [])
    print_subsection(f"NegBio Findings ({len(xray_findings)} detected)")
    if xray_findings:
        unique_findings = list(set(xray_findings))
        for finding in unique_findings[:10]:
            count = xray_findings.count(finding)
            print(f"  • {finding.title()} (×{count})")
    else:
        print("  [No X-ray findings detected]")

    # X-ray image
    image_path = sample.get("image")
    print_subsection("X-Ray Image")
    if image_path:
        print(f"  Image path: {image_path}")
        full_path = image_path
        if cxr_root and not os.path.isabs(image_path):
            full_path = os.path.join(cxr_root, image_path)

        # Check if the nested path exists, otherwise try flattened structure
        if not os.path.exists(full_path):
            # Extract dicom_id from the path (filename without extension)
            # Path format: files/p10/p10000032/s50414267/<dicom_id>.jpg
            dicom_id = Path(image_path).stem
            if cxr_root:
                # Try flattened structure: images/<dicom_id>.jpg
                flattened_path = os.path.join(
                    cxr_root, "images", f"{dicom_id}.jpg"
                )
                if os.path.exists(flattened_path):
                    print(f"  Using flattened path: images/{dicom_id}.jpg")
                    full_path = flattened_path

        if os.path.exists(full_path):
            display_image(full_path, save_image)
        else:
            print("  [Image file not accessible - set --cxr-root to view]")
    else:
        print("  [No X-ray image available for this sample]")

    # =========================================================================
    # Patient Profile Summary
    # =========================================================================
    print_section("Patient Profile Summary (Aggregated Across Visits)")

    print(f"\n  Patient ID: {sample.get('patient_id', 'N/A')}")
    print(f"  Mortality Label: {sample.get('mortality', 'N/A')}")
    print(f"\n  ✓ EHR Codes: {len(conditions)} unique diagnoses, "
          f"{len(procedures)} unique procedures, {len(drugs)} unique drugs")
    print(f"  ✓ Clinical Notes: Discharge ({len(discharge_text.split())} words), "
          f"Radiology ({len(radiology_text.split())} words)")
    if labs_data:
        time_span = max(labs_data[0]) - min(labs_data[0])
        print(f"  ✓ Lab Events: {len(labs_data[0])} measurements over "
              f"{time_span:.1f} hours, 10 dimensions")
    print(f"  ✓ X-Ray: {len(set(xray_findings))} unique NegBio findings, "
          f"Image: {'Available' if image_path else 'N/A'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyHealth Multimodal MIMIC-IV Demo: Benchmark + Showcase"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev mode (smaller subset)",
    )
    parser.add_argument(
        "--ehr-root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.2",
        help="Path to MIMIC-IV EHR root (hosp module)",
    )
    parser.add_argument(
        "--note-root",
        type=str,
        default="/srv/local/data/MIMIC-IV/2.0/",
        help="Path to MIMIC-IV notes root (note module)",
    )
    parser.add_argument(
        "--cxr-root",
        type=str,
        default="/srv/local/data/MIMIC-CXR",
        help="Path to MIMIC-CXR root for images",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/srv/local/data/jw3/pyhealth_cache",
        help="Cache directory for processed data",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Path to save X-ray visualization",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of sample to display (default: 0)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmarking, only show sample",
    )
    parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Skip loading clinical notes (test EHR+CXR only)",
    )
    parser.add_argument(
        "--no-cxr",
        action="store_true",
        help="Skip loading CXR data (test EHR+notes only)",
    )
    parser.add_argument(
        "--save-labs",
        type=str,
        default=None,
        help="Directory to save lab events CSV file",
    )
    args = parser.parse_args()

    num_workers = 8

    print_section("PyHealth: Multimodal Medical Data Loading Demo")
    print("\nThis demo showcases PyHealth's ability to load and process")
    print("multimodal medical data from MIMIC-IV dataset.")
    print("\nConfiguration:")
    print(f"  EHR root:  {args.ehr_root}")
    print(f"  Note root: {args.note_root} {'[DISABLED]' if args.no_notes else ''}")
    print(f"  CXR root:  {args.cxr_root} {'[DISABLED]' if args.no_cxr else ''}")
    print(f"  Dev mode:  {args.dev}")
    print(f"  Workers:   {num_workers}")
    if args.no_notes:
        print("\n  ⚠ Notes disabled: Testing EHR + CXR only")
    if args.no_cxr:
        print("\n  ⚠ CXR disabled: Testing EHR + Notes only")
    if args.no_notes and args.no_cxr:
        print("\n  ⚠ Both notes and CXR disabled: EHR only mode")

    # =========================================================================
    # Benchmark: Load Dataset
    # =========================================================================
    print_section("Step 1: Loading MIMIC-IV Multimodal Dataset")

    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4

    cache_root = Path(args.cache_dir)
    
    # Create cache directory name based on configuration
    cache_suffix = "_dev" if args.dev else ""
    if args.no_notes and args.no_cxr:
        cache_suffix += "_ehr_only"
    elif args.no_notes:
        cache_suffix += "_ehr_cxr"
    elif args.no_cxr:
        cache_suffix += "_ehr_notes"
    else:
        cache_suffix += "_full"
    
    base_cache_dir = cache_root / f"base_dataset{cache_suffix}"
    task_cache_dir = cache_root / f"task_samples{cache_suffix}"

    # Initialize memory tracker
    tracker = PeakMemoryTracker(poll_interval_s=0.1)
    tracker.start()
    tracker.reset()

    run_start = time.time()

    # Load base dataset
    print("\n[1/2] Loading base dataset...")
    print("  Loading EHR tables from:", args.ehr_root)
    if not args.no_notes:
        print("  Loading note tables from:", args.note_root)
    if not args.no_cxr:
        print("  Loading CXR tables from:", args.cxr_root)
    dataset_start = time.time()

    # Build dataset configuration based on flags
    dataset_config = {
        "ehr_root": args.ehr_root,
        "ehr_tables": [
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
            "labevents",
        ],
        "dev": args.dev,
        "cache_dir": str(base_cache_dir),
        "num_workers": num_workers,
    }

    # Add notes if not disabled
    if not args.no_notes:
        dataset_config["note_root"] = args.note_root
        dataset_config["note_tables"] = ["discharge", "radiology"]

    # Add CXR if not disabled
    if not args.no_cxr:
        dataset_config["cxr_root"] = args.cxr_root
        dataset_config["cxr_tables"] = ["metadata", "negbio"]

    # MIMIC4Dataset uses separate roots for different data sources:
    # - ehr_root: hosp module (patients, admissions, diagnoses, procedures, etc.)
    # - note_root: note module (discharge, radiology notes)
    # - cxr_root: MIMIC-CXR (images and metadata)
    base_dataset = MIMIC4Dataset(**dataset_config)

    dataset_load_s = time.time() - dataset_start
    base_cache_bytes = get_directory_size(base_cache_dir)
    print(f"  ✓ Base dataset loaded in {dataset_load_s:.2f}s")
    print(f"  ✓ Base cache size: {format_size(base_cache_bytes)}")

    # Apply multimodal task
    print("\n[2/2] Applying MultimodalMortalityPredictionMIMIC4 task...")
    task_start = time.time()

    task = MultimodalMortalityPredictionMIMIC4(cxr_root=args.cxr_root)
    sample_dataset = base_dataset.set_task(
        task,
        num_workers=num_workers,
        cache_dir=str(task_cache_dir),
    )

    task_process_s = time.time() - task_start
    total_s = time.time() - run_start
    peak_rss_bytes = tracker.peak_bytes()
    
    print(f"  ✓ Task completed in {task_process_s:.2f}s", flush=True)
    
    # Get sample count first (faster, uses cached count if available)
    print("  Getting sample count...", flush=True)
    try:
        # Try to get count without loading samples
        if hasattr(sample_dataset, '_samples'):
            num_samples = len(sample_dataset._samples) if sample_dataset._samples else 0
        else:
            num_samples = len(sample_dataset)
        print(f"  ✓ Total samples: {num_samples}")
    except Exception as e:
        print(f"  ✗ Error getting sample count: {e}")
        num_samples = 0
    
    # Calculate cache size (can be slow for large directories)
    if not args.skip_benchmark:
        print("  Calculating cache size...", flush=True)
        cache_calc_start = time.time()
        task_cache_bytes = get_directory_size(task_cache_dir)
        cache_calc_time = time.time() - cache_calc_start
        print(f"  ✓ Task cache size: {format_size(task_cache_bytes)} "
              f"(calculated in {cache_calc_time:.1f}s)")
    else:
        task_cache_bytes = 0

    # =========================================================================
    # Benchmark Results
    # =========================================================================
    if not args.skip_benchmark:
        print_section("Benchmark Results")

        print(f"\n  Dataset load time:    {dataset_load_s:>10.2f}s")
        print(f"  Task processing time: {task_process_s:>10.2f}s")
        print(f"  Total time:           {total_s:>10.2f}s")
        print(f"  Peak memory (RSS):    {format_size(peak_rss_bytes):>10}")
        print(f"  Base cache size:      {format_size(base_cache_bytes):>10}")
        print(f"  Task cache size:      {format_size(task_cache_bytes):>10}")
        print(f"  Number of samples:    {num_samples:>10}")

        print("\n  Dataset Schema:")
        print(f"    Input: {list(sample_dataset.input_schema.keys())}")
        print(f"    Output: {list(sample_dataset.output_schema.keys())}")

    tracker.stop()

    # =========================================================================
    # Showcase: Display First Sample
    # =========================================================================
    if num_samples == 0:
        print("\n⚠ No samples generated. This may be because:")
        print("  - The dataset is too small (try without --dev)")
        print("  - Missing required modalities (all must be present)")
        return

    print_section("Step 2: Showcasing Patient Profile (Aggregated Multimodal Data)")

    sample_idx = min(args.sample_idx, num_samples - 1)
    print(f"  Loading sample {sample_idx}...", flush=True)
    load_start = time.time()
    try:
        sample = sample_dataset.samples[sample_idx]
        load_time = time.time() - load_start
        print(f"  ✓ Sample loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        print("\n  Skipping sample showcase due to error.")
        return

    print(f"\n  Sample index: {sample_idx}")
    print(f"  Patient ID: {sample.get('patient_id', 'N/A')}")
    print(f"  Mortality label: {sample.get('mortality', 'N/A')}")
    print("  (Patient-level aggregation: data from all visits before death/end)")

    showcase_sample(
        sample,
        cxr_root=args.cxr_root,
        save_image=args.save_image,
        save_labs_dir=args.save_labs
    )

    # =========================================================================
    # Final Summary
    # =========================================================================
    print_section("Demo Complete")
    print("\n  PyHealth: Your one-stop solution for multimodal medical data!")
    print("\n  Key features demonstrated:")
    print("    • Patient-level data aggregation across visits")
    print("    • EHR code loading with medcode lookup")
    print("    • Clinical note extraction (discharge, radiology)")
    print("    • Time-series lab event processing with CSV export")
    print("    • Chest X-ray image integration")
    print("    • Memory-efficient parallel processing")
    print("=" * 80)


if __name__ == "__main__":
    main()

