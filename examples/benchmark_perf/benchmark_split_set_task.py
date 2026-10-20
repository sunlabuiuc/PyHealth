"""Benchmark set_task with and without split=PatientSplit(...) at scale.

Generates synthetic MIMIC-III-format tables (no real patient data) for N
patients, then runs MIMIC3Dataset + MortalityPredictionMIMIC3 twice in separate
processes, each with a fresh cache:

  nosplit : dataset.set_task(task)                         (current behaviour)
  split   : dataset.set_task(task, split=PatientSplit(...)) (fit on train only)

For each run it reports wall time per stage and the peak resident memory of the
process plus its workers, sampled every 0.1 s. Use it to check that fitting on
the training split streams the samples (no extra memory) on your machine.

Usage:
    python examples/benchmark_perf/benchmark_split_set_task.py --patients 50000
    python examples/benchmark_perf/benchmark_split_set_task.py --patients 300000 --num-workers 4

Data and caches go under --workdir (default: a temporary folder, deleted at the
end; pass a path to keep them). Generation is reused when the folder exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


# ---------------------------------------------------------------- data ----
def generate_mimic3(root: Path, n_patients: int, seed: int = 0) -> dict:
    """Writes synthetic MIMIC-III CSVs with the columns PyHealth reads."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    subject = np.arange(10_000, 10_000 + n_patients)

    pd.DataFrame(
        {
            "row_id": np.arange(n_patients),
            "subject_id": subject,
            "gender": rng.choice(["F", "M"], n_patients),
            "dob": "2080-01-01 00:00:00",
            "dod": "",
            "dod_hosp": "",
            "dod_ssn": "",
            "expire_flag": 0,
        }
    ).to_csv(root / "PATIENTS.csv", index=False)

    n_adm = rng.integers(1, 7, n_patients)  # 1-6 admissions per patient
    adm_subject = np.repeat(subject, n_adm)
    n_a = adm_subject.size
    hadm = np.arange(100_000, 100_000 + n_a)
    order = np.concatenate([np.arange(k) for k in n_adm])
    admit = pd.Timestamp("2100-01-01") + pd.to_timedelta(order * 60 + rng.integers(0, 30, n_a), unit="D")
    disch = admit + pd.to_timedelta(rng.integers(1, 15, n_a), unit="D")
    pd.DataFrame(
        {
            "row_id": np.arange(n_a), "subject_id": adm_subject, "hadm_id": hadm,
            "admittime": admit.strftime("%Y-%m-%d %H:%M:%S"),
            "dischtime": disch.strftime("%Y-%m-%d %H:%M:%S"),
            "deathtime": "", "admission_type": "EMERGENCY",
            "admission_location": "EMERGENCY ROOM ADMIT", "discharge_location": "HOME",
            "insurance": "Medicare", "language": "ENGL", "religion": "", "marital_status": "",
            "ethnicity": "WHITE", "edregtime": "", "edouttime": "", "diagnosis": "",
            "hospital_expire_flag": (rng.random(n_a) < 0.03).astype(int),
            "has_chartevents_data": 1,
        }
    ).to_csv(root / "ADMISSIONS.csv", index=False)

    pd.DataFrame(
        {
            "subject_id": adm_subject, "intime": admit.strftime("%Y-%m-%d %H:%M:%S"),
            "icustay_id": np.arange(200_000, 200_000 + n_a), "first_careunit": "MICU",
            "dbsource": "carevue", "last_careunit": "MICU",
            "outtime": disch.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ).to_csv(root / "ICUSTAYS.csv", index=False)

    def codes_table(name, lo, hi, vocab, extra):
        k = rng.integers(lo, hi + 1, n_a)
        frame = pd.DataFrame(
            {
                "row_id": np.arange(k.sum()),
                "subject_id": np.repeat(adm_subject, k),
                "hadm_id": np.repeat(hadm, k),
                **extra(k),
            }
        )
        frame.to_csv(root / name, index=False)
        return int(k.sum())

    icd = np.array([f"{c:05d}" for c in rng.choice(100_000, 6_000, replace=False)])
    proc = np.array([f"{c:04d}" for c in rng.choice(10_000, 2_000, replace=False)])
    ndc = np.array([f"{c:011d}" for c in rng.choice(10**11, 3_000, replace=False)])
    n_diag = codes_table(
        "DIAGNOSES_ICD.csv", 3, 15, icd,
        lambda k: {"seq_num": np.concatenate([np.arange(1, j + 1) for j in k]),
                   "icd9_code": rng.choice(icd, k.sum())},
    )
    n_proc = codes_table(
        "PROCEDURES_ICD.csv", 1, 5, proc,
        lambda k: {"seq_num": np.concatenate([np.arange(1, j + 1) for j in k]),
                   "icd9_code": rng.choice(proc, k.sum())},
    )

    def presc_cols(k):
        start = (np.repeat(admit, k)).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "icustay_id": "", "startdate": start, "enddate": start, "drug_type": "MAIN",
            "drug": "drug", "drug_name_poe": "", "drug_name_generic": "", "formulary_drug_cd": "",
            "gsn": "", "ndc": rng.choice(ndc, k.sum()), "prod_strength": "", "dose_val_rx": "1",
            "dose_unit_rx": "mg", "form_val_disp": "1", "form_unit_disp": "TAB", "route": "PO",
        }

    n_presc = codes_table("PRESCRIPTIONS.csv", 2, 15, ndc, presc_cols)
    return {"patients": n_patients, "admissions": int(n_a), "diagnoses": n_diag,
            "procedures": n_proc, "prescriptions": n_presc}


# ------------------------------------------------------------- memory ----
class PeakMemory:
    """Peak RSS of this process plus its children, sampled in a thread."""

    def __init__(self, interval: float = 0.1):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _rss(self) -> int:
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self._rss()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())


# --------------------------------------------------------------- child ----
def run_child(root: str, cache: str, mode: str, num_workers: int) -> dict:
    from pyhealth.datasets import MIMIC3Dataset, PatientSplit
    from pyhealth.tasks import MortalityPredictionMIMIC3

    out = {"mode": mode, "baseline_rss_mib": psutil.Process().memory_info().rss / 2**20}
    t0 = time.perf_counter()
    with PeakMemory() as mem:
        dataset = MIMIC3Dataset(
            root=root, tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cache, num_workers=num_workers,
        )
        _ = dataset.global_event_df
    out["events_s"], out["events_peak_mib"] = time.perf_counter() - t0, mem.peak / 2**20

    split = PatientSplit(ratios=(0.7, 0.1, 0.2), seed=42) if mode == "split" else None
    t1 = time.perf_counter()
    with PeakMemory() as mem:
        result = dataset.set_task(MortalityPredictionMIMIC3(), num_workers=num_workers, split=split)
    out["set_task_s"], out["set_task_peak_mib"] = time.perf_counter() - t1, mem.peak / 2**20

    parts = result if isinstance(result, tuple) else (result,)
    out["samples"] = [len(p) for p in parts]
    out["patients_in_index"] = len(parts[0].patient_to_index) if mode == "nosplit" else None
    schema = Path(parts[0].path) / "schema.pkl"
    out["schema_pkl_mib"] = schema.stat().st_size / 2**20
    split_file = Path(parts[0].path) / "split.npz"
    out["split_npz_mib"] = split_file.stat().st_size / 2**20 if split_file.exists() else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=20_000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--modes", type=str, default="nosplit,split")
    parser.add_argument("--child", nargs=4, metavar=("ROOT", "CACHE", "MODE", "WORKERS"))
    args = parser.parse_args()

    if args.child:
        root, cache, mode, workers = args.child
        print("RESULT " + json.dumps(run_child(root, cache, mode, int(workers))), flush=True)
        return

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="pyhealth_split_bench_"))
    data = workdir / f"mimic3_synthetic_{args.patients}"
    try:
        if not (data / "PRESCRIPTIONS.csv").exists():
            t = time.perf_counter()
            sizes = generate_mimic3(data, args.patients)
            print(f"generated {sizes} in {time.perf_counter() - t:.0f} s", flush=True)
        for mode in args.modes.split(","):
            cache = workdir / f"cache_{args.patients}_{mode}"
            shutil.rmtree(cache, ignore_errors=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", str(data), str(cache), mode, str(args.num_workers)],
                capture_output=True, text=True,
            )
            result = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode or not result:
                print(proc.stderr[-3000:], file=sys.stderr)
                raise SystemExit(f"{mode} run failed (exit {proc.returncode})")
            print(result[-1][7:], flush=True)
    finally:
        if not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
