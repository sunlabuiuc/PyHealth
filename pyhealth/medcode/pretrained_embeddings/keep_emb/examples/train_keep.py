"""Train KEEP embeddings on MIMIC-III or MIMIC-IV.

Generates medical code embeddings via the KEEP two-stage pipeline (Node2Vec
on SNOMED hierarchy + regularized GloVe on patient co-occurrence). Outputs
``keep_snomed.txt`` ready for use with ``pretrained_emb_path`` in any
downstream PyHealth model.

Use this script when you want embeddings only — no downstream task training.
For end-to-end downstream examples (KEEP + GRASP + mortality prediction),
see ``examples/mortality_prediction/mortality_mimic4_grasp_keep.py``.

Intended workflows:
    1. Generate embeddings once, use them for multiple downstream tasks
    2. Run hyperparameter sweeps (KEEP_VARIANT, min_occurrences, λ, etc.)
    3. Experiment with filter thresholds (min_occurrences=1 vs 2)
    4. Produce artifacts to distribute (Google Drive, HF Hub, etc.)

Prerequisites:
    - Athena OMOP vocabularies (SNOMED + ICD9CM + ICD10CM) downloaded
      from https://athena.ohdsi.org/ and unzipped to ATHENA_DIR below
    - pip install pyhealth[keep]
    - pip install codecarbon pynvml  (optional, for compute tracking)

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with
       Clinical Data for Robust Code Embeddings", CHIL 2025.
"""

import json
import platform
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, MIMIC4EHRDataset

# ── Compute Tracking (optional) ───────────────────────────
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False


def print_hardware_info(compute_tracking_enabled: bool):
    """Print hardware and tracking availability.

    Args:
        compute_tracking_enabled: User's toggle from config. Shown alongside
            library availability so it's obvious why tracking may be off
            (disabled vs not installed).
    """
    print("=" * 60)
    print("Hardware Information")
    print("=" * 60)
    print(f"Platform:     {platform.system()} {platform.machine()}")
    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA:         {torch.version.cuda}")
    else:
        print("GPU:          None (CPU only)")

    def _status(available: bool) -> str:
        if not compute_tracking_enabled:
            return "disabled via config" if available else "not installed"
        return "enabled" if available else "not installed"

    print(f"pynvml:       {_status(NVML_AVAILABLE)}")
    print(f"codecarbon:   {_status(CODECARBON_AVAILABLE)}")
    print("=" * 60)


# ── Configuration ─────────────────────────────────────────
ATHENA_DIR = "data/athena"           # path to Athena OMOP vocabulary download
KEEP_VARIANT = "paper"               # "paper" or "code" — see KEEP_VARIANTS below
RUN_INTRINSIC_EVAL = True            # compute Resnik + co-occ correlations after pipeline
ENABLE_COMPUTE_TRACKING = True       # record CO2 (codecarbon) + GPU stats (pynvml) if available

# Device for GloVe training (Node2Vec stage is always CPU — gensim limitation).
# "auto" picks cuda > mps > cpu automatically based on what's available.
# Override explicitly if you want to force a specific device (reproducibility,
# debugging MPS issues, forcing CPU on a server with CUDA, etc.).
DEVICE = "auto"                      # "auto" | "cuda" | "mps" | "cpu"

# Keep `cooc_matrix.npy` (~247 MB) and `cooc_index.json` after the run?
# The cooc matrix is needed for intrinsic eval's co-occurrence correlation,
# so it's always created and used during this script. This toggle only
# controls whether it's retained on disk after the script exits.
# Set False to save disk space during sweeps; True if you want to
# re-run intrinsic eval later without retraining.
SAVE_COOC_ARTIFACTS = True

# Also export Node2Vec (Stage 1) embeddings as node2vec_snomed.txt?
# Format matches keep_snomed.txt so it can be used as a pretrained_emb_path
# baseline for downstream ablation comparisons (KEEP vs Node2Vec-only).
# The underlying node2vec_embeddings.npy is always saved for diagnostics
# regardless of this flag.
EXPORT_NODE2VEC_SNOMED_TXT = True

# Data source
MIMIC_VERSION = "mimic4"             # "mimic3" (ICD-9 only) or "mimic4" (mixed ICD-9/ICD-10)
LOCAL_MIMIC_ROOTS = {
    "mimic3": "data/mimic3",
    "mimic4": "data/mimic4",
}
DEV_MODE = False                     # True = 1000-patient subset + tiny pipeline for smoke tests

# Filter knob (the paper uses 2; MIMIC-IV likely benefits from 1 — see
# docs/plans/keep/keep-learning-journal-filter-investigation.md)
MIN_OCCURRENCES = 2

# Output layout (flat, Trainer-style, timestamped; namespaced by method):
#   OUTPUT_ROOT/{timestamp}/
#       keep_snomed.txt             pretrained embeddings (text, for pretrained_emb_path)
#       keep_embeddings.npy         final KEEP embeddings (numpy, aligned to cooc rows)
#       node2vec_embeddings.npy     Stage 1 init (aligned) — for drift diagnostics
#       cooc_matrix.npy             (optional) co-occurrence matrix
#       cooc_index.json             (optional) SNOMED id → matrix row index
#       config.json                 inputs: hyperparameters, data sources, variant
#       results.json                outputs: wall time, energy, intrinsic eval
#
#   e.g. output/embeddings/keep/20260420-143042/keep_snomed.txt
#        output/embeddings/keep/20260421-090000/keep_snomed.txt
#
# The `embeddings/keep/` namespace leaves room for sibling methods used as
# baselines for the paper (e.g. output/embeddings/node2vec/, cui2vec/,
# medbert/). Config details (variant, mimic version, min_occurrences)
# live in each run's config.json — NOT in the directory name.
#
# To identify a run:
#   cat output/embeddings/keep/*/config.json | jq '.run_name'
OUTPUT_ROOT = "output/embeddings/keep"
# ──────────────────────────────────────────────────────────

# Paper-faithful vs G2Lab code-faithful hyperparameter variants.
# Both are valid KEEP; they exercise different points in the
# ontology-anchoring vs data-driven trade-off space.
KEEP_VARIANTS = {
    "paper": {
        "reg_distance": "l2",        # squared L2 per paper Eq 4
        "optimizer": "adamw",        # paper Algorithm 1
        "lambd": 1e-3,               # paper Table 6
    },
    "code": {
        "reg_distance": "cosine",
        "optimizer": "adagrad",
        "lambd": 1e-5,
    },
}
# Note: reg reduction is always `sum` (paper Eq 4: Σᵢ₌₁^V). It is not
# exposed as a variant knob because it is mathematically coupled to `lambd`
# — mean would be equivalent to scaling `lambd` by 1/V, making the two
# hyperparameters ambiguous. The library hardcodes the paper-faithful sum.


def main():
    print_hardware_info(ENABLE_COMPUTE_TRACKING)

    variant_params = KEEP_VARIANTS[KEEP_VARIANT]

    # Output layout (flat, Trainer-style): {OUTPUT_ROOT}/{timestamp}/
    # Config lives in manifest.json, not the path.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{KEEP_VARIANT}_emb_{MIMIC_VERSION}_minocc{MIN_OCCURRENCES}_{timestamp}"
    print(f"\nRun: {run_name}")
    print(f"  Variant params: {variant_params}")
    print(f"  Dev mode: {DEV_MODE}")

    # ── Load MIMIC dataset ────────────────────────────────
    mimic_root = LOCAL_MIMIC_ROOTS[MIMIC_VERSION]
    print(f"\n[1/3] Loading {MIMIC_VERSION} dataset from {mimic_root}...")

    if MIMIC_VERSION == "mimic3":
        base_dataset = MIMIC3Dataset(
            root=mimic_root,
            tables=["DIAGNOSES_ICD"],
            cache_dir=tempfile.TemporaryDirectory().name,
            dev=DEV_MODE,
        )
    elif MIMIC_VERSION == "mimic4":
        base_dataset = MIMIC4EHRDataset(
            root=mimic_root,
            tables=["diagnoses_icd"],
            cache_dir=tempfile.TemporaryDirectory().name,
            dev=DEV_MODE,
        )
    else:
        raise ValueError(
            f"MIMIC_VERSION must be 'mimic3' or 'mimic4', got {MIMIC_VERSION!r}"
        )
    base_dataset.stats()

    # ── Run KEEP pipeline ─────────────────────────────────
    print(f"\n[2/3] Training KEEP embeddings (variant={KEEP_VARIANT}, min_occ={MIN_OCCURRENCES})...")
    from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
        run_keep_pipeline,
    )

    output_dir = Path(OUTPUT_ROOT) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Start compute tracking ────────────────────────────
    emissions_tracker = None
    if ENABLE_COMPUTE_TRACKING and CODECARBON_AVAILABLE:
        emissions_tracker = EmissionsTracker(
            project_name=run_name, log_level="error", save_to_file=False,
        )
        emissions_tracker.start()

    # Resolve DEVICE='auto' early so we can log the actual device picked
    from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
        resolve_device,
    )
    resolved_device = resolve_device(DEVICE)
    if DEVICE == "auto":
        print(f"  Device: auto → {resolved_device}")
    else:
        print(f"  Device: {resolved_device}")

    start_time = time.time()
    keep_emb_path = run_keep_pipeline(
        athena_dir=ATHENA_DIR,
        dataset=base_dataset,
        output_dir=str(output_dir),
        dev=DEV_MODE,
        min_occurrences=MIN_OCCURRENCES,
        device=resolved_device,
        export_node2vec_text=EXPORT_NODE2VEC_SNOMED_TXT,
        **variant_params,
    )
    wall_time = time.time() - start_time

    emissions_data = None
    if emissions_tracker:
        emissions_tracker.stop()
        emissions_data = emissions_tracker.final_emissions_data

    print(f"  Embeddings saved to: {keep_emb_path}")
    print(f"  Wall time: {wall_time:.1f}s ({wall_time / 60:.2f} min)")
    if emissions_data:
        print(f"  Energy: {emissions_data.energy_consumed:.6f} kWh")
        print(f"  CO2: {emissions_data.emissions:.6f} kg")

    # ── Intrinsic evaluation ──────────────────────────────
    intrinsic_results = None
    if RUN_INTRINSIC_EVAL:
        print("\n[3/3] Intrinsic evaluation (Resnik + co-occurrence correlations)...")
        from pyhealth.medcode.pretrained_embeddings.keep_emb import (
            build_hierarchy_graph,
            cooccurrence_correlation,
            load_keep_embeddings,
            resnik_correlation,
        )

        # Rebuild graph for Resnik evaluation
        athena_concept = Path(ATHENA_DIR) / "CONCEPT.csv"
        athena_rel = Path(ATHENA_DIR) / "CONCEPT_RELATIONSHIP.csv"
        athena_ancestor = Path(ATHENA_DIR) / "CONCEPT_ANCESTOR.csv"
        if not athena_ancestor.exists():
            athena_ancestor = None
        eval_graph = build_hierarchy_graph(
            athena_concept, athena_rel, ancestor_csv=athena_ancestor,
        )

        # Load embeddings keyed by SNOMED concept_code
        eval_emb, token_strings = load_keep_embeddings(
            keep_emb_path, embedding_dim=100,
        )
        code_to_id = {
            str(eval_graph.nodes[n].get("concept_code", n)): n
            for n in eval_graph.nodes()
        }
        eval_node_ids = [
            code_to_id[tok] for tok in token_strings if tok in code_to_id
        ]
        valid_mask = [tok in code_to_id for tok in token_strings]
        eval_emb = eval_emb[valid_mask]

        if len(eval_node_ids) >= 11:
            k1 = min(10, len(eval_node_ids) // 10)
            k2 = min(150, len(eval_node_ids) - k1 - 1)
            runs = 50  # paper uses 250; 50 is enough for a smoke check

            # ── KEEP's Resnik + cooc (paper Table 2 targets: 0.68 / 0.62) ──
            resnik_results = resnik_correlation(
                eval_emb, eval_node_ids, eval_graph,
                k1=k1, k2=k2, num_runs=runs, seed=42,
            )
            print(
                f"  KEEP Resnik correlation (median): {resnik_results['median']:.4f} "
                f"(paper target: 0.68)"
            )

            cooc_matrix_path = output_dir / "cooc_matrix.npy"
            cooc_index_path = output_dir / "cooc_index.json"
            cooc_results = None
            cooc_matrix = None
            code_to_idx_saved = None
            if cooc_matrix_path.exists() and cooc_index_path.exists():
                cooc_matrix = np.load(cooc_matrix_path)
                with open(cooc_index_path) as f:
                    idx_to_code_saved = json.load(f)
                code_to_idx_saved = {
                    int(c): i for i, c in enumerate(idx_to_code_saved)
                }
                cooc_results = cooccurrence_correlation(
                    eval_emb, eval_node_ids, cooc_matrix, code_to_idx_saved,
                    k1=k1, k2=k2, num_runs=runs, seed=42,
                )
                print(
                    f"  KEEP Co-occurrence correlation (median): "
                    f"{cooc_results['median']:.4f} (paper target: 0.62)"
                )
            else:
                print(
                    f"  KEEP Co-occurrence correlation skipped: cooc matrix not "
                    f"found at {cooc_matrix_path}."
                )

            # ── Node2Vec-only baselines (answers Desmond's question) ──
            # Load Node2Vec init (saved by run_pipeline.py, aligned to cooc rows)
            n2v_baseline_resnik = None
            n2v_baseline_cooc = None
            stage2_effectiveness = None
            n2v_npy_path = output_dir / "node2vec_embeddings.npy"
            keep_npy_path = output_dir / "keep_embeddings.npy"
            if n2v_npy_path.exists() and keep_npy_path.exists():
                n2v_emb_full = np.load(n2v_npy_path)    # (V, 100)
                keep_emb_full = np.load(keep_npy_path)  # (V, 100)

                # Reconstruct which rows of the npy correspond to eval_node_ids
                # (both npy files are aligned to cooc_index order)
                if code_to_idx_saved is not None:
                    eval_row_indices = [
                        code_to_idx_saved[nid]
                        for nid in eval_node_ids
                        if nid in code_to_idx_saved
                    ]
                    n2v_eval_emb = n2v_emb_full[eval_row_indices]
                    eval_node_ids_filtered = [
                        nid for nid in eval_node_ids if nid in code_to_idx_saved
                    ]

                    # Node2Vec-only Resnik
                    n2v_resnik = resnik_correlation(
                        n2v_eval_emb, eval_node_ids_filtered, eval_graph,
                        k1=k1, k2=k2, num_runs=runs, seed=42,
                    )
                    n2v_baseline_resnik = n2v_resnik
                    print(
                        f"  Node2Vec-only Resnik (median):    "
                        f"{n2v_resnik['median']:.4f}"
                    )
                    print(
                        f"  KEEP lift over Node2Vec (Resnik): "
                        f"{resnik_results['median'] - n2v_resnik['median']:+.4f}"
                    )

                    # Node2Vec-only co-occurrence
                    if cooc_matrix is not None:
                        n2v_cooc = cooccurrence_correlation(
                            n2v_eval_emb, eval_node_ids_filtered,
                            cooc_matrix, code_to_idx_saved,
                            k1=k1, k2=k2, num_runs=runs, seed=42,
                        )
                        n2v_baseline_cooc = n2v_cooc
                        print(
                            f"  Node2Vec-only cooc corr (median): "
                            f"{n2v_cooc['median']:.4f}"
                        )
                        print(
                            f"  KEEP lift over Node2Vec (cooc):   "
                            f"{cooc_results['median'] - n2v_cooc['median']:+.4f}"
                        )

                # ── Embedding drift (Stage 2 effectiveness) ──
                # Per-concept relative change from Node2Vec init to KEEP final
                n2v_norms = np.linalg.norm(n2v_emb_full, axis=1)
                diff_norms = np.linalg.norm(
                    keep_emb_full - n2v_emb_full, axis=1
                )
                # Guard against division by zero for zero-init rows (shouldn't
                # happen for observed concepts but be safe)
                safe_n2v = np.where(n2v_norms > 1e-10, n2v_norms, 1.0)
                rel_drift = diff_norms / safe_n2v

                mean_drift = float(rel_drift.mean())
                median_drift = float(np.median(rel_drift))
                max_drift = float(rel_drift.max())
                pct_stuck = float((rel_drift < 0.01).mean() * 100)

                # Heuristic verdict based on both drift and cooc lift
                cooc_lift = None
                if cooc_results is not None and n2v_baseline_cooc is not None:
                    cooc_lift = (
                        cooc_results['median'] - n2v_baseline_cooc['median']
                    )

                if mean_drift < 0.01:
                    verdict = "embeddings_frozen_stage2_did_nothing"
                elif cooc_lift is not None and cooc_lift <= 0.01:
                    verdict = "moved_but_no_empirical_signal_gained"
                elif cooc_lift is not None and cooc_lift > 0.10:
                    verdict = "strong_empirical_signal_captured"
                else:
                    verdict = "partial_empirical_signal_captured"

                stage2_effectiveness = {
                    "mean_relative_drift": mean_drift,
                    "median_relative_drift": median_drift,
                    "max_relative_drift": max_drift,
                    "pct_concepts_stuck_below_1pct": pct_stuck,
                    "cooc_lift_over_node2vec": cooc_lift,
                    "verdict": verdict,
                }
                print(
                    f"  Embedding drift (mean): {mean_drift:.4f}  "
                    f"(stuck <1%: {pct_stuck:.1f}% of concepts)"
                )
                print(f"  Stage 2 verdict: {verdict}")
            else:
                print(
                    f"  Skipped Node2Vec baseline + drift: "
                    f"{n2v_npy_path} or {keep_npy_path} not found "
                    f"(pipeline may have been run before the n2v save was added)."
                )

            intrinsic_results = {
                "resnik": resnik_results,
                "cooccurrence": cooc_results,
                "resnik_node2vec_baseline": n2v_baseline_resnik,
                "cooccurrence_node2vec_baseline": n2v_baseline_cooc,
                "stage2_effectiveness": stage2_effectiveness,
            }
        else:
            print(
                f"  Skipped: only {len(eval_node_ids)} in-graph concepts "
                "(need >= 11 for K1=10 + K2>=1). DEV mode produces tiny vocab."
            )

    # ── Clean up cooc artifacts if user opted out ────────
    # Always created by run_keep_pipeline() + used by intrinsic eval above.
    # Deleted here if SAVE_COOC_ARTIFACTS=False to save disk space.
    cooc_artifacts_kept = SAVE_COOC_ARTIFACTS
    if not SAVE_COOC_ARTIFACTS:
        cooc_matrix_path = output_dir / "cooc_matrix.npy"
        cooc_index_path = output_dir / "cooc_index.json"
        bytes_freed = 0
        for path in (cooc_matrix_path, cooc_index_path):
            if path.exists():
                bytes_freed += path.stat().st_size
                path.unlink()
        if bytes_freed > 0:
            print(
                f"\nRemoved cooc artifacts "
                f"({bytes_freed / 1e6:.1f} MB) "
                f"— set SAVE_COOC_ARTIFACTS=True to keep them."
            )

    # ── Save config.json (inputs — what you asked for) ────
    # Everything deterministic that goes INTO the run. Copy this file into
    # train_keep.py's config block and rerun to reproduce this run.
    config = {
        "run_name": run_name,
        "timestamp": timestamp,                         # matches output_dir suffix
        "timestamp_iso": datetime.now().isoformat(),    # ISO format for parsers
        "mimic_version": MIMIC_VERSION,
        "mimic_root": mimic_root,
        "athena_dir": ATHENA_DIR,
        "dev_mode": DEV_MODE,
        "keep_variant": KEEP_VARIANT,
        "min_occurrences": MIN_OCCURRENCES,
        "hyperparameters": {
            "embedding_dim": 100,
            "num_walks": 10 if DEV_MODE else 750,
            "walk_length": 30,
            "glove_epochs": 10 if DEV_MODE else 300,
            **variant_params,
        },
        "device_config": DEVICE,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to:  {config_path}")

    # ── Save results.json (outputs — what the run produced) ────
    # Non-deterministic across reruns of the same config (seeds, hardware).
    results_data = {
        "wall_time_s": wall_time,
        "wall_time_min": wall_time / 60,
        "device_resolved": resolved_device,
        "keep_emb_path": keep_emb_path,
        "cooc_artifacts_kept": cooc_artifacts_kept,
    }
    if emissions_data:
        results_data["energy_kwh"] = emissions_data.energy_consumed
        results_data["co2_kg"] = emissions_data.emissions
    if intrinsic_results:
        keep_resnik = intrinsic_results.get("resnik") or {}
        keep_cooc = intrinsic_results.get("cooccurrence") or {}
        n2v_resnik = intrinsic_results.get("resnik_node2vec_baseline") or {}
        n2v_cooc = intrinsic_results.get("cooccurrence_node2vec_baseline") or {}
        stage2 = intrinsic_results.get("stage2_effectiveness")

        def _lift(keep_metric, baseline_metric, key="median"):
            if not keep_metric or not baseline_metric:
                return None
            k = keep_metric.get(key)
            b = baseline_metric.get(key)
            if k is None or b is None:
                return None
            return float(k - b)

        results_data["intrinsic_eval"] = {
            "resnik": {
                "keep": keep_resnik,
                "node2vec_baseline": n2v_resnik if n2v_resnik else None,
                "lift_over_baseline_median": _lift(keep_resnik, n2v_resnik),
                "paper_target": 0.68,
            },
            "cooccurrence": {
                "keep": keep_cooc if keep_cooc else None,
                "node2vec_baseline": n2v_cooc if n2v_cooc else None,
                "lift_over_baseline_median": _lift(keep_cooc, n2v_cooc),
                "paper_target": 0.62,
            },
            "stage2_effectiveness": stage2,
        }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {results_path}")

    print(f"\n{'=' * 60}")
    print(f"Done. Artifacts in: {output_dir}")
    print(f"{'=' * 60}")
    print("  keep_snomed.txt          ← KEEP (Stage 2) embeddings, text format")
    if EXPORT_NODE2VEC_SNOMED_TXT:
        print("  node2vec_snomed.txt      ← Node2Vec (Stage 1) embeddings, text format (baseline)")
    print("  keep_embeddings.npy      ← KEEP embeddings, numpy (aligned to cooc rows)")
    print("  node2vec_embeddings.npy  ← Node2Vec embeddings, numpy (aligned to cooc rows)")
    if cooc_artifacts_kept:
        print("  cooc_matrix.npy          ← co-occurrence matrix")
        print("  cooc_index.json          ← SNOMED concept_id → matrix row index")
    print("  config.json              ← inputs (hyperparameters, data sources, variant)")
    print("  results.json             ← outputs (timing, intrinsic eval, stage2 effectiveness)")


if __name__ == "__main__":
    main()
