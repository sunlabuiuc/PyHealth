"""GRASP + KEEP embeddings on MIMIC-III mortality prediction.

Demonstrates the full KEEP embedding pipeline (Node2Vec + regularized
GloVe) integrated with GRASP for mortality prediction.

Prerequisites:
    - Athena OMOP vocabularies (SNOMED + ICD9CM + ICD10CM) downloaded
      from https://athena.ohdsi.org/ and unzipped to ATHENA_DIR below
    - pip install pyhealth[keep]
    - pip install codecarbon pynvml  (optional, for compute tracking)

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with
       Clinical Data for Robust Code Embeddings", CHIL 2025.
"""

import time
import tempfile
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC3Dataset,
    MIMIC4EHRDataset,
    split_by_patient,
    get_dataloader,
)
from pyhealth.models import GRASP
from pyhealth.tasks import (
    MortalityPredictionMIMIC3,
    MortalityPredictionMIMIC4,
)
from pyhealth.trainer import Trainer

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


def gpu_snapshot():
    """Capture current GPU power, utilization, and memory."""
    if not NVML_AVAILABLE:
        return {}
    return {
        "power_w": pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE) / 1000,
        "util_pct": pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE).gpu,
        "mem_gb": pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE).used / 1e9,
    }


def print_compute_report(name, wall_time, emissions_data, gpu_readings):
    """Print compute cost summary."""
    print(f"\n{'=' * 60}")
    print(f"Compute Report: {name}")
    print(f"{'=' * 60}")
    print(f"Wall time: {wall_time:.1f}s ({wall_time / 60:.2f} min)")

    if emissions_data:
        print(f"Energy consumed: {emissions_data.energy_consumed:.6f} kWh")
        print(f"CO2 emissions: {emissions_data.emissions:.6f} kg CO2eq")

    if gpu_readings:
        powers = [r["power_w"] for r in gpu_readings]
        utils = [r["util_pct"] for r in gpu_readings]
        mems = [r["mem_gb"] for r in gpu_readings]
        print(f"Avg GPU power: {sum(powers)/len(powers):.0f} W")
        print(f"Avg GPU util: {sum(utils)/len(utils):.0f}%")
        print(f"Peak GPU mem: {max(mems):.2f} GB")

    print(f"{'=' * 60}")

def print_hardware_info():
    """Print hardware and tracking availability."""
    import platform
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
    print(f"pynvml:       {'available' if NVML_AVAILABLE else 'not installed'}")
    print(f"codecarbon:   {'available' if CODECARBON_AVAILABLE else 'not installed'}")
    print("=" * 60)


# ── Configuration ─────────────────────────────────────────
USE_KEEP = True                 # False = random embeddings, True = KEEP pipeline
ATHENA_DIR = "data/athena"       # path to Athena OMOP vocabulary download
KEEP_VARIANT = "paper"           # "paper" (L2+1e-3+AdamW+mean) or "code" (cosine+1e-5+Adagrad+sum)
RUN_INTRINSIC_EVAL = True        # compute Resnik/co-occ correlations after pipeline

# KEEP embedding caching: reuse previously-trained embeddings if they exist.
# Cache layout matches train_keep.py — flat + Trainer-style timestamps:
#   {KEEP_CACHE_ROOT}/{timestamp}/keep_snomed.txt
#   {KEEP_CACHE_ROOT}/{timestamp}/manifest.json  ← config lives here, not in the path
#
# Cache lookup reads each run's manifest.json and picks one matching the CURRENT
# config (KEEP_VARIANT + MIMIC_VERSION + MIN_OCCURRENCES):
#   - KEEP_CACHE_RUN_ID = None    → newest matching run
#   - KEEP_CACHE_RUN_ID = "<ts>"  → that specific timestamp folder (no manifest check)
USE_KEEP_CACHE = False                # True = reuse cached embeddings; False = always rebuild
KEEP_CACHE_ROOT = "output/keep_emb_output"
KEEP_CACHE_RUN_ID = None              # None = newest matching; or e.g. "20260420-143042"

# Data source toggle: local real MIMIC vs GCS synthetic
MIMIC_VERSION = "mimic4"              # "mimic3" (ICD-9 only) or "mimic4" (mixed ICD-9/ICD-10)
USE_LOCAL_MIMIC = True                # True = real data at LOCAL_MIMIC_ROOT, False = GCS synthetic
LOCAL_MIMIC_ROOTS = {
    "mimic3": "data/mimic3",
    "mimic4": "data/mimic4",
}
DEV_MODE = False                     # True = subset + tiny pipeline, False = full run (real experiment)

# Device for KEEP GloVe training (Node2Vec stage is always CPU — gensim limitation).
# "auto" picks cuda > mps > cpu automatically based on what's available.
# Only used when rebuilding embeddings (USE_KEEP_CACHE=False or no cache exists).
# GRASP training device is picked independently by the PyHealth Trainer.
DEVICE = "auto"                      # "auto" | "cuda" | "mps" | "cpu"

# Filter knob — must match the cached run's min_occurrences for cache lookup.
# Paper default is 2; MIMIC-IV may benefit from 1 (see keep-learning-journal-filter-investigation.md)
MIN_OCCURRENCES = 2
# ──────────────────────────────────────────────────────────

# Paper-faithful vs G2Lab code-faithful variants.
# Both are valid KEEP; we don't know which produced the published Table 4
# numbers.
KEEP_VARIANTS = {
    "paper": {
        "reg_distance": "l2",
        "optimizer": "adamw",
        "lambd": 1e-3,
    },
    "code": {
        "reg_distance": "cosine",
        "optimizer": "adagrad",
        "lambd": 1e-5,
    },
}
# Note: reg reduction is always `sum` per paper Eq 4. It's not a variant
# knob because it's mathematically coupled to `lambd`.

if __name__ == "__main__":
    print_hardware_info()

    # STEP 1: load data
    if MIMIC_VERSION == "mimic3":
        mimic_root = (
            LOCAL_MIMIC_ROOTS["mimic3"]
            if USE_LOCAL_MIMIC
            else "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
        )
        print(f"MIMIC-III root: {mimic_root} (dev={DEV_MODE})")
        base_dataset = MIMIC3Dataset(
            root=mimic_root,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            cache_dir=tempfile.TemporaryDirectory().name,
            dev=DEV_MODE,
        )
    elif MIMIC_VERSION == "mimic4":
        if not USE_LOCAL_MIMIC:
            raise ValueError(
                "MIMIC-IV has no public synthetic fallback. "
                "Set USE_LOCAL_MIMIC=True and provide local data."
            )
        mimic_root = LOCAL_MIMIC_ROOTS["mimic4"]
        print(f"MIMIC-IV root: {mimic_root} (dev={DEV_MODE})")
        base_dataset = MIMIC4EHRDataset(
            root=mimic_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=tempfile.TemporaryDirectory().name,
            dev=DEV_MODE,
        )
    else:
        raise ValueError(
            f"MIMIC_VERSION must be 'mimic3' or 'mimic4', got {MIMIC_VERSION!r}"
        )
    base_dataset.stats()

    # STEP 2: build KEEP embeddings (skip if USE_KEEP=False)
    keep_emb_path = None

    intrinsic_results = None

    if USE_KEEP:
        import json as _json
        from datetime import datetime as _dt
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            run_keep_pipeline,
        )
        variant_params = KEEP_VARIANTS[KEEP_VARIANT]
        print(f"KEEP variant: {KEEP_VARIANT} ({variant_params})")

        # Cache layout (shared with train_keep.py, flat):
        #   {KEEP_CACHE_ROOT}/{timestamp}/keep_snomed.txt
        # Config lives in each run's manifest.json — we read manifests to
        # find cache runs matching the CURRENT config (variant + mimic + minocc).
        cache_root = Path(KEEP_CACHE_ROOT)

        def _resolve_cached_run():
            """Find a cached run matching current config via manifest lookup."""
            if not cache_root.exists():
                return None
            # List timestamp subfolders, newest first (lexicographic sort works
            # because timestamps are YYYYMMDD-HHMMSS)
            all_runs = sorted(
                [p for p in cache_root.iterdir() if p.is_dir()],
                reverse=True,
            )
            if not all_runs:
                return None

            # If user pinned a specific timestamp, use it directly
            if KEEP_CACHE_RUN_ID is not None:
                match = next(
                    (p for p in all_runs if p.name == KEEP_CACHE_RUN_ID),
                    None,
                )
                if match is None:
                    print(
                        f"  WARN: KEEP_CACHE_RUN_ID='{KEEP_CACHE_RUN_ID}' not "
                        f"found under {cache_root}. Available: "
                        f"{[p.name for p in all_runs]}"
                    )
                return match

            # Otherwise, read config.json (or legacy manifest.json) in each
            # run and match on config. Supports both the new split-file layout
            # and legacy manifest.json from earlier runs.
            for run_dir in all_runs:
                config_path = run_dir / "config.json"
                legacy_manifest = run_dir / "manifest.json"
                source = None
                if config_path.exists():
                    source = config_path
                elif legacy_manifest.exists():
                    source = legacy_manifest  # backward-compat
                if source is None:
                    continue
                try:
                    with open(source) as f:
                        m = _json.load(f)
                except (OSError, _json.JSONDecodeError):
                    continue
                if (
                    m.get("keep_variant") == KEEP_VARIANT
                    and m.get("mimic_version") == MIMIC_VERSION
                    and m.get("min_occurrences") == MIN_OCCURRENCES
                ):
                    return run_dir
            return None

        cached_run_dir = _resolve_cached_run() if USE_KEEP_CACHE else None
        if cached_run_dir is not None:
            cached_emb = cached_run_dir / "keep_snomed.txt"
            if cached_emb.exists():
                print(f"Using cached KEEP embeddings: {cached_emb}")
                print(f"  (cache selected: {cached_run_dir.name}; set "
                      f"USE_KEEP_CACHE=False to rebuild)")
                keep_emb_path = str(cached_emb)
                variant_output_dir = cached_run_dir  # used downstream for cooc
            else:
                print(f"  WARN: Found run dir {cached_run_dir} but no "
                      f"keep_snomed.txt. Rebuilding.")
                cached_run_dir = None

        if cached_run_dir is None:
            # No cache (or USE_KEEP_CACHE=False) — rebuild into a fresh
            # timestamped folder matching train_keep.py's flat layout.
            from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
                resolve_device,
            )
            resolved_device = resolve_device(DEVICE)
            if DEVICE == "auto":
                print(f"  KEEP GloVe device: auto → {resolved_device}")
            else:
                print(f"  KEEP GloVe device: {resolved_device}")

            timestamp = _dt.now().strftime("%Y%m%d-%H%M%S")
            variant_output_dir = cache_root / timestamp
            variant_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Rebuilding KEEP embeddings → {variant_output_dir}")
            keep_emb_path = run_keep_pipeline(
                athena_dir=ATHENA_DIR,
                dataset=base_dataset,
                output_dir=str(variant_output_dir),
                dev=DEV_MODE,
                min_occurrences=MIN_OCCURRENCES,
                device=resolved_device,
                **variant_params,
            )

        # STEP 2b: intrinsic evaluation against paper Table 2 targets
        if RUN_INTRINSIC_EVAL:
            print("\nKEEP intrinsic eval: Resnik/co-occurrence correlations...")
            import json
            from pyhealth.medcode.pretrained_embeddings.keep_emb import (
                build_hierarchy_graph, load_keep_embeddings,
                resnik_correlation, cooccurrence_correlation,
            )
            # Rebuild graph (fast since Athena files are cached in memory)
            athena_concept = Path(ATHENA_DIR) / "CONCEPT.csv"
            athena_rel = Path(ATHENA_DIR) / "CONCEPT_RELATIONSHIP.csv"
            athena_ancestor = Path(ATHENA_DIR) / "CONCEPT_ANCESTOR.csv"
            if not athena_ancestor.exists():
                athena_ancestor = None
            eval_graph = build_hierarchy_graph(
                athena_concept, athena_rel,
                ancestor_csv=athena_ancestor,
            )

            # Load exported embeddings (keyed by SNOMED concept_code string)
            eval_emb, token_strings = load_keep_embeddings(
                keep_emb_path, embedding_dim=100,
            )
            # Map concept_code back to concept_id for graph lookup
            code_to_id = {
                str(eval_graph.nodes[n].get("concept_code", n)): n
                for n in eval_graph.nodes()
            }
            eval_node_ids = [
                code_to_id[tok] for tok in token_strings if tok in code_to_id
            ]
            # Filter embeddings to matching subset
            valid_mask = [tok in code_to_id for tok in token_strings]
            eval_emb = eval_emb[valid_mask]

            if len(eval_node_ids) >= 11:
                # Smaller K to keep eval tractable in dev mode
                k1 = min(10, len(eval_node_ids) // 10)
                k2 = min(150, len(eval_node_ids) - k1 - 1)
                runs = 50  # paper uses 250; 50 is enough for a smoke check

                # Resnik correlation (paper Table 2 target: 0.68)
                resnik_results = resnik_correlation(
                    eval_emb, eval_node_ids, eval_graph,
                    k1=k1, k2=k2, num_runs=runs, seed=42,
                )
                print(
                    f"  Resnik correlation (median): {resnik_results['median']:.4f} "
                    f"(paper target: 0.68)"
                )

                # Co-occurrence correlation (paper Table 2 target: 0.62)
                # Load the saved cooc matrix + index from the pipeline output.
                cooc_matrix_path = variant_output_dir / "cooc_matrix.npy"
                cooc_index_path = variant_output_dir / "cooc_index.json"
                cooc_results = None
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
                        f"  Co-occurrence correlation (median): "
                        f"{cooc_results['median']:.4f} (paper target: 0.62)"
                    )
                else:
                    print(
                        "  Co-occurrence correlation skipped: saved cooc matrix "
                        f"not found at {cooc_matrix_path}. Cached embeddings "
                        "from before 2026-04-13 don't include the matrix — "
                        "rerun the pipeline to regenerate."
                    )

                intrinsic_results = {
                    "resnik": resnik_results,
                    "cooccurrence": cooc_results,
                }
            else:
                print(
                    f"  Skipped: only {len(eval_node_ids)} in-graph concepts "
                    "(need >= 11 for K1=10 + K2>=1). Real MIMIC runs will have "
                    "thousands of concepts."
                )
                intrinsic_results = None
    else:
        print("USE_KEEP=False, using random embeddings.")

    # STEP 3: set task
    task_cls = (
        MortalityPredictionMIMIC3 if MIMIC_VERSION == "mimic3"
        else MortalityPredictionMIMIC4
    )
    if keep_emb_path is not None:
        task = task_cls(
            code_mapping={
                "conditions": ("ICD9CM", "SNOMED"),
                "procedures": ("ICD9PROC", "CCSPROC"),
                "drugs": ("NDC", "ATC"),
            }
        )
    else:
        task = task_cls()

    sample_dataset = base_dataset.set_task(task)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

    # STEP 4: define model
    model = GRASP(
        dataset=sample_dataset,
        embedding_dim=100,
        hidden_dim=32,
        cluster_num=8,
        block="GRU",
        pretrained_emb_path=keep_emb_path,
    )

    print(f"\n{'=' * 60}")
    print(f"Model: GRASP + GRU {'+ KEEP' if keep_emb_path else '(random init)'}")
    print(f"{'=' * 60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Embedding dim: {model.embedding_dim}")
    print(f"Hidden dim: {model.hidden_dim}")
    print(f"Cluster num: 8")
    print(f"Pretrained: {keep_emb_path or 'None (random)'}")
    print(model)

    # STEP 5: define trainer
    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "accuracy", "f1"],
    )

    # ── Start compute tracking ────────────────────────────
    run_name = f"GRASP_GRU_{'KEEP' if keep_emb_path else 'random'}"
    emissions_tracker = None
    if CODECARBON_AVAILABLE:
        emissions_tracker = EmissionsTracker(
            project_name=run_name, log_level="error", save_to_file=False,
        )
        emissions_tracker.start()

    gpu_readings = []
    gpu_readings.append(gpu_snapshot())
    start_time = time.time()

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=50,
        monitor="pr_auc",
        optimizer_params={"lr": 1e-3},
        weight_decay=1e-4,
    )

    wall_time = time.time() - start_time
    gpu_readings.append(gpu_snapshot())
    emissions_data = None
    if emissions_tracker:
        emissions_tracker.stop()
        emissions_data = emissions_tracker.final_emissions_data

    print_compute_report(run_name, wall_time, emissions_data,
                         [r for r in gpu_readings if r])

    # STEP 6: evaluate
    results = trainer.evaluate(test_dataloader)
    print(f"\n{'=' * 60}")
    print(f"Test Results ({run_name}):")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print(f"{'=' * 60}")

    # STEP 7: save run artifacts
    import json
    from datetime import datetime

    run_dir = Path(trainer.exp_path) if trainer.exp_path else Path(f"output/{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config (inputs — deterministic given the script's config block)
    config = {
        "run_name": run_name,
        # Data source
        "mimic_version": MIMIC_VERSION,
        "mimic_root": LOCAL_MIMIC_ROOTS[MIMIC_VERSION],
        "dev_mode": DEV_MODE,
        # Embedding source
        "embedding": "KEEP" if keep_emb_path else "random",
        "keep_variant": KEEP_VARIANT if keep_emb_path else None,
        "keep_variant_params": (
            KEEP_VARIANTS[KEEP_VARIANT] if keep_emb_path else None
        ),
        "min_occurrences": MIN_OCCURRENCES if keep_emb_path else None,
        "use_keep_cache": USE_KEEP_CACHE,
        "keep_cache_run_id": KEEP_CACHE_RUN_ID,
        "device_config": DEVICE,
        "pretrained_emb_path": keep_emb_path,
        # GRASP model
        "embedding_dim": 100,
        "hidden_dim": 32,
        "cluster_num": 8,
        "block": "GRU",
        # Training
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 50,
        "monitor": "pr_auc",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save results (outputs — non-deterministic across reruns)
    run_results = {
        "metrics": {k: float(v) for k, v in results.items()},
        "wall_time_s": wall_time,
        "wall_time_min": wall_time / 60,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    if keep_emb_path:
        # Record which embeddings file actually got loaded (may differ from
        # the config's pretrained_emb_path if cache lookup resolved a newer run)
        run_results["keep_embeddings_used"] = keep_emb_path
    if emissions_data:
        run_results["energy_kwh"] = emissions_data.energy_consumed
        run_results["co2_kg"] = emissions_data.emissions
    if intrinsic_results:
        run_results["intrinsic_eval"] = {
            "resnik": intrinsic_results.get("resnik"),
            "cooccurrence": intrinsic_results.get("cooccurrence"),
            "paper_resnik_target": 0.68,
            "paper_cooccurrence_target": 0.62,
        }
    with open(run_dir / "results.json", "w") as f:
        json.dump(run_results, f, indent=2)

    # Copy KEEP embedding artifacts into run_dir so everything is in one place
    if keep_emb_path is not None:
        import shutil
        keep_emb_src = Path(keep_emb_path).parent
        for artifact in ("keep_snomed.txt", "cooc_matrix.npy", "cooc_index.json"):
            src = keep_emb_src / artifact
            if src.exists():
                shutil.copy2(src, run_dir / artifact)
                print(f"  Copied {artifact} -> {run_dir / artifact}")

    # Loss landscape visualization
    print(f"\nGenerating loss landscape -> {run_dir / 'loss_landscape.png'}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def random_direction(state_dict):
            direction = {}
            for k, v in state_dict.items():
                if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    d = torch.randn_like(v)
                    w_norm, d_norm = v.norm(), d.norm()
                    direction[k] = d * (w_norm / d_norm) if d_norm > 1e-10 and w_norm > 1e-10 else d
                else:
                    direction[k] = torch.zeros_like(v)
            return direction

        model.eval()
        center = {k: v.clone() for k, v in model.state_dict().items()}
        dir1 = random_direction(center)
        dir2 = random_direction(center)

        steps = 20
        distance = 0.5
        alphas = np.linspace(-distance, distance, steps)
        betas = np.linspace(-distance, distance, steps)
        losses = np.zeros((steps, steps))

        landscape_batch = next(iter(test_dataloader))
        landscape_batch = {k: v for k, v in landscape_batch.items() if k != "embed"}

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                perturbed = {
                    k: center[k] + alpha * dir1[k] + beta * dir2[k]
                    for k in center
                }
                model.load_state_dict(perturbed)
                with torch.no_grad():
                    try:
                        out = model(**landscape_batch)
                        losses[i, j] = out["loss"].item()
                    except Exception:
                        losses[i, j] = float("nan")

        model.load_state_dict(center)

        A, B = np.meshgrid(alphas, betas)
        Z = np.nan_to_num(losses.T, nan=np.nanmax(losses))

        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(f"Loss Landscape: {run_name}", fontsize=13, fontweight="bold")

        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot_surface(A, B, Z, cmap="plasma", alpha=0.85, linewidth=0)
        ax1.set_xlabel("Direction 1")
        ax1.set_ylabel("Direction 2")
        ax1.set_zlabel("Loss")
        ax1.set_title("3D Loss Surface")

        ax2 = fig.add_subplot(132)
        contour = ax2.contourf(A, B, Z, levels=30, cmap="plasma")
        plt.colorbar(contour, ax=ax2)
        ax2.scatter([0], [0], color="cyan", s=120, zorder=5, label="Trained weights")
        ax2.set_xlabel("Direction 1")
        ax2.set_ylabel("Direction 2")
        ax2.set_title("Contour")
        ax2.legend()

        ax3 = fig.add_subplot(133)
        mid = steps // 2
        ax3.plot(alphas, losses[:, mid], "b-", lw=2, label="Direction 1")
        ax3.plot(betas, losses[mid, :], "r-", lw=2, label="Direction 2")
        ax3.axvline(0, color="gray", ls="--", alpha=0.5)
        ax3.set_xlabel("Perturbation")
        ax3.set_ylabel("Loss")
        ax3.set_title("1D Cross-sections")
        ax3.legend()

        valid = losses[~np.isnan(losses)]
        sharpness = valid.max() - valid.min()
        stats = (
            f"Current loss: {losses[mid, mid]:.4f}\n"
            f"Sharpness: {sharpness:.4f}\n"
            f"{'Wide basin (stable)' if sharpness < 0.1 else 'Moderate basin' if sharpness < 0.5 else 'Sharp basin (oscillation risk)'}"
        )
        fig.text(0.01, 0.02, stats, fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        plt.savefig(run_dir / "loss_landscape.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved loss landscape to {run_dir / 'loss_landscape.png'}")
        print(f"  Sharpness: {sharpness:.4f}")
    except Exception as e:
        print(f"  Loss landscape skipped: {e}")

    print(f"\nAll artifacts saved to {run_dir}")
