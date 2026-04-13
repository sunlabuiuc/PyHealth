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

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import GRASP
from pyhealth.tasks import MortalityPredictionMIMIC3
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
# ──────────────────────────────────────────────────────────

# Paper-faithful vs G2Lab code-faithful variants.
# Both are valid KEEP; we don't know which produced the published Table 4
# numbers. See docs/plans/keep/keep-implementation-comparison.md.
KEEP_VARIANTS = {
    "paper": {
        "reg_distance": "l2",
        "reg_reduction": "mean",
        "optimizer": "adamw",
        "lambd": 1e-3,
    },
    "code": {
        "reg_distance": "cosine",
        "reg_reduction": "sum",
        "optimizer": "adagrad",
        "lambd": 1e-5,
    },
}

if __name__ == "__main__":
    print_hardware_info()

    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: build KEEP embeddings (skip if USE_KEEP=False)
    keep_emb_path = None

    intrinsic_results = None

    if USE_KEEP:
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            run_keep_pipeline,
        )
        variant_params = KEEP_VARIANTS[KEEP_VARIANT]
        print(f"KEEP variant: {KEEP_VARIANT} ({variant_params})")
        keep_emb_path = run_keep_pipeline(
            athena_dir=ATHENA_DIR,
            dataset=base_dataset,
            output_dir="keep_output",
            dev=True,  # fast params for testing; set False for real runs
            **variant_params,
        )

        # STEP 2b: intrinsic evaluation against paper Table 2 targets
        if RUN_INTRINSIC_EVAL:
            print("\nKEEP intrinsic eval: Resnik/co-occurrence correlations...")
            from pyhealth.medcode.pretrained_embeddings.keep_emb import (
                build_hierarchy_graph, load_keep_embeddings,
                resnik_correlation,
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
                intrinsic_results = resnik_correlation(
                    eval_emb, eval_node_ids, eval_graph,
                    k1=k1, k2=k2, num_runs=runs, seed=42,
                )
                print(
                    f"  Resnik correlation (median): {intrinsic_results['median']:.4f} "
                    f"(paper target: 0.68)"
                )
            else:
                print(
                    f"  Skipped: only {len(eval_node_ids)} in-graph concepts "
                    "(need >= 11 for K1=10 + K2>=1). Real MIMIC runs will have "
                    "thousands of concepts."
                )
    else:
        print("USE_KEEP=False, using random embeddings.")

    # STEP 3: set task
    if keep_emb_path is not None:
        task = MortalityPredictionMIMIC3(
            code_mapping={
                "conditions": ("ICD9CM", "SNOMED"),
                "procedures": ("ICD9PROC", "CCSPROC"),
                "drugs": ("NDC", "ATC"),
            }
        )
    else:
        task = MortalityPredictionMIMIC3()

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

    # Save config
    config = {
        "run_name": run_name,
        "embedding": "KEEP" if keep_emb_path else "random",
        "keep_variant": KEEP_VARIANT if keep_emb_path else None,
        "keep_variant_params": (
            KEEP_VARIANTS[KEEP_VARIANT] if keep_emb_path else None
        ),
        "embedding_dim": 100,
        "hidden_dim": 32,
        "cluster_num": 8,
        "block": "GRU",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 50,
        "monitor": "pr_auc",
        "pretrained_emb_path": keep_emb_path,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save results
    run_results = {
        "metrics": {k: float(v) for k, v in results.items()},
        "wall_time_s": wall_time,
        "wall_time_min": wall_time / 60,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    if emissions_data:
        run_results["energy_kwh"] = emissions_data.energy_consumed
        run_results["co2_kg"] = emissions_data.emissions
    if intrinsic_results:
        run_results["intrinsic_eval"] = {
            "resnik": intrinsic_results,
            "paper_resnik_target": 0.68,
        }
    with open(run_dir / "results.json", "w") as f:
        json.dump(run_results, f, indent=2)

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
