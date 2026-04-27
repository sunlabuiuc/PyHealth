"""
LLMSYN: Synthetic EHR Generation on MIMIC-III (Hao et al., MLHC 2024).

Demonstrates all three ablation variants from the paper:
  - LLMSYNfull  : prior statistics + Mayo Clinic RAG (best performance)
  - LLMSYNprior : prior statistics only
  - LLMSYNbase  : minimal prompts, single sampled disease per record

Prerequisites:
  - MIMIC-III 1.4 access via PhysioNet (ADMISSIONS.csv, DIAGNOSES_ICD.csv,
    D_ICD_DIAGNOSES.csv, PATIENTS.csv)
  - For real LLM generation: set OPENAI_API_KEY or ANTHROPIC_API_KEY
  - For RAG (LLMSYNfull): pip install requests beautifulsoup4

Reference:
    Hao et al., "LLMSYN: Generating Synthetic Electronic Health Records
    Without Patient-Level Data", MLHC 2024, PMLR 252.
    https://proceedings.mlr.press/v252/hao24a.html
"""

import json

from pyhealth.datasets import MIMIC3Dataset, create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.models import LLMSYNModel
from pyhealth.tasks import SyntheticEHRGenerationTask
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# Configuration — update these paths for your environment
# ---------------------------------------------------------------------------

# Path to pre-computed stats JSON (run compute_all_stats() once to generate it,
# or use the bundled file from test-resources/llmsyn/stats.json)
STATS_PATH = "test-resources/llmsyn/stats.json"

# Path to MIMIC-III data directory (must contain .csv.gz files incl. ICUSTAYS.csv.gz)
# Using bundled demo data for quick testing — replace with full MIMIC-III 1.4 for real use
MIMIC3_ROOT = "test-resources/core/mimic3demo"


LLM_PROVIDER = "mock"      # "mock" | "openai" | "claude"
API_KEY = None             # set if using "openai" or "claude"
N_GENERATE = 10            # synthetic records to generate per variant


# ---------------------------------------------------------------------------
# Step 1: Compute or load prior statistics from MIMIC-III
# ---------------------------------------------------------------------------

def get_stats(stats_path: str) -> dict:
    print(f"Loading stats from {stats_path}")
    with open(stats_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 2a: Load MIMIC-III dataset and apply SyntheticEHRGenerationTask
#           (requires full MIMIC-III 1.4 download with ICUSTAYS.csv.gz)
# ---------------------------------------------------------------------------

def load_sample_dataset(mimic3_root: str):
    print("\n=== Step 2a: Loading MIMIC-III and applying task ===")
    base_dataset = MIMIC3Dataset(
        root=mimic3_root,
        tables=["diagnoses_icd"],
        dev=True,  # remove for full dataset
    )
    task = SyntheticEHRGenerationTask()
    sample_dataset = base_dataset.set_task(task)
    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema : {task.input_schema}")
    print(f"Output schema: {task.output_schema}")
    return sample_dataset


# ---------------------------------------------------------------------------
# Step 2b: Create a small in-memory dataset for quick testing / TSTR demo
#           (no MIMIC-III download required — uses create_sample_dataset)
# ---------------------------------------------------------------------------

_ICD9_POOL = [
    ["250.00", "401.9", "428.0"],   # diabetes + hypertension + heart failure
    ["410.71", "414.01", "272.4"],  # MI + coronary artery disease + hyperlipidemia
    ["486", "496", "491.21"],       # pneumonia + COPD + acute exacerbation
    ["584.9", "585.9", "403.90"],   # AKI + CKD + hypertensive CKD
    ["038.9", "995.91", "785.52"],  # sepsis + SIRS + septic shock
]


def create_mock_sample_dataset():
    """Build a 20-patient in-memory dataset using create_sample_dataset.

    Returns a SampleDataset compatible with split_by_patient and LLMSYNModel
    without requiring any MIMIC-III files.  Each of the 20 mock patients has
    1–2 admissions drawn from a small pool of common ICD-9 clusters.
    """
    print("\n=== Step 2b: Creating mock in-memory dataset ===")
    task = SyntheticEHRGenerationTask()
    samples = []
    for i in range(20):
        pid = f"P{i:03d}"
        for j in range(1 + (i % 2)):          # 1 or 2 visits per patient
            samples.append({
                "patient_id": pid,
                "visit_id": f"V{i:03d}_{j}",
                "conditions": _ICD9_POOL[i % len(_ICD9_POOL)],
                "mortality": 1 if i % 10 == 0 else 0,
            })
    sample_dataset = create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        task_name=task.task_name,
    )
    print(f"Mock samples : {len(sample_dataset)}")
    return sample_dataset


# ---------------------------------------------------------------------------
# Step 3: Run all three ablation variants
# ---------------------------------------------------------------------------

VARIANTS = {
    "LLMSYNfull" : {"prior_mode": "full",    "enable_rag": True},
    "LLMSYNprior": {"prior_mode": "full",    "enable_rag": False},
    "LLMSYNbase" : {"prior_mode": "sampled", "enable_rag": False},
}


def run_variants(sample_dataset, stats: dict):
    print(f"\n=== Step 3: Generating {N_GENERATE} records per variant ===")
    results = {}
    for name, cfg in VARIANTS.items():
        print(f"\n--- {name} ---")
        model = LLMSYNModel(
            dataset=sample_dataset,
            llm_provider=LLM_PROVIDER,
            api_key=API_KEY,
            stats=stats,
            prior_mode=cfg["prior_mode"],
            enable_rag=cfg["enable_rag"],
            noise_scale=0.05,
            seed=42,
        )
        records = model.generate(n=N_GENERATE)
        results[name] = records

        # Print first record summary
        r = records[0]
        print(f"  Age        : {r['Age']}")
        print(f"  Gender     : {r['Gender']}")
        print(f"  Survived   : {r['Survived']}")
        print(f"  MainDx     : {r['MainDiagnosis']}")
        print(f"  Comps      : {r['Complications']}")
        print(f"  Procedures : {r['Procedures']}")

    return results


# ---------------------------------------------------------------------------
# Step 4: TSTR forward pass (optional — shows PyHealth training integration)
# ---------------------------------------------------------------------------

def run_tstr(sample_dataset, stats: dict):
    print("\n=== Step 4: TSTR forward pass (LLMSYNprior) ===")
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.7, 0.15, 0.15])
    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=16, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=16, shuffle=False)

    model = LLMSYNModel(
        dataset=sample_dataset,
        llm_provider=LLM_PROVIDER,
        api_key=API_KEY,
        stats=stats,
        prior_mode="full",
        enable_rag=False,
        seed=0,
    )

    trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,
        monitor="pr_auc",
    )

    print("\nTest Results (TSTR):")
    results = trainer.evaluate(test_loader)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LLMSYN: SYNTHETIC EHR GENERATION ON MIMIC-III")
    print("=" * 60)

    # Step 1: stats
    print("\n=== Step 1: Prior statistics ===")
    stats = get_stats(STATS_PATH)
    print(f"Mortality rate : {stats['mortality_rate']:.4f}")
    print(f"Top disease    : {stats['top_diseases'][0]['icd9_code']} — "
          f"{stats['top_diseases'][0]['desc']}")

    # Step 2: dataset
    # Option A — real MIMIC-III (requires download + ICUSTAYS.csv.gz):
    sample_dataset = load_sample_dataset(MIMIC3_ROOT)
    # Option B — quick in-memory mock (no download needed):
    # sample_dataset = create_mock_sample_dataset()

    # Step 3: generate with all 3 variants
    run_variants(sample_dataset, stats)

    # Step 4: TSTR training loop — uses mock dataset so it always runs
    # To use real data instead: run_tstr(sample_dataset, stats)
    run_tstr(sample_dataset, stats)   # real data
    # mock_ds = create_mock_sample_dataset()
    # run_tstr(mock_ds, stats)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
