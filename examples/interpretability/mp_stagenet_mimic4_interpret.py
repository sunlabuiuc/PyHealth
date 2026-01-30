"""Evaluate all interpretability methods on StageNet + MIMIC-IV dataset using comprehensiveness 
and sufficiency metrics.

This example demonstrates:
1. Loading a pre-trained StageNet model with processors and MIMIC-IV dataset
2. Computing attributions with various interpretability methods
3. Evaluating attribution faithfulness with Comprehensiveness & Sufficiency for each method
4. Presenting results in a summary table
"""

from datetime import datetime
import torch

import argparse
from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.interpret.methods import BaseInterpreter, IntegratedGradients, DeepLift, GIM, ShapExplainer, LimeExplainer
from pyhealth.metrics.interpretability import Evaluator, evaluate_attribution
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer
from pyhealth.datasets.utils import load_processors
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Comma separated list of interpretability methods to evaluate"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ig,deeplift,gim,shap,lime",
        help="Comma-separated list of interpretability methods to evaluate (default: ig,deeplift,gim,shap,lime)",
    )
    
    """Main execution function."""
    print("=" * 70)
    print("Interpretability Metrics Example: StageNet + MIMIC-IV")
    print("=" * 70)

    # Set path
    CACHE_DIR = Path("/shared/eng/pyhealth_dka/cache/mp_stagenet_mimic4")
    CKPTS_DIR = Path("/shared/eng/pyhealth_dka/ckpts/mp_stagenet_mimic4")
    OUTPUT_DIR = Path("/shared/eng/pyhealth_dka/output/mp_stagenet_mimic4")
    print(f"\nUsing cache dir: {CACHE_DIR}")
    print(f"Using checkpoints dir: {CKPTS_DIR}")
    print(f"Using output dir: {OUTPUT_DIR}")

    # Set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load MIMIC-IV dataset
    print("\n Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        cache_dir=str(CACHE_DIR),
        num_workers=16,
    )

    # Apply mortality prediction task
    if not (CKPTS_DIR / "input_processors.pkl").exists():
        raise FileNotFoundError(f"Input processors not found in {CKPTS_DIR}. ")
    if not (CKPTS_DIR / "output_processors.pkl").exists():
        raise FileNotFoundError(f"Output processors not found in {CKPTS_DIR}. ")
    input_processors, output_processors = load_processors(str(CKPTS_DIR))
    print("✓ Loaded input and output processors from checkpoint directory.")
    
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=16,
        input_processors=input_processors,
        output_processors=output_processors,
    )
    print(f"✓ Loaded {len(sample_dataset)} samples")

    # Split dataset and get test loader
    _, _, test_dataset = split_by_patient(sample_dataset, [0.9, 0.09, 0.01], seed=233)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)
    print(f"✓ Test set: {len(test_dataset)} samples")

    # Initialize and load pre-trained model
    print("\n Loading pre-trained StageNet model...")
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    trainer = Trainer(model=model, device=device)
    trainer.load_ckpt(str(CKPTS_DIR / "best.ckpt"))
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded checkpoint: {CKPTS_DIR / 'best.ckpt'}")
    print(f"✓ Model moved to {device}")

    methods: dict[str, BaseInterpreter] = {
        # "ig": IntegratedGradients(model, use_embeddings=True),
        # "deeplift": DeepLift(model, use_embeddings=True),
        "gim": GIM(model),
        "shap": ShapExplainer(model, use_embeddings=True),
        "lime": LimeExplainer(model, use_embeddings=True),
    }
    methods = {k: v for k, v in methods.items() if k in parser.parse_args().methods.split(",")}
    print(f"\nEvaluating methods: {list(methods.keys())}")
    
    res = {}
    for name, method in methods.items():
        print(f"\n Initializing {name}...")
        print("=" * 70)
        
        # Option 1: Functional API (simple one-off evaluation)
        print("\nEvaluating with Functional API on full dataset...")
        print("Using: evaluate_attribution(model, dataloader, method, ...)")

        results_functional = evaluate_attribution(
            model,
            test_loader,
            method,
            metrics=["comprehensiveness", "sufficiency"],
            percentages=[25, 50, 99],
        )

        print("\n" + "=" * 70)
        print("Dataset-Wide Results (Functional API)")
        print("=" * 70)
        comp = results_functional["comprehensiveness"]
        suff = results_functional["sufficiency"]
        print(f"\nComprehensiveness: {comp:.4f}")
        print(f"Sufficiency:       {suff:.4f}")
        
        res[name] = {
            "comp": comp,
            "suff": suff,
        }
    
    print("")
    print("=" * 70)
    print("Summary of Results for All Methods")
    print(res)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUTPUT_DIR / "results.csv").exists():
        df = pd.read_csv(OUTPUT_DIR / "results.csv")
    else:
        df = pd.DataFrame({
            'Method': pd.Series(dtype='str'),
            'Comp': pd.Series(dtype='float'),
            'Suff': pd.Series(dtype='float'),
        })
    
    for name, scores in res.items():
        df.loc[len(df)] = [name, scores['comp'], scores['suff']]
    
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

if __name__ == "__main__":
    main()
