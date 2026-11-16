"""DeepLift attributions for a StageNet mortality model on MIMIC-IV.

Loads a trained StageNet for ``MortalityPredictionStageNetMIMIC4``, computes
DeepLift attributions (``use_embeddings=True``) for one test batch, then scores
DeepLift on the test split with comprehensiveness and sufficiency
(``pyhealth.metrics.interpretability.evaluate_attribution``). Change ``name``
in ``main()`` to run IG, GIM, SHAP or LIME the same way.

Data: MIMIC-IV v2.2 hosp tables (credentialed PhysioNet access:
https://physionet.org/content/mimiciv/2.2/). ``WORK_DIR/ckpts/mp_stagenet_mimic4/``
should hold ``input_processors.pkl``, ``output_processors.pkl`` and
``best.ckpt`` from training (e.g. with
``examples/mortality_prediction/mortality_mimic4_stagenet_v2.py``).

Run (set MIMIC4_ROOT and WORK_DIR below first; a GPU is used if available):
    python examples/interpretability/deeplift_stagenet_mimic4.py
To try it without credentials, set MIMIC4_ROOT to
``test-resources/core/mimic4demo`` and train on it first; with only a few
patients the scores are not meaningful.
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

# Paths: edit before running.
MIMIC4_ROOT = "/path/to/mimiciv/2.2/"
WORK_DIR = Path("/path/to/pyhealth_dka")  # holds cache/, ckpts/, output/


def main():    
    """Main execution function."""
    print("=" * 70)
    print("Interpretability Metrics Example: StageNet + MIMIC-IV")
    print("=" * 70)

    # Set path
    CACHE_DIR = WORK_DIR / "cache" / "mp_stagenet_mimic4"
    CKPTS_DIR = WORK_DIR / "ckpts" / "mp_stagenet_mimic4"
    OUTPUT_DIR = WORK_DIR / "output" / "mp_stagenet_mimic4"
    print(f"\nUsing cache dir: {CACHE_DIR}")
    print(f"Using checkpoints dir: {CKPTS_DIR}")
    print(f"Using output dir: {OUTPUT_DIR}")

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load MIMIC-IV dataset
    print("\n Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
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
        "ig": IntegratedGradients(model, use_embeddings=True),
        "deeplift": DeepLift(model, use_embeddings=True),
        "gim": GIM(model),
        "shap": ShapExplainer(model, use_embeddings=True),
        "lime": LimeExplainer(model, use_embeddings=True),
    }
    
    name = "deeplift"
    method = methods[name]
    
    print(f"\nMove batch to {device}...")
    batch = next(iter(test_loader))
    batch0 = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch0[key] = value.to(device)
        elif isinstance(value, tuple):
            # StageNet format: (time, values) or similar tuples
            batch0[key] = tuple(
                v.to(device) if isinstance(v, torch.Tensor) else v for v in value
            )
        else:
            # Keep non-tensor values as-is (labels, metadata, etc.)
            batch0[key] = value
    batch = batch0
    
    attributions = method.attribute(**batch)
    print(f"{name} attributions for one batch:")
    for key, attr in attributions.items():
        print(f"  {key}: shape={tuple(attr.shape)}, sum={attr.sum().item():+.4f}")

    # # Initialize evaluator
    # evaluator = Evaluator(model, percentages=[1, 99])

    # # Compute metrics for single batch
    # print("\nComputing metrics on single batch...")
    # comp_metric = evaluator.metrics["comprehensiveness"]
    # comp_scores, comp_mask = comp_metric.compute(batch, attributions)

    # suff_metric = evaluator.metrics["sufficiency"]
    # suff_scores, suff_mask = suff_metric.compute(batch, attributions)

    # print("\n" + "=" * 70)
    # print("Single Batch Results Summary")
    # print("=" * 70)
    # if comp_mask.sum() > 0:
    #     valid_comp = comp_scores[comp_mask]
    #     print(f"Comprehensiveness (valid samples): {valid_comp.mean():.4f}")
    #     print(f"  Valid samples: {comp_mask.sum()}/{len(comp_mask)}")
    # else:
    #     print("Comprehensiveness: No valid samples")

    # if suff_mask.sum() > 0:
    #     valid_suff = suff_scores[suff_mask]
    #     print(f"Sufficiency (valid samples): {valid_suff.mean():.4f}")
    #     print(f"  Valid samples: {suff_mask.sum()}/{len(suff_mask)}")
    # else:
    #     print("Sufficiency: No valid samples")

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

if __name__ == "__main__":
    main()
