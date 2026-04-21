"""Complete pipeline for wav2sleep training on SHHS sleep staging task.

This example demonstrates the full training pipeline for the wav2sleep model
following PyHealth implementation requirements:

1. **Data Loading**: Load SHHS dataset or fall back to synthetic data
2. **Dataset Configuration**: Multi-modal setup with ECG, ABD, THX signals
3. **Model Initialization**: Wav2Sleep with configurable architecture
4. **Training Loop**: Using the wav2sleep-specific trainer with:
   - AdamW optimizer
   - Linear warmup to max learning rate
   - Exponential decay schedule
   - Early stopping when validation loss plateaus (5 epochs patience)
5. **Evaluation**: Cross-modal evaluation and subset-based analysis
6. **Reproducibility**: Seed control, configurable subset sizes

Wav2Sleep paper link:
    https://doi.org/10.48550/arXiv.2411.04644

Wav2Sleep paper citation:
    Carter, J. F.; and Tarassenko, L. 2024. wav2sleep: A unified multi-modal approach
    to sleep stage classification from physiological signals. arXiv preprint arXiv:2411.04644.

Authors:
    Justin Cheok (jcheok2@illinois.edu)
    Austin Jarrett (ajj7@illinois.edu)
    Jimmy Scray (escray2@illinois.edu)

**Requirements:**
- SHHS data (optional): Set SHHS_ROOT to use real data (must be requested from NSRR)
- Or: Script will use fully synthetic data for demonstration

**Usage:**
    python examples/shhs_sleep_staging_wav2sleep.py \\
        --shhs-root /path/to/shhs/polysomnography \\
        --max-recordings 50 \\
        --train-fraction 1.0 \\
        --val-fraction 1.0 \\
        --epochs 80 \\
        --batch-size 32 \\
        --max-lr 1e-3 \\
        --patience 5
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Wav2Sleep, load_shhs_samples
from pyhealth.training.wav2sleep_trainer import Wav2SleepTrainer, create_subset_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Sleep stage label definitions (AASM 4-class)
SLEEP_STAGES = {0: "Wake", 1: "Light (N1+N2)", 2: "N3 (Deep)", 3: "REM"}
NUM_CLASSES = len(SLEEP_STAGES)

# Signal sampling parameters (matching paper)
ECG_SAMPLES_PER_EPOCH = 256  # ~34 Hz × 30 s (paper uses 1024 with higher resolution)
RESP_SAMPLES_PER_EPOCH = 128  # ~8 Hz × 30 s (paper uses 256 with higher resolution)


# =============================================================================
# Dataset Construction
# =============================================================================

def build_synthetic_dataset(
    n_patients: int = 10,
    epochs_per_patient: int = 100,
    dataset_name: str = "synthetic_shhs_wav2sleep",
    seed: int = 42,
) -> object:
    """Create synthetic SHHS-like dataset for testing.

    This function generates a fully synthetic dataset that mirrors the SHHS
    data structure, useful for development, testing, and when real data
    is unavailable.

    Each sample corresponds to one 30-second polysomnography epoch with:
    - ECG  : electrocardiogram signal (1 channel)
    - ABD  : abdominal respiratory effort (1 channel)
    - THX  : thoracic respiratory effort (1 channel)
    - Label: sleep stage (0=Wake, 1=Light, 2=N3, 3=REM)

    Args:
        n_patients: Number of synthetic patients.
        epochs_per_patient: Number of 30-second epochs per patient.
        dataset_name: Name for the created dataset.
        seed: Random seed for reproducibility.

    Returns:
        PyHealth SampleDataset ready for training.
    """
    logger.info(f"Building synthetic dataset ({n_patients} patients, "
                f"{epochs_per_patient} epochs each)")

    rng = np.random.RandomState(seed)
    samples = []

    for pid in range(n_patients):
        for epoch_idx in range(epochs_per_patient):
            # Randomly assign sleep stage
            label = rng.randint(0, NUM_CLASSES)

            samples.append({
                "patient_id": f"patient-{pid:03d}",
                "visit_id": f"epoch-{epoch_idx:04d}",
                # ECG signal: 1 channel, ECG_SAMPLES_PER_EPOCH samples
                "ecg": rng.randn(1, ECG_SAMPLES_PER_EPOCH).astype(np.float32),
                # Abdominal respiratory effort
                "abd": rng.randn(1, RESP_SAMPLES_PER_EPOCH).astype(np.float32),
                # Thoracic respiratory effort
                "thx": rng.randn(1, RESP_SAMPLES_PER_EPOCH).astype(np.float32),
                "label": label,
            })

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "ecg": "tensor",  # Electrocardiogram
            "abd": "tensor",  # Abdominal respiratory effort
            "thx": "tensor",  # Thoracic respiratory effort
        },
        output_schema={"label": "multiclass"},
        dataset_name=dataset_name,
    )
    logger.info(f"✓ Created synthetic dataset with {len(dataset)} samples")
    return dataset


def load_shhs_or_synthetic(
    shhs_root: Optional[str] = None,
    max_recordings: Optional[int] = None,
) -> object:
    """Load SHHS dataset if available, otherwise create synthetic fallback.

    Args:
        shhs_root: Path to SHHS polysomnography directory.
            If None or invalid, synthetic data is used.
        max_recordings: Maximum number of recordings to load from SHHS.
            If None, all available recordings are loaded.

    Returns:
        PyHealth SampleDataset.
    """
    if shhs_root and os.path.isdir(shhs_root):
        logger.info(f"Loading SHHS dataset from {shhs_root}")
        # Map SHHS 5-class labels to 4-class (merge N1+N2 as "Light")
        # SHHS: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        # Target: 0=Wake, 1=Light(N1+N2), 2=N3, 3=REM
        label_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}

        samples = load_shhs_samples(
            shhs_root=shhs_root,
            epoch_seconds=30,
            ecg_samples_per_epoch=ECG_SAMPLES_PER_EPOCH,
            resp_samples_per_epoch=RESP_SAMPLES_PER_EPOCH,
            max_recordings=max_recordings,
            label_map=label_map,
        )

        if not samples:
            logger.warning("No samples loaded from SHHS, falling back to synthetic data")
            return build_synthetic_dataset()

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "ecg": "tensor",
                "abd": "tensor",
                "thx": "tensor",
            },
            output_schema={"label": "multiclass"},
            dataset_name="shhs_wav2sleep",
        )
        logger.info(f"✓ Loaded {len(dataset)} samples from SHHS")
        return dataset
    else:
        logger.info("SHHS_ROOT not set or invalid — using synthetic data")
        return build_synthetic_dataset()


# =============================================================================
# Model Training
# =============================================================================

def train_wav2sleep_model(
    dataset,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    batch_size: int = 32,
    epochs: int = 80,
    warmup_fraction: float = 0.1,
    max_lr: float = 1e-3,
    decay_rate: float = 0.9999,
    weight_decay: float = 1e-2,
    patience: int = 5,
    device: Optional[str] = None,
    output_dir: str = "output",
) -> dict:
    """Train wav2sleep model with specified hyperparameters.

    This function implements the complete training pipeline described in the
    wav2sleep paper with:
    - AdamW optimizer (no weight decay on bias/norm parameters)
    - Linear warmup phase: 0 → max_lr over warmup_fraction of training
    - Exponential decay phase: max_lr → ~0 with decay_rate^step
    - Early stopping: patience=5 (as per paper) means training stops if
      validation loss doesn't improve for 5 consecutive epochs
    - Optional subset training for memory-constrained environments

    Args:
        dataset: PyHealth SampleDataset.
        train_fraction: Fraction of training set to use (0.0 to 1.0).
            Default: 1.0 (use full dataset).
        val_fraction: Fraction of validation set to use (0.0 to 1.0).
            Default: 1.0 (use full dataset).
        batch_size: Batch size for training. Default: 32.
        epochs: Maximum number of epochs. Default: 80.
        warmup_fraction: Fraction of training steps for linear warmup.
            Default: 0.1 (10% of training are warmup).
        max_lr: Maximum learning rate (reached at end of warmup).
            Default: 1e-3.
        decay_rate: Exponential decay factor per step.
            Default: 0.9999 (smooth decay).
            Higher values → slower decay, lower values → faster decay.
        weight_decay: L2 regularization coefficient. Default: 1e-2.
        patience: Early stopping patience. Default: 5 (as per wav2sleep paper).
        device: Device for training ('cuda' or 'cpu'). Auto-detect if None.
        output_dir: Directory for saving checkpoints and logs. Default: 'output'.

    Returns:
        Dictionary with training history:
        - 'train_loss': List of per-epoch training losses
        - 'val_loss': List of per-epoch validation losses
        - 'val_accuracy': List of per-epoch validation accuracies
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("")
    logger.info("=" * 70)
    logger.info("Wav2Sleep Model Training")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Create data loaders (with subset support)
    # -----------------------------------------------------------------------
    logger.info("\n[1/4] Preparing data loaders...")
    train_loader, val_loader = create_subset_loaders(
        dataset=dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        batch_size=batch_size,
        seed=42,
    )
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader) if val_loader else 'N/A'}")

    # -----------------------------------------------------------------------
    # 2. Create model
    # -----------------------------------------------------------------------
    logger.info("\n[2/4] Initializing model...")
    model = Wav2Sleep(
        dataset=dataset,
        feature_dim=128,
        n_transformer_layers=2,
        n_attention_heads=8,
        transformer_ff_dim=512,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # 3. Create trainer and configure schedule
    # -----------------------------------------------------------------------
    logger.info("\n[3/4] Configuring trainer...")
    trainer = Wav2SleepTrainer(
        model=model,
        device=device,
        enable_logging=True,
        output_path=output_dir,
        exp_name="wav2sleep_training",
    )

    # Log training configuration
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Max learning rate: {max_lr:.2e}")
    logger.info(f"  Warmup fraction: {warmup_fraction:.1%}")
    logger.info(f"  Decay rate: {decay_rate:.6f}")
    logger.info(f"  Weight decay: {weight_decay:.2e}")
    logger.info(f"  Early stopping patience: {patience} epochs")

    # -----------------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------------
    logger.info("\n[4/4] Training model...")
    history = trainer.train_with_schedule(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        warmup_fraction=warmup_fraction,
        max_lr=max_lr,
        decay_rate=decay_rate,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        patience=patience,
    )

    return history, trainer, model


# =============================================================================
# Evaluation & Analysis
# =============================================================================

def evaluate_modality_subset(
    model,
    val_loader,
    subset_keys: list,
    device: str = "cuda",
) -> tuple:
    """Evaluate model on subset of modalities (cross-modal generalization).

    This demonstrates the key feature of wav2sleep: the same trained model
    can work with any subset of input modalities without retraining.

    Args:
        model: Trained Wav2Sleep model.
        val_loader: Validation data loader.
        subset_keys: List of modality names to keep (e.g., ['ecg']).
        device: Device for evaluation.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            # Keep only requested modalities
            filtered_batch = {
                k: v for k, v in batch.items()
                if k in subset_keys or k == "label"
            }

            output = model(**filtered_batch)
            loss = output["loss"].item() * len(output["y_true"])
            total_loss += loss

            preds = output["y_prob"].argmax(dim=-1)
            total_correct += (preds == output["y_true"]).sum().item()
            total_samples += len(output["y_true"])

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return mean_loss, accuracy


def print_results_summary(history, model, val_loader, device="cuda"):
    """Print comprehensive training results summary with wav2sleep metrics.
    
    This includes Cohen's Kappa (the primary metric from the wav2sleep paper),
    accuracy, per-class metrics, and confusion matrices. Results are logged via
    the Python logger, including overall training progress, model performance
    on all modalities, and cross-modal evaluation showing robustness with
    different signal subsets.
    
    Args:
        history: Training history dictionary with keys:
            - 'train_loss': List of per-epoch training losses.
            - 'val_loss': List of per-epoch validation losses (optional).
            - 'val_accuracy': List of per-epoch validation accuracies (optional).
        model: Trained Wav2Sleep model instance with evaluate() and
            evaluate_modalities() methods.
        val_loader: PyHealth DataLoader containing validation dataset samples.
            Each batch should include signal modalities and labels.
        device: Device for evaluation, either 'cuda' or 'cpu'. Default: 'cuda'.
    
    Returns:
        None. Results are printed via logger.info() calls.
    """
    from pyhealth.metrics import format_evaluation_report
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Results Summary")
    logger.info("=" * 70)

    # Training curves
    logger.info("\nTraining Progress:")
    logger.info(f"  Initial train loss: {history['train_loss'][0]:.6f}")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    if history.get('val_loss'):
        logger.info(f"  Best val loss: {min(history['val_loss']):.6f}")
    if history.get('val_accuracy'):
        logger.info(f"  Best val accuracy: {max(history['val_accuracy']):.4f}")

    # =========================================================================
    # Comprehensive Evaluation Metrics (wav2sleep study)
    # =========================================================================
    if val_loader:
        logger.info("\n" + "=" * 70)
        logger.info("WAV2SLEEP EVALUATION METRICS (Primary: Cohen's Kappa)")
        logger.info("=" * 70)
        
        # Overall evaluation on all modalities
        logger.info("\n[1] Overall Model Performance (All Modalities)")
        logger.info("-" * 70)
        eval_results = model.evaluate(
            val_loader,
            class_names=list(SLEEP_STAGES.values())
        )
        
        logger.info(f"  Accuracy (Correct/Total):        {eval_results['accuracy']:.4f}")
        logger.info(f"  Cohen's Kappa (Primary Metric):  {eval_results['kappa']:.4f}")
        logger.info(f"  Macro-averaged Precision:        {eval_results['macro_precision']:.4f}")
        logger.info(f"  Macro-averaged Recall:           {eval_results['macro_recall']:.4f}")
        logger.info(f"  Macro-averaged F1-score:         {eval_results['macro_f1']:.4f}")
        logger.info(f"  Weighted F1-score:               {eval_results['weighted_f1']:.4f}")
        
        # Print full evaluation report with per-class metrics
        logger.info("\n" + format_evaluation_report(eval_results))
        
        # =====================================================================
        # Cross-modal evaluation: same model with different modality subsets
        # =====================================================================
        logger.info("\n[2] Cross-Modal Evaluation (Key wav2sleep Feature)")
        logger.info("-" * 70)
        logger.info("Same trained model evaluated on different modality subsets:")
        logger.info("(demonstrates graceful degradation with missing modalities)\n")
        
        modality_subsets = {
            "All (ECG + ABD + THX)": ["ecg", "abd", "thx"],
            "ECG only": ["ecg"],
            "ABD + THX (Respiratory)": ["abd", "thx"],
            "ECG + ABD": ["ecg", "abd"],
            "ECG + THX": ["ecg", "thx"],
        }
        
        cross_modal_results = model.evaluate_modalities(
            val_loader,
            modality_subsets=modality_subsets,
            class_names=list(SLEEP_STAGES.values())
        )
        
        # Print results for each subset
        results_table = []
        for subset_name in modality_subsets.keys():
            if subset_name in cross_modal_results:
                metrics = cross_modal_results[subset_name]
                results_table.append((
                    subset_name,
                    metrics['accuracy'],
                    metrics['kappa'],
                    metrics['macro_f1']
                ))
        
        # Sort by Kappa (descending) to show best/worst combinations
        results_table.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"{'Modality Subset':<30} {'Accuracy':<12} {'Kappa':<12} {'Macro F1':<12}")
        logger.info("-" * 66)
        for subset_name, acc, kappa, f1 in results_table:
            logger.info(f"{subset_name:<30} {acc:<12.4f} {kappa:<12.4f} {f1:<12.4f}")
        
        # Analysis of robustness
        logger.info("\nKey Insights:")
        best_subset = results_table[0]
        worst_subset = results_table[-1]
        logger.info(f"  Best Configuration: {best_subset[0]} (Kappa={best_subset[2]:.4f})")
        logger.info(f"  Most Challenging: {worst_subset[0]} (Kappa={worst_subset[2]:.4f})")
        logger.info(f"  Robustness: Model maintains {worst_subset[2]/best_subset[2]*100:.1f}% of best performance with limited modalities")

    logger.info("=" * 70)


# =============================================================================
# Main Pipeline
# =============================================================================

def main(args):
    """Execute complete training pipeline."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("=" * 70)
    logger.info("WAV2SLEEP: Sleep Stage Classification Pipeline")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Paper: Carter & Tarassenko, arXiv:2411.04644")
    logger.info("Implementation: PyHealth")
    logger.info("")

    # =========================================================================
    # 1. Load Dataset
    # =========================================================================
    logger.info("[Pipeline Step 1/4] Loading dataset...")
    dataset = load_shhs_or_synthetic(
        shhs_root=args.shhs_root,
        max_recordings=args.max_recordings,
    )
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Input modalities: {list(dataset.input_processors.keys())}")
    logger.info(f"Sleep stages: {NUM_CLASSES} classes")

    # =========================================================================
    # 2. Train Model
    # =========================================================================
    logger.info("\n[Pipeline Step 2/4] Training model with paper-specific schedule...")
    history, trainer, model = train_wav2sleep_model(
        dataset=dataset,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_fraction=args.warmup_fraction,
        max_lr=args.max_lr,
        decay_rate=args.decay_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        output_dir=args.output_dir,
    )

    # =========================================================================
    # 3. Evaluate & Analyze
    # =========================================================================
    logger.info("\n[Pipeline Step 3/4] Evaluating on validation set...")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Create validation loader for evaluation
    _, val_loader = create_subset_loaders(
        dataset=dataset,
        train_fraction=0.8,
        val_fraction=1.0,
        batch_size=args.batch_size,
    )

    print_results_summary(history, model, val_loader, device=device)

    # =========================================================================
    # 4. Save Model & Results
    # =========================================================================
    logger.info("\n[Pipeline Step 4/4] Saving artifacts...")
    output_path = Path(args.output_dir) / "wav2sleep_training"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save final model
    final_model_path = output_path / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✓ Model saved: {final_model_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wav2Sleep training pipeline for sleep stage classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--shhs-root",
        type=str,
        default=None,
        help="Path to SHHS polysomnography directory. If not provided, synthetic data is used.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Maximum number of SHHS recordings to load (None = load all).",
    )

    # Subset options (for large datasets)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training set to use (0.0 to 1.0). Useful for large datasets.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=1.0,
        help="Fraction of validation set to use (0.0 to 1.0). Useful for large datasets.",
    )

    # Training hyperparameters (from wav2sleep paper)
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Fraction of training steps used for linear warmup (0.0 to 1.0).",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-3,
        help="Maximum learning rate (at end of warmup phase).",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.9999,
        help="Exponential decay rate per step (lambda^step). Higher = slower decay.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="L2 regularization coefficient (AdamW weight decay).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs with no improvement). Paper uses 5.",
    )

    # System options
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for training ('cuda' or 'cpu'). If None, auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for saving checkpoints and logs.",
    )

    args = parser.parse_args()
    main(args)
