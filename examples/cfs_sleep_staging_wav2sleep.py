"""Complete pipeline for wav2sleep training on CFS sleep staging task.

This example demonstrates the full training pipeline for the wav2sleep model
on the Cleveland Family Study (CFS) dataset following PyHealth implementation
requirements:

1. **Data Loading**: Load CFS dataset with automatic sleep staging task
2. **Dataset Configuration**: Multi-modal setup with EEG, EOG, EMG, ECG signals
3. **Model Initialization**: Wav2Sleep with configurable architecture for CFS modalities
4. **Training Loop**: Using the wav2sleep-specific trainer with:
   - AdamW optimizer
   - Linear warmup to max learning rate
   - Exponential decay schedule
   - Early stopping when validation loss plateaus (5 epochs patience)
5. **Evaluation**: Cross-modal evaluation showing model robustness
6. **Reproducibility**: Seed control, configurable subset sizes

**Paper Reference:**
Jonathan F. Carter & Lionel Tarassenko. "wav2sleep: A Unified Multi-Modal
Approach to Sleep Stage Classification from Physiological Signals."
arXiv:2411.04644, 2024. https://arxiv.org/abs/2411.04644

**CFS Dataset:**
The Cleveland Family Study (CFS) is a longitudinal cohort study with
polysomnography recordings. Each recording contains multiple physiological
signals: EEG, left/right EOG, chin EMG, and ECG.

**Requirements:**
- CFS data: Available at `pyhealth/datasets/cfs/` or can be downloaded
- Or: Script will use fully synthetic data for demonstration

**Usage:**
    python examples/cfs_sleep_staging_wav2sleep.py \\
        --cfs-root /path/to/cfs/data \\
        --max-recordings 100 \\
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
from pyhealth.models import Wav2Sleep
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

# Sleep stage label definitions (AASM 5-class)
SLEEP_STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
NUM_CLASSES = len(SLEEP_STAGES)

# Signal sampling parameters for CFS
# CFS typically samples physiological signals at high frequencies (200+ Hz)
# We downsample to reasonable lengths for computational efficiency
EEG_SAMPLES_PER_EPOCH = 256    # ~8 Hz × 30 s (downsampled from 200 Hz)
EOG_SAMPLES_PER_EPOCH = 256    # ~8 Hz × 30 s
EMG_SAMPLES_PER_EPOCH = 128    # ~4 Hz × 30 s
ECG_SAMPLES_PER_EPOCH = 256    # ~8 Hz × 30 s

# CFS channel mapping (standard ordering in CFS dataset)
CFS_CHANNEL_NAMES = ("EEG", "EOG-L", "EOG-R", "EMG-Chin", "ECG")
CFS_CHANNEL_MAPPING = {
    "eeg": 0,           # EEG channel
    "eog_left": 1,      # Left EOG channel
    "eog_right": 2,     # Right EOG channel
    "emg": 3,           # Chin EMG channel
    "ecg": 4,           # ECG channel
}


# =============================================================================
# Dataset Construction
# =============================================================================

def build_synthetic_cfs_dataset(
    n_patients: int = 10,
    epochs_per_patient: int = 100,
    dataset_name: str = "synthetic_cfs_wav2sleep",
    seed: int = 42,
) -> object:
    """Create synthetic CFS-like dataset for testing.

    This function generates a fully synthetic dataset that mirrors the CFS
    data structure, useful for development, testing, and when real data
    is unavailable.

    Each sample corresponds to one 30-second polysomnography epoch with:
    - EEG    : electroencephalogram signal (1 channel)
    - EOG-L  : left electrooculogram (1 channel)
    - EOG-R  : right electrooculogram (1 channel)
    - EMG    : chin electromyogram (1 channel)
    - ECG    : electrocardiogram signal (1 channel)
    - Label  : sleep stage (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

    Args:
        n_patients: Number of synthetic patients.
        epochs_per_patient: Number of 30-second epochs per patient.
        dataset_name: Name for the created dataset.
        seed: Random seed for reproducibility.

    Returns:
        PyHealth SampleDataset ready for training.
    """
    logger.info(f"Building synthetic CFS dataset ({n_patients} patients, "
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
                # EEG signal: 1 channel, EEG_SAMPLES_PER_EPOCH samples
                "eeg": rng.randn(1, EEG_SAMPLES_PER_EPOCH).astype(np.float32),
                # Left EOG signal
                "eog_left": rng.randn(1, EOG_SAMPLES_PER_EPOCH).astype(np.float32),
                # Right EOG signal
                "eog_right": rng.randn(1, EOG_SAMPLES_PER_EPOCH).astype(np.float32),
                # Chin EMG signal
                "emg": rng.randn(1, EMG_SAMPLES_PER_EPOCH).astype(np.float32),
                # ECG signal
                "ecg": rng.randn(1, ECG_SAMPLES_PER_EPOCH).astype(np.float32),
                "label": label,
            })

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "eeg": "tensor",       # Electroencephalogram
            "eog_left": "tensor",  # Left electrooculogram
            "eog_right": "tensor", # Right electrooculogram
            "emg": "tensor",       # Chin electromyogram
            "ecg": "tensor",       # Electrocardiogram
        },
        output_schema={"label": "multiclass"},
        dataset_name=dataset_name,
    )
    logger.info(f"✓ Created synthetic dataset with {len(dataset)} samples")
    return dataset


def load_cfs_or_synthetic(
    cfs_root: Optional[str] = None,
    max_recordings: Optional[int] = None,
) -> object:
    """Load CFS dataset if available, otherwise create synthetic fallback.

    Args:
        cfs_root: Path to CFS dataset root directory.
            If None or invalid, synthetic data is used.
        max_recordings: Maximum number of recordings to load from CFS.
            If None, all available recordings are loaded.

    Returns:
        PyHealth SampleDataset.
    """
    if cfs_root and os.path.isdir(cfs_root):
        logger.info(f"Loading CFS dataset from {cfs_root}")
        try:
            from pyhealth.datasets import CFSDataset
            from pyhealth.tasks import SleepStagingCFS
            from pyhealth.models import load_cfs_samples
            
            # Load CFS dataset
            cfs_dataset = CFSDataset(root=cfs_root)
            cfs_dataset.set_task(SleepStagingCFS())
            
            # Convert to Wav2Sleep format with subset of modalities
            # Using all modalities: EEG, EOG-L, EOG-R, EMG, ECG
            samples = load_cfs_samples(
                cfs_dataset=cfs_dataset,
                channel_mapping={
                    "eeg": 0,           # EEG
                    "eog_left": 1,      # EOG-L
                    "eog_right": 2,     # EOG-R
                    "emg": 3,           # EMG-Chin
                    "ecg": 4,           # ECG
                },
                max_recordings=max_recordings,
            )

            if not samples:
                logger.warning("No samples loaded from CFS, falling back to synthetic data")
                return build_synthetic_cfs_dataset()

            dataset = create_sample_dataset(
                samples=samples,
                input_schema={
                    "eeg": "tensor",
                    "eog_left": "tensor",
                    "eog_right": "tensor",
                    "emg": "tensor",
                    "ecg": "tensor",
                },
                output_schema={"label": "multiclass"},
                dataset_name="cfs_wav2sleep",
            )
            logger.info(f"✓ Loaded {len(dataset)} samples from CFS")
            return dataset

        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load CFS dataset ({type(e).__name__}), falling back to synthetic data")
            return build_synthetic_cfs_dataset()
    else:
        logger.info("CFS_ROOT not set or invalid — using synthetic data")
        return build_synthetic_cfs_dataset()


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
    """Train wav2sleep model on CFS dataset with specified hyperparameters.

    This function implements the complete training pipeline described in the
    wav2sleep paper with:
    - AdamW optimizer (no weight decay on bias/norm parameters)
    - Linear warmup phase: 0 → max_lr over warmup_fraction of training
    - Exponential decay phase: max_lr → ~0 with decay_rate^step
    - Early stopping: patience=5 (as per paper) means training stops if
      validation loss doesn't improve for 5 consecutive epochs
    - Optional subset training for memory-constrained environments

    Args:
        dataset: PyHealth SampleDataset with CFS data.
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
        Tuple of (history dict, trainer, model):
        - history: Training history with 'train_loss', 'val_loss', 'val_accuracy'
        - trainer: Wav2SleepTrainer instance
        - model: Trained Wav2Sleep model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("")
    logger.info("=" * 70)
    logger.info("Wav2Sleep Model Training (CFS Dataset)")
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
        exp_name="wav2sleep_cfs_training",
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
        subset_keys: List of modality names to keep (e.g., ['eeg']).
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
            "All (EEG + EOG + EMG + ECG)": [
                "eeg", "eog_left", "eog_right", "emg", "ecg"
            ],
            "EEG only": ["eeg"],
            "ECG only": ["ecg"],
            "EEG + ECG": ["eeg", "ecg"],
            "EEG + EOG + ECG": ["eeg", "eog_left", "eog_right", "ecg"],
            "EOG + EMG": ["eog_left", "eog_right", "emg"],
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
        
        logger.info(f"{'Modality Subset':<40} {'Accuracy':<12} {'Kappa':<12} {'Macro F1':<12}")
        logger.info("-" * 76)
        for subset_name, acc, kappa, f1 in results_table:
            logger.info(f"{subset_name:<40} {acc:<12.4f} {kappa:<12.4f} {f1:<12.4f}")
        
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
    logger.info("WAV2SLEEP: Sleep Stage Classification on CFS Dataset")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Paper: Carter & Tarassenko, arXiv:2411.04644")
    logger.info("Dataset: Cleveland Family Study (CFS)")
    logger.info("Implementation: PyHealth")
    logger.info("")

    # =========================================================================
    # 1. Load Dataset
    # =========================================================================
    logger.info("[Pipeline Step 1/4] Loading dataset...")
    dataset = load_cfs_or_synthetic(
        cfs_root=args.cfs_root,
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
    output_path = Path(args.output_dir) / "wav2sleep_cfs_training"
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
        description="Wav2Sleep training pipeline for CFS sleep stage classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--cfs-root",
        type=str,
        default=None,
        help="Path to CFS dataset root. If not provided, synthetic data is used.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Maximum number of CFS recordings to load (None = load all).",
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
