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

Wav2Sleep paper link:
    https://doi.org/10.48550/arXiv.2411.04644

Wav2Sleep paper citation:
    Carter, J. F.; and Tarassenko, L. 2024. wav2sleep: A unified multi-modal approach
    to sleep stage classification from physiological signals. arXiv preprint arXiv:2411.04644.

Authors:
    Austin Jarrett (ajj7@illinois.edu)
    Justin Cheok (jcheok2@illinois.edu)
    Jimmy Scray (escray2@illinois.edu)

CFS Dataset:
    The Cleveland Family Study (CFS) is a longitudinal cohort study with
    polysomnography recordings. Each recording contains multiple physiological
    signals: EEG, left/right EOG, chin EMG, ECG, and optionally PPG.
    Note: PPG (plethysmograph) is available in ~43% of CFS files.

Requirements:
    - CFS data: Set --cfs-root to use real data (must be requested from NSRR)
    - Or: Script will use fully synthetic data for demonstration

**Usage Examples:**

    # Quick test with synthetic data (2 min)
    python examples/cfs_sleep_staging_wav2sleep.py --epochs 1

    # Full training with 2% of CFS data (~37 patients, ~20 min on CPU)
    # This is a working configuration that produces proper loss curves:
    python examples/cfs_sleep_staging_wav2sleep.py \\
        --cfs-root /path/to/cfs \\
        --subset-fraction 0.02 \\
        --train-fraction 0.8 \\
        --val-fraction 0.2 \\
        --epochs 50 \\
        --batch-size 8 \\
        --max-lr 1e-3 \\
        --decay-rate 0.9999 \\
        --weight-decay 1e-2 \\
        --patience 20 \\
        --device cpu

    # Training with PPG as additional modality (missing PPG zero-padded)
    python examples/cfs_sleep_staging_wav2sleep.py \\
        --cfs-root /path/to/cfs \\
        --subset-fraction 0.02 \\
        --include-ppg \\
        --ppg-samples 256 \\
        --epochs 50 \\
        --batch-size 8 \\
        --device cpu

    # Production training with full dataset
    python examples/cfs_sleep_staging_wav2sleep.py \\
        --cfs-root /path/to/cfs \\
        --epochs 80 \\
        --batch-size 32 \\
        --patient-fraction 1.0 \\
        --device cuda
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Wav2Sleep
from pyhealth.training.wav2sleep_trainer import Wav2SleepTrainer, create_subset_loaders
from pyhealth.metrics.wav2sleep import compute_confusion_matrix, cohens_kappa

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

# Signal sampling parameters handled by SleepStagingCFS task:
# - EEG: 256 samples (~8 Hz × 30 s, downsampled from 200 Hz)
# - EOG-L/R: 256 samples (~8 Hz × 30 s)
# - EMG: 128 samples (~4 Hz × 30 s)
# - ECG: 256 samples (~8 Hz × 30 s)
# These values should match the task constructor if customized.
EEG_SAMPLES_PER_EPOCH = 256    # For reference, matches task default
EOG_SAMPLES_PER_EPOCH = 256    # For reference, matches task default
EMG_SAMPLES_PER_EPOCH = 128    # For reference, matches task default
ECG_SAMPLES_PER_EPOCH = 256    # For reference, matches task default
PPG_SAMPLES_PER_EPOCH = 256    # For reference, PPG sample count (optional)


# =============================================================================
# Dataset Construction
# =============================================================================

def build_synthetic_cfs_dataset(n_samples: int = 100, include_ppg: bool = False) -> object:
    """Create a synthetic CFS dataset for testing/demonstration.
    
    Generates random multi-modal polysomnography signals (EEG, EOG, EMG, ECG, optionally PPG)
    with random sleep stage labels. Useful for testing the training pipeline
    when real CFS data is unavailable.
    
    Args:
        n_samples: Number of synthetic samples to generate. Default: 100.
        include_ppg: Whether to include PPG modality. Default: False.
        
    Returns:
        PyHealth SampleDataset with synthetic CFS data in wav2sleep format.
    """
    logger.info(f"Generating {n_samples} synthetic samples (include_ppg={include_ppg})...")
    
    synthetic_samples = []
    for i in range(n_samples):
        sample = {
            "patient_id": f"synthetic_{i // 10}",  # 10 samples per synthetic patient
            "study_id": f"synthetic_{i // 10}_study",
            "patient_age": np.random.randint(30, 80),
            "patient_sex": np.random.randint(0, 2),
            "eeg": np.random.randn(EEG_SAMPLES_PER_EPOCH).astype(np.float32),
            "eog_left": np.random.randn(EOG_SAMPLES_PER_EPOCH).astype(np.float32),
            "eog_right": np.random.randn(EOG_SAMPLES_PER_EPOCH).astype(np.float32),
            "emg": np.random.randn(EMG_SAMPLES_PER_EPOCH).astype(np.float32),
            "ecg": np.random.randn(ECG_SAMPLES_PER_EPOCH).astype(np.float32),
            "label": np.random.randint(0, 5),  # 5 sleep stages: 0-4
        }
        if include_ppg:
            sample["ppg"] = np.random.randn(PPG_SAMPLES_PER_EPOCH).astype(np.float32)
        synthetic_samples.append(sample)
    
    input_schema = {
        "eeg": "tensor",
        "eog_left": "tensor",
        "eog_right": "tensor",
        "emg": "tensor",
        "ecg": "tensor",
    }
    if include_ppg:
        input_schema["ppg"] = "tensor"
    
    dataset = create_sample_dataset(
        samples=synthetic_samples,
        input_schema=input_schema,
        output_schema={"label": "multiclass"},
        dataset_name="synthetic_cfs_wav2sleep",
    )
    
    logger.info(f"✓ Generated {len(synthetic_samples)} synthetic samples")
    return dataset


def load_cfs_or_synthetic(
    cfs_root: Optional[str] = None,
    max_recordings: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    include_ppg: bool = False,
    ppg_samples: int = 256,
) -> object:
    """Load CFS dataset if available, otherwise create synthetic fallback.
    
    The NSRR CFS dataset must have:
    - polysomnography/edfs/*.edf (signal files)
    - polysomnography/annotations-events-nsrr/*.xml (sleep stage annotations)
    - polysomnography/annotations-events-profusion/*.xml (alternative annotations)
    
    This function generates polysomnography-metadata-pyhealth.csv from the NSRR
    dataset structure and passes it to PyHealth's CFSDataset.
    
    Args:
        cfs_root: Path to CFS dataset root directory.
            If None or invalid, synthetic data is used.
        max_recordings: Maximum number of recordings to load from CFS.
            If None, all available recordings are loaded.
        subset_fraction: Fraction of dataset to load (0.0 to 1.0).
            If specified, limits the dataset size to this fraction.
            Applied at the dataset level before loading, more efficient than 
            post-hoc subsampling.
        include_ppg: Whether to include PPG modality (only in ~43% of CFS files,
            missing values are zero-padded). Default: False.
        ppg_samples: Target sample count for PPG after resampling. Default: 256.
    Returns:
        PyHealth SampleDataset in wav2sleep format.
    """
    if cfs_root and os.path.isdir(cfs_root):
        try:
            from pyhealth.datasets import CFSDataset
            from pyhealth.tasks import SleepStagingCFS
            
            logger.info(f"Attempting to load CFS dataset from: {cfs_root}")
            
            # Load CFS dataset with optional subset
            cfs_dataset = CFSDataset(root=cfs_root, dev=False)
            
            # Apply subset if requested (limits patient count efficiently)
            if subset_fraction is not None and subset_fraction < 1.0:
                patients_to_load = cfs_dataset.unique_patient_ids
                n_patients = max(1, int(len(patients_to_load) * subset_fraction))
                sampled_patients = np.random.choice(
                    patients_to_load, size=n_patients, replace=False
                )
                logger.info(f"Loading {n_patients} / {len(patients_to_load)} patients ({subset_fraction*100:.1f}%)")
                
                # Process only sampled patients
                samples = []
                task = SleepStagingCFS(preload=False, include_ppg=include_ppg, ppg_samples=ppg_samples)
                for pid in sampled_patients:
                    patient = cfs_dataset.get_patient(pid)
                    samples.extend(task(patient))
            else:
                # Load all records (via task)
                logger.info(f"Loading all available records")
                task = SleepStagingCFS(preload=False, include_ppg=include_ppg, ppg_samples=ppg_samples)
                samples = []
                for pid in cfs_dataset.unique_patient_ids:
                    patient = cfs_dataset.get_patient(pid)
                    samples.extend(task(patient))
                    if max_recordings and len(samples) >= max_recordings:
                        samples = samples[:max_recordings]
                        break
            
            if not samples:
                logger.warning(f"CFS loader returned 0 samples. Falling back to synthetic data.")
                return build_synthetic_cfs_dataset()
            
            logger.info(f"✓ Loaded {len(samples)} CFS samples in wav2sleep format")
            
            # Task already outputs wav2sleep format
            # Build input schema based on whether PPG is included
            input_schema = {
                "eeg": "tensor",
                "eog_left": "tensor",
                "eog_right": "tensor",
                "emg": "tensor",
                "ecg": "tensor",
            }
            if include_ppg:
                input_schema["ppg"] = "tensor"
                logger.info(f"  (PPG included in {sum(1 for s in samples if 'ppg' in s)}/{len(samples)} samples)")
            
            dataset = create_sample_dataset(
                samples=samples,
                input_schema=input_schema,
                output_schema={"label": "multiclass"},
                dataset_name="cfs_wav2sleep",
            )
            return dataset
        except Exception as e:
            logger.error(f"Error loading CFS dataset: {type(e).__name__}: {e}", exc_info=True)
            logger.info("Falling back to synthetic data.")
            return build_synthetic_cfs_dataset()
    else:
        if cfs_root:
            logger.warning(f"CFS_ROOT directory not found: {cfs_root}")
        logger.info("Using synthetic data for training.")
        return build_synthetic_cfs_dataset(include_ppg=include_ppg)


# =============================================================================
# Model Training
# =============================================================================

def train_wav2sleep_model(
    dataset,
    train_fraction: float = 0.8,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    epochs: int = 80,
    warmup_fraction: float = 0.1,
    max_lr: float = 1e-3,
    decay_rate: float = 0.9999,
    weight_decay: float = 1e-2,
    patience: int = 5,
    device: Optional[str] = None,
    output_dir: str = "output",
) -> tuple:
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
            Default: 0.8 (use 80% for training).
        val_fraction: Fraction of validation set to use (0.0 to 1.0).
            Default: 0.2 (use 20% for validation).
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
    # Create data loaders
    train_loader, val_loader = create_subset_loaders(
        dataset=dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        batch_size=batch_size,
        seed=42,
    )
    # Create model
    model = Wav2Sleep(
        dataset=dataset,
        feature_dim=128,
        n_transformer_layers=2,
        n_attention_heads=8,
        transformer_ff_dim=512,
        dropout=0.1,
    )
    # Create trainer
    trainer = Wav2SleepTrainer(
        model=model,
        device=device,
        enable_logging=True,
        output_path=output_dir,
        exp_name="wav2sleep_cfs_training",
    )
    logger.info(f"Training: {sum(p.numel() for p in model.parameters()):,} parameters, {epochs} epochs, device={device}")
    # Train
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


def print_results_summary(history, model, val_loader, device="cpu"):
    """Print concise training results summary."""
    logger.info("\nTraining Results:")
    if history.get('train_loss'):
        logger.info(f"  Initial loss: {history['train_loss'][0]:.6f}, Final loss: {history['train_loss'][-1]:.6f}")
    if history.get('val_loss'):
        logger.info(f"  Best val loss: {min(history['val_loss']):.6f}")
    if history.get('val_accuracy'):
        logger.info(f"  Best val accuracy: {max(history['val_accuracy']):.4f}")
    
    # Compute Cohen's Kappa on validation set
    if val_loader:
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                output = model(**batch)
                preds = output["y_prob"].argmax(dim=-1)
                labels = output["y_true"]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute confusion matrix and Cohen's Kappa
        cmat = compute_confusion_matrix(all_preds, all_labels, num_classes=NUM_CLASSES)
        kappa = cohens_kappa(cmat)
        
        logger.info(f"  Cohen's Kappa: {kappa:.4f}")


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

    logger.info("=" * 70)
    logger.info("WAV2SLEEP: Sleep Stage Classification Pipeline")
    logger.info("=" * 70)
    logger.info("[Step 1/4] Loading dataset...")

    dataset = load_cfs_or_synthetic(
        cfs_root=args.cfs_root,
        max_recordings=args.max_recordings,
        subset_fraction=args.subset_fraction,
        include_ppg=args.include_ppg,
        ppg_samples=args.ppg_samples,
    )
    logger.info(f"  ✓ Loaded {len(dataset)} samples")

    logger.info("\n[Step 2/4] Training model with paper-specific schedule...")
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

    logger.info("\n[Step 3/4] Evaluating on validation set...")

    # Create validation loader for evaluation
    _, val_loader = create_subset_loaders(
        dataset=dataset,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
    )

    # Print training results summary
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print_results_summary(history, model, val_loader, device=device)

    logger.info("\n[Step 4/4] Saving artifacts...")
    output_path = Path(args.output_dir) / "wav2sleep_training"
    output_path.mkdir(parents=True, exist_ok=True)
    # Save final model with timestamp to preserve multiple training runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = output_path / f"wav2sleep_model_{timestamp}.pt"
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
        "--subset-fraction",
        type=float,
        default=None,
        help="Fraction of CFS dataset to load (0.0 to 1.0). Applied at dataset level for efficiency.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of loaded data to use for training (0.0 to 1.0).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of loaded data to use for validation (0.0 to 1.0).",
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

    # PPG options (optional modality)
    parser.add_argument(
        "--include-ppg",
        action="store_true",
        help="Include PPG (plethysmograph) as additional modality. Note: PPG is available in ~43% of CFS files; missing values are zero-padded.",
    )
    parser.add_argument(
        "--ppg-samples",
        type=int,
        default=256,
        help="Target sample count for PPG after resampling (default: 256).",
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
    
    # Log the configuration
    logger.info(f"Configuration: include_ppg={args.include_ppg}, ppg_samples={args.ppg_samples}")
    
    main(args)
