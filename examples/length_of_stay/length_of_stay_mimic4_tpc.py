"""
TPC Model Example with Ablation Study for MIMIC-IV Remaining Length-of-Stay Prediction

This script demonstrates the Temporal Pointwise Convolution (TPC) model for ICU length-of-stay
prediction with comprehensive ablation studies including:

1. Baseline TPC model training
2. Hyperparameter variations (layers, loss functions, dropout)
3. Monte Carlo Dropout uncertainty estimation (novel ablation)
4. Performance comparison across configurations

Paper: Rocheteau et al., "Temporal Pointwise Convolutional Networks for Length of Stay 
       Prediction in the ICU", CHIL 2021

NOTE: Set dev=True for testing with small subset. For full dataset, set dev=False.
"""

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader
from pyhealth.tasks import RemainingLOSMIMIC4
from pyhealth.models import TPC
from pyhealth.trainer import Trainer
import torch
import numpy as np
from pathlib import Path
import json

# ============================================================================
# Configuration
# ============================================================================

# Update this path once you download from Google Drive
MIMIC_ROOT = r"C:\cs598\mimic-iv"  # Path to your downloaded MIMIC-IV data
CACHE_PATH = r"C:\cs598\.cache_dir"
OUTPUT_DIR = Path("./tpc_ablation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Training configuration - adjust based on your hardware
EPOCHS = 10  # Increase for better results (20-50 for full training)
BATCH_SIZE = 32  # Adjust based on your GPU memory
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Ablation Study Configurations
# ============================================================================

ABLATION_CONFIGS = {
    "baseline": {
        "name": "TPC Baseline (3 layers, MSLE)",
        "params": {
            "n_layers": 3,
            "temp_kernels": [8, 8, 8],
            "point_sizes": [14, 14, 14],
            "use_msle": True,
            "main_dropout_rate": 0.3,
        }
    },
    "shallow": {
        "name": "TPC Shallow (2 layers, MSLE)",
        "params": {
            "n_layers": 2,
            "temp_kernels": [8, 8],
            "point_sizes": [14, 14],
            "use_msle": True,
            "main_dropout_rate": 0.3,
        }
    },
    "mse_loss": {
        "name": "TPC with MSE Loss (3 layers)",
        "params": {
            "n_layers": 3,
            "temp_kernels": [8, 8, 8],
            "point_sizes": [14, 14, 14],
            "use_msle": False,  # Use MSE instead of MSLE
            "main_dropout_rate": 0.3,
        }
    },
    "low_dropout": {
        "name": "TPC Low Dropout (3 layers, 0.1 dropout)",
        "params": {
            "n_layers": 3,
            "temp_kernels": [8, 8, 8],
            "point_sizes": [14, 14, 14],
            "use_msle": True,
            "main_dropout_rate": 0.1,  # Reduced dropout
        }
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_data(dev=True):
    """Load MIMIC-IV dataset and apply RemainingLOSMIMIC4 task.
    
    Args:
        dev: If True, uses development subset for faster testing.
        
    Returns:
        SampleDataset with timeseries, static, conditions, and los features.
    """
    print("=" * 80)
    print("LOADING MIMIC-IV DATA")
    print("=" * 80)
    
    # Use minimal tables to reduce memory usage for large dataset
    # chartevents is essential for vital signs, diagnoses for conditions
    mimic4 = MIMIC4EHRDataset(
        root=MIMIC_ROOT,
        tables=["diagnoses_icd", "chartevents"],  # Reduced tables for memory efficiency
        dev=dev,
        cache_dir=CACHE_PATH
    )
    
    print(f"\nDataset statistics:")
    mimic4.stats()
    
    # Apply remaining LoS task
    print(f"\nApplying RemainingLOSMIMIC4 task...")
    sample_dataset = mimic4.set_task(RemainingLOSMIMIC4())
    
    print(f"Total samples: {len(sample_dataset)}")
    
    # Inspect first sample
    first_sample = sample_dataset[0]
    print(f"\nSample structure:")
    for key, value in first_sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    return sample_dataset


def train_model(dataset, config_name, model_params, epochs=5):
    """Train TPC model with given configuration.
    
    Args:
        dataset: SampleDataset from RemainingLOSMIMIC4 task
        config_name: Name of configuration for logging
        model_params: Dictionary of TPC model parameters
        epochs: Number of training epochs
        
    Returns:
        Trained model and training metrics
    """
    print("\n" + "=" * 80)
    print(f"TRAINING: {ABLATION_CONFIGS[config_name]['name']}")
    print("=" * 80)
    
    # Initialize model
    model = TPC(dataset=dataset, **model_params)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Split dataset: 70% train, 15% val, 15% test
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.15, 0.15], seed=42
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=DEVICE,
        metrics=["mae", "mse"],  # Mean Absolute Error and MSE
    )
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    trainer.train(
        train_dataloader=get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        val_dataloader=get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        epochs=epochs,
        monitor="mae",  # Monitor MAE for early stopping
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = trainer.evaluate(
        get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    )
    
    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, test_metrics


def run_mc_dropout_ablation(model, dataset, mc_samples=30):
    """Run Monte Carlo Dropout ablation study (novel contribution).
    
    This demonstrates predictive uncertainty estimation using MC Dropout,
    which is not in the original TPC paper.
    
    Args:
        model: Trained TPC model
        dataset: Test dataset
        mc_samples: Number of MC dropout samples
        
    Returns:
        Dictionary with uncertainty statistics
    """
    print("\n" + "=" * 80)
    print("ABLATION: Monte Carlo Dropout Uncertainty Estimation")
    print("=" * 80)
    print("\nThis is a NOVEL extension beyond the original TPC paper.")
    print(f"Running {mc_samples} stochastic forward passes with dropout active...")
    
    # Get a batch of test samples
    test_loader = get_dataloader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(test_loader))
    
    # Run MC Dropout
    with torch.no_grad():
        uncertainty_output = model.predict_with_uncertainty(
            mc_samples=mc_samples,
            **batch
        )
    
    # Compute statistics
    mean_predictions = uncertainty_output["mean"]  # (B, T)
    std_predictions = uncertainty_output["std"]    # (B, T)
    
    print(f"\nUncertainty Statistics:")
    print(f"  Mean prediction std: {std_predictions.mean().item():.4f} hours")
    print(f"  Max prediction std: {std_predictions.max().item():.4f} hours")
    print(f"  Min prediction std: {std_predictions.min().item():.4f} hours")
    
    # Compute coefficient of variation (std/mean) as relative uncertainty
    cv = std_predictions / (mean_predictions + 1e-8)
    print(f"  Mean coefficient of variation: {cv.mean().item():.4f}")
    
    results = {
        "mean_std": std_predictions.mean().item(),
        "max_std": std_predictions.max().item(),
        "mean_cv": cv.mean().item(),
        "mc_samples": mc_samples,
    }
    
    print("\nInterpretation:")
    print("  - Higher std = higher prediction uncertainty")
    print("  - Useful for identifying high-risk patients needing attention")
    print("  - Can be used for confidence intervals in clinical decision support")
    
    return results


def compare_configurations(results):
    """Compare performance across all configurations.
    
    Args:
        results: Dictionary mapping config_name to metrics
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS COMPARISON")
    print("=" * 80)
    
    print("\n{:<30} {:<15} {:<15}".format("Configuration", "Test MAE", "Test MSE"))
    print("-" * 60)
    
    for config_name, metrics in results.items():
        config_display = ABLATION_CONFIGS[config_name]["name"]
        mae = metrics.get("mae", float('nan'))
        mse = metrics.get("mse", float('nan'))
        print("{:<30} {:<15.4f} {:<15.4f}".format(config_display, mae, mse))
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1].get("mae", float('inf')))
    print(f"\n✓ Best configuration: {ABLATION_CONFIGS[best_config[0]]['name']}")
    print(f"  MAE: {best_config[1]['mae']:.4f} hours")
    
    # Save results to JSON
    output_file = OUTPUT_DIR / "ablation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()}
            for name, metrics in results.items()
        }, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run complete ablation study."""
    
    print("\n" + "=" * 80)
    print("TPC MODEL ABLATION STUDY FOR MIMIC-IV REMAINING LOS PREDICTION")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data - using dev mode with reduced tables to avoid memory issues
    dataset = load_data(dev=True)
    
    # Train all configurations
    all_results = {}
    trained_models = {}
    
    for config_name in ABLATION_CONFIGS.keys():
        model, metrics = train_model(
            dataset,
            config_name,
            ABLATION_CONFIGS[config_name]["params"],
            epochs=EPOCHS
        )
        all_results[config_name] = metrics
        trained_models[config_name] = model
    
    # Compare results
    compare_configurations(all_results)
    
    # Run MC Dropout ablation on best model
    best_config_name = min(all_results.items(), key=lambda x: x[1].get("mae", float('inf')))[0]
    best_model = trained_models[best_config_name]
    
    # Get test dataset
    _, _, test_dataset = dataset.split(ratios=[0.7, 0.15, 0.15], seed=42)
    
    mc_results = run_mc_dropout_ablation(best_model, test_dataset, mc_samples=30)
    
    # Save MC Dropout results
    mc_output_file = OUTPUT_DIR / "mc_dropout_results.json"
    with open(mc_output_file, 'w') as f:
        json.dump(mc_results, f, indent=2)
    print(f"\n✓ MC Dropout results saved to: {mc_output_file}")
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Compared 4 TPC configurations with varying architectures and losses")
    print("2. Demonstrated Monte Carlo Dropout for uncertainty quantification")
    print("3. Identified best hyperparameter configuration")
    print(f"4. Results saved to {OUTPUT_DIR}/")
    

def inspect_only():
    """Quick inspection function for testing data loading."""
    print("Running quick data inspection...")
    dataset = load_data(dev=True)
    print("\n✓ Data loading successful!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  First sample keys: {list(dataset[0].keys())}")


if __name__ == "__main__":
    # For full ablation study:
    main()
    
    # For quick data inspection only, uncomment:
    # inspect_only()
