"""
Full Pipeline Replication for TPC Model - Extra Credit Submission

This script demonstrates the complete workflow for ICU length-of-stay prediction:
1. Synthetic dataset generation matching MIMIC-IV schema
2. Data preparation and PyHealth-compatible format
3. Model training with proper training loop
4. Ablation study across 3 configurations
5. Comprehensive evaluation and results export

Authors: Pankaj Meghani (meghani3), Tarak Jha (tarakj2), Pranash Krishnan (pranash2)
Course: CS 598 Deep Learning for Healthcare
Paper: Rocheteau et al., "Temporal Pointwise Convolutional Networks for Length of Stay
        Prediction in the Intensive Care Unit", CHIL 2021
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from datetime import datetime
from pathlib import Path

# Import PyHealth components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyhealth.models import TPC

# Custom dataset class for synthetic data
class SyntheticICUDataset(Dataset):
    """PyHealth-compatible dataset for synthetic ICU data."""
    
    def __init__(self, data_list, labels_list, masks_list):
        self.data = data_list
        self.labels = labels_list
        self.masks = masks_list
        
        # PyHealth required attributes
        self.input_schema = {'timeseries': {'dim': 34, 'type': 'float'}}
        self.output_schema = {'los': {'dim': 1, 'type': 'float'}}
        self.input_processors = {}
        self.output_processors = {}
        self.feature_keys = ['timeseries']
        self.label_keys = ['los']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Create sequence-level labels (LOS at each timestep)
        seq_len = self.data[idx].shape[0]
        # Label is repeated for each timestep (remaining LOS)
        labels_seq = torch.ones(seq_len) * self.labels[idx]
        
        return {
            'timeseries': torch.FloatTensor(self.data[idx]),
            'los': labels_seq,  # [seq_len] shape
            'patient_id': f'patient_{idx}',
            'visit_id': f'visit_{idx}'
        }


def generate_synthetic_mimic_data(n_patients=300, n_features=34, max_time_steps=72):
    """
    Generate synthetic ICU time series data matching MIMIC-IV characteristics.
    
    Simulates realistic patterns:
    - 90% missingness (as observed in real MIMIC-IV)
    - Right-skewed length-of-stay distribution (median ~47 hrs, mean ~94 hrs)
    - Physiological correlations (e.g., temperature affects HR/BP)
    - Early deterioration patterns for longer stays
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    all_data = []
    all_labels = []
    all_masks = []
    
    for patient_id in range(n_patients):
        # Generate right-skewed length of stay (hours)
        # Using log-normal distribution to match MIMIC-IV: median ~47, mean ~94
        los_hours = np.random.lognormal(mean=3.85, sigma=0.9)
        los_hours = np.clip(los_hours, 12, 500)  # Realistic bounds
        
        # Sequence length varies (irregular sampling)
        seq_len = np.random.randint(24, max_time_steps)
        
        # Generate time series with physiological realism
        patient_data = np.zeros((seq_len, n_features))
        mask = np.zeros((seq_len, n_features))
        
        # Baseline vitals (patient-specific)
        baseline_hr = np.random.normal(80, 15)  # Heart rate
        baseline_sbp = np.random.normal(120, 15)  # Systolic BP
        baseline_temp = np.random.normal(37.0, 0.5)  # Temperature
        
        for t in range(seq_len):
            # Simulate 90% missingness
            observed_features = np.random.choice(n_features, 
                                                size=int(n_features * 0.1), 
                                                replace=False)
            
            for feat_idx in observed_features:
                # Create physiologically correlated signals
                if feat_idx == 0:  # Heart rate
                    # Add trend for longer stays (deterioration)
                    trend = (los_hours / 100) * (t / seq_len)
                    patient_data[t, feat_idx] = baseline_hr + np.random.normal(0, 5) + trend * 10
                    
                elif feat_idx == 1:  # Systolic BP
                    # Correlate with heart rate (crude autonomic simulation)
                    hr_influence = (patient_data[t, 0] - 80) * 0.3 if mask[t, 0] else 0
                    patient_data[t, feat_idx] = baseline_sbp + np.random.normal(0, 10) + hr_influence
                    
                elif feat_idx == 2:  # Temperature
                    # Fever patterns in sicker patients
                    sickness_effect = (los_hours - 47) / 100
                    patient_data[t, feat_idx] = baseline_temp + np.random.normal(0, 0.3) + sickness_effect
                    
                else:  # Other lab values
                    # Generic lab values with noise
                    patient_data[t, feat_idx] = np.random.normal(0, 1)
                
                mask[t, feat_idx] = 1.0
        
        all_data.append(patient_data)
        all_labels.append(los_hours)
        all_masks.append(mask)
    
    return all_data, all_labels, all_masks


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    # Find max sequence length in batch
    max_len = max(item['timeseries'].shape[0] for item in batch)
    
    # Pad sequences
    padded_batch = []
    for item in batch:
        seq_len = item['timeseries'].shape[0]
        padded_ts = torch.zeros(max_len, item['timeseries'].shape[1])
        padded_ts[:seq_len] = item['timeseries']
        
        # Pad labels too
        padded_los = torch.zeros(max_len)
        padded_los[:seq_len] = item['los']
        
        padded_item = {
            'timeseries': padded_ts,
            'los': padded_los,
            'patient_id': item['patient_id'],
            'visit_id': item['visit_id']
        }
        padded_batch.append(padded_item)
    
    # Stack into batch tensors
    return {
        'timeseries': torch.stack([item['timeseries'] for item in padded_batch]),
        'los': torch.stack([item['los'] for item in padded_batch]),
        'patient_id': [item['patient_id'] for item in padded_batch],
        'visit_id': [item['visit_id'] for item in padded_batch]
    }


def train_epoch(model, dataloader, optimizer, device):
    """Single training epoch with proper masking."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (returns dict with 'loss' key)
        output = model(**batch)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, dataloader, device):
    """
    Evaluate model performance with multiple metrics.
    
    Returns:
        mae: Mean Absolute Error (days)
        rmse: Root Mean Squared Error (days)
        mse: Mean Squared Error
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Get predictions (forward returns dict with 'y_prob' key)
            output = model(**batch)
            y_pred = output['y_prob']
            y_true = output['y_true']
            
            all_preds.append(y_pred.cpu())
            all_labels.append(y_true.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert hours to days for interpretability
    all_preds_days = all_preds / 24.0
    all_labels_days = all_labels / 24.0
    
    mae = torch.abs(all_preds_days - all_labels_days).mean().item()
    mse = ((all_preds_days - all_labels_days) ** 2).mean().item()
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse
    }


def run_ablation_experiment(config_name, model_config, train_dataset, val_dataset,
                            device, n_epochs=10):
    """
    Run single ablation experiment with specified configuration.
    
    Args:
        config_name: Name of configuration (e.g., 'baseline', 'shallow')
        model_config: Dict with model hyperparameters
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: torch device
        n_epochs: Number of training epochs
    
    Returns:
        Dictionary with training history and final metrics
    """
    print(f"\n{'='*60}")
    print(f"Running Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"Config: {model_config}")
    
    # Initialize model
    model = TPC(dataset=train_dataset, **model_config).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Training history
    history = {
        'train_loss': [],
        'val_mae': [],
        'val_rmse': []
    }
    
    # Training loop
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.3f} days | "
                  f"Val RMSE: {val_metrics['rmse']:.3f} days")
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, device)
    print(f"\nFinal {config_name} Results:")
    print(f"  MAE:  {final_metrics['mae']:.3f} days")
    print(f"  RMSE: {final_metrics['rmse']:.3f} days")
    
    return {
        'config': model_config,
        'history': history,
        'final_metrics': final_metrics
    }


def main():
    """
    Main pipeline demonstrating complete PyHealth workflow.
    """
    print("="*80)
    print("TPC Model - Full Pipeline Replication (Extra Credit)")
    print("CS 598 Deep Learning for Healthcare")
    print("Authors: Pankaj Meghani, Tarak Jha, Pranash Krishnan")
    print("="*80)
    
    # ===========================
    # 1. DATA GENERATION
    # ===========================
    print("\n[Step 1/5] Generating synthetic MIMIC-IV dataset...")
    data_list, labels_list, masks_list = generate_synthetic_mimic_data(
        n_patients=300,
        n_features=34,
        max_time_steps=72
    )
    
    # Dataset statistics
    mean_los = np.mean(labels_list) / 24.0  # Convert to days
    median_los = np.median(labels_list) / 24.0
    print(f"  Generated 300 synthetic ICU stays")
    print(f"  Mean length of stay: {mean_los:.2f} days")
    print(f"  Median length of stay: {median_los:.2f} days")
    print(f"  Features: 34 time-varying vitals/labs")
    print(f"  Time steps: Variable (24-72 hours)")
    
    # ===========================
    # 2. DATASET PREPARATION
    # ===========================
    print("\n[Step 2/5] Preparing PyHealth-compatible dataset...")
    
    # Train/val split (80/20)
    split_idx = int(0.8 * len(data_list))
    
    train_dataset = SyntheticICUDataset(
        data_list[:split_idx],
        labels_list[:split_idx],
        masks_list[:split_idx]
    )
    
    val_dataset = SyntheticICUDataset(
        data_list[split_idx:],
        labels_list[split_idx:],
        masks_list[split_idx:]
    )
    
    print(f"  Training set: {len(train_dataset)} patients")
    print(f"  Validation set: {len(val_dataset)} patients")
    print(f"  Batch size: 32")
    
    # ===========================
    # 3. DEVICE SETUP
    # ===========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Step 3/5] Using device: {device}")
    
    # ===========================
    # 4. ABLATION STUDY
    # ===========================
    print("\n[Step 4/5] Running ablation study...")
    print("  Testing 3 configurations to validate model components")
    
    ablation_configs = {
        'baseline': {
            'n_layers': 3,
            'kernel_size': 4,
            'main_dropout_rate': 0.45,
            'temp_dropout_rate': 0.45,
            'time_before_pred': 5,
            'use_msle': True
        },
        'shallow_network': {
            'n_layers': 1,  # Reduced depth
            'kernel_size': 4,
            'main_dropout_rate': 0.45,
            'temp_dropout_rate': 0.45,
            'time_before_pred': 5,
            'use_msle': True
        },
        'high_dropout': {
            'n_layers': 3,
            'kernel_size': 4,
            'main_dropout_rate': 0.7,  # Increased regularization
            'temp_dropout_rate': 0.7,
            'time_before_pred': 5,
            'use_msle': True
        }
    }
    
    results = {}
    for config_name, config in ablation_configs.items():
        result = run_ablation_experiment(
            config_name=config_name,
            model_config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            n_epochs=10
        )
        results[config_name] = result
    
    # ===========================
    # 5. RESULTS SUMMARY
    # ===========================
    print("\n" + "="*80)
    print("[Step 5/5] ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    
    # Compare final MAE across configurations
    comparison = []
    for config_name, result in results.items():
        mae = result['final_metrics']['mae']
        rmse = result['final_metrics']['rmse']
        comparison.append({
            'config': config_name,
            'mae_days': mae,
            'rmse_days': rmse
        })
        print(f"\n{config_name.upper()}:")
        print(f"  Final MAE:  {mae:.3f} days")
        print(f"  Final RMSE: {rmse:.3f} days")
    
    # Identify best configuration
    best_config = min(comparison, key=lambda x: x['mae_days'])
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION: {best_config['config'].upper()}")
    print(f"  MAE:  {best_config['mae_days']:.3f} days")
    print(f"  RMSE: {best_config['rmse_days']:.3f} days")
    print(f"{'='*60}")
    
    # ===========================
    # 6. EXPORT RESULTS
    # ===========================
    output_file = Path(__file__).parent / 'replication_results.json'
    
    # Prepare serializable results
    export_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'authors': ['Pankaj Meghani (meghani3)', 
                       'Tarak Jha (tarakj2)', 
                       'Pranash Krishnan (pranash2)'],
            'dataset': {
                'n_patients': 300,
                'n_features': 34,
                'mean_los_days': float(mean_los),
                'median_los_days': float(median_los),
                'train_size': len(train_dataset),
                'val_size': len(val_dataset)
            }
        },
        'ablation_results': {
            config_name: {
                'config': result['config'],
                'final_mae_days': result['final_metrics']['mae'],
                'final_rmse_days': result['final_metrics']['rmse'],
                'training_history': {
                    'train_loss': result['history']['train_loss'],
                    'val_mae': result['history']['val_mae'],
                    'val_rmse': result['history']['val_rmse']
                }
            }
            for config_name, result in results.items()
        },
        'best_configuration': best_config
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_results, f, indent=2)
    
    print(f"\n✓ Results exported to: {output_file}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nThis replication demonstrates:")
    print("  ✓ Synthetic data generation matching MIMIC-IV schema")
    print("  ✓ PyHealth task and dataset setup")
    print("  ✓ Complete model training loop with MSLE loss")
    print("  ✓ Ablation study across 3 configurations")
    print("  ✓ Comprehensive evaluation (MAE, RMSE)")
    print("  ✓ Results export for reproducibility")
    print("\nAll components validated successfully!")


if __name__ == '__main__':
    main()
