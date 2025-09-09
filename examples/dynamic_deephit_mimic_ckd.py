#!/usr/bin/env python3
"""
Dynamic DeepHit Model Training Script for CKD Survival Analysis

This script trains a Dynamic DeepHit model using PyHealth's MIMIC4Dataset
and the MIMIC4CKDSurvAnalysis task for predicting ESRD (End-Stage Renal Disease)
in chronic kidney disease patients.

Usage:
    python train_dynamic_deephit_ckd.py --data_root /path/to/mimic4 --setting time_variant --epochs 50 --device cuda:0
"""

import argparse
import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Add parent directory to Python path to make pyhealth accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# PyHealth imports
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.models import BaseModel

# Custom task import - assuming it's in the parent directory
sys.path.append(os.path.join(parent_dir, "."))  # Add project root
from pyhealth.tasks.ckd_surv import MIMIC4CKDSurvAnalysis  # Adjust import as needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicDeepHit(BaseModel):
    """
    Dynamic DeepHit model for survival analysis with time-varying covariates.
    Inherits from PyHealth's BaseModel for compatibility.
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "binary",
        hidden_dims: List[int] = [128, 64],
        dropout_lstm: float = 0.2,
        dropout_cause: float = 0.2,
        pred_times: int = 365 * 5,
        **kwargs,
    ):
        super(DynamicDeepHit, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            **kwargs,
        )

        self.hidden_dims = hidden_dims
        self.dropout_lstm = dropout_lstm
        self.dropout_cause = dropout_cause
        self.pred_times = pred_times
        self.num_risks = 1  # Only ESRD

        # Determine input dimension based on feature keys and setting
        self.input_dim = self._get_input_dim()

        # LSTM layers (bidirectional)
        num_layer_lstm = 2
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dims[0],
            num_layers=num_layer_lstm,
            batch_first=True,
            dropout=dropout_lstm if num_layer_lstm > 1 else 0,
            bidirectional=True,
        )

        # FC layer after LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[0]),  # *2 for bidirectional
            nn.Tanh(),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], 1),
        )

        # Cause-specific fully connected layers
        layers = []
        prev_dim = hidden_dims[0]
        if len(hidden_dims) > 1:
            for hidden_dim in hidden_dims[1:]:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_cause))
                prev_dim = hidden_dim

        self.cause_specific_fc = nn.Sequential(*layers) if layers else nn.Identity()

        # Risk-specific output heads
        self.risk_heads = nn.ModuleList(
            [nn.Linear(prev_dim, self.pred_times) for _ in range(self.num_risks)]
        )

    def _get_input_dim(self) -> int:
        """Determine input dimension based on feature keys."""
        # This will be set based on the actual features in the dataset
        # For now, return a default that will be updated when we see the first batch
        return 1  # Will be updated dynamically

    def attention_net(
        self, fc_output: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention mechanism with masking."""
        attention_weights = self.attention(fc_output)
        mask = mask.unsqueeze(-1)
        attention_weights = attention_weights.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * fc_output, dim=1)
        return context, attention_weights

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with PyHealth's BaseModel interface.

        Expected kwargs from PyHealth:
        - For time_invariant: patient features as tensors
        - For time_variant/heterogeneous: lab_measurements as sequences
        """
        # Extract data based on feature keys
        if "lab_measurements" in kwargs:
            # Time-varying data
            x, mask = self._process_time_varying_data(kwargs)
        else:
            # Time-invariant data
            x, mask = self._process_time_invariant_data(kwargs)

        # Update input_dim if this is the first forward pass
        if hasattr(self, "_first_forward") and not self._first_forward:
            if x.shape[-1] != self.input_dim:
                self.input_dim = x.shape[-1]
                # Reinitialize LSTM with correct input dimension
                self.lstm = nn.LSTM(
                    input_size=self.input_dim,
                    hidden_size=self.hidden_dims[0],
                    num_layers=2,
                    batch_first=True,
                    dropout=self.dropout_lstm,
                    bidirectional=True,
                ).to(x.device)
            self._first_forward = True

        # LSTM forward pass
        lstm_output, _ = self.lstm(x)

        # FC layer
        fc_output = self.fc(lstm_output)

        # Attention mechanism
        context, attention_weights = self.attention_net(fc_output, mask)

        # Cause-specific layers
        x = self.cause_specific_fc(context)

        # Risk predictions
        hazard_preds = [torch.sigmoid(risk_head(x)) for risk_head in self.risk_heads]
        result = torch.stack(hazard_preds, dim=1)

        # Return in PyHealth format
        return {
            "logits": result.squeeze(
                1
            ),  # Remove risk dimension since we only have 1 risk
            "hazard_preds": result,
            "attention_weights": attention_weights,
        }

    def _process_time_invariant_data(self, kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process time-invariant data (baseline features only)."""
        batch_size = len(kwargs[self.label_key])

        # Collect features
        features = []
        for key in self.feature_keys:
            if key in kwargs:
                if key == "baseline_egfr":
                    features.append(kwargs[key].unsqueeze(-1))
                elif key == "demographics":
                    # Handle demographics (should be converted to numerical)
                    demo_features = self._process_demographics(kwargs[key])
                    features.append(demo_features)
                elif key == "comorbidities":
                    # Handle comorbidities (convert to count or binary features)
                    comorbid_features = self._process_comorbidities(kwargs[key])
                    features.append(comorbid_features)
                elif key in ["age", "gender"]:
                    features.append(kwargs[key].unsqueeze(-1))

        if features:
            x = torch.cat(features, dim=-1)
        else:
            # Fallback
            x = torch.zeros(batch_size, 3)  # Default 3 features

        # Add sequence dimension and create mask
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        mask = torch.ones(batch_size, 1)  # [batch_size, 1]

        return x, mask

    def _process_time_varying_data(self, kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process time-varying lab measurements."""
        lab_measurements = kwargs["lab_measurements"]
        batch_size = len(lab_measurements)

        # Find max sequence length
        max_seq_len = max(len(measurements) for measurements in lab_measurements)

        # Initialize tensors
        features = []
        masks = []

        for measurements in lab_measurements:
            seq_len = len(measurements)

            # Pad measurements to max_seq_len
            padded_measurements = measurements + [{}] * (max_seq_len - seq_len)

            # Extract features from each measurement
            measurement_features = []
            for measurement in padded_measurements:
                feature_vec = self._extract_measurement_features(measurement)
                measurement_features.append(feature_vec)

            features.append(torch.stack(measurement_features))

            # Create mask
            mask = torch.zeros(max_seq_len)
            mask[:seq_len] = 1
            masks.append(mask)

        x = torch.stack(features)  # [batch_size, seq_len, feature_dim]
        mask = torch.stack(masks)  # [batch_size, seq_len]

        return x, mask

    def _extract_measurement_features(
        self, measurement: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract numerical features from a single measurement."""
        features = []

        # Standard features that should be in measurements
        if "egfr" in measurement:
            features.append(measurement.get("egfr", 0.0))

        if "missing_egfr" in measurement:
            features.append(float(measurement.get("missing_egfr", True)))

        if "protein" in measurement:
            features.append(measurement.get("protein", 0.0))

        if "missing_protein" in measurement:
            features.append(float(measurement.get("missing_protein", True)))

        if "albumin" in measurement:
            features.append(measurement.get("albumin", 0.0))

        if "missing_albumin" in measurement:
            features.append(float(measurement.get("missing_albumin", True)))

        if "creatinine" in measurement:
            features.append(measurement.get("creatinine", 0.0))

        # If no features found, return a default
        if not features:
            features = [0.0]

        return torch.tensor(features, dtype=torch.float32)

    def _process_demographics(self, demographics_list) -> torch.Tensor:
        """Convert demographics to numerical features."""
        batch_size = len(demographics_list)
        demo_features = torch.zeros(batch_size, 3)  # age_group, gender, race

        for i, demographics in enumerate(demographics_list):
            # Age group: adult=0, elderly=1
            if "elderly" in demographics:
                demo_features[i, 0] = 1.0

            # Gender: female=0, male=1
            if "male" in demographics:
                demo_features[i, 1] = 1.0

            # Race (simplified): other=0, white=1
            if any(
                "white" in demo.lower()
                for demo in demographics
                if isinstance(demo, str)
            ):
                demo_features[i, 2] = 1.0

        return demo_features

    def _process_comorbidities(self, comorbidities_list) -> torch.Tensor:
        """Convert comorbidities to numerical features."""
        batch_size = len(comorbidities_list)
        # Simple approach: just count comorbidities
        comorbid_counts = torch.zeros(batch_size, 1)

        for i, comorbidities in enumerate(comorbidities_list):
            comorbid_counts[i, 0] = len(comorbidities) if comorbidities else 0

        return comorbid_counts


def create_survival_loss_function(alpha: float = 0.5):
    """
    Create a survival analysis loss function compatible with PyHealth.

    Args:
        alpha: Weight balance between likelihood loss and ranking loss

    Returns:
        Loss function that works with PyHealth's training loop
    """

    def survival_loss(y_true, y_prob, **kwargs):
        """
        Survival loss function.

        Args:
            y_true: Ground truth labels (duration_days, has_esrd)
            y_prob: Model predictions (hazard predictions)
            **kwargs: Additional arguments from PyHealth
        """
        # Extract duration and event information
        if isinstance(y_true, dict):
            duration_days = y_true.get("duration_days", torch.zeros(len(y_prob)))
            has_esrd = y_true.get("has_esrd", torch.zeros(len(y_prob)))
        else:
            # Fallback: assume y_true contains the event indicator
            has_esrd = y_true.float()
            duration_days = torch.ones_like(has_esrd) * 365  # Default duration

        # Simple negative log-likelihood loss for now
        # In practice, you'd want a more sophisticated survival loss
        hazard_preds = torch.sigmoid(y_prob)

        # Basic likelihood loss
        event_loss = has_esrd * torch.log(hazard_preds + 1e-8)
        no_event_loss = (1 - has_esrd) * torch.log(1 - hazard_preds + 1e-8)

        loss = -(event_loss + no_event_loss).mean()

        return loss

    return survival_loss


def main():
    """Main training function using PyHealth's standard workflow."""
    parser = argparse.ArgumentParser(
        description="Train Dynamic DeepHit for CKD Survival Analysis"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to MIMIC-IV data"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="time_variant",
        choices=["time_invariant", "time_variant", "heterogeneous"],
        help="Analysis setting",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    parser.add_argument(
        "--dev", action="store_true", help="Use dev mode (smaller dataset)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--dropout_lstm", type=float, default=0.2, help="LSTM dropout rate"
    )
    parser.add_argument(
        "--dropout_cause", type=float, default=0.2, help="Cause-specific dropout rate"
    )
    parser.add_argument(
        "--loss_alpha", type=float, default=0.5, help="Loss weighting parameter"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load MIMIC4 dataset
    logger.info(f"Loading MIMIC4Dataset from {args.data_root}")

    tables = ["patients", "admissions", "diagnoses_icd", "labevents"]

    base_dataset = MIMIC4Dataset(
        ehr_root=args.data_root,
        ehr_tables=tables,
        dataset_name="mimic4_ckd",
        dev=args.dev,
    )

    logger.info(f"Dataset loaded successfully")
    # Print dataset statistics
    base_dataset.stats()

    # Apply CKD survival analysis task
    logger.info(f"Applying CKD survival task with setting: {args.setting}")

    ckd_task = MIMIC4CKDSurvAnalysis(
        setting=args.setting, min_age=18, prediction_window_days=365 * 5
    )

    sample_dataset = base_dataset.set_task(ckd_task)
    logger.info(f"Generated {len(sample_dataset.samples)} samples")

    # Split dataset using PyHealth's splitter
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.7, 0.15, 0.15], seed=42
    )

    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create data loaders using PyHealth's get_dataloader
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Determine feature keys based on setting
    if args.setting == "time_invariant":
        feature_keys = [
            "baseline_egfr",
            "demographics",
            "comorbidities",
            "age",
            "gender",
        ]
    elif args.setting == "time_variant":
        feature_keys = ["lab_measurements", "demographics", "age", "gender"]
    elif args.setting == "heterogeneous":
        feature_keys = [
            "lab_measurements",
            "missing_indicators",
            "demographics",
            "age",
            "gender",
        ]

    # Initialize model using PyHealth's interface
    logger.info("Initializing Dynamic DeepHit model...")

    model = DynamicDeepHit(
        dataset=sample_dataset,
        feature_keys=feature_keys,
        label_key="has_esrd",  # Primary outcome
        mode="binary",
        hidden_dims=args.hidden_dims,
        dropout_lstm=args.dropout_lstm,
        dropout_cause=args.dropout_cause,
    )

    # Set flag for first forward pass
    model._first_forward = False

    logger.info(f"Model initialized: {model}")

    # Initialize PyHealth trainer
    logger.info("Starting training with PyHealth Trainer...")

    trainer = Trainer(
        model=model,
        device=device,
        metrics=["accuracy", "f1"],  # Will be overridden for survival analysis
        output_path=args.output_dir,
        exp_name=f"dynamic_deephit_{args.setting}",
    )

    # Custom loss function for survival analysis
    survival_loss_fn = create_survival_loss_function(alpha=args.loss_alpha)

    # Train the model
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        loss_fn=survival_loss_fn,  # Custom survival loss
        monitor="val_loss",
        monitor_criterion="min",
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")

    # Save final model
    final_model_path = os.path.join(
        args.output_dir, f"dynamic_deephit_{args.setting}_final.pth"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
