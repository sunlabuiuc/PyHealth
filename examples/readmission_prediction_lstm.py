"""
Hospital Readmission Prediction using LSTM on MIMIC-III

This script demonstrates how to use PyHealth to predict 30-day hospital 
readmission using LSTM on the MIMIC-III dataset.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import torch
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import readmission_prediction_mimic3_fn
from pyhealth.models import RNN
from pyhealth.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hospital Readmission Prediction using LSTM on MIMIC-III"
    )
    parser.add_argument(
        "--mimic3_path", type=str, required=True, help="Path to MIMIC-III dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension of LSTM"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--window_size", type=int, default=30, help="Window size for readmission in days"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development mode (smaller dataset)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    """Load and prepare MIMIC-III dataset."""
    print("Loading MIMIC-III dataset...")
    dataset = MIMIC3Dataset(
        root=args.mimic3_path,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": "ATC"},
        dev=args.dev,
    )

    # Split the dataset
    train_dataset, val_dataset, test_dataset = dataset.split([0.7, 0.1, 0.2])

    # Generate samples for each split
    train_samples = train_dataset.set_task(
        readmission_prediction_mimic3_fn, window_size=args.window_size
    )
    val_samples = val_dataset.set_task(
        readmission_prediction_mimic3_fn, window_size=args.window_size
    )
    test_samples = test_dataset.set_task(
        readmission_prediction_mimic3_fn, window_size=args.window_size
    )

    # Print dataset statistics
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    return train_samples, val_samples, test_samples


def build_model(train_samples, args):
    """Build LSTM model for readmission prediction."""
    # Get the feature keys and label key from the samples
    feature_keys = list(train_samples[0]["x"].keys())
    label_key = list(train_samples[0]["y"].keys())[0]

    print(f"Feature keys: {feature_keys}")
    print(f"Label key: {label_key}")

    # Configure the model
    model = RNN(
        feature_keys=feature_keys,
        label_key=label_key,
        mode="binary",
        rnn_type="lstm",
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    return model


def train_model(model, train_samples, val_samples, test_samples, args):
    """Train the model using PyHealth's Trainer."""
    print("Training model...")
    # Configure the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_samples,
        val_dataset=val_samples,
        test_dataset=test_samples,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        evaluation_metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    # Train the model
    trainer.train()
    return trainer


def evaluate_model(trainer, test_samples):
    """Evaluate the model on the test set."""
    print("Evaluating model...")
    # Evaluate on test set
    test_result = trainer.evaluate(test_samples)
    print("Test results:")
    for metric, value in test_result.items():
        print(f"{metric}: {value:.4f}")
    return test_result


def visualize_results(trainer, test_samples, args):
    """Create visualizations of model performance."""
    print("Creating visualizations...")
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get predictions on test set
    y_true, y_prob = trainer.predict(test_samples, return_y_true=True)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"))
    plt.close()

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.output_dir, "pr_curve.png"))
    plt.close()

    # Save metrics to file
    test_result = trainer.evaluate(test_samples)
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for metric, value in test_result.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"Visualizations saved to {args.output_dir}")


def main():
    """Main function to run the example."""
    args = parse_args()
    set_seed(args.seed)

    # Load and prepare data
    train_samples, val_samples, test_samples = load_data(args)

    # Build model
    model = build_model(train_samples, args)

    # Train model
    trainer = train_model(model, train_samples, val_samples, test_samples, args)

    # Evaluate model
    evaluate_model(trainer, test_samples)

    # Visualize results
    visualize_results(trainer, test_samples, args)

    # Save model
    torch.save(
        model.state_dict(), os.path.join(args.output_dir, "readmission_lstm_model.pt")
    )
    print(f"Model saved to {os.path.join(args.output_dir, 'readmission_lstm_model.pt')}")

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
