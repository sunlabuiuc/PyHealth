"""
COVID-RED Dataset Example for PyHealth

This script demonstrates how to:
1. Load the COVID-RED wearable device dataset
2. Define a COVID-19 detection task
3. Train a simple LSTM classifier for early COVID-19 detection

Dataset: Remote Early Detection of SARS-CoV-2 infections (COVID-RED)
Source: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import PyHealth components (adjust imports based on actual PyHealth structure)
try:
    from pyhealth.datasets import COVIDREDDataset
    from pyhealth.tasks import covidred_detection_fn, covidred_prediction_fn
except ImportError:
    # For standalone testing, import from local files
    import sys
    sys.path.insert(0, '.')
    from covidred_dataset import COVIDREDDataset
    from covidred_tasks import covidred_detection_fn, covidred_prediction_fn


class LSTMClassifier(nn.Module):
    """
    Simple LSTM classifier for COVID-19 detection from time series data.
    
    This model processes multivariate time series of wearable device measurements
    (heart rate, steps, sleep) to predict COVID-19 infection.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        """
        Parameters
        ----------
        input_size : int
            Number of features per time step (e.g., 8 for COVID-RED)
        hidden_size : int
            Number of LSTM hidden units
        num_layers : int
            Number of LSTM layers
        num_classes : int
            Number of output classes (2 for binary classification)
        dropout : float
            Dropout probability
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features, seq_len)
        
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        # Transpose to (batch_size, seq_len, n_features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def collate_fn(batch):
    """
    Custom collate function to batch samples from COVIDREDDataset.
    
    Parameters
    ----------
    batch : list
        List of sample dictionaries from the dataset.
    
    Returns
    -------
    dict
        Batched data with stacked tensors.
    """
    signals = torch.stack([item["signal"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    patient_ids = [item["patient_id"] for item in batch]
    visit_ids = [item["visit_id"] for item in batch]
    
    return {
        "signal": signals,
        "label": labels,
        "patient_id": patient_ids,
        "visit_id": visit_ids,
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    for batch in dataloader:
        signals = batch["signal"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Calculate AUC if there are both positive and negative samples
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probabilities)
    else:
        auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc


def main():
    """Main function to demonstrate COVID-RED dataset usage."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    DATA_ROOT = "/path/to/covidred"  # Update this path
    WINDOW_DAYS = 7
    TASK_TYPE = "prediction"  # "detection" or "prediction"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("COVID-RED Dataset Example for PyHealth")
    print("=" * 80)
    
    # Load datasets
    print("\n[1/5] Loading COVID-RED dataset...")
    print(f"  - Root directory: {DATA_ROOT}")
    print(f"  - Window size: {WINDOW_DAYS} days")
    print(f"  - Task type: {TASK_TYPE}")
    
    try:
        train_dataset = COVIDREDDataset(
            root=DATA_ROOT,
            split="train",
            window_days=WINDOW_DAYS,
            task=TASK_TYPE,
        )
        
        test_dataset = COVIDREDDataset(
            root=DATA_ROOT,
            split="test",
            window_days=WINDOW_DAYS,
            task=TASK_TYPE,
        )
        
        print(f"\n  Dataset loaded successfully!")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
        
        # Show label distribution
        train_dist = train_dataset.get_label_distribution()
        test_dist = test_dataset.get_label_distribution()
        
        print(f"\n  Training set distribution:")
        print(f"    - Positive: {train_dist['positive_samples']} ({train_dist['positive_ratio']:.2%})")
        print(f"    - Negative: {train_dist['negative_samples']}")
        
        print(f"\n  Test set distribution:")
        print(f"    - Positive: {test_dist['positive_samples']} ({test_dist['positive_ratio']:.2%})")
        print(f"    - Negative: {test_dist['negative_samples']}")
        
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("\n  Please download the COVID-RED dataset from:")
        print("  https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7")
        return
    
    # Create data loaders
    print("\n[2/5] Creating data loaders...")
    
    # Apply task function to samples
    task_fn = covidred_prediction_fn if TASK_TYPE == "prediction" else covidred_detection_fn
    
    # Wrap samples with task function
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, task_fn):
            self.base_dataset = base_dataset
            self.task_fn = task_fn
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            sample = self.base_dataset[idx]
            return self.task_fn(sample)
    
    train_task_dataset = TaskDataset(train_dataset, task_fn)
    test_task_dataset = TaskDataset(test_dataset, task_fn)
    
    train_loader = DataLoader(
        train_task_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_task_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n[3/5] Initializing LSTM model...")
    
    # Get feature dimension from first sample
    sample = train_dataset[0]
    input_size = len(train_dataset.get_feature_names())
    
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
    ).to(DEVICE)
    
    print(f"  - Input features: {input_size}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {DEVICE}")
    
    # Loss and optimizer
    # Use weighted loss for imbalanced datasets
    pos_weight = train_dist['negative_samples'] / max(train_dist['positive_samples'], 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("\n[4/5] Training model...")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Class weight (positive): {pos_weight:.2f}")
    
    best_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
            model, test_loader, criterion, DEVICE
        )
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), "best_covidred_model.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"\n  Epoch [{epoch+1}/{NUM_EPOCHS}]")
            print(f"    Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"    Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
                  f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    # Final evaluation
    print("\n[5/5] Final evaluation on test set...")
    model.load_state_dict(torch.load("best_covidred_model.pt"))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n  Final Test Metrics:")
    print(f"    - Accuracy:  {test_acc:.4f}")
    print(f"    - Precision: {test_prec:.4f}")
    print(f"    - Recall:    {test_rec:.4f}")
    print(f"    - F1-Score:  {test_f1:.4f}")
    print(f"    - AUC:       {test_auc:.4f}")
    
    print("\n" + "=" * 80)
    print("Training complete! Best model saved to 'best_covidred_model.pt'")
    print("=" * 80)


if __name__ == "__main__":
    main()
