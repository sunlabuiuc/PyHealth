"""
Your Name: Ritwik Biswas
Your NetId: rbiswas2
Paper Title: Daily Physical Activity Monitoring: Adaptive Learning from Multi-source Motion Sensor Data
Paper Link: https://proceedings.mlr.press/v248/zhang24a.html
Description: A reproduction of the IPD-guided adaptive transfer learning framework on the UCI DSA dataset.
             Includes independent implementations in PyTorch and Keras to isolate framework-specific artifacts.
             Note: This reproduction observes negative transfer (56.01% vs 70.04% baseline).
"""

import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
# Note: You may need to install fastdtw: pip install fastdtw
from fastdtw import fastdtw
import requests
import zipfile
import io
from pathlib import Path

# PyHealth imports
from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

# Global Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------
# 1. Dataset Implementation (UCI DSA)
# ---------------------------------------------------------
class UCIDSADataset(BaseDataset):
    """
    Dataset loader for the UCI Daily and Sports Activity (DSA) dataset.
    Handles downloading, extraction, and subject-aware parsing.
    """
    def __init__(self, root="data", **kwargs):
        self.root = root
        # Automatically download and process on init
        self.download_and_extract_dsa()
        self.data = self.load_dsa_dataset_by_subject()
        
        # PyHealth BaseDataset requires a 'samples' attribute, but since
        # your pipeline uses a custom subject-split loader, we store the 
        # dictionary structure in self.data for downstream access.
        self.samples = [] 

    def download_and_extract_dsa(self):
        """Downloads the UCI DSA dataset if not present."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip"
        if Path(self.root).exists():
            print("Dataset already downloaded and extracted.")
            return

        print("Downloading DSA dataset...")
        try:
            r = requests.get(url)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            print("Extracting dataset...")
            z.extractall() # Extracts to 'data' folder by default in zip
            print("Done.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")

    def load_dsa_dataset_by_subject(self):
        """Parses the raw text files into a Subject-Sensor dictionary."""
        dataset = {}
        activities = range(1, 20)
        subjects = range(1, 9)
        segments = range(1, 61)
        
        sensor_map = {
            "T": "torso", "RA": "right_arm", "LA": "left_arm",
            "RL": "right_leg", "LL": "left_leg"
        }

        print("Parsing data files (Subject-Aware)...")
        for activity_id in activities:
            for subject_id in subjects:
                dir_path = os.path.join(self.root, f"a{activity_id:02}", f"p{subject_id}")
                if not os.path.exists(dir_path): continue
                
                activity_label = activity_id - 1
                
                for segment_id in segments:
                    file_path = os.path.join(dir_path, f"s{segment_id:02}.txt")
                    if not os.path.exists(file_path): continue
                    
                    try:
                        raw_data = np.loadtxt(file_path, delimiter=",")
                    except: 
                        continue

                    # Split raw 45 columns into 5 sensors (9 cols each)
                    for i, (prefix, name) in enumerate(sensor_map.items()):
                        if name not in dataset: dataset[name] = {}
                        if subject_id not in dataset[name]: dataset[name][subject_id] = []
                        
                        start_col = i * 9
                        sensor_data = raw_data[:, start_col : start_col + 9]
                        dataset[name][subject_id].append((sensor_data, activity_label))
        return dataset

# ---------------------------------------------------------
# 2. Model Implementation (IPD-Guided)
# ---------------------------------------------------------
class AdaptiveLSTM(BaseModel):
    """
    LSTM classifier with Dropout and Softmax head.
    Formula: f(x) = Softmax(Wd · Dropout(LSTM(x)))
    """
    def __init__(self, input_dim, hidden_dim, num_classes, **kwargs):
        super(AdaptiveLSTM, self).__init__(**kwargs)
        # Note: PyTorch LSTM default num_layers is 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, Time, Feat]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out

def init_weights(model):
    """
    Custom initialization to match Keras 'glorot_uniform' and 'orthogonal' defaults.
    Verified necessary for convergence in PyTorch vs Keras reproduction.
    """
    for name, param in model.named_parameters():
        if 'weight_ih' in name:  # Input-Hidden weights (LSTM)
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:  # Hidden-Hidden weights (LSTM)
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:  # Biases
            nn.init.zeros_(param.data)
        elif 'fc.weight' in name:  # Dense layer weights
            nn.init.xavier_uniform_(param.data)
        elif 'fc.bias' in name:  # Dense layer bias
            nn.init.zeros_(param.data)

# ---------------------------------------------------------
# 3. Helper Functions (Data Processing & IPD)
# ---------------------------------------------------------
def prepare_loaders_subject_split(domain_data, train_subjects, test_subjects, batch_size=16):
    """
    Splits data by subject and applies Per-Sample MinMax Scaling (-1, 1).
    Batch size 16 used for better generalization (Regularization noise).
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Aggregate Train Data
    for subj in train_subjects:
        if subj in domain_data:
            X_train.extend([item[0] for item in domain_data[subj]])
            y_train.extend([item[1] for item in domain_data[subj]])

    # Aggregate Test Data
    for subj in test_subjects:
        if subj in domain_data:
            X_test.extend([item[0] for item in domain_data[subj]])
            y_test.extend([item[1] for item in domain_data[subj]])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    # Normalization (Per-Sample MinMax)
    N, T, F = X_train.shape
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Reshape to (N*T, F) for scaling, then back
    X_train_reshaped = X_train.reshape(-1, F)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(N, T, F)
    
    N_test, _, _ = X_test.shape
    X_test_reshaped = X_test.reshape(-1, F)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(N_test, T, F)

    # Create TensorDatasets
    train_ds = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test_scaled), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, F, 19 # 19 Classes

def calculate_ipd_proxy(target_data, source_data, subjects, max_pairs=100):
    """
    Calculates Inter-domain Pairwise Distance (IPD) using Dynamic Time Warping (DTW).
    Used to rank source domains by kinematic similarity.
    """
    rng = np.random.default_rng(42)
    t_pool, s_pool = [], []

    for subj in subjects:
        if subj in target_data: t_pool.extend([x[0] for x in target_data[subj]])
        if subj in source_data: s_pool.extend([x[0] for x in source_data[subj]])

    num_samples = min(len(t_pool), len(s_pool))
    use_pairs = min(max_pairs, num_samples)
    
    if use_pairs == 0: return np.inf

    indices = rng.choice(num_samples, size=use_pairs, replace=False)
    distances = []

    for idx in indices:
        # DTW distance using Euclidean norm
        dist, _ = fastdtw(t_pool[idx], s_pool[idx], dist=lambda x, y: np.linalg.norm(x - y))
        distances.append(dist)

    return np.mean(distances) if distances else np.inf

def eval_model(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            pred = torch.max(out, 1)[1]
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

# ---------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting reproduction of Zhang et al. (2024)...")
    
    # --- Config ---
    TARGET_DOMAIN = "right_arm"
    SOURCE_DOMAINS = ["torso", "left_arm", "right_leg", "left_leg"]
    TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
    VAL_SUBJECTS = [7, 8]
    
    # 1. Load Data
    dsa_dataset = UCIDSADataset(root="data")
    dsa_data = dsa_dataset.data # Access the internal dict

    # 2. Calculate IPD
    ipd_scores = {}
    print(f"Calculating IPD against '{TARGET_DOMAIN}'...")
    for source_name in SOURCE_DOMAINS:
        ipd = calculate_ipd_proxy(dsa_data[TARGET_DOMAIN], dsa_data[source_name], TRAIN_SUBJECTS)
        ipd_scores[source_name] = ipd
        print(f" IPD({source_name}): {ipd:.4f}")

    # Prepare Domain Weights
    valid_ipds = {k: v for k, v in ipd_scores.items() if not np.isinf(v)}
    total_ipd = sum(valid_ipds.values())
    domain_weights = {k: v / total_ipd for k, v in valid_ipds.items()}
    
    # Sort Descending (Dissimilar -> Similar for pre-training)
    sorted_sources = sorted(valid_ipds.keys(), key=lambda k: valid_ipds[k], reverse=True)
    print(f"\nTraining Order (Dissimilar -> Similar): {sorted_sources}")

    # 3. Setup Model
    # Get dims from target domain loader
    target_train_l, target_test_l, F_dim, C_classes = prepare_loaders_subject_split(
        dsa_data[TARGET_DOMAIN], TRAIN_SUBJECTS, VAL_SUBJECTS
    )
    
    model = AdaptiveLSTM(input_dim=F_dim, hidden_dim=64, num_classes=C_classes).to(DEVICE)
    model.apply(init_weights) # Apply Keras Init
    criterion = nn.CrossEntropyLoss()

    # --- Phase 1: Source Pre-training ---
    print("\n--- Phase 1: Source Pre-training ---")
    current_lr = 0.005 # High initial LR per repo
    
    for domain in sorted_sources:
        print(f"\nTraining on {domain} (LR: {current_lr:.5f})")
        train_l, val_l, _, _ = prepare_loaders_subject_split(
            dsa_data[domain], TRAIN_SUBJECTS, VAL_SUBJECTS
        )
        
        # Reset Optimizer (Clear Momentum between domains)
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        
        for ep in range(30): # 30 Epochs per domain
            model.train()
            total_loss, correct, total = 0, 0, 0
            for X, y in train_l:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                # STABILIZATION: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * X.size(0)
                pred = torch.max(out, 1)[1]
                correct += (pred == y).sum().item()
                total += y.size(0)
            
            t_acc = correct / total
            v_loss, v_acc = eval_model(model, val_l, criterion)
            
            if (ep+1) % 10 == 0:
                print(f" Ep {ep+1}: Train Acc {t_acc:.3f} | Val Acc {v_acc:.3f}")

        # Adaptive Decay based on IPD
        alpha = domain_weights[domain]
        current_lr = current_lr * (1 - alpha)

    # --- Phase 2: Target Fine-tuning ---
    print("\n--- Phase 2: Target Fine-tuning ---")
    ft_lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=ft_lr)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for ep in range(40):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in target_train_l:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            pred = torch.max(out, 1)[1]
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        t_acc = correct / total
        v_loss, v_acc = eval_model(model, target_test_l, criterion)
        
        if v_acc > best_acc:
            best_acc = v_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        if (ep+1) % 5 == 0:
            print(f" Ep {ep+1}: Train Acc {t_acc:.3f} | Val Acc {v_acc:.3f}")

    print(f"Final Transfer Accuracy (Best): {best_acc:.4f}")