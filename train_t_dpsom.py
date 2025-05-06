import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.datasets import make_classification
import numpy as np
from pyhealth.models.t_dpsom import TDPSOM

# Generate synthetic sequential data
def generate_synthetic_sequence_data(n_samples=500, seq_len=10, input_dim=5, n_classes=5):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=input_dim,
        n_informative=min(input_dim, 3),
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,  # ✅ Fix here
        random_state=42
    )
    X_seq = np.repeat(X[:, np.newaxis, :], seq_len, axis=1)
    return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Hyperparameters
input_dim = 5
seq_len = 10
hidden_dim = 16
n_prototypes = 5
batch_size = 32
epochs = 10

# Dataset
X, y = generate_synthetic_sequence_data()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = TDPSOM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    n_prototypes=n_prototypes
)


optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, _ in loader:
        assignments = model(batch_x)
        loss = model.loss(batch_x, assignments)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# Evaluation
model.eval()
all_assignments = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in loader:
        assignments = model(batch_x)
        all_assignments.append(assignments)
        all_labels.append(batch_y)

all_assignments = torch.cat(all_assignments).numpy()
all_labels = torch.cat(all_labels).numpy()

nmi = normalized_mutual_info_score(all_labels, all_assignments)
purity = accuracy_score(all_labels, all_assignments)

print(f"NMI: {nmi:.4f}, Purity: {purity:.4f}")

