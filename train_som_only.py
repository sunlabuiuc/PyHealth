# train_som_only.py

from sklearn.datasets import make_classification
import torch
from torch.utils.data import DataLoader, TensorDataset
from pyhealth.models.som_only import SOMOnlyModel
from sklearn.metrics import normalized_mutual_info_score, accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, n_clusters_per_class=1, n_informative=2, random_state=42)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
input_dim = X.shape[1]
n_clusters = 4
model = SOMOnlyModel(input_dim=input_dim, n_clusters=n_clusters)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_X, _ in train_loader:
        assignments = model(batch_X)
        loss = model.loss(batch_X, assignments)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
all_assignments = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in train_loader:
        assignments = model(batch_X)
        all_assignments.extend(assignments.numpy())
        all_labels.extend(batch_y.numpy())

nmi = normalized_mutual_info_score(all_labels, all_assignments)
purity = accuracy_score(all_labels, all_assignments)
print(f"NMI: {nmi:.4f}, Purity: {purity:.4f}")

