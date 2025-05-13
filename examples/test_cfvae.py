import numpy as np
import torch
from datetime import datetime, timedelta
from pyhealth.datasets import SampleDataset
from pyhealth.models import CFVAE

# ----- Step 1: Create Dummy Time Series Data -----
samples = []
for i in range(8):  # 8 samples
    # Generate 5 time points (1 hour apart)
    timestamps = [datetime(2024, 1, 1, 0, 0) + timedelta(hours=j) for j in range(5)]
    values = np.random.rand(5, 784).astype(np.float32)  # shape (T=5, F=784)

    samples.append({
        "patient_id": f"p{i}",
        "visit_id": f"v{i}",
        "image": (timestamps, values),   # 👈 format expected by timeseries processor
        "label": np.random.randint(0, 2)
    })

input_schema = {"image": "timeseries"}
output_schema = {"label": "binary"}

# ----- Step 2: Wrap in SampleDataset -----
dataset = SampleDataset(
    samples=samples,
    input_schema=input_schema,
    output_schema=output_schema,
    dataset_name="DummyTimeSeries",
    task_name="cfvae_test"
)

# ----- Step 3: Initialize CFVAE Model -----
model = CFVAE(
    dataset=dataset,
    feat_dim=784,
    emb_dim1=128,
    _mlp_dim1=0, _mlp_dim2=0, _mlp_dim3=0,
    mlp_inpemb=64,
    f_dim1=32,
    f_dim2=16
)

model.eval()

# ----- Step 4: Forward Pass -----
with torch.no_grad():
    batch = [dataset[i] for i in range(4)]
    output = model(batch)

# ----- Step 5: Inspect Output -----
print("Logits shape:", output["logits"].shape)
print("Reconstruction shape:", output["reconstruction"].shape)
print("Mu shape:", output["mu"].shape)
print("Log var shape:", output["log_var"].shape)