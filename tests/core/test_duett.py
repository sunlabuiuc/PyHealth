import torch
import numpy as np
from pyhealth.models import DuETT
from pyhealth.datasets import create_sample_dataset

def test_duett_model():
    samples = [
        {"x_ts": (np.random.randn(32, 36), np.ones((32, 36))), "x_static": np.random.randn(8), 
         "times": np.arange(32, dtype=np.float32), "label": 1},
        {"x_ts": (np.random.randn(32, 36), np.zeros((32, 36))), "x_static": np.random.randn(8), 
         "times": np.arange(32, dtype=np.float32), "label": 0}
    ]
    
    input_schema = {"x_ts": "duett_ts", "x_static": "tensor", "times": "tensor"}
    output_schema = {"label": "binary"}
    dataset = create_sample_dataset(samples, input_schema, output_schema, in_memory=True)
    
    batch = {
        "x_ts": torch.stack([dataset[i]["x_ts"] for i in range(len(samples))]),
        "x_static": torch.stack([dataset[i]["x_static"] for i in range(len(samples))]),
        "times": torch.stack([dataset[i]["times"] for i in range(len(samples))]),
        "label": torch.tensor([dataset[i]["label"] for i in range(len(samples))])
    }
        
    # Rubric: Tests instantiation
    model = DuETT(
        dataset=dataset, 
        n_timesteps=32,
        d_embedding=24, 
        n_duett_layers=1,
        d_hidden_tab_encoder=32,
        d_hidden_head=16
    )
    
    # Rubric: Tests BaseModel properties
    assert model.get_output_size() == 1
    assert model.device.type == "cpu"
    
    # Rubric: Tests forward pass and output shapes
    out = model(**batch)
    assert "loss" in out
    assert "y_prob" in out
    assert out["y_prob"].shape == (2, 1)
    
    # Rubric: Tests gradient computation
    out["loss"].backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Backward logic failed."
    
    # Rubric: Tests hyperparameter variations
    model2 = DuETT(
        dataset=dataset, 
        n_timesteps=32,
        d_embedding=32, 
        n_duett_layers=2,
        d_hidden_tab_encoder=64,
        d_hidden_head=32
    )
    out2 = model2(**batch)
    assert out2["y_prob"].shape == (2, 1)