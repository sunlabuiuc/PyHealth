import torch
from pyhealth.datasets import PTBXLDataset
from pyhealth.models import TaskAug

def test_taskaug():
    dataset = PTBXLDataset(
        root="tests/data/ptbxl_synthetic",
        cache_dir="tests/data/my_clean_cache"
    )
    samples = dataset.set_task(dataset.default_task)
    
    model = TaskAug(
        dataset=samples,
        feature_keys=["signal"],
        label_key="labels",
        mode="multilabel",
        num_stages=2
    )
    
    dummy_batch = {
        "signal": torch.randn(2, 12, 1000),
        "labels": torch.tensor([[0, 1, 0], [1, 0, 1]])
    }
    
    output = model(**dummy_batch)
    
    assert "loss" in output, "Model output must contain 'loss'"
    assert "y_prob" in output, "Model output must contain 'y_prob'"
    
    assert output["y_prob"].shape == (2, 3), f"Expected shape (2, 3), got {output['y_prob'].shape}"
    
    assert output["loss"].requires_grad, "Loss tensor must require gradients for training"

if __name__ == "__main__":
    test_taskaug()
    print("All TaskAug tests passed successfully!")