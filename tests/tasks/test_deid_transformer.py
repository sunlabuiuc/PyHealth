from pyhealth.tasks import DeidTransformer
import torch

def test_forward_shape():
    task = DeidTransformer(model_name="bert-base-uncased", num_labels=2)
    out = task(
        input_ids=torch.zeros((1, 8), dtype=torch.long),
        attention_mask=torch.ones((1, 8), dtype=torch.long),
        labels=torch.zeros((1, 8), dtype=torch.long),
    )
    assert out["logits"].shape == (1, 8, 2)
