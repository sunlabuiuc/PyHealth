import torch
from pyhealth.models import LabTOPTransformer
from pyhealth.tasks import LabTOPNextTokenTask

def test_forward():
    """
    Test that LabTOPTransformer forward pass returns correct shape
    """
    model = LabTOPTransformer(vocab=50)
    x = torch.randint(0, 50, (2, 32))    # B=2, L=32
    out = model(x)

    # Expected shape: (batch, sequence_length, vocab_size)
    assert out.shape == (2, 32, 50)


def test_task_loss():
    """
    Test that task loss runs without crashing
    """
    task = LabTOPNextTokenTask()

    logits = torch.randn(2, 32, 50)
    labels = torch.randint(0, 50, (2, 32))

    loss = task.get_loss(logits, labels)

    assert loss.item() > 0
