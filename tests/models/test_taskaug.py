import torch

from pyhealth.datasets import PTBXLDataset
from pyhealth.models import TaskAug
from pyhealth.models.taskaug import BasicBlock1D


def test_basic_block_1d_identity_and_downsample():
    identity_block = BasicBlock1D(in_channels=8, out_channels=8, stride=1)
    x = torch.randn(2, 8, 64)
    out_identity = identity_block(x)
    assert out_identity.shape == (2, 8, 64)

    downsample_block = BasicBlock1D(in_channels=8, out_channels=16, stride=2)
    out_down = downsample_block(x)
    assert out_down.shape == (2, 16, 32)


def test_taskaug_multilabel_train_and_eval():
    dataset = PTBXLDataset(
        root="tests/data/ptbxl_synthetic",
        cache_dir="tests/data/my_clean_cache",
    )
    samples = dataset.set_task(dataset.default_task)

    model = TaskAug(
        dataset=samples,
        feature_keys=["signal"],
        label_key="labels",
        mode="multilabel",
        num_stages=2,
    )

    model.train()
    train_batch = {
        "signal": [torch.randn(12, 1000), torch.randn(12, 1000)],
        "labels": [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])],
    }
    train_out = model(**train_batch)
    assert "loss" in train_out and train_out["loss"] is not None
    assert train_out["loss"].requires_grad
    assert train_out["y_prob"].shape == (2, 3)

    model.eval()
    eval_batch = {
        "signal": torch.randn(2, 12, 1000),
        "labels": torch.tensor([[0, 1, 0], [1, 0, 1]]),
    }
    eval_out = model(**eval_batch)
    assert eval_out["y_prob"].shape == (2, 3)


def test_taskaug_multiclass_mode():
    dataset = PTBXLDataset(
        root="tests/data/ptbxl_synthetic",
        cache_dir="tests/data/my_clean_cache",
    )
    samples = dataset.set_task(dataset.default_task)

    model = TaskAug(
        dataset=samples,
        feature_keys=["signal"],
        label_key="labels",
        mode="multiclass",
        num_stages=1,
    )
    model.fc = torch.nn.Linear(512, 3)

    batch = {
        "signal": torch.randn(2, 12, 1000),
        "labels": torch.tensor([1, 2]),
    }
    out = model(**batch)
    assert out["loss"] is not None
    assert out["loss"].requires_grad
    assert out["y_prob"].shape == (2, 3)


if __name__ == "__main__":
    test_basic_block_1d_identity_and_downsample()
    test_taskaug_multilabel_train_and_eval()
    test_taskaug_multiclass_mode()
    print("All TaskAug tests passed successfully!")