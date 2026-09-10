import pytest
import torch
from PIL import Image

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Graph_TorchvisionModel
from pyhealth.processors import ImageProcessor, MultiClassLabelProcessor
from pyhealth.sampler.sage_sampler import EdgeIndex


@pytest.mark.parametrize(
    "processor,labels,output_size,mode",
    [
        ("binary", [0, 1], 1, "L"),
        (MultiClassLabelProcessor, ["healthy", "ill"], 2, "RGB"),
        ("multilabel", [["a"], ["b"]], 2, "L"),
    ],
)
def test_graph_torchvision_processed_batch(tmp_path, processor, labels, output_size, mode):
    path = tmp_path / "scan.png"
    Image.new(mode, (32, 32), color=128).save(path)
    dataset = create_sample_dataset(
        samples=[{"scan": str(path), "target": label} for label in labels],
        input_schema={"scan": ImageProcessor(image_size=32, mode=mode)},
        output_schema={"target": processor},
    )
    model = Graph_TorchvisionModel(
        dataset,
        model_name="resnet18",
        model_config={"weights": None},
        gnn_config={"input_dim": 8, "hidden_dim": 4},
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2)))
    adjacency = EdgeIndex(torch.tensor([[0, 1], [0, 1]]), None, (2, 2))
    # Keep the classifier deterministic and away from inactive ReLUs.
    model.eval()
    with torch.no_grad():
        model.gnn.gc1.bias.fill_(1)
        model.gnn.gc2.weight.fill_(0.1)
        model.gnn.gc2.bias.fill_(0.2)
    result = model(**batch, adjacencies=[adjacency, adjacency])

    assert model.feature_keys == ["scan"]
    assert model.label_keys == ["target"]
    assert result["y_prob"].shape == (2, output_size)
    torch.testing.assert_close(result["y_true"], batch["target"])
    torch.testing.assert_close(
        result["y_prob"], model.prepare_y_prob(result["logit"])
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for parameter in (model.model.fc.weight, model.gnn.gc2.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    if processor == "binary":
        assert not torch.all(result["y_prob"] == 0.5)
        assert model.gnn.gc2.weight.grad.abs().sum() > 0

    model.train()
    model.zero_grad()
    loss = model(**batch, adjacencies=[adjacency, adjacency])["loss"]
    assert torch.isfinite(loss)
    loss.backward()
