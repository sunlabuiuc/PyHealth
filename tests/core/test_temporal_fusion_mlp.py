import torch

from pyhealth.models.temporal_fusion_mlp import TemporalFusionMLP


class DummyDataset:
    def __init__(self):
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
            "drugs": "sequence",
            "admission_year": "tensor",
        }
        self.output_schema = {"mortality": "binary"}


def make_batch():
    return {
        "conditions": [["A", "B"], ["C"]],
        "procedures": [["P1"], ["P2", "P3"]],
        "drugs": [["D1"], ["D2"]],
        "admission_year": [[0.5], [0.8]],
        "mortality": [0, 1],
    }


def test_temporal_fusion_mlp_instantiation():
    model = TemporalFusionMLP(
        dataset=DummyDataset(),
        feature_keys=["conditions", "procedures", "drugs", "admission_year"],
        label_key="mortality",
    )
    assert model.input_dim == 4


def test_temporal_fusion_mlp_forward_shape():
    model = TemporalFusionMLP(
        dataset=DummyDataset(),
        feature_keys=["conditions", "procedures", "drugs", "admission_year"],
        label_key="mortality",
    )
    out = model(**make_batch())
    assert out["logit"].shape == (2,)


def test_temporal_fusion_mlp_gradient():
    model = TemporalFusionMLP(
        dataset=DummyDataset(),
        feature_keys=["conditions", "procedures", "drugs", "admission_year"],
        label_key="mortality",
    )
    out = model(**make_batch())
    out["loss"].backward()