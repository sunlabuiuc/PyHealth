"""Synthetic strict conversion and saved-source reconstruction checks.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., MLHC 2025.
ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning.
https://proceedings.mlr.press/v298/sethi25a.html
"""

import copy

import pytest
import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ProtoECGNetH1
from pyhealth.models.protoecgnet_h1_reference import (
    REFERENCE_COMMIT,
    bind_reconstructed_reference_sources,
    convert_reference_checkpoint,
)
from torch import nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 4, 1)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


class TinyBackbone(nn.Module):
    def __init__(self, in_leads, pretrained=False):
        super().__init__()
        assert in_leads == 2 and not pretrained
        self.out_channels = 4
        self.net = TinyNet()

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr("pyhealth.models.protoecgnet.ResNet2dBackbone", TinyBackbone)
    torch.manual_seed(59841)
    dataset = create_sample_dataset(
        [
            {
                "signal": torch.randn(2, 9),
                "labels": labels,
                "ecg_id": 11 + i,
                "patient_id": str(21 + i),
                "strat_fold": i + 1,
            }
            for i, labels in enumerate([["B"], ["A"], []])
        ],
        {"signal": "tensor"},
        {"labels": ("multilabel", {"labels": ["B", "A"]})},
    )
    model = ProtoECGNetH1(
        dataset,
        in_channels=2,
        proto_dim=4,
        prototypes_per_label=2,
        class_weights=torch.tensor([2.0, 3.0]),
        label_cooccurrence=torch.tensor([[1.0, 0.2], [0.2, 1.0]]),
    )
    # Make the saved projected bank from known patches without a search.
    model.set_stage("classifier")
    model.eval()
    batch = next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))
    branch = model.branches["morphology"]
    features = branch.feature_map(batch["signal"])
    with torch.no_grad():
        for i in range(4):
            branch.prototype_vectors[i].copy_(features[i // 2, :, i % 2 : i % 2 + 3])
    state = {
        "prototype_vectors": branch.prototype_vectors.detach().unsqueeze(2).clone(),
        "prototype_class_identity": branch.prototype_class_identity.clone(),
        "label_cooccurrence": model.label_cooccurrence.clone(),
        "ones": torch.ones(4, 4, 1, 3),
        "feature_extractor.resnet.fc.weight": torch.randn(1000, 512),
        "feature_extractor.resnet.fc.bias": torch.randn(1000),
        "classifier.weight": branch.classifier.weight.detach().clone(),
    }
    state.update(
        {
            "feature_extractor.resnet." + k: v.clone()
            for k, v in branch.backbone.net.state_dict().items()
        }
    )
    state.update(
        {
            "add_on_layers." + k: v.unsqueeze(-1).clone()
            if k.endswith("weight")
            else v.clone()
            for k, v in branch.add_on.state_dict().items()
        }
    )
    spec = {
        "reference_commit": REFERENCE_COMMIT,
        "label_names": ["B", "A"],
        "train_ecg_ids": [11, 12, 13],
        "pooling": "top5_mean",
        "model_config": {
            "backbone": "resnet18",
            "num_classes": 2,
            "single_class_prototype_per_class": 2,
            "joint_prototypes_per_border": 0,
            "proto_dim": 4,
            "proto_time_len": 3,
            "latent_space_type": "arc",
            "add_on_layers_type": "linear",
            "last_layer_connection_weight": 1.0,
            "custom_groups": True,
            "label_set": "3",
            "pretrained_weights": None,
        },
        "loss_settings": {
            "lam_clst": 0.004,
            "lam_sep": 0.0004,
            "lam_spars": 0.0,
            "lam_div": 250.0,
            "lam_cnrst": 300.0,
            "use_contrastive": True,
        },
    }
    sources = [
        {
            "prototype_id": i,
            "assigned_label": ["B", "A"][i // 2],
            "source_ecg_id": 11 + i // 2,
            "latent_start": i % 2,
            "latent_end_exclusive": i % 2 + 3,
            "pre_projection_similarity": -0.1 + i,
        }
        for i in range(4)
    ]
    checkpoint = {
        "stage": "h1_refitted",
        "selection_role": "best",
        "completed_refit_epochs": 35,
        "arm": "paired_consistency",
        "run_spec": spec,
        "h1_training_protocol": {
            "run_spec": spec,
            "lambda_by_arm": {"paired_lambda0": 0.0, "paired_consistency": 117608.0},
        },
        "model_state_dict": state,
        "class_weights": model.class_weights.clone(),
        "prototype_class_identity": branch.prototype_class_identity.clone(),
        "projection_sources": sources,
    }
    return model, checkpoint, batch


def source_batch(batch):
    return {k: v[:2] for k, v in batch.items()}


def test_conversion_strict_reload_preserves_outputs_and_does_not_mutate(fixture):
    model, checkpoint, batch = fixture
    before = copy.deepcopy(model.state_dict())
    native, record = convert_reference_checkpoint(model, checkpoint)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, model.state_dict()[name])
    clone = copy.deepcopy(model)
    clone.load_state_dict(native, strict=True)
    clone.eval()
    assert clone.lambda_cons == 117608 and clone.stage == "classifier"
    assert clone.get_projection_provenance() is None
    assert record["projection_sources"] == checkpoint["projection_sources"]
    for key in ("logit", "similarity", "y_prob"):
        assert torch.equal(model(**batch)[key], clone(**batch)[key])
    native["class_weights"][0] = 900
    assert checkpoint["class_weights"][0] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_key",
        "missing_key",
        "shape",
        "dtype",
        "labels",
        "assignment",
        "epoch",
        "lambda",
        "ones",
        "source_id",
        "cooccurrence",
    ],
)
def test_conversion_rejects_incompatible_or_ambiguous_inputs(fixture, kind):
    model, ck, _ = fixture
    ck = copy.deepcopy(ck)
    if kind == "unknown_key":
        ck["model_state_dict"]["unrecognized.weight"] = torch.ones(1)
    if kind == "missing_key":
        del ck["model_state_dict"]["add_on_layers.0.bias"]
    if kind == "shape":
        ck["model_state_dict"]["classifier.weight"] = torch.ones(2, 5)
    if kind == "dtype":
        ck["model_state_dict"]["prototype_vectors"] = ck["model_state_dict"][
            "prototype_vectors"
        ].double()
    if kind == "labels":
        ck["run_spec"]["label_names"] = ["A", "B"]
    if kind == "assignment":
        ck["prototype_class_identity"] = torch.zeros(4, 2)
    if kind == "epoch":
        ck["completed_refit_epochs"] = 34
    if kind == "lambda":
        ck["h1_training_protocol"]["lambda_by_arm"]["paired_consistency"] = 1
    if kind == "ones":
        ck["model_state_dict"]["ones"][0, 0, 0, 0] = 0
    if kind == "source_id":
        ck["projection_sources"][0]["source_ecg_id"] = 99
    if kind == "cooccurrence":
        ck["model_state_dict"]["label_cooccurrence"][0, 0] = 2
    with pytest.raises((ValueError, KeyError)):
        convert_reference_checkpoint(model, ck)


def test_reconstruction_binds_every_saved_source_without_reprojection(
    fixture, tmp_path
):
    model, ck, batch = fixture
    native, _ = convert_reference_checkpoint(model, ck)
    model.load_state_dict(native, strict=True)
    model.eval()
    before = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if isinstance(v, torch.Tensor)
    }
    provenance = bind_reconstructed_reference_sources(
        model,
        ck["projection_sources"],
        [source_batch(batch)],
        source_context={"checkpoint_sha256": "a" * 64},
    )
    assert provenance["covered_prototypes"] == 4
    assert provenance["input_records"] == 2
    assert provenance["source_context"]["patch_max_absolute_error"] == 0
    assert provenance["source_records"][2]["source_ecg_id"] == 12
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])
    path = tmp_path / "native.pt"
    torch.save(model.state_dict(), path)
    restored = copy.deepcopy(model)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert restored.get_projection_provenance() == model.get_projection_provenance()


@pytest.mark.parametrize(
    "kind",
    ["wrong_patch", "wrong_label", "heldout", "duplicate", "missing", "bad_coordinate"],
)
def test_failed_reconstruction_does_not_attach_sources_or_change_bank(fixture, kind):
    model, ck, batch = fixture
    batch = source_batch(batch)
    sources = copy.deepcopy(ck["projection_sources"])
    before = model.branches["morphology"].prototype_vectors.detach().clone()
    if kind == "wrong_patch":
        batch["signal"] = batch["signal"] + 10
    if kind == "wrong_label":
        batch["labels"] = torch.zeros_like(batch["labels"])
    if kind == "heldout":
        batch["strat_fold"] = [10, 2]
    if kind == "duplicate":
        batch["ecg_id"] = [11, 11]
    if kind == "missing":
        batch = {k: v[:1] for k, v in batch.items()}
    if kind == "bad_coordinate":
        sources[0]["latent_start"] = 8
    with pytest.raises(ValueError):
        bind_reconstructed_reference_sources(model, sources, [batch], source_context={})
    assert model.get_projection_provenance() is None
    assert torch.equal(before, model.branches["morphology"].prototype_vectors)
    assert not model.training


def test_source_fold_guard_precedes_feature_forward(fixture, monkeypatch):
    model, ck, batch = fixture
    batch = source_batch(batch)
    batch["strat_fold"] = [10, 2]
    calls = []
    monkeypatch.setattr(
        model.branches["morphology"], "feature_map", lambda x: calls.append(x)
    )
    with pytest.raises(ValueError, match="outside training folds"):
        bind_reconstructed_reference_sources(
            model, ck["projection_sources"], [batch], source_context={}
        )
    assert calls == []


def test_import_rebuilds_label_derived_routing_metadata(fixture):
    model, ck, _ = fixture
    model.index_morphology.zero_()
    model.fusion_mask.zero_()
    native, _ = convert_reference_checkpoint(model, ck)
    assert torch.equal(native["index_morphology"], torch.arange(2))
    assert torch.equal(native["fusion_mask"], ck["prototype_class_identity"].t())
