"""Synthetic bank/source checkpoint integrity and standard projection checks.

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
from pyhealth.models import ProtoECGNet, ProtoECGNetH1
from pyhealth.models.protoecgnet_h1_checkpoint import (
    bank_state_digest,
    validate_projection_provenance,
)
from pyhealth.trainer import Trainer
from torch import nn


class TinyBankBackbone(nn.Module):
    def __init__(self, in_leads, pretrained=False):
        super().__init__()
        assert not pretrained
        self.out_channels = 4
        self.conv = nn.Conv1d(in_leads, 4, 1)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


@pytest.fixture(scope="module", autouse=True)
def small_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def factory(monkeypatch):
    from pyhealth.models import protoecgnet

    monkeypatch.setattr(protoecgnet, "ResNet2dBackbone", TinyBankBackbone)

    def make(seed=412, missing_label=False):
        generator = torch.Generator().manual_seed(seed)
        samples = [
            {
                "patient_id": str(101 + index),
                "ecg_id": [17, 29, 41][index],
                "strat_fold": [1, 2, 8][index],
                "signal": torch.randn(2, 9, generator=generator),
                "labels": ["B"] if missing_label else [["B"], ["A"], ["B", "A"]][index],
            }
            for index in range(3)
        ]
        dataset = create_sample_dataset(
            samples,
            {"signal": "tensor"},
            {"labels": ("multilabel", {"labels": ["B", "A"]})},
        )
        with torch.random.fork_rng():
            torch.manual_seed(681)
            model = ProtoECGNetH1(
                dataset,
                in_channels=2,
                proto_dim=4,
                prototypes_per_label=2,
                lambda_cons=117608.0,
            )
        return model, dataset

    return make


def project(model, dataset):
    return model.project_prototypes(
        get_dataloader(dataset, batch_size=3, shuffle=False),
        source_context={"protocol_sha256": "a" * 64, "preprocessing": "synthetic"},
    )


def tensor_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def test_unprojected_checkpoint_has_no_sources(factory):
    model, _ = factory()
    assert model.get_projection_provenance() is None
    assert model.get_extra_state()["projection_provenance"] is None


def test_native_projection_matches_existing_algorithm_exactly(factory):
    model, dataset = factory()
    baseline = copy.deepcopy(model)
    expected = ProtoECGNet.project_prototypes(
        baseline, get_dataloader(dataset, batch_size=3, shuffle=False)
    )
    actual = project(model, dataset)
    torch.testing.assert_close(
        model.branches["morphology"].prototype_vectors,
        baseline.branches["morphology"].prototype_vectors,
        rtol=0,
        atol=0,
    )
    for actual_row, expected_row in zip(actual["morphology"], expected["morphology"]):
        for key in ("sample", "window", "similarity"):
            assert actual_row[key] == expected_row[key]


def test_sources_bind_ids_folds_labels_windows_and_reconstruct_patches(factory):
    model, dataset = factory()
    sources = project(model, dataset)["morphology"]
    assert model.training
    model.eval()
    branch = model.branches["morphology"]
    for index, source in enumerate(sources):
        assert source["prototype_id"] == index
        assert source["assigned_label"] == ["B", "A"][index // 2]
        assert source["source_ecg_id"] == dataset[source["sample"]]["ecg_id"]
        assert source["source_patient_id"] == dataset[source["sample"]]["patient_id"]
        assert source["source_fold"] in range(1, 9)
        assert len(source["source_signal_sha256"]) == 64
        assert source["latent_num_windows"] == 7
        features = branch.feature_map(dataset[source["sample"]]["signal"].unsqueeze(0))
        patch = features[0, :, source["latent_start"] : source["latent_end_exclusive"]]
        torch.testing.assert_close(
            branch.prototype_vectors[index], patch, rtol=1e-5, atol=1e-6
        )
    binding = model.get_projection_provenance()
    assert binding["covered_prototypes"] == 4
    assert binding["input_records"] == 3
    assert binding["source_context"]["protocol_sha256"] == "a" * 64
    assert binding["bank_state_sha256"] == bank_state_digest(
        model.state_dict(), model.label_names, model._architecture()
    )


def test_projection_returns_copies_not_mutable_internal_source_claims(factory):
    model, dataset = factory()
    returned = project(model, dataset)
    returned["morphology"][0]["source_ecg_id"] = 999
    first = model.get_projection_provenance()
    first["source_records"][0]["source_ecg_id"] = 888
    assert model.get_projection_provenance()["source_records"][0][
        "source_ecg_id"
    ] not in (888, 999)


def test_uncovered_prototypes_are_explicitly_none(factory):
    model, dataset = factory(missing_label=True)
    sources = project(model, dataset)["morphology"]
    assert sources[0] is not None and sources[1] is not None
    assert sources[2:] == [None, None]
    assert model.get_projection_provenance()["covered_prototypes"] == 2


def test_strict_weights_only_checkpoint_preserves_bound_sources(factory, tmp_path):
    model, dataset = factory()
    project(model, dataset)
    model.set_stage("classifier")
    model.eval()
    expected = model(signal=dataset[0]["signal"].unsqueeze(0))["logit"]
    path = tmp_path / "bound.pt"
    torch.save(model.state_dict(), path)
    restored, _ = factory()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.eval()
    assert restored.get_projection_provenance() == model.get_projection_provenance()
    assert restored.projection_info == model.projection_info
    torch.testing.assert_close(
        restored(signal=dataset[0]["signal"].unsqueeze(0))["logit"],
        expected,
        rtol=0,
        atol=0,
    )


def test_actual_trainer_checkpoint_preserves_source_binding(factory, tmp_path):
    model, dataset = factory()
    project(model, dataset)
    path = tmp_path / "trainer.ckpt"
    Trainer(model, device="cpu", enable_logging=False).save_ckpt(str(path))
    restored, _ = factory()
    Trainer(restored, device="cpu", enable_logging=False).load_ckpt(str(path))
    assert restored.get_projection_provenance() == model.get_projection_provenance()


def test_classifier_refit_changes_head_and_keeps_binding_valid(factory):
    model, dataset = factory()
    project(model, dataset)
    expected = model.get_projection_provenance()
    model.set_stage("classifier")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    batch = next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))
    before = model.branches["morphology"].classifier.weight.detach().clone()
    model(**batch)["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, model.branches["morphology"].classifier.weight)
    assert model.get_projection_provenance() == expected
    assert model.state_dict()["_extra_state"]["projection_provenance"] == expected


@pytest.mark.parametrize(
    "change",
    ["prototype", "feature_weight", "bn_buffer", "assignment", "epsilon", "topk"],
)
def test_stale_bank_feature_or_math_state_cannot_export_sources(factory, change):
    model, dataset = factory()
    project(model, dataset)
    branch = model.branches["morphology"]
    with torch.no_grad():
        if change == "prototype":
            branch.prototype_vectors[0, 0, 0] += 0.1
        elif change == "feature_weight":
            branch.backbone.conv.weight[0, 0, 0] += 0.1
        elif change == "bn_buffer":
            branch.backbone.bn.running_mean[0] += 0.1
        elif change == "assignment":
            branch.prototype_class_identity[0] = branch.prototype_class_identity[2]
        elif change == "epsilon":
            branch.epsilon = 0.01
        else:
            branch.topk = 1
    with pytest.raises(ValueError):
        model.get_projection_provenance()
    with pytest.raises(ValueError):
        model.state_dict()
    with pytest.raises(ValueError):
        model.prototype_contributions(dataset[0]["signal"].unsqueeze(0))


def test_mismatched_bank_binding_is_rejected_before_tensor_copy(factory):
    first, first_data = factory(seed=1)
    second, second_data = factory(seed=2)
    project(first, first_data)
    project(second, second_data)
    state = copy.deepcopy(first.state_dict())
    state["_extra_state"]["projection_provenance"] = second.get_projection_provenance()
    target, _ = factory(seed=3)
    before = tensor_state(target)
    with pytest.raises(ValueError, match="stale|another bank"):
        target.load_state_dict(state)
    for name, value in before.items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change", ["source_id", "window", "assigned_label", "coverage", "context"]
)
def test_source_bundle_edits_are_rejected(factory, change):
    model, dataset = factory()
    project(model, dataset)
    state = copy.deepcopy(model.state_dict())
    binding = state["_extra_state"]["projection_provenance"]
    record = binding["source_records"][0]
    if change == "source_id":
        record["source_ecg_id"] = 999
    elif change == "window":
        record["latent_start"] = 999
    elif change == "assigned_label":
        record["assigned_label"] = "A"
    elif change == "coverage":
        binding["covered_prototypes"] = 0
    else:
        binding["source_context"]["preprocessing"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        model.load_state_dict(state)


def test_legacy_unbound_checkpoint_clears_existing_sources(factory):
    target, dataset = factory()
    project(target, dataset)
    unbound, _ = factory()
    state = copy.deepcopy(unbound.state_dict())
    state["_extra_state"].pop("projection_provenance")
    target.load_state_dict(state)
    assert target.get_projection_provenance() is None
    assert target.projection_info is None


def test_recursive_parent_module_load_also_validates_sources(factory):
    model, dataset = factory()
    project(model, dataset)
    parent = nn.ModuleDict({"model": model})
    state = copy.deepcopy(parent.state_dict())
    state["model.branches.morphology.prototype_vectors"][0, 0, 0] += 0.1
    restored, _ = factory()
    with pytest.raises(ValueError, match="stale|another bank"):
        nn.ModuleDict({"model": restored}).load_state_dict(state)


def test_bound_partial_load_requires_explicit_clear(factory):
    model, dataset = factory()
    project(model, dataset)
    with pytest.raises(ValueError, match="strict"):
        model.load_state_dict(model.state_dict(), strict=False)
    before = model.branches["morphology"].prototype_vectors.detach().clone()
    model.clear_projection_provenance()
    assert model.get_projection_provenance() is None
    assert model.projection_info is None
    torch.testing.assert_close(
        model.branches["morphology"].prototype_vectors, before, rtol=0, atol=0
    )
    model.load_state_dict(model.state_dict(), strict=False)


@pytest.mark.parametrize(
    "invalid", ["fold9", "fold10", "missing_ids", "unsorted", "duplicates", "empty"]
)
def test_invalid_projection_rolls_back_bank_sources_and_training_mode(factory, invalid):
    model, dataset = factory()
    project(model, dataset)
    before = model.get_projection_provenance()
    before_bank = model.branches["morphology"].prototype_vectors.detach().clone()
    batch = next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))
    if invalid == "fold9":
        batch["strat_fold"] = [1, 2, 9]
    elif invalid == "fold10":
        batch["strat_fold"] = [1, 2, 10]
    elif invalid == "missing_ids":
        del batch["ecg_id"]
    elif invalid == "unsorted":
        batch["ecg_id"] = [17, 41, 29]
    elif invalid == "duplicates":
        batch["ecg_id"] = [17, 17, 41]
    was_training = model.training
    with pytest.raises(ValueError):
        model.project_prototypes([] if invalid == "empty" else [batch])
    assert model.training == was_training
    torch.testing.assert_close(
        model.branches["morphology"].prototype_vectors, before_bank, rtol=0, atol=0
    )
    assert model.get_projection_provenance() == before


def test_source_schema_validator_is_callable_on_saved_state(factory):
    model, dataset = factory()
    project(model, dataset)
    state = model.state_dict()
    validate_projection_provenance(
        state["_extra_state"]["projection_provenance"],
        state,
        model.label_names,
        model._architecture(),
    )


def test_exact_ties_use_first_eligible_ecg_and_first_latent_window(factory):
    model, dataset = factory()
    branch = model.branches["morphology"]
    # Force every latent feature and prototype to 0.5, producing exact ties.
    with torch.no_grad():
        branch.add_on[0].weight.zero_()
        branch.add_on[0].bias.zero_()
        branch.add_on[2].weight.zero_()
        branch.add_on[2].bias.zero_()
        branch.prototype_vectors.fill_(0.5)
    records = project(model, dataset)["morphology"]
    assert [record["source_ecg_id"] for record in records] == [17, 17, 29, 29]
    assert all(record["latent_start"] == 0 for record in records)


def test_reprojection_refreshes_a_stale_source_binding(factory):
    model, dataset = factory()
    project(model, dataset)
    with torch.no_grad():
        model.branches["morphology"].prototype_vectors[0] += 0.01
    with pytest.raises(ValueError, match="stale"):
        model.get_projection_provenance()
    project(model, dataset)
    assert model.get_projection_provenance()["covered_prototypes"] == 4
