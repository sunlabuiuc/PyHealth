"""Synthetic H1 model/Trainer checks, using four samples and a tiny backbone."""

import copy

import pytest
import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel, ProtoECGNet, ProtoECGNetH1
from pyhealth.models.protoecgnet_h1 import MorphologyPrototypeBranch
from pyhealth.trainer import Trainer
from torch import nn


class TinyBNBackbone(nn.Module):
    """Exercise BN/dropout policies without a full ResNet in unit tests."""

    def __init__(self, in_leads: int, pretrained: bool = False) -> None:
        super().__init__()
        assert pretrained is False
        self.out_channels = 4
        self.conv = nn.Conv1d(in_leads, 4, 1)
        self.bn = nn.BatchNorm1d(4)
        self.buffer_bn = nn.BatchNorm1d(4, affine=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.buffer_bn(self.bn(self.conv(x))))


@pytest.fixture(scope="module", autouse=True)
def small_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset():
    generator = torch.Generator().manual_seed(598)
    samples = []
    for index, labels in enumerate((["B"], ["A"], ["A", "B"], [])):
        clean = torch.randn(2, 9, generator=generator)
        samples.append(
            {
                "patient_id": f"patient_{index}",
                "ecg_id": 100 + index,
                "signal": clean,
                "labels": labels,
                "noisy_signal": clean + 0.01 * torch.randn(2, 9, generator=generator),
            }
        )
    return create_sample_dataset(
        samples,
        input_schema={"signal": "tensor"},
        output_schema={"labels": ("multilabel", {"labels": ["B", "A"]})},
    )


@pytest.fixture
def model_factory(monkeypatch, dataset):
    from pyhealth.models import protoecgnet

    monkeypatch.setattr(protoecgnet, "ResNet2dBackbone", TinyBNBackbone)

    def make(**kwargs):
        options = dict(
            dataset=dataset,
            in_channels=2,
            proto_dim=4,
            prototypes_per_label=2,
            class_weights=torch.tensor([2.0, 3.0]),
            label_cooccurrence=torch.tensor([[1.0, 0.2], [0.2, 1.0]]),
        )
        options.update(kwargs)
        with torch.random.fork_rng():
            torch.manual_seed(681)
            return ProtoECGNetH1(**options)

    return make


@pytest.fixture
def batch(dataset):
    return next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))


def tensor_state(model):
    return {
        key: value.clone()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def test_public_export_inheritance_order_and_all_zero_targets(model_factory, batch):
    model = model_factory().eval()
    assert isinstance(model, (BaseModel, ProtoECGNet))
    assert isinstance(model.branches["morphology"], MorphologyPrototypeBranch)
    assert model.label_names == ["B", "A"]
    assert model.branch_index("morphology").tolist() == [0, 1]
    result = model(**batch)
    assert result["logit"].shape == (4, 2)
    assert result["similarity"].shape == result["best_window"].shape == (4, 4)
    assert result["y_true"][3].sum().item() == 0
    assert result["y_true"].shape[0] == 4


def test_processor_indices_define_order_not_dict_insertion(model_factory, dataset):
    dataset.output_processors["labels"].label_vocab = {"A": 1, "B": 0}
    model = model_factory()
    assert model.label_names == ["B", "A"]
    assert model.branch_index("morphology").tolist() == [0, 1]


def test_clean_three_and_four_dimensional_inputs_agree(model_factory, batch):
    model = model_factory().eval()
    x = batch["signal"]
    first = model(signal=x)
    second = model(signal=x.unsqueeze(1))
    for key in ("logit", "y_prob", "similarity", "best_window"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_joint_combines_views_once_and_preserves_mode_policy(model_factory, batch):
    model = model_factory(lambda_cons=117608.0)
    branch = model.branches["morphology"]
    seen = []
    hook = branch.backbone.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].clone())
    )
    model.train()
    assert branch.backbone.bn.training
    assert branch.backbone.buffer_bn.training
    assert not branch.backbone.dropout.training
    assert not branch.classifier.weight.requires_grad
    before = branch.backbone.bn.num_batches_tracked.clone()
    result = model(**batch)
    hook.remove()
    assert len(seen) == 1
    torch.testing.assert_close(
        seen[0], torch.cat([batch["signal"], batch["noisy_signal"]])
    )
    assert branch.backbone.bn.num_batches_tracked.item() == before.item() + 1
    result["loss"].backward()
    assert branch.prototype_vectors.grad.abs().sum() > 0
    assert branch.backbone.conv.weight.grad.abs().sum() > 0
    assert branch.classifier.weight.grad is None


@pytest.mark.parametrize("coefficient", [0.0, 117608.0])
def test_model_loss_and_gradients_match_combined_reference(
    model_factory, batch, coefficient
):
    model = model_factory(lambda_cons=coefficient)
    reference = copy.deepcopy(model)
    actual = model(**batch)
    branch = reference.branches["morphology"]
    logits, scores, _ = branch(torch.cat([batch["signal"], batch["noisy_signal"]]))
    clean = reference._original(logits[:4], scores[:4], batch["labels"])["loss"]
    noisy = reference._original(logits[4:], scores[4:], batch["labels"])["loss"]
    expected = (
        0.5 * (clean + noisy) + coefficient * (scores[:4] - scores[4:]).square().mean()
    )
    torch.testing.assert_close(actual["loss"], expected)
    first = [p for p in model.parameters() if p.requires_grad]
    second = [p for p in reference.parameters() if p.requires_grad]
    for a, b in zip(
        torch.autograd.grad(actual["loss"], first),
        torch.autograd.grad(expected, second),
    ):
        torch.testing.assert_close(a, b)


def test_consistency_keeps_both_input_and_prototype_gradients(model_factory, batch):
    model = model_factory(lambda_cons=117608.0)
    clean = batch["signal"].clone().requires_grad_()
    noisy = batch["noisy_signal"].clone().requires_grad_()
    result = model(signal=clean, noisy_signal=noisy, labels=batch["labels"])
    gradients = torch.autograd.grad(
        result["weighted_consistency_loss"],
        (
            clean,
            noisy,
            model.branches["morphology"].prototype_vectors,
        ),
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )


def test_identical_views_have_zero_consistency(model_factory, batch):
    model = model_factory(lambda_cons=117608.0)
    result = model(
        signal=batch["signal"], noisy_signal=batch["signal"], labels=batch["labels"]
    )
    assert result["consistency_loss"].item() == 0


def test_classifier_stage_clears_stale_gradients_and_freezes_buffers(
    model_factory, batch
):
    model = model_factory()
    model(**batch)["loss"].backward()
    model.set_stage("classifier")
    model.train()
    branch = model.branches["morphology"]
    assert not branch.backbone.bn.training
    assert not branch.backbone.buffer_bn.training
    assert branch.backbone.conv.weight.grad is None
    before = tensor_state(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
    result = model(signal=batch["signal"], labels=batch["labels"])
    torch.testing.assert_close(
        result["loss"], result["classification_loss"] + model.lam_l1 * result["l1_loss"]
    )
    result["loss"].backward()
    optimizer.step()
    after = tensor_state(model)
    assert not torch.equal(
        before["branches.morphology.classifier.weight"],
        after["branches.morphology.classifier.weight"],
    )
    for name in before:
        if name != "branches.morphology.classifier.weight":
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_strict_checkpoint_restores_coefficients_stage_and_statistics(
    model_factory, batch, tmp_path
):
    model = model_factory(lambda_cons=117608.0, lam_spars=0.001).eval()
    model.set_stage("classifier")
    expected = model(signal=batch["signal"], labels=batch["labels"])
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = model_factory(lambda_cons=0.0, lam_spars=0.0).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert restored.stage == "classifier"
    assert restored.lambda_cons == 117608.0
    assert restored.lam_spars == 0.001
    for name, value in tensor_state(model).items():
        torch.testing.assert_close(value, tensor_state(restored)[name], rtol=0, atol=0)
    actual = restored(signal=batch["signal"], labels=batch["labels"])
    for key in ("logit", "similarity", "loss"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    restored.train()
    assert not restored.branches["morphology"].backbone.buffer_bn.training


@pytest.mark.parametrize("field", ["label_names", "architecture", "h1_format_version"])
def test_checkpoint_rejects_incompatible_metadata(model_factory, field):
    state = copy.deepcopy(model_factory().state_dict())
    metadata = state["_extra_state"]
    if field == "label_names":
        metadata[field] = ["A", "B"]
    elif field == "architecture":
        metadata[field]["topk"] = 1
    else:
        metadata[field] = 999
    with pytest.raises(ValueError):
        model_factory().load_state_dict(state)


def test_zero_cooccurrence_enabled_flag_survives_checkpoint(model_factory):
    model = model_factory(label_cooccurrence=torch.zeros(2, 2))
    restored = model_factory(label_cooccurrence=None)
    assert not restored._contrastive_enabled
    restored.load_state_dict(model.state_dict())
    assert restored._contrastive_enabled


def test_inherited_contributions_and_projection_work_on_image_inputs(
    model_factory, dataset, batch
):
    model = model_factory().eval()
    x = batch["signal"].unsqueeze(1)
    contributions = model.prototype_contributions(x)["morphology"]
    torch.testing.assert_close(contributions.sum(-1), model(signal=x)["logit"])
    sources = model.project_prototypes(
        get_dataloader(dataset, batch_size=4, shuffle=False)
    )
    assert len(sources["morphology"]) == 4
    for index, entry in enumerate(sources["morphology"]):
        assert entry is not None
        assert batch["labels"][entry["sample"], index // 2] == 1


def test_joint_requires_a_pair_even_at_lambda_zero(model_factory, batch):
    with pytest.raises(ValueError, match="requires noisy_signal"):
        model_factory(lambda_cons=0.0)(signal=batch["signal"], labels=batch["labels"])


def test_classifier_rejects_paired_input(model_factory, batch):
    model = model_factory()
    model.set_stage("classifier")
    with pytest.raises(ValueError, match="only in the joint stage"):
        model(**batch)


@pytest.mark.parametrize("stage", ["all", "fusion", "prototypes", "feature_extractor"])
def test_unsupported_stage_is_rejected(model_factory, stage):
    with pytest.raises(ValueError, match="stage must be"):
        model_factory().set_stage(stage)


@pytest.mark.parametrize(
    "weights",
    [torch.ones(3), torch.tensor([0.0, 1.0]), torch.tensor([float("nan"), 1.0])],
)
def test_invalid_label_weights_are_rejected(model_factory, weights):
    with pytest.raises(ValueError):
        model_factory(class_weights=weights)


def test_real_trainer_joint_refit_and_checkpoint(
    model_factory, dataset, batch, tmp_path
):
    """Two synthetic optimizer steps through the actual PyHealth Trainer."""
    model = model_factory(lambda_cons=117608.0)
    trainer = Trainer(model=model, device="cpu", enable_logging=False)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    classifier = model.branches["morphology"].classifier.weight.detach().clone()
    trainer.train(
        loader,
        epochs=1,
        steps_per_epoch=1,
        optimizer_params={"lr": 1e-4},
        load_best_model_at_last=False,
    )
    torch.testing.assert_close(
        model.branches["morphology"].classifier.weight, classifier, rtol=0, atol=0
    )
    model.set_stage("classifier")
    frozen = tensor_state(model)
    clean_records = [
        {key: value for key, value in dataset[index].items() if key != "noisy_signal"}
        for index in range(len(dataset))
    ]
    clean_loader = torch.utils.data.DataLoader(clean_records, batch_size=4)
    trainer.train(
        clean_loader,
        epochs=1,
        steps_per_epoch=1,
        optimizer_params={"lr": 1e-4},
        load_best_model_at_last=False,
    )
    for name, value in frozen.items():
        if name != "branches.morphology.classifier.weight":
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    ckpt = tmp_path / "trainer.ckpt"
    trainer.save_ckpt(str(ckpt))
    restored = model_factory()
    restored_trainer = Trainer(model=restored, device="cpu", enable_logging=False)
    restored_trainer.load_ckpt(str(ckpt))
    restored.eval()
    model.eval()
    torch.testing.assert_close(
        restored(signal=batch["signal"])["logit"],
        model(signal=batch["signal"])["logit"],
        rtol=0,
        atol=0,
    )
    truth, probabilities, loss = restored_trainer.inference(clean_loader)
    assert truth.shape == probabilities.shape == (4, 2)
    assert torch.isfinite(torch.tensor(loss))
