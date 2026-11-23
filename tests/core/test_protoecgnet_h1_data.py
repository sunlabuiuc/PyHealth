"""Synthetic ordered-data and ID/epoch noise checks; no real PTB-XL records."""

import copy

import numpy as np
import pytest
import torch
from pyhealth.datasets import get_dataloader
from pyhealth.models import ProtoECGNetH1
from pyhealth.models.protoecgnet_h1_data import (
    PTBXL_MORPHOLOGY_LABEL_ORDER,
    PairedECGLoader,
    create_h1_ptbxl_dataset,
    make_ecg_noisy_view,
)
from pyhealth.trainer import Trainer
from torch import nn


@pytest.fixture(scope="module", autouse=True)
def small_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tensors():
    generator = torch.Generator().manual_seed(4598)
    signals = torch.randn(3, 12, 1000, generator=generator)
    signals[:, 0] = 2.0
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    return signals, targets


def dataset_for(tensors, split="train", in_memory=True):
    signals, targets = tensors
    fold = {"train": 1, "validation": 9, "test": 10}[split]
    return create_h1_ptbxl_dataset(
        signals,
        targets,
        ecg_ids=[17, 29, 41],
        patient_ids=[101, 101, 102],
        folds=[fold] * 3,
        split=split,
        label_names=["B", "A"],
        in_memory=in_memory,
    )


def test_category_three_order_is_explicit():
    assert len(PTBXL_MORPHOLOGY_LABEL_ORDER) == 52
    assert len(set(PTBXL_MORPHOLOGY_LABEL_ORDER)) == 52
    assert PTBXL_MORPHOLOGY_LABEL_ORDER[:5] == ("NDT", "NST_", "LNGQT", "IMI", "ASMI")
    assert PTBXL_MORPHOLOGY_LABEL_ORDER[-1] == "PRC(S)"


def test_native_dataset_preserves_order_zero_targets_ids_and_waveforms(tensors):
    dataset = dataset_for(tensors)
    assert dataset.output_processors["labels"].label_vocab == {"B": 0, "A": 1}
    assert dataset.input_schema == {"signal": "tensor"}
    assert len(dataset) == 3
    for index in range(3):
        sample = dataset[index]
        torch.testing.assert_close(sample["signal"], tensors[0][index], rtol=0, atol=0)
        torch.testing.assert_close(sample["labels"], tensors[1][index], rtol=0, atol=0)
        assert sample["ecg_id"] == [17, 29, 41][index]
        assert sample["record_id"] == str(sample["ecg_id"])
        assert sample["patient_id"] == str([101, 101, 102][index])
        assert sample["strat_fold"] == 1
    assert dataset[2]["labels"].sum().item() == 0


def test_reference_image_shape_and_integral_csv_ids_are_adapted(tensors):
    dataset = create_h1_ptbxl_dataset(
        tensors[0].unsqueeze(1),
        tensors[1],
        ecg_ids=["17", "29.0", 41.0],
        patient_ids=["101", "101.0", 102.0],
        folds=[1, 2, 8],
        split="train",
        label_names=["B", "A"],
    )
    assert dataset[0]["signal"].shape == (12, 1000)
    assert [dataset[index]["ecg_id"] for index in range(3)] == [17, 29, 41]
    assert dataset[1]["patient_id"] == "101"


def test_disk_backed_native_adapter_preserves_zero_rows_and_cleans_up(tensors):
    from pathlib import Path

    dataset = dataset_for(tensors, in_memory=False)
    path = Path(dataset.path)
    try:
        batch = next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))
        assert batch["ecg_id"] == [17, 29, 41]
        torch.testing.assert_close(batch["signal"], tensors[0], rtol=0, atol=0)
        torch.testing.assert_close(batch["labels"], tensors[1], rtol=0, atol=0)
    finally:
        dataset.close()
    assert not path.exists()


def test_default_52_label_order_keeps_unobserved_labels(tensors):
    targets = torch.zeros(3, 52)
    targets[0, 0] = 1
    dataset = create_h1_ptbxl_dataset(
        tensors[0],
        targets,
        ecg_ids=[17, 29, 41],
        patient_ids=[101, 101, 102],
        folds=[1, 2, 8],
        split="train",
    )
    vocab = dataset.output_processors["labels"].label_vocab
    assert list(vocab) == list(PTBXL_MORPHOLOGY_LABEL_ORDER)
    assert dataset[2]["labels"].shape == (52,)
    assert dataset[2]["labels"].sum().item() == 0


def test_noise_is_exactly_batch_order_and_partition_invariant(tensors):
    clean = tensors[0]
    whole = make_ecg_noisy_view(clean, [17, 29, 41], stream_seed=598, epoch_or_draw=1)
    permuted = make_ecg_noisy_view(
        clean[[2, 0, 1]], [41, 17, 29], stream_seed=598, epoch_or_draw=1
    )
    torch.testing.assert_close(whole[[2, 0, 1]], permuted, rtol=0, atol=0)
    chunks = torch.cat(
        [
            make_ecg_noisy_view(clean[:1], [17], stream_seed=598, epoch_or_draw=1),
            make_ecg_noisy_view(clean[1:], [29, 41], stream_seed=598, epoch_or_draw=1),
        ]
    )
    torch.testing.assert_close(whole, chunks, rtol=0, atol=0)


def test_noise_shape_variants_have_identical_values(tensors):
    clean = tensors[0]
    kwargs = dict(stream_seed=598, epoch_or_draw=1)
    first = make_ecg_noisy_view(clean, [17, 29, 41], **kwargs)
    second = make_ecg_noisy_view(clean.unsqueeze(1), [17, 29, 41], **kwargs)
    torch.testing.assert_close(first, second.squeeze(1), rtol=0, atol=0)


def test_noise_zero_ac_rms_zero_alpha_and_clean_immutability(tensors):
    clean = tensors[0]
    original = clean.clone()
    noisy = make_ecg_noisy_view(clean, [17, 29, 41], stream_seed=598, epoch_or_draw=1)
    torch.testing.assert_close(noisy[:, 0], clean[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(clean, original, rtol=0, atol=0)
    zero = make_ecg_noisy_view(
        clean, [17, 29, 41], stream_seed=598, epoch_or_draw=1, alpha=0.0
    )
    torch.testing.assert_close(zero, clean, rtol=0, atol=0)
    assert noisy.data_ptr() != clean.data_ptr()


def test_noise_uses_local_rng_and_separate_streams(tensors):
    torch_state = torch.get_rng_state().clone()
    numpy_state = copy.deepcopy(np.random.get_state())
    clean = tensors[0]
    first = make_ecg_noisy_view(clean, [17, 29, 41], stream_seed=598, epoch_or_draw=1)
    epoch2 = make_ecg_noisy_view(clean, [17, 29, 41], stream_seed=598, epoch_or_draw=2)
    evaluation = make_ecg_noisy_view(
        clean, [17, 29, 41], stream_seed=599, epoch_or_draw=1
    )
    assert not torch.equal(first, epoch2)
    assert not torch.equal(first, evaluation)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


def test_common_noise_is_identical_across_arm_loader_orders(tensors):
    dataset = dataset_for(tensors)
    first = PairedECGLoader(dataset, 2, stream_seed=598, seed=1)
    second = PairedECGLoader(dataset, 1, stream_seed=598, seed=2)

    def collect(loader):
        return {
            ecg_id: noisy.clone()
            for batch in loader
            for ecg_id, noisy in zip(batch["ecg_id"], batch["noisy_signal"])
        }

    control = collect(first)
    consistency = collect(second)
    assert control.keys() == consistency.keys()
    for ecg_id in control:
        torch.testing.assert_close(control[ecg_id], consistency[ecg_id], rtol=0, atol=0)


def test_loader_full_pass_advances_epoch_without_changing_clean_data(tensors):
    dataset = dataset_for(tensors)
    loader = PairedECGLoader(dataset, 3, stream_seed=598, shuffle=False)
    for epoch in (1, 2):
        batch = next(iter(loader))
        expected = make_ecg_noisy_view(
            tensors[0], [17, 29, 41], stream_seed=598, epoch_or_draw=epoch
        )
        torch.testing.assert_close(batch["noisy_signal"], expected, rtol=0, atol=0)
        torch.testing.assert_close(batch["signal"], tensors[0], rtol=0, atol=0)
    assert loader.next_epoch == 3


@pytest.mark.parametrize("split", ["validation", "test"])
def test_paired_loader_rejects_nontraining_splits(tensors, split):
    with pytest.raises(ValueError, match="train split"):
        PairedECGLoader(dataset_for(tensors, split=split), 3, stream_seed=598)


@pytest.mark.parametrize("folds", [[1, 2, 9], [1, 2, 10], [1, 2, 0]])
def test_dataset_rejects_fold_mixing(tensors, folds):
    with pytest.raises(ValueError, match="declared split"):
        create_h1_ptbxl_dataset(
            tensors[0],
            tensors[1],
            ecg_ids=[17, 29, 41],
            patient_ids=[101, 101, 102],
            folds=folds,
            split="train",
            label_names=["B", "A"],
        )


@pytest.mark.parametrize("ids", [[17, 17, 41], [17, 29.5, 41], [17, float("nan"), 41]])
def test_dataset_rejects_duplicate_or_fractional_ids(tensors, ids):
    with pytest.raises((ValueError, TypeError)):
        create_h1_ptbxl_dataset(
            tensors[0],
            tensors[1],
            ecg_ids=ids,
            patient_ids=[101, 101, 102],
            folds=[1, 2, 8],
            split="train",
            label_names=["B", "A"],
        )


@pytest.mark.parametrize("alpha", [-0.01, float("nan"), float("inf"), 1e100])
def test_noise_rejects_invalid_strength(tensors, alpha):
    with pytest.raises(ValueError):
        make_ecg_noisy_view(
            tensors[0], [17, 29, 41], stream_seed=598, epoch_or_draw=1, alpha=alpha
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ecg_ids": [17, 29]},
        {"ecg_ids": [17, 29.1, 41]},
        {"stream_seed": -1},
        {"epoch_or_draw": True},
    ],
)
def test_noise_rejects_invalid_id_seed_or_epoch(tensors, kwargs):
    settings = dict(ecg_ids=[17, 29, 41], stream_seed=598, epoch_or_draw=1)
    settings.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        make_ecg_noisy_view(tensors[0], **settings)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_noise_rejects_unexpected_dtype(tensors, dtype):
    with pytest.raises(TypeError, match="float32"):
        make_ecg_noisy_view(
            tensors[0].to(dtype), [17, 29, 41], stream_seed=598, epoch_or_draw=1
        )


@pytest.mark.parametrize(
    "case",
    [
        "signal_dtype",
        "signal_shape",
        "signal_nan",
        "label_order",
        "target_shape",
        "target_value",
        "id_count",
        "split",
    ],
)
def test_dataset_rejects_incompatible_inputs(tensors, case):
    signals, targets = (value.clone() for value in tensors)
    settings = dict(
        ecg_ids=[17, 29, 41],
        patient_ids=[101, 101, 102],
        folds=[1, 2, 8],
        split="train",
        label_names=["B", "A"],
    )
    if case == "signal_dtype":
        signals = signals.double()
    elif case == "signal_shape":
        signals = signals[:, :, :-1]
    elif case == "signal_nan":
        signals[0, 0, 0] = float("nan")
    elif case == "label_order":
        settings["label_names"] = ["B", "B"]
    elif case == "target_shape":
        targets = targets[:, :1]
    elif case == "target_value":
        targets[0, 0] = 0.5
    elif case == "id_count":
        settings["ecg_ids"] = [17, 29]
    else:
        settings["split"] = "unknown"
    with pytest.raises((ValueError, TypeError)):
        create_h1_ptbxl_dataset(signals, targets, **settings)


class TinyDataBackbone(nn.Module):
    def __init__(self, in_leads, pretrained=False):
        super().__init__()
        assert not pretrained
        self.out_channels = 4
        self.conv = nn.Conv1d(in_leads, 4, 1)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_two_epoch_trainer_uses_distinct_id_based_noise_without_resetting_loader(
    tensors, monkeypatch
):
    from pyhealth.models import protoecgnet

    monkeypatch.setattr(protoecgnet, "ResNet2dBackbone", TinyDataBackbone)
    dataset = dataset_for(tensors)
    loader = PairedECGLoader(dataset, 3, stream_seed=598, shuffle=False)
    model = ProtoECGNetH1(
        dataset, proto_dim=4, prototypes_per_label=1, lambda_cons=117608.0
    )
    trainer = Trainer(model, device="cpu", enable_logging=False)
    seen = []
    hook = model.branches["morphology"].backbone.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].clone())
    )
    trainer.train(
        loader, epochs=2, optimizer_params={"lr": 1e-4}, load_best_model_at_last=False
    )
    hook.remove()
    assert len(seen) == 2
    assert loader.next_epoch == 3
    for epoch, combined in enumerate(seen, 1):
        expected = make_ecg_noisy_view(
            tensors[0], [17, 29, 41], stream_seed=598, epoch_or_draw=epoch
        )
        torch.testing.assert_close(combined[:3], tensors[0], rtol=0, atol=0)
        torch.testing.assert_close(combined[3:], expected, rtol=0, atol=0)
    assert not torch.equal(seen[0][3:], seen[1][3:])
    clean_loader = get_dataloader(dataset, batch_size=3, shuffle=False)
    truth, probabilities, _ = trainer.inference(clean_loader)
    assert truth.shape == probabilities.shape == (3, 2)
