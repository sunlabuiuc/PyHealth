"""Tests for the TemporalFeatureProcessor ABC, TemporalTimeseriesProcessor,
collate_temporal helper, and UnifiedMultimodalEmbeddingModel.

Run with:
    TOKENIZERS_PARALLELISM=false pytest tests/test_unified_multimodal.py -v
"""
import math
from datetime import datetime, timedelta

import pytest
import torch
import numpy as np


# ── 1. TemporalFeatureProcessor ABC & ModalityType ────────────────────────────

def test_modality_type_values():
    from pyhealth.processors import ModalityType
    assert ModalityType.CODE    == "code"
    assert ModalityType.TEXT    == "text"
    assert ModalityType.IMAGE   == "image"
    assert ModalityType.NUMERIC == "numeric"


def test_stagenet_is_temporal():
    from pyhealth.processors import StageNetProcessor, TemporalFeatureProcessor, ModalityType
    p = StageNetProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.CODE


def test_stagenet_tensor_is_temporal():
    from pyhealth.processors import StageNetTensorProcessor, TemporalFeatureProcessor, ModalityType
    p = StageNetTensorProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.NUMERIC


def test_tuple_time_text_is_temporal():
    from pyhealth.processors import TupleTimeTextProcessor, TemporalFeatureProcessor, ModalityType
    p = TupleTimeTextProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.TEXT


def test_time_image_is_temporal():
    from pyhealth.processors import TimeImageProcessor, TemporalFeatureProcessor, ModalityType
    p = TimeImageProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.IMAGE


# ── 2. StageNetProcessor.process_temporal() ───────────────────────────────────

def test_stagenet_process_temporal():
    from pyhealth.processors import StageNetProcessor
    samples = [{"codes": (None, ["A", "B", "C"])}]
    p = StageNetProcessor()
    p.fit(samples, "codes")

    sample_value = ([0.0, 1.5, 3.0], ["A", "B", "C"])
    out = p.process_temporal(sample_value)

    assert set(out.keys()) == {"value", "time"}
    assert out["value"].dtype == torch.long
    assert out["time"].dtype  == torch.float32
    assert out["value"].shape == (3,)
    assert out["time"].shape  == (3,)


def test_stagenet_tensor_process_temporal():
    from pyhealth.processors import StageNetTensorProcessor
    samples = [{"vitals": ([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])}]
    p = StageNetTensorProcessor()
    p.fit(samples, "vitals")

    out = p.process_temporal(([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]]))
    assert set(out.keys()) == {"value", "time"}
    assert out["value"].shape == (2, 2)
    assert out["time"].shape  == (2,)
    assert p.value_dim() == 2
    assert p.modality().value == "numeric"


# ── 3. TemporalTimeseriesProcessor ────────────────────────────────────────────

def test_temporal_timeseries_basic():
    from pyhealth.processors import TemporalTimeseriesProcessor

    p = TemporalTimeseriesProcessor(sampling_rate=timedelta(hours=2))
    ts = [
        datetime(2023, 1, 1,  0),
        datetime(2023, 1, 1,  4),
        datetime(2023, 1, 1,  8),
    ]
    val = np.array([[120.0, 80.0], [115.0, 78.0], [118.0, 82.0]])

    out = p.process((ts, val))
    # 8 h window / 2 h step + 1 = 5 steps
    assert out["value"].shape == (5, 2)
    assert out["time"].shape  == (5,)
    # Times should be [0, 2, 4, 6, 8]
    expected_times = torch.tensor([0., 2., 4., 6., 8.])
    assert torch.allclose(out["time"], expected_times)


def test_temporal_timeseries_fit():
    from pyhealth.processors import TemporalTimeseriesProcessor
    p = TemporalTimeseriesProcessor()
    ts = [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 1)]
    val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    samples = [{"ts": (ts, val)}]
    p.fit(samples, "ts")
    assert p.value_dim() == 3
    assert p.size() == 3


def test_temporal_timeseries_imputation():
    from pyhealth.processors import TemporalTimeseriesProcessor
    p = TemporalTimeseriesProcessor(sampling_rate=timedelta(hours=1))
    ts = [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 2)]  # gap at h=1
    val = np.array([[10.0], [20.0]])
    out = p.process((ts, val))
    # 3 steps: h=0 → 10, h=1 → forward-filled to 10, h=2 → 20
    assert out["value"].shape == (3, 1)
    assert float(out["value"][1, 0]) == pytest.approx(10.0)


# ── 4. collate_temporal ───────────────────────────────────────────────────────

def test_collate_temporal_basic():
    from pyhealth.datasets.collate import collate_temporal

    batch = [
        {
            "codes": {"value": torch.tensor([1, 2, 3], dtype=torch.long),
                      "time":  torch.tensor([0., 1., 2.])},
            "label": torch.tensor(1),
        },
        {
            "codes": {"value": torch.tensor([4, 5, 3], dtype=torch.long),
                      "time":  torch.tensor([0.5, 1.5, 2.5])},
            "label": torch.tensor(0),
        },
    ]

    collated = collate_temporal(batch)

    assert collated["codes"]["value"].shape == (2, 3)
    assert collated["codes"]["time"].shape  == (2, 3)
    assert collated["label"].shape == (2,)


def test_collate_temporal_variable_length():
    """When sequences have different lengths, pad_sequence should be used."""
    from pyhealth.datasets.collate import collate_temporal

    batch = [
        {"codes": {"value": torch.tensor([1, 2], dtype=torch.long),
                   "time":  torch.tensor([0., 1.])}},
        {"codes": {"value": torch.tensor([3, 4, 5], dtype=torch.long),
                   "time":  torch.tensor([0., 1., 2.])}},
    ]
    collated = collate_temporal(batch)
    # Padded to length 3
    assert collated["codes"]["value"].shape == (2, 3)


# ── 5. SinusoidalTimeEmbedding ────────────────────────────────────────────────

def test_sinusoidal_time_embedding_shape():
    from pyhealth.models.unified_embedding import SinusoidalTimeEmbedding
    emb = SinusoidalTimeEmbedding(dim=64, max_hours=720.0)
    t   = torch.tensor([[0.0, 12.0, 24.0], [0.0, 6.0, 48.0]])  # (2, 3)
    out = emb(t)
    assert out.shape == (2, 3, 64)


def test_sinusoidal_different_times_differ():
    from pyhealth.models.unified_embedding import SinusoidalTimeEmbedding
    emb = SinusoidalTimeEmbedding(dim=32)
    t0  = emb(torch.tensor([0.0]))
    t1  = emb(torch.tensor([24.0]))
    assert not torch.allclose(t0, t1)


# ── 6. UnifiedMultimodalEmbeddingModel — code-only smoke test ─────────────────

def _make_code_processors_and_inputs(batch_size=2, seq_len=5):
    """Build a minimal dataset mock with a single CODE-modality field."""
    from pyhealth.processors import StageNetProcessor

    samples = [{"codes": (None, [f"c{i}" for i in range(seq_len)])}]
    proc = StageNetProcessor()
    proc.fit(samples, "codes")
    vocab_size = proc.value_dim()  # <pad>, <unk>, c0..c4 → 7

    processors = {"codes": proc}

    # Fake batch dict (as produced by collate_temporal)
    value = torch.randint(1, vocab_size, (batch_size, seq_len))
    time  = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    inputs = {"codes": {"value": value, "time": time}}

    return processors, inputs


def test_unified_model_code_only():
    from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel

    processors, inputs = _make_code_processors_and_inputs()
    model = UnifiedMultimodalEmbeddingModel(processors=processors, embedding_dim=64)

    out = model(inputs)

    assert "sequence" in out
    assert "time"     in out
    assert "mask"     in out

    B, S, E = out["sequence"].shape
    assert B == 2
    assert S == 5
    assert E == 64


def test_unified_model_rejects_non_temporal():
    from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import SequenceProcessor

    bad_proc = SequenceProcessor()
    with pytest.raises(TypeError, match="TemporalFeatureProcessor"):
        UnifiedMultimodalEmbeddingModel(processors={"field": bad_proc}, embedding_dim=64)


def test_unified_model_gradient_flow():
    """Loss.backward() should propagate through time + type embeddings."""
    from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel

    processors, inputs = _make_code_processors_and_inputs()
    model = UnifiedMultimodalEmbeddingModel(processors=processors, embedding_dim=32)

    out  = model(inputs)
    loss = out["sequence"].mean()
    loss.backward()

    # type_embedding grad should be non-zero
    assert model.type_embedding.weight.grad is not None
    assert model.time_embed.freqs.grad is None  # buffer, not parameter — OK


def test_unified_model_time_sort():
    """Events should be sorted by time ascending in the output."""
    from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import StageNetProcessor

    samples = [{"c": (None, ["a", "b"])}]
    proc = StageNetProcessor()
    proc.fit(samples, "c")

    model = UnifiedMultimodalEmbeddingModel(
        processors={"c": proc}, embedding_dim=16
    )
    # Reverse-order times
    value = torch.tensor([[2, 1]])   # (1, 2)
    time  = torch.tensor([[10.0, 0.0]])  # t=10 then t=0 → should sort to [0, 10]
    out   = model({"c": {"value": value, "time": time}})
    assert out["time"][0, 0].item() == pytest.approx(0.0)
    assert out["time"][0, 1].item() == pytest.approx(10.0)
