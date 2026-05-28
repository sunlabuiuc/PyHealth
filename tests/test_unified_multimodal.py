"""Tests for the TemporalFeatureProcessor ABC, TemporalTimeseriesProcessor,
collate_temporal helper, and UnifiedMultimodalEmbeddingModel.

Run with:
    TOKENIZERS_PARALLELISM=false python tests/test_unified_multimodal.py
"""

import math
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import torch
import numpy as np


# ── 1. TemporalFeatureProcessor ABC & ModalityType ────────────────────────────


def test_modality_type_values():
    from pyhealth.processors import ModalityType

    assert ModalityType.CODE == "code"
    assert ModalityType.TEXT == "text"
    assert ModalityType.IMAGE == "image"
    assert ModalityType.NUMERIC == "numeric"


def test_stagenet_is_temporal():
    from pyhealth.processors import (
        StageNetProcessor,
        TemporalFeatureProcessor,
        ModalityType,
    )

    p = StageNetProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.CODE


def test_stagenet_tensor_is_temporal():
    from pyhealth.processors import (
        StageNetTensorProcessor,
        TemporalFeatureProcessor,
        ModalityType,
    )

    p = StageNetTensorProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.NUMERIC


def test_tuple_time_text_is_temporal():
    from pyhealth.processors import (
        TupleTimeTextProcessor,
        TemporalFeatureProcessor,
        ModalityType,
    )

    p = TupleTimeTextProcessor()
    assert isinstance(p, TemporalFeatureProcessor)
    assert p.modality() == ModalityType.TEXT


def test_time_image_is_temporal():
    from pyhealth.processors import (
        TimeImageProcessor,
        TemporalFeatureProcessor,
        ModalityType,
    )

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
    assert out["time"].dtype == torch.float32
    assert out["value"].shape == (3,)
    assert out["time"].shape == (3,)


def test_stagenet_tensor_process_temporal():
    from pyhealth.processors import StageNetTensorProcessor

    samples = [{"vitals": ([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])}]
    p = StageNetTensorProcessor()
    p.fit(samples, "vitals")

    out = p.process_temporal(([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]]))
    assert set(out.keys()) == {"value", "time"}
    assert out["value"].shape == (2, 2)
    assert out["time"].shape == (2,)
    assert p.value_dim() == 2
    assert p.modality().value == "numeric"


# ── 3. TemporalTimeseriesProcessor ────────────────────────────────────────────


def test_temporal_timeseries_basic():
    from pyhealth.processors import TemporalTimeseriesProcessor

    p = TemporalTimeseriesProcessor(sampling_rate=timedelta(hours=2))
    ts = [
        datetime(2023, 1, 1, 0),
        datetime(2023, 1, 1, 4),
        datetime(2023, 1, 1, 8),
    ]
    val = np.array([[120.0, 80.0], [115.0, 78.0], [118.0, 82.0]])

    out = p.process((ts, val))
    # 8 h window / 2 h step + 1 = 5 steps
    assert out["value"].shape == (5, 2)
    assert out["time"].shape == (5,)
    # Times should be [0, 2, 4, 6, 8]
    expected_times = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
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
    assert math.isclose(float(out["value"][1, 0]), 10.0, rel_tol=1e-6)


# ── 4. collate_temporal ───────────────────────────────────────────────────────


def test_collate_temporal_basic():
    from pyhealth.datasets.collate import collate_temporal

    batch = [
        {
            "codes": {
                "value": torch.tensor([1, 2, 3], dtype=torch.long),
                "time": torch.tensor([0.0, 1.0, 2.0]),
            },
            "label": torch.tensor(1),
        },
        {
            "codes": {
                "value": torch.tensor([4, 5, 3], dtype=torch.long),
                "time": torch.tensor([0.5, 1.5, 2.5]),
            },
            "label": torch.tensor(0),
        },
    ]

    collated = collate_temporal(batch)

    assert collated["codes"]["value"].shape == (2, 3)
    assert collated["codes"]["time"].shape == (2, 3)
    assert collated["label"].shape == (2,)


def test_collate_temporal_variable_length():
    """When sequences have different lengths, pad_sequence should be used."""
    from pyhealth.datasets.collate import collate_temporal

    batch = [
        {
            "codes": {
                "value": torch.tensor([1, 2], dtype=torch.long),
                "time": torch.tensor([0.0, 1.0]),
            }
        },
        {
            "codes": {
                "value": torch.tensor([3, 4, 5], dtype=torch.long),
                "time": torch.tensor([0.0, 1.0, 2.0]),
            }
        },
    ]
    collated = collate_temporal(batch)
    # Padded to length 3
    assert collated["codes"]["value"].shape == (2, 3)


# ── 5. SinusoidalTimeEmbedding ────────────────────────────────────────────────


def test_sinusoidal_time_embedding_shape():
    from pyhealth.models.embedding import SinusoidalTimeEmbedding

    emb = SinusoidalTimeEmbedding(dim=64, max_hours=720.0)
    t = torch.tensor([[0.0, 12.0, 24.0], [0.0, 6.0, 48.0]])  # (2, 3)
    out = emb(t)
    assert out.shape == (2, 3, 64)


def test_sinusoidal_different_times_differ():
    from pyhealth.models.embedding import SinusoidalTimeEmbedding

    emb = SinusoidalTimeEmbedding(dim=32)
    t0 = emb(torch.tensor([0.0]))
    t1 = emb(torch.tensor([24.0]))
    assert not torch.allclose(t0, t1)


# ── 6. UnifiedMultimodalEmbeddingModel, code-only smoke test ─────────────────


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
    time = (
        torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    )
    inputs = {"codes": {"value": value, "time": time}}

    return processors, inputs


def test_unified_model_code_only():
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel

    processors, inputs = _make_code_processors_and_inputs()
    model = UnifiedMultimodalEmbeddingModel(processors=processors, embedding_dim=64)

    out = model(inputs)

    assert "sequence" in out
    assert "time" in out
    assert "mask" in out

    B, S, E = out["sequence"].shape
    assert B == 2
    assert S == 5
    assert E == 64


def test_unified_model_rejects_non_temporal():
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import SequenceProcessor

    bad_proc = SequenceProcessor()
    try:
        UnifiedMultimodalEmbeddingModel(
            processors={"field": bad_proc},
            embedding_dim=64,
        )
    except TypeError as exc:
        assert "TemporalFeatureProcessor" in str(exc)
    else:
        raise AssertionError("Expected TypeError for non-temporal processor")


def test_unified_model_gradient_flow():
    """Loss.backward() should propagate through time + type embeddings."""
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel

    processors, inputs = _make_code_processors_and_inputs()
    model = UnifiedMultimodalEmbeddingModel(processors=processors, embedding_dim=32)

    out = model(inputs)
    loss = out["sequence"].mean()
    loss.backward()

    # type_embedding grad should be non-zero
    assert model.type_embedding.weight.grad is not None
    assert model.time_embed.freqs.grad is None  # buffer, not parameter, OK


def test_unified_model_time_sort():
    """Events should be sorted by time ascending in the output."""
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import StageNetProcessor

    samples = [{"c": (None, ["a", "b"])}]
    proc = StageNetProcessor()
    proc.fit(samples, "c")

    model = UnifiedMultimodalEmbeddingModel(processors={"c": proc}, embedding_dim=16)
    # Reverse-order times
    value = torch.tensor([[2, 1]])  # (1, 2)
    time = torch.tensor([[10.0, 0.0]])  # t=10 then t=0 → should sort to [0, 10]
    out = model({"c": {"value": value, "time": time}})
    assert math.isclose(out["time"][0, 0].item(), 0.0, rel_tol=1e-6)
    assert math.isclose(out["time"][0, 1].item(), 10.0, rel_tol=1e-6)


# ── 7. field_embeddings: reuse pre-built unimodal encoder ─────────────────────


def test_unified_field_embeddings_reuses_encoder():
    """field_embeddings: encoder from a pre-built model is used in-place."""
    import torch.nn as nn
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import StageNetProcessor

    proc = StageNetProcessor()
    proc.fit([{"codes": (None, [f"c{i}" for i in range(5)])}], "codes")
    vocab_size = proc.value_dim()

    # Simulate a pre-built EmbeddingModel via a lightweight mock
    pre_emb = nn.Embedding(vocab_size, 32)

    class _MockEmbedModel:
        embedding_dim = 32
        embedding_layers = {"codes": pre_emb}

    model = UnifiedMultimodalEmbeddingModel(
        processors={"codes": proc},
        embedding_dim=32,
        field_embeddings={"codes": _MockEmbedModel()},
    )
    # The encoder registered should be the exact same object
    assert model.encoders["codes"] is pre_emb


def test_unified_field_embeddings_projection_added_on_dim_mismatch():
    """When pre-built embedding_dim != unified embedding_dim, a projection is added."""
    import torch.nn as nn
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import StageNetProcessor

    proc = StageNetProcessor()
    proc.fit([{"codes": (None, ["A", "B"])}], "codes")
    vocab_size = proc.value_dim()

    pre_emb = nn.Embedding(vocab_size, 16)  # pre-built dim=16

    class _MockEmbedModel:
        embedding_dim = 16
        embedding_layers = {"codes": pre_emb}

    model = UnifiedMultimodalEmbeddingModel(
        processors={"codes": proc},
        embedding_dim=32,  # different from pre-built
        field_embeddings={"codes": _MockEmbedModel()},
    )
    # A Sequential(pre_emb, nn.Linear(16→32)) should be built
    assert isinstance(model.encoders["codes"], nn.Sequential)
    # Forward should produce embedding_dim=32
    value = torch.randint(1, vocab_size, (2, 3))
    time = torch.arange(3, dtype=torch.float32).unsqueeze(0).expand(2, -1)
    out = model({"codes": {"value": value, "time": time}})
    assert out["sequence"].shape[-1] == 32


def test_unified_field_embeddings_forward():
    """End-to-end forward with field_embeddings reusing a CODE encoder."""
    import torch.nn as nn
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import StageNetProcessor

    proc = StageNetProcessor()
    proc.fit([{"codes": (None, [f"c{i}" for i in range(4)])}], "codes")
    vocab_size = proc.value_dim()

    pre_emb = nn.Embedding(vocab_size, 64)

    class _MockEmbedModel:
        embedding_dim = 64
        embedding_layers = {"codes": pre_emb}

    model = UnifiedMultimodalEmbeddingModel(
        processors={"codes": proc},
        embedding_dim=64,
        field_embeddings={"codes": _MockEmbedModel()},
    )
    value = torch.randint(1, vocab_size, (3, 4))
    time = torch.arange(4, dtype=torch.float32).unsqueeze(0).expand(3, -1)
    out = model({"codes": {"value": value, "time": time}})

    assert out["sequence"].shape == (3, 4, 64)
    assert out["mask"].shape == (3, 4)


def test_unified_text_encoder_shared_by_tokenizer():
    """Token-based TEXT fields with the same tokenizer share one encoder."""
    from types import SimpleNamespace
    import torch.nn as nn
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.processors import ModalityType, TemporalFeatureProcessor

    class _DummyTemporalTextProcessor(TemporalFeatureProcessor):
        def __init__(self, tokenizer_model: str):
            self.tokenizer_model = tokenizer_model

        def process(self, value):
            return value

        def modality(self):
            return ModalityType.TEXT

        def value_dim(self):
            return 0

        def is_token(self):
            return True

        def schema(self):
            return ("value", "mask", "time")

        def dim(self):
            return (2, 2, 1)

        def spatial(self):
            return (False, False)

    class _DummyBert(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)

        def forward(self, input_ids=None, attention_mask=None):
            if input_ids is None:
                raise ValueError("input_ids is required")
            b, l = input_ids.shape
            hidden = self.config.hidden_size
            out = torch.zeros(b, l, hidden)
            return SimpleNamespace(last_hidden_state=out)

    call_count = {"n": 0}

    def _fake_from_pretrained(_name):
        call_count["n"] += 1
        return _DummyBert(hidden_size=48)

    with patch("transformers.AutoModel.from_pretrained", _fake_from_pretrained):
        processors = {
            "discharge_note_times": _DummyTemporalTextProcessor("bert-base-uncased"),
            "radiology_note_times": _DummyTemporalTextProcessor("bert-base-uncased"),
        }

        model = UnifiedMultimodalEmbeddingModel(
            processors=processors,
            embedding_dim=32,
        )

    # Both text fields reuse the same encoder instance.
    assert (
        model.encoders["discharge_note_times"] is model.encoders["radiology_note_times"]
    )
    assert call_count["n"] == 1

    # Projections remain field-specific.
    assert "discharge_note_times" in model.projections
    assert "radiology_note_times" in model.projections
    assert (
        model.projections["discharge_note_times"]
        is not model.projections["radiology_note_times"]
    )


# ── 8. Downstream models in unified mode ──────────────────────────────────────


def _make_stagenet_dataset(n_codes: int = 5):
    """Build a minimal SampleDataset with one StageNetProcessor field.

    Time arrays are kept the same length as the code arrays so that the
    temporal batch has consistent shapes.
    """
    from pyhealth.datasets import create_sample_dataset

    codes_p0 = [f"c{i}" for i in range(n_codes)]
    times_p0 = [float(i) for i in range(n_codes)]
    codes_p1 = [f"c{i}" for i in range(2)]
    times_p1 = [0.0, 1.0]

    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "codes": (times_p0, codes_p0),
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "codes": (times_p1, codes_p1),
            "label": 0,
        },
    ]
    return create_sample_dataset(
        samples,
        input_schema={"codes": "stagenet"},
        output_schema={"label": "binary"},
        dataset_name="test_unified_downstream",
    )


def test_transformer_unified_mode():
    """Transformer with unified_embedding uses a single backbone + forward works."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.transformer import Transformer

    dataset = _make_stagenet_dataset()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    model = Transformer(dataset=dataset, embedding_dim=32, unified_embedding=unified)

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)

    assert "loss" in out and "y_prob" in out and "logit" in out
    out["loss"].backward()
    # fc input size should be embedding_dim, not n_fields * embedding_dim
    assert model.fc.in_features == 32


def test_ehrmamba_unified_mode():
    """EHRMamba with unified_embedding uses a single Mamba stack."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.ehrmamba import EHRMamba

    dataset = _make_stagenet_dataset()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    model = EHRMamba(
        dataset=dataset,
        embedding_dim=32,
        num_layers=1,
        unified_embedding=unified,
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)

    assert "loss" in out and "y_prob" in out
    out["loss"].backward()
    assert model.fc.in_features == 32


def test_jamba_ehr_unified_mode():
    """JambaEHR with unified_embedding uses a single JambaLayer."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.jamba_ehr import JambaEHR

    dataset = _make_stagenet_dataset()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    model = JambaEHR(
        dataset=dataset,
        embedding_dim=32,
        num_transformer_layers=1,
        num_mamba_layers=1,
        heads=2,
        unified_embedding=unified,
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)

    assert "loss" in out and "y_prob" in out
    out["loss"].backward()
    assert model.fc.in_features == 32


def test_mlp_unified_mode():
    """MLP with unified_embedding mean-pools the event sequence and produces valid output."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.mlp import MLP

    dataset = _make_stagenet_dataset()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    model = MLP(
        dataset=dataset, embedding_dim=32, hidden_dim=32, unified_embedding=unified
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)

    assert "loss" in out and "y_prob" in out and "logit" in out
    out["loss"].backward()
    # fc input size should be hidden_dim, not n_fields * hidden_dim
    assert model.fc.in_features == 32


def test_rnn_unified_mode():
    """RNN with unified_embedding uses a single RNN over the event sequence."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.rnn import RNN

    dataset = _make_stagenet_dataset()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    model = RNN(
        dataset=dataset, embedding_dim=32, hidden_dim=32, unified_embedding=unified
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)

    assert "loss" in out and "y_prob" in out and "logit" in out
    out["loss"].backward()
    # fc input size should be hidden_dim, not n_fields * hidden_dim
    assert model.fc.in_features == 32


def test_bottleneck_transformer_unified_mode():
    """BottleneckTransformer with unified_embedding uses n_modality=1 encoder and a single CLS token."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.bottleneck_transformer import BottleneckTransformer

    dataset = _make_stagenet_dataset()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    model = BottleneckTransformer(
        dataset=dataset,
        embedding_dim=32,
        bottlenecks_n=2,
        fusion_startidx=1,
        num_layers=2,
        heads=2,
        unified_embedding=unified,
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)

    assert "loss" in out and "y_prob" in out and "logit" in out
    out["loss"].backward()
    # fc input must be embedding_dim (CLS token), not n_fields * embedding_dim
    assert model.fc.in_features == 32
    # Encoder must be configured for n_modality=1
    assert model.encoder.n_modality == 1


def test_unified_per_field_backward_compat():
    """Models without unified_embedding still work in per-field mode."""
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.models.transformer import Transformer

    dataset = _make_stagenet_dataset()
    # Uses SequenceProcessor-style input_schema for per-field mode
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {"patient_id": "p0", "visit_id": "v0", "codes": ["A", "B", "C"], "label": 1},
        {"patient_id": "p1", "visit_id": "v1", "codes": ["D", "E"], "label": 0},
    ]
    ds = create_sample_dataset(
        samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="test_compat",
    )
    model = Transformer(dataset=ds, embedding_dim=32)
    loader = get_dataloader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)
    assert "loss" in out
    out["loss"].backward()


def load_tests(loader, tests, pattern):
    """Expose top-level test_ functions to unittest discovery."""
    suite = unittest.TestSuite()
    namespace = globals()
    for name in sorted(namespace):
        if name.startswith("test_") and callable(namespace[name]):
            suite.addTest(unittest.FunctionTestCase(namespace[name]))
    return suite


if __name__ == "__main__":
    unittest.main()
