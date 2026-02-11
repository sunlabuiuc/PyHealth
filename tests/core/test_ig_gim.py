"""Unit tests for IntegratedGradientGIM (IG + GIM combined method)."""

import math
import unittest
from typing import Dict

import torch
import torch.nn as nn

from pyhealth.interpret.methods import GIM, IntegratedGradients
from pyhealth.interpret.methods.ig_gim import IntegratedGradientGIM
from pyhealth.interpret.api import Interpretable
from pyhealth.models import BaseModel
from pyhealth.models.transformer import Attention


# ---------------------------------------------------------------------------
# Mock helpers (reused from test_gim.py style)
# ---------------------------------------------------------------------------

class _MockProcessor:
    """Mock feature processor with configurable schema."""

    def __init__(self, schema_tuple=("value",), is_token_val=True):
        self._schema = schema_tuple
        self._is_token = is_token_val

    def schema(self):
        return self._schema

    def is_token(self):
        return self._is_token


class _ContinuousProcessor(_MockProcessor):
    """Processor for continuous (non-token) features."""

    def __init__(self, schema_tuple=("value",)):
        super().__init__(schema_tuple=schema_tuple, is_token_val=False)


class _MockDataset:
    """Lightweight stand-in for SampleDataset in unit tests."""

    def __init__(self, input_schema, output_schema, processors=None):
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_processors = processors or {
            k: _MockProcessor() for k in input_schema
        }
        self.output_processors = {"label": _BinaryProcessor()}
        self.dataset_name = "dummy"


class _BinaryProcessor:
    def size(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------

class _ToyEmbeddingModel(nn.Module):
    """Deterministic embedding lookup."""

    def __init__(self, vocab_size: int = 32, embedding_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        with torch.no_grad():
            weights = torch.arange(vocab_size * embedding_dim).float()
            weights = weights.view(vocab_size, embedding_dim)
            self.embedding.weight.copy_(weights / float(vocab_size))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: self.embedding(val.long()) for key, val in inputs.items()}


class _ToyModel(BaseModel, Interpretable):
    """Small model with softmax attention for testing IG-GIM."""

    def __init__(self, vocab_size=32, embedding_dim=4, schema=("value",)):
        dataset = _MockDataset(
            input_schema={"codes": "sequence"},
            output_schema={"label": "binary"},
            processors={"codes": _MockProcessor(schema)},
        )
        super().__init__(dataset=dataset)
        self.feature_keys = ["codes"]
        self.label_keys = ["label"]
        self.mode = "binary"

        self.embedding_model = _ToyEmbeddingModel(vocab_size, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc = nn.Linear(embedding_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            identity = torch.eye(self.query.in_features)
            self.query.weight.copy_(identity)
            self.key.weight.copy_(identity)
            self.value.weight.copy_(identity)
            self.fc.weight.copy_(torch.tensor([[0.2, -0.3, 0.4, 0.1]]))

    def forward_from_embedding(self, **kwargs):
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            emb = codes[schema.index("value")]
        else:
            emb = codes
        q = self.query(emb)
        k = self.key(emb)
        v = self.value(emb)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        weights = self.softmax(scores)
        context = torch.matmul(weights, v)
        pooled = context.mean(dim=1)
        logits = self.fc(pooled)
        label = kwargs.get("label")
        if label is None:
            label = torch.zeros_like(logits)
        return {
            "logit": logits,
            "y_prob": self.sigmoid(logits),
            "y_true": label,
            "loss": torch.zeros((), device=logits.device),
        }

    def forward(self, **kwargs):
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            val = codes[schema.index("value")]
        else:
            val = codes
        embedded = self.embedding_model({"codes": val})["codes"]
        if isinstance(codes, tuple):
            i = schema.index("value")
            kwargs["codes"] = codes[:i] + (embedded,) + codes[i + 1:]
        else:
            kwargs["codes"] = (embedded,)
        return self.forward_from_embedding(**kwargs)

    def get_embedding_model(self):
        return self.embedding_model


class _ToyModelWithLayerNorm(_ToyModel):
    """Extends _ToyModel with nn.LayerNorm after attention pooling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = nn.LayerNorm(self.fc.in_features)
        with torch.no_grad():
            self.layer_norm.weight.fill_(1.0)
            self.layer_norm.bias.fill_(0.0)

    def forward_from_embedding(self, **kwargs):
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            emb = codes[schema.index("value")]
        else:
            emb = codes
        q = self.query(emb)
        k = self.key(emb)
        v = self.value(emb)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        weights = self.softmax(scores)
        context = torch.matmul(weights, v)
        pooled = context.mean(dim=1)
        pooled = self.layer_norm(pooled)
        logits = self.fc(pooled)
        label = kwargs.get("label")
        if label is None:
            label = torch.zeros_like(logits)
        return {
            "logit": logits,
            "y_prob": self.sigmoid(logits),
            "y_true": label,
            "loss": torch.zeros((), device=logits.device),
        }


class _ToyModelWithAttention(_ToyModel):
    """Model using PyHealth's real Attention module so GIM can swap it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn = Attention()
        dim = self.fc.in_features
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            for lin in (self.q_proj, self.k_proj, self.v_proj):
                lin.weight.copy_(torch.eye(dim))

    def forward_from_embedding(self, **kwargs):
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            emb = codes[schema.index("value")]
        else:
            emb = codes
        q = self.q_proj(emb)
        k = self.k_proj(emb)
        v = self.v_proj(emb)
        context, _attn = self.attn(q, k, v)
        pooled = context.mean(dim=1)
        logits = self.fc(pooled)
        label = kwargs.get("label")
        if label is None:
            label = torch.zeros_like(logits)
        return {
            "logit": logits,
            "y_prob": self.sigmoid(logits),
            "y_true": label,
            "loss": torch.zeros((), device=logits.device),
        }


class _ToyModelWithMixed(_ToyModel):
    """Model with both token (discrete) and continuous features."""

    def __init__(self, vocab_size=32, embedding_dim=4):
        dataset = _MockDataset(
            input_schema={"codes": "sequence", "vitals": "tensor"},
            output_schema={"label": "binary"},
            processors={
                "codes": _MockProcessor(("value",)),
                "vitals": _ContinuousProcessor(("value",)),
            },
        )
        # Skip _ToyModel.__init__ and call BaseModel directly
        BaseModel.__init__(self, dataset=dataset)
        self.feature_keys = ["codes", "vitals"]
        self.label_keys = ["label"]
        self.mode = "binary"

        self.embedding_model = _ToyMixedEmbeddingModel(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.fc.weight.fill_(0.1)

    def forward_from_embedding(self, **kwargs):
        codes = kwargs["codes"]
        vitals = kwargs["vitals"]
        if isinstance(codes, tuple):
            schema_c = self.dataset.input_processors["codes"].schema()
            emb_codes = codes[schema_c.index("value")]
        else:
            emb_codes = codes
        if isinstance(vitals, tuple):
            schema_v = self.dataset.input_processors["vitals"].schema()
            emb_vitals = vitals[schema_v.index("value")]
        else:
            emb_vitals = vitals

        # Simple: pool codes, concat with vitals embedding, then FC
        pooled_codes = emb_codes.mean(dim=1)  # [B, emb_dim]
        # vitals is already [B, emb_dim] from embedding model
        if emb_vitals.dim() == 3:
            pooled_vitals = emb_vitals.mean(dim=1)
        else:
            pooled_vitals = emb_vitals
        combined = torch.cat([pooled_codes, pooled_vitals], dim=-1)
        logits = self.fc(combined)

        label = kwargs.get("label")
        if label is None:
            label = torch.zeros_like(logits)
        return {
            "logit": logits,
            "y_prob": self.sigmoid(logits),
            "y_true": label,
            "loss": torch.zeros((), device=logits.device),
        }

    def forward(self, **kwargs):
        codes = kwargs["codes"]
        vitals = kwargs["vitals"]
        if isinstance(codes, tuple):
            schema_c = self.dataset.input_processors["codes"].schema()
            val_c = codes[schema_c.index("value")]
        else:
            val_c = codes
        if isinstance(vitals, tuple):
            schema_v = self.dataset.input_processors["vitals"].schema()
            val_v = vitals[schema_v.index("value")]
        else:
            val_v = vitals

        embedded = self.embedding_model({"codes": val_c, "vitals": val_v})

        if isinstance(codes, tuple):
            i = schema_c.index("value")
            kwargs["codes"] = codes[:i] + (embedded["codes"],) + codes[i + 1:]
        else:
            kwargs["codes"] = (embedded["codes"],)
        if isinstance(vitals, tuple):
            i = schema_v.index("value")
            kwargs["vitals"] = vitals[:i] + (embedded["vitals"],) + vitals[i + 1:]
        else:
            kwargs["vitals"] = (embedded["vitals"],)

        return self.forward_from_embedding(**kwargs)

    def get_embedding_model(self):
        return self.embedding_model


class _ToyMixedEmbeddingModel(nn.Module):
    """Embedding model that handles both token and continuous features."""

    def __init__(self, vocab_size=32, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            weights = torch.arange(vocab_size * embedding_dim).float()
            weights = weights.view(vocab_size, embedding_dim)
            self.embedding.weight.copy_(weights / float(vocab_size))
            self.linear.weight.copy_(torch.eye(embedding_dim))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in inputs.items():
            if val.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                out[key] = self.embedding(val.long())
            else:
                # Continuous: treat as pre-formed features, apply linear
                out[key] = self.linear(val.float())
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntegratedGradientGIM(unittest.TestCase):
    """Tests for the combined IG-GIM method."""

    def setUp(self):
        torch.manual_seed(42)
        self.tokens = torch.tensor([[2, 5, 3, 1]])
        self.labels = torch.zeros((1, 1))

    # ----- Initialization -----

    def test_init_success(self):
        """IntegratedGradientGIM should initialize with a valid model."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=2.0, steps=10)
        self.assertIsInstance(ig_gim, IntegratedGradientGIM)
        self.assertEqual(ig_gim.temperature, 2.0)
        self.assertEqual(ig_gim.steps, 10)

    def test_init_rejects_model_without_forward_from_embedding(self):
        """Should raise if model lacks forward_from_embedding."""
        # Create a minimal object that does not have forward_from_embedding
        class _NoFwdEmb:
            pass

        obj = _NoFwdEmb()
        obj.feature_keys = ["codes"]
        obj.label_keys = ["label"]
        obj.get_embedding_model = lambda: _ToyEmbeddingModel()

        with self.assertRaises((AssertionError, AttributeError, ValueError)):
            IntegratedGradientGIM(obj)

    def test_init_rejects_model_without_embedding_model(self):
        """Should raise if get_embedding_model returns None during attribution."""
        model = _ToyModel()
        model.get_embedding_model = lambda: None
        ig_gim = IntegratedGradientGIM(model)
        with self.assertRaises(AssertionError):
            ig_gim.attribute(
                codes=self.tokens, label=self.labels, target_class_idx=0,
            )

    def test_temperature_clamped_to_one(self):
        """Temperatures below 1.0 should be clamped to 1.0."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=0.5)
        self.assertEqual(ig_gim.temperature, 1.0)

    # ----- Basic attribution -----

    def test_basic_attribution_returns_correct_keys(self):
        """Attribution dict should contain the model's feature keys."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=5)
        attrs = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )
        self.assertIn("codes", attrs)
        self.assertEqual(len(attrs), 1)

    def test_attribution_shape_matches_input(self):
        """Attribution tensor shape should match the raw input shape."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=5)
        attrs = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )
        self.assertEqual(attrs["codes"].shape, self.tokens.shape)

    def test_attribution_values_are_finite(self):
        """Attributions should contain no NaN or Inf values."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=2.0, steps=10)
        attrs = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )
        self.assertTrue(torch.isfinite(attrs["codes"]).all())

    # ----- Temperature effect -----

    def test_temperature_changes_attributions(self):
        """Higher temperature should produce different attributions (TSG effect).

        Uses a model with the real Attention module so the TSG rule has
        something to act on.
        """
        model = _ToyModelWithAttention()
        ig_gim_t1 = IntegratedGradientGIM(model, temperature=1.0, steps=10)
        ig_gim_t3 = IntegratedGradientGIM(model, temperature=3.0, steps=10)

        attrs_t1 = ig_gim_t1.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]
        attrs_t3 = ig_gim_t3.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]

        self.assertTrue(
            torch.any(~torch.isclose(attrs_t1, attrs_t3, atol=1e-6)),
            "Different temperatures should produce different attributions",
        )

    # ----- Comparison: IG-GIM vs plain IG -----

    def test_differs_from_plain_ig_on_attention_model(self):
        """IG-GIM should differ from plain IG when GIM hooks are active.

        Uses a model with the real Attention module so that the GIM
        gradient normalisation & TSG rules have an effect.
        """
        model = _ToyModelWithAttention()

        ig = IntegratedGradients(model, use_embeddings=True, steps=10)
        ig_gim = IntegratedGradientGIM(model, temperature=2.0, steps=10)

        attrs_ig = ig.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]
        attrs_ig_gim = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]

        self.assertTrue(
            torch.any(~torch.isclose(attrs_ig, attrs_ig_gim, atol=1e-6)),
            "IG-GIM should differ from plain IG on attention models",
        )

    def test_matches_plain_ig_at_temperature_one_simple_model(self):
        """On a model without Attention / LN, IG-GIM(T=1) ≈ plain IG.

        When there is no Attention module or LayerNorm to swap, the
        GIM context is a no-op, so both methods should agree.
        """
        model = _ToyModelNoAttention()
        ig = IntegratedGradients(model, use_embeddings=True, steps=10)
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=10)

        attrs_ig = ig.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]
        attrs_ig_gim = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]

        torch.testing.assert_close(
            attrs_ig, attrs_ig_gim, atol=1e-5, rtol=1e-4,
        )

    # ----- LayerNorm freeze -----

    def test_layernorm_freeze_affects_attributions(self):
        """With LayerNorm in the model, GIM hooks should change attributions.

        Compare IG-GIM against plain IG to confirm the LN freeze rule
        has an effect.
        """
        model = _ToyModelWithLayerNorm()
        ig = IntegratedGradients(model, use_embeddings=True, steps=10)
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=10)

        attrs_ig = ig.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]
        attrs_ig_gim = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]

        self.assertTrue(
            torch.any(~torch.isclose(attrs_ig, attrs_ig_gim, atol=1e-6)),
            "LayerNorm freeze in IG-GIM should produce different "
            "attributions than plain IG",
        )

    # ----- Model restoration -----

    def test_model_restored_after_attribution(self):
        """Attention and LayerNorm modules should be restored after attribution."""
        model = _ToyModelWithAttention()
        original_attn = model.attn
        self.assertIsInstance(original_attn, Attention)

        ig_gim = IntegratedGradientGIM(model, temperature=2.0, steps=5)
        ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )

        self.assertIs(
            model.attn, original_attn,
            "Attention module not restored after IG-GIM",
        )
        self.assertIsInstance(model.attn, Attention)

    def test_layernorm_restored_after_attribution(self):
        """LayerNorm module should be restored after attribution."""
        model = _ToyModelWithLayerNorm()
        original_ln = model.layer_norm
        self.assertIsInstance(original_ln, nn.LayerNorm)

        ig_gim = IntegratedGradientGIM(model, temperature=2.0, steps=5)
        ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )

        self.assertIs(
            model.layer_norm, original_ln,
            "LayerNorm not restored after IG-GIM",
        )
        self.assertIsInstance(model.layer_norm, nn.LayerNorm)

    # ----- Target class -----

    def test_auto_target_class(self):
        """When target_class_idx is None, should use predicted class."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=5)
        attrs = ig_gim.attribute(codes=self.tokens, label=self.labels)
        self.assertIn("codes", attrs)
        self.assertEqual(attrs["codes"].shape, self.tokens.shape)

    def test_different_target_classes(self):
        """Attributions for different target classes should differ."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=10)

        attrs_0 = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )["codes"]
        attrs_1 = ig_gim.attribute(
            codes=self.tokens, label=self.labels, target_class_idx=1,
        )["codes"]

        self.assertFalse(
            torch.allclose(attrs_0, attrs_1),
            "Different target classes should give different attributions",
        )

    # ----- Temporal tuple inputs -----

    def test_handles_temporal_tuple_inputs(self):
        """StageNet-style (time, value) tuples should work seamlessly."""
        model = _ToyModel(schema=("time", "value"))
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=5)

        time_indices = torch.arange(self.tokens.numel()).view_as(self.tokens).float()
        attrs = ig_gim.attribute(
            codes=(time_indices, self.tokens),
            label=self.labels,
            target_class_idx=0,
        )
        self.assertIn("codes", attrs)
        self.assertEqual(attrs["codes"].shape, self.tokens.shape)

    # ----- Steps override -----

    def test_steps_override(self):
        """Steps argument in attribute() should override init default."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=50)

        # Use steps=5 override — should return in reasonable time
        attrs = ig_gim.attribute(
            codes=self.tokens, label=self.labels,
            target_class_idx=0, steps=5,
        )
        self.assertIn("codes", attrs)

    # ----- Mixed features -----

    def test_mixed_token_and_continuous_features(self):
        """Should handle both token and continuous features correctly."""
        model = _ToyModelWithMixed()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=5)

        tokens = torch.tensor([[2, 5, 3, 1]])
        vitals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        labels = torch.zeros((1, 1))

        attrs = ig_gim.attribute(
            codes=tokens, vitals=vitals, label=labels, target_class_idx=0,
        )
        self.assertIn("codes", attrs)
        self.assertIn("vitals", attrs)
        self.assertEqual(attrs["codes"].shape, tokens.shape)
        self.assertEqual(attrs["vitals"].shape, vitals.shape)
        self.assertTrue(torch.isfinite(attrs["codes"]).all())
        self.assertTrue(torch.isfinite(attrs["vitals"]).all())

    # ----- Callable interface -----

    def test_callable_interface(self):
        """IG-GIM should be callable via __call__ (inherited from BaseInterpreter)."""
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model, temperature=1.0, steps=5)
        attrs = ig_gim(
            codes=self.tokens, label=self.labels, target_class_idx=0,
        )
        self.assertIn("codes", attrs)

    # ----- repr -----

    def test_repr(self):
        model = _ToyModel()
        ig_gim = IntegratedGradientGIM(model)
        self.assertIn("IntegratedGradientGIM", repr(ig_gim))


# ---------------------------------------------------------------------------
# Additional toy model without Attention (no GIM-swappable modules)
# ---------------------------------------------------------------------------

class _ToyModelNoAttention(BaseModel, Interpretable):
    """Simple model without Attention or LayerNorm — GIM hooks are no-ops."""

    def __init__(self, vocab_size=32, embedding_dim=4):
        dataset = _MockDataset(
            input_schema={"codes": "sequence"},
            output_schema={"label": "binary"},
            processors={"codes": _MockProcessor(("value",))},
        )
        super().__init__(dataset=dataset)
        self.feature_keys = ["codes"]
        self.label_keys = ["label"]
        self.mode = "binary"

        self.embedding_model = _ToyEmbeddingModel(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.2, -0.3, 0.4, 0.1]]))

    def forward_from_embedding(self, **kwargs):
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            emb = codes[schema.index("value")]
        else:
            emb = codes
        pooled = emb.mean(dim=1)
        logits = self.fc(pooled)
        label = kwargs.get("label")
        if label is None:
            label = torch.zeros_like(logits)
        return {
            "logit": logits,
            "y_prob": self.sigmoid(logits),
            "y_true": label,
            "loss": torch.zeros((), device=logits.device),
        }

    def forward(self, **kwargs):
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            val = codes[schema.index("value")]
        else:
            val = codes
        embedded = self.embedding_model({"codes": val})["codes"]
        if isinstance(codes, tuple):
            i = schema.index("value")
            kwargs["codes"] = codes[:i] + (embedded,) + codes[i + 1:]
        else:
            kwargs["codes"] = (embedded,)
        return self.forward_from_embedding(**kwargs)

    def get_embedding_model(self):
        return self.embedding_model


if __name__ == "__main__":
    unittest.main()
