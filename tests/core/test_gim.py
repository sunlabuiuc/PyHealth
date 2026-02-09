import math
import unittest
from typing import Dict

import torch
import torch.nn as nn

from pyhealth.interpret.methods import GIM
from pyhealth.models import BaseModel


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockProcessor:
    """Mock feature processor with configurable schema."""

    def __init__(self, schema_tuple=("value",)):
        self._schema = schema_tuple

    def schema(self):
        return self._schema

    def is_token(self):
        return True


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
    """Deterministic embedding lookup ensuring reproducible gradients."""

    def __init__(self, vocab_size: int = 32, embedding_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        with torch.no_grad():
            weights = torch.arange(vocab_size * embedding_dim).float()
            weights = weights.view(vocab_size, embedding_dim)
            self.embedding.weight.copy_(weights / float(vocab_size))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: self.embedding(val.long()) for key, val in inputs.items()}


class _ToyGIMModel(BaseModel):
    """Small attention-style model with module-based nonlinearities.

    Follows the new API conventions: ``forward_from_embedding(**kwargs)``
    receives tuples keyed by feature name, and ``get_embedding_model()``
    returns the embedding module.
    """

    def __init__(self, vocab_size: int = 32, embedding_dim: int = 4, schema=("value",)):
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

        # Nonlinearities as modules so interpretability can swap them
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

    def forward_from_embedding(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        codes = kwargs["codes"]
        # Unpack from tuple using processor schema
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

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        codes = kwargs["codes"]
        if isinstance(codes, tuple):
            schema = self.dataset.input_processors["codes"].schema()
            val = codes[schema.index("value")]
        else:
            val = codes

        embedded = self.embedding_model({"codes": val})["codes"]

        # Replace value in input tuple
        if isinstance(codes, tuple):
            i = schema.index("value")
            kwargs["codes"] = codes[:i] + (embedded,) + codes[i + 1:]
        else:
            kwargs["codes"] = (embedded,)

        return self.forward_from_embedding(**kwargs)

    def get_embedding_model(self) -> nn.Module | None:
        return self.embedding_model


# ---------------------------------------------------------------------------
# Manual reference implementation
# ---------------------------------------------------------------------------

def _manual_token_attribution(
    model: _ToyGIMModel,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation mimicking GIM without temperature scaling."""

    embeddings = model.embedding_model({"codes": tokens})["codes"].detach()
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    # Build tuple matching the processor schema
    schema = model.dataset.input_processors["codes"].schema()
    parts: list[torch.Tensor] = []
    for s in schema:
        if s == "value":
            parts.append(embeddings)
        elif s == "time":
            parts.append(torch.zeros(tokens.shape, dtype=torch.float32))
        else:
            parts.append(torch.zeros(tokens.shape, dtype=torch.int32))
    output = model.forward_from_embedding(codes=tuple(parts), label=labels)
    logits = output["logit"]

    # Binary mode: target class 0 → sign = -1 (2*0 - 1)
    target = (-1.0 * logits).sum()

    model.zero_grad(set_to_none=True)
    if embeddings.grad is not None:
        embeddings.grad.zero_()
    target.backward()

    grad = embeddings.grad.detach()
    token_attr = grad.sum(dim=-1)
    return token_attr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGIM(unittest.TestCase):
    """Unit tests validating the PyHealth GIM interpreter."""

    def setUp(self):
        torch.manual_seed(7)
        self.tokens = torch.tensor([[2, 5, 3, 1]])
        self.labels = torch.zeros((1, 1))

    def test_matches_manual_gradient_when_temperature_one(self):
        """Temperature=1 should collapse to plain gradients."""

        model = _ToyGIMModel()
        gim = GIM(model, temperature=1.0)

        attributions = gim.attribute(
            target_class_idx=0,
            codes=self.tokens,
            label=self.labels,
        )
        manual = _manual_token_attribution(model, self.tokens, self.labels)
        torch.testing.assert_close(attributions["codes"], manual, atol=1e-6, rtol=1e-5)

    def test_temperature_hooks_modify_gradients(self):
        """Raising the temperature should change attributions."""

        model = _ToyGIMModel()
        baseline_gim = GIM(model, temperature=1.0)
        hot_gim = GIM(model, temperature=2.0)

        baseline_attr = baseline_gim.attribute(
            target_class_idx=0,
            codes=self.tokens,
            label=self.labels,
        )["codes"]
        hot_attr = hot_gim.attribute(
            target_class_idx=0,
            codes=self.tokens,
            label=self.labels,
        )["codes"]

        self.assertTrue(torch.any(~torch.isclose(baseline_attr, hot_attr)))

    def test_attributions_match_input_shape(self):
        """Collapsed gradients should align with the token tensor shape."""

        model = _ToyGIMModel()
        gim = GIM(model, temperature=1.0)

        attrs = gim.attribute(target_class_idx=0, codes=self.tokens, label=self.labels)
        self.assertEqual(tuple(attrs["codes"].shape), tuple(self.tokens.shape))

    def test_handles_temporal_tuple_inputs(self):
        """StageNet-style (time, value) tuples should be processed seamlessly."""

        model = _ToyGIMModel(schema=("time", "value"))
        gim = GIM(model, temperature=1.0)

        time_indices = torch.arange(self.tokens.numel()).view_as(self.tokens).float()
        attributions = gim.attribute(
            target_class_idx=0,
            codes=(time_indices, self.tokens),
            label=self.labels,
        )

        # Compute manual reference (time is ignored by the toy model)
        manual = _manual_token_attribution(model, self.tokens, self.labels)
        torch.testing.assert_close(attributions["codes"], manual, atol=1e-6, rtol=1e-5)

    def test_auto_target_class(self):
        """When target_class_idx is None, GIM should use predicted class."""

        model = _ToyGIMModel()
        gim = GIM(model, temperature=1.0)

        # Should not raise
        attrs = gim.attribute(codes=self.tokens, label=self.labels)
        self.assertIn("codes", attrs)
        self.assertEqual(tuple(attrs["codes"].shape), tuple(self.tokens.shape))


if __name__ == "__main__":
    unittest.main()
