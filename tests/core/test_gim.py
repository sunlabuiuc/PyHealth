import math
import unittest
from typing import Dict

import torch
import torch.nn as nn

from pyhealth.interpret.methods import GIM
from pyhealth.models import BaseModel


class _BinaryProcessor:
    def size(self) -> int:
        return 1


class _DummyBinaryDataset:
    """Minimal dataset stub that mimics the pieces BaseModel expects."""

    def __init__(self):
        self.input_schema = {"codes": "sequence"}
        self.output_schema = {"label": "binary"}
        self.output_processors = {"label": _BinaryProcessor()}


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
    """Small attention-style model exposing StageNet-compatible hooks."""

    def __init__(self, vocab_size: int = 32, embedding_dim: int = 4):
        super().__init__(dataset=_DummyBinaryDataset())
        self.feature_keys = ["codes"]
        self.label_keys = ["label"]
        self.mode = "binary"

        self.embedding_model = _ToyEmbeddingModel(vocab_size, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc = nn.Linear(embedding_dim, 1, bias=False)

        self._activation_hooks = None
        self.deeplift_hook_calls = 0

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            identity = torch.eye(self.query.in_features)
            self.query.weight.copy_(identity)
            self.key.weight.copy_(identity)
            self.value.weight.copy_(identity)
            self.fc.weight.copy_(torch.tensor([[0.2, -0.3, 0.4, 0.1]]))

    def set_deeplift_hooks(self, hooks) -> None:
        self.deeplift_hook_calls += 1
        self._activation_hooks = hooks

    def clear_deeplift_hooks(self) -> None:
        self._activation_hooks = None

    def _apply_activation(self, name: str, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._activation_hooks is not None and hasattr(self._activation_hooks, "apply"):
            return self._activation_hooks.apply(name, tensor, **kwargs)
        fn = getattr(torch, name)
        return fn(tensor, **kwargs)

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Dict[str, torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        emb = feature_embeddings["codes"]
        q = self.query(emb)
        k = self.key(emb)
        v = self.value(emb)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        weights = self._apply_activation("softmax", scores, dim=-1)
        context = torch.matmul(weights, v)
        pooled = context.mean(dim=1)
        logits = self.fc(pooled)

        label = kwargs.get("label")
        if label is None:
            label = torch.zeros_like(logits)

        return {
            "logit": logits,
            "y_prob": torch.sigmoid(logits),
            "y_true": label,
            "loss": torch.zeros((), device=logits.device),
        }


class _ToyGIMModelWithCustomHooks(_ToyGIMModel):
    """Variant that exposes dedicated set_gim_hooks/clear_gim_hooks."""

    def __init__(self):
        super().__init__()
        self.gim_hook_calls = 0

    def set_gim_hooks(self, hooks) -> None:
        self.gim_hook_calls += 1
        self._activation_hooks = hooks

    def clear_gim_hooks(self) -> None:
        self._activation_hooks = None


def _manual_token_attribution(
    model: _ToyGIMModel,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation mimicking GIM without temperature scaling."""

    embeddings = model.embedding_model({"codes": tokens})["codes"].detach()
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    output = model.forward_from_embedding({"codes": embeddings}, label=labels)
    logits = output["logit"].squeeze(-1)
    target = logits.sum()

    model.zero_grad(set_to_none=True)
    if embeddings.grad is not None:
        embeddings.grad.zero_()
    target.backward()

    grad = embeddings.grad.detach()
    token_attr = grad.sum(dim=-1)
    return token_attr


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
        """Raising the temperature must both attach hooks and change attributions."""

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

        self.assertEqual(model.deeplift_hook_calls, 1)
        self.assertFalse(torch.allclose(baseline_attr, hot_attr))

    def test_prefers_custom_gim_hooks(self):
        """Models exposing set_gim_hooks should bypass the DeepLIFT surface."""

        model = _ToyGIMModelWithCustomHooks()
        gim = GIM(model, temperature=2.0)
        gim.attribute(target_class_idx=0, codes=self.tokens, label=self.labels)

        self.assertEqual(model.gim_hook_calls, 1)
        self.assertEqual(model.deeplift_hook_calls, 0)

    def test_attributions_match_input_shape(self):
        """Collapsed gradients should align with the token tensor shape."""

        model = _ToyGIMModel()
        gim = GIM(model, temperature=1.0)

        attrs = gim.attribute(target_class_idx=0, codes=self.tokens, label=self.labels)
        self.assertEqual(tuple(attrs["codes"].shape), tuple(self.tokens.shape))

    def test_handles_temporal_tuple_inputs(self):
        """StageNet-style (time, value) tuples should be processed seamlessly."""

        model = _ToyGIMModel()
        gim = GIM(model, temperature=1.0)

        time_indices = torch.arange(self.tokens.numel()).view_as(self.tokens).float()
        attributions = gim.attribute(
            target_class_idx=0,
            codes=(time_indices, self.tokens),
            label=self.labels,
        )
        manual = _manual_token_attribution(model, self.tokens, self.labels)
        torch.testing.assert_close(attributions["codes"], manual, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
