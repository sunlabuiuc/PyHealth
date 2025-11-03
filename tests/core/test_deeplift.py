import unittest

import torch
import torch.nn as nn

from pyhealth.interpret.methods import DeepLift
from pyhealth.models import BaseModel


class _ToyDeepLiftModel(BaseModel):
    """Minimal model that exposes the DeepLIFT hook surface."""

    def __init__(self):
        super().__init__(dataset=None)
        self.feature_keys = ["x"]
        self.label_keys = ["y"]
        self.mode = "binary"

        self.linear1 = nn.Linear(2, 1, bias=True)
        self.linear2 = nn.Linear(1, 1, bias=True)

        self._activation_hooks = None

    # ------------------------------------------------------------------
    # Hook utilities mirroring StageNet integration
    # ------------------------------------------------------------------
    def set_deeplift_hooks(self, hooks) -> None:
        self._activation_hooks = hooks

    def clear_deeplift_hooks(self) -> None:
        self._activation_hooks = None

    def _apply_activation(self, name: str, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._activation_hooks is not None and hasattr(self._activation_hooks, "apply"):
            return self._activation_hooks.apply(name, tensor, **kwargs)
        fn = getattr(torch, name)
        return fn(tensor, **kwargs)

    # ------------------------------------------------------------------
    # Forward definition compatible with DeepLift(use_embeddings=False)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        pre_relu = self.linear1(x)
        hidden = self._apply_activation("relu", pre_relu)
        logit = self.linear2(hidden)
        y_prob = self._apply_activation("sigmoid", logit)

        return {
            "logit": y_prob,
            "y_prob": y_prob,
            "y_true": y.to(y_prob.device),
            "loss": torch.zeros((), device=y_prob.device),
        }


def _safe_division(numerator: torch.Tensor, denominator: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mask = denominator.abs() > eps
    safe_denominator = torch.where(mask, denominator, torch.ones_like(denominator))
    quotient = numerator / safe_denominator
    return torch.where(mask, quotient, fallback)


class TestDeepLift(unittest.TestCase):
    """Unit tests validating DeepLIFT against analytical expectations."""

    def setUp(self):
        self.model = _ToyDeepLiftModel()
        self.model.eval()

        with torch.no_grad():
            self.model.linear1.weight.copy_(torch.tensor([[1.5, -2.0]]))
            self.model.linear1.bias.copy_(torch.tensor([0.5]))
            self.model.linear2.weight.copy_(torch.tensor([[0.8]]))
            self.model.linear2.bias.copy_(torch.tensor([-0.2]))

        self.baseline = torch.tensor([[-0.5, 0.0]])
        self.labels = torch.zeros((1, 1))
        self.deeplift = DeepLift(self.model, use_embeddings=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _manual_deeplift(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute DeepLIFT contributions by hand using the Rescale rule."""

        w1 = self.model.linear1.weight.detach()
        b1 = self.model.linear1.bias.detach()
        w2 = self.model.linear2.weight.detach()
        b2 = self.model.linear2.bias.detach()

        a = torch.nn.functional.linear(inputs, w1, b1)
        a0 = torch.nn.functional.linear(self.baseline, w1, b1)
        delta_a = a - a0
        h = torch.relu(a)
        h0 = torch.relu(a0)
        delta_h = h - h0
        relu_deriv = (a > 0).to(inputs.dtype)
        relu_secant = _safe_division(delta_h, delta_a, relu_deriv)

        z = torch.nn.functional.linear(h, w2, b2)
        z0 = torch.nn.functional.linear(h0, w2, b2)
        delta_z = z - z0
        y = torch.sigmoid(z)
        y0 = torch.sigmoid(z0)
        delta_y = y - y0
        sigmoid_deriv = y * (1 - y)
        sigmoid_secant = _safe_division(delta_y, delta_z, sigmoid_deriv)

        delta_x = (inputs - self.baseline).squeeze(0)
        chain_multiplier = (
            w1.squeeze(0) * relu_secant.squeeze(0) * w2.squeeze(0) * sigmoid_secant.squeeze(0)
        )
        expected = delta_x * chain_multiplier
        return expected, delta_y.squeeze()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_rescale_matches_manual_chain(self):
        """DeepLIFT contributions should match the analytical Rescale solution."""

        inputs = torch.tensor([[1.2, -0.3]])
        attributions = self.deeplift.attribute(
            baseline={"x": self.baseline}, x=inputs, y=self.labels
        )

        contrib = attributions["x"].squeeze(0)
        expected, delta_y = self._manual_deeplift(inputs)

        torch.testing.assert_close(contrib, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(contrib.sum(), delta_y, atol=1e-5, rtol=1e-5)

    def test_state_reset_between_calls(self):
        """Multiple DeepLIFT calls should not leak activation state."""

        first_input = torch.tensor([[0.2, 0.1]])
        second_input = torch.tensor([[1.0, -1.0]])

        first_attr = self.deeplift.attribute(
            baseline={"x": self.baseline}, x=first_input, y=self.labels
        )
        second_attr = self.deeplift.attribute(
            baseline={"x": self.baseline}, x=second_input, y=self.labels
        )

        first_expected, first_delta_y = self._manual_deeplift(first_input)
        second_expected, second_delta_y = self._manual_deeplift(second_input)

        torch.testing.assert_close(first_attr["x"].squeeze(0), first_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(second_attr["x"].squeeze(0), second_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first_attr["x"].sum(), first_delta_y, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(second_attr["x"].sum(), second_delta_y, atol=1e-5, rtol=1e-5)

    def test_zero_delta_input_returns_zero_attribution(self):
        """If inputs equal the baseline, contributions must be zero."""

        inputs = self.baseline.clone()
        attributions = self.deeplift.attribute(
            baseline={"x": self.baseline}, x=inputs, y=self.labels
        )

        self.assertTrue(torch.allclose(attributions["x"], torch.zeros_like(inputs)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
