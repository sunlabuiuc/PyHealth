"""Integrated Gradients with Gradient Interaction Modifications (IG-GIM).

This module combines Integrated Gradients (IG) as the core attribution
method with GIM's backward-pass modifications for attention and
LayerNorm.  The result is a path-integral attribution that is not
distorted by the self-repair artefacts that attention and LayerNorm
introduce in the backward pass.

Concretely, during every interpolation step of IG the following GIM
rules are active:

1. **Temperature-Scaled Gradients (TSG):** The softmax Jacobian inside
   attention is recomputed at a higher temperature, exposing interactions
   that are otherwise suppressed by softmax redistribution.
2. **Frozen LayerNorm:** ``nn.LayerNorm`` backward passes treat the
   running mean / variance as constants, preventing the normalisation
   from masking feature contributions.
3. **Gradient normalisation (uniform division):** ``Q·K^T`` and
   ``attn·V`` matrix multiplications in attention divide gradients by
   the fan-in (=2), which compounds to the standard Q÷4, K÷4, V÷2
   scaling from the GIM paper.

References:
    * Sundararajan et al., "Axiomatic Attribution for Deep Networks",
      ICML 2017. https://arxiv.org/abs/1703.01365
    * Edin et al., "Gradient Interaction Modifications", 2025.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pyhealth.models import BaseModel

from .base_interpreter import BaseInterpreter
from pyhealth.interpret.api import Interpretable
from .gim import _GIMHookContext


class IntegratedGradientGIM(BaseInterpreter):
    """Integrated Gradients with GIM backward-pass modifications.

    This interpreter runs the standard IG path-integral computation but
    wraps every forward/backward step inside a GIM context so that the
    accumulated gradients are free of the self-repair artefacts caused
    by softmax redistribution and LayerNorm re-centring.

    Args:
        model: Trained PyHealth model that implements
            ``forward_from_embedding()`` and ``get_embedding_model()``.
        temperature: Softmax temperature used for the TSG backward rule.
            Values > 1 flatten the softmax Jacobian, counteracting
            gradient suppression.  ``2.0`` is the paper's recommended
            default.
        steps: Default number of Riemann-sum interpolation steps.  Can
            be overridden per call in :meth:`attribute`.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods.ig_gim import IntegratedGradientGIM
        >>>
        >>> # Assume ``dataset`` and a trained Transformer are available.
        >>> model = Transformer(dataset=dataset).eval()
        >>> ig_gim = IntegratedGradientGIM(model, temperature=2.0, steps=50)
        >>>
        >>> batch = next(iter(get_dataloader(dataset, batch_size=1)))
        >>> attributions = ig_gim.attribute(**batch)
        >>> print({k: v.shape for k, v in attributions.items()})
    """

    def __init__(
        self,
        model: BaseModel,
        temperature: float = 2.0,
        steps: int = 50,
    ):
        super().__init__(model)
        if not isinstance(model, Interpretable):
            raise ValueError("Model must implement Interpretable interface")
        self.model = model

        self.temperature = max(float(temperature), 1.0)
        self.steps = steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: Optional[int] = None,
        target_class_idx: Optional[int] = None,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute IG-GIM attributions for a batch.

        Args:
            baseline: Per-feature baseline tensors.  ``None`` uses the
                default strategy (UNK-token for discrete features,
                near-zero for continuous features).
            steps: Number of interpolation steps.  Overrides the instance
                default when given.
            target_class_idx: Target class for attribution.  ``None``
                uses the model's predicted class.
            **kwargs: Dataloader batch (feature tensors + optional labels).

        Returns:
            Dictionary mapping each feature key to an attribution tensor
            whose shape matches the raw input value tensor.
        """
        if steps is None:
            steps = self.steps

        device = next(self.model.parameters()).device

        # ----- unpack inputs -----
        inputs = {
            k: (v,) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if k in self.model.feature_keys
        }

        values: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            schema = self.model.dataset.input_processors[k].schema()
            values[k] = v[schema.index("value")]
            if "mask" in schema:
                masks[k] = v[schema.index("mask")]
            else:
                val = v[schema.index("value")]
                processor = self.model.dataset.input_processors[k]
                if processor.is_token():
                    masks[k] = (val != 0).int()
                else:
                    # For continuous features, check whether the entire
                    # feature vector at each timestep is zero (padding)
                    # rather than per-element, so valid 0.0 values are
                    # not masked out.
                    if val.dim() >= 3:
                        masks[k] = (val.abs().sum(dim=-1) != 0).int()
                    else:
                        masks[k] = (val != 0).int()

        for k, v in inputs.items():
            if "mask" not in self.model.dataset.input_processors[k].schema():
                inputs[k] = (*v, masks[k])

        # ----- target class -----
        with torch.no_grad():
            base_logits = self.model.forward(**inputs)["logit"]

        mode = self._prediction_mode()
        target = self._resolve_target(
            base_logits, mode, target_class_idx, device
        )

        # ----- baselines -----
        if baseline is None:
            baselines = self._generate_baseline(values)
        else:
            baselines = {
                k: v.to(device)
                for k, v in baseline.items()
                if k in self.model.feature_keys
            }

        # Save raw shapes for final re-mapping
        shapes = {k: v.shape for k, v in values.items()}

        # ----- embed token features -----
        embedding_model = self.model.get_embedding_model()
        assert embedding_model is not None

        token_keys = {
            k for k in values
            if self.model.dataset.input_processors[k].is_token()
        }
        continuous_keys = set(values.keys()) - token_keys

        if token_keys:
            embedded_vals = embedding_model({k: values[k] for k in token_keys})
            for k in token_keys:
                values[k] = embedded_vals[k]
            token_baselines = {
                k: baselines[k] for k in token_keys if k in baselines
            }
            if token_baselines:
                embedded_bl = embedding_model(token_baselines)
                for k in token_baselines:
                    baselines[k] = embedded_bl[k]

        # ----- IG loop with GIM hooks -----
        attributions = self._integrated_gradients_gim(
            inputs=inputs,
            xs=values,
            bs=baselines,
            steps=steps,
            target=target,
            token_keys=token_keys,
            continuous_keys=continuous_keys,
        )

        return self._map_to_input_shapes(attributions, shapes)

    # ------------------------------------------------------------------
    # Core IG + GIM loop
    # ------------------------------------------------------------------
    def _integrated_gradients_gim(
        self,
        inputs: Dict[str, tuple[torch.Tensor, ...]],
        xs: Dict[str, torch.Tensor],
        bs: Dict[str, torch.Tensor],
        steps: int,
        target: torch.Tensor,
        token_keys: set[str],
        continuous_keys: set[str],
    ) -> Dict[str, torch.Tensor]:
        """Run the Riemann-sum IG loop with GIM's backward modifications.

        At every interpolation step the forward/backward pass is wrapped
        in ``_GIMHookContext`` so that:

        * Attention softmax uses temperature-scaled gradients (TSG).
        * LayerNorm treats statistics as constants.
        * Q·K^T / attn·V gradients are uniformly divided.

        Returns:
            Per-feature attribution tensors (embedding dim already summed
            out for token features).
        """
        keys = sorted(xs.keys())
        avg_gradients = {key: torch.zeros_like(xs[key]) for key in keys}

        for step_idx in range(steps + 1):
            alpha = step_idx / steps

            # ---- interpolate ----
            interpolated: dict[str, torch.Tensor] = {}
            for key in keys:
                interp = bs[key] + alpha * (xs[key] - bs[key])
                interp = interp.detach().requires_grad_(True)
                interp.retain_grad()
                interpolated[key] = interp

            # ---- build forward_inputs ----
            forward_inputs = inputs.copy()
            for k in forward_inputs.keys():
                schema = self.model.dataset.input_processors[k].schema()
                val_idx = schema.index("value")
                if k in continuous_keys:
                    embedding_model = self.model.get_embedding_model()
                    assert embedding_model is not None
                    embedded_cont = embedding_model(
                        {k: interpolated[k]}
                    )[k]
                    forward_inputs[k] = (
                        *forward_inputs[k][:val_idx],
                        embedded_cont,
                        *forward_inputs[k][val_idx + 1:],
                    )
                else:
                    forward_inputs[k] = (
                        *forward_inputs[k][:val_idx],
                        interpolated[k],
                        *forward_inputs[k][val_idx + 1:],
                    )

            # ---- forward + backward inside GIM context ----
            self.model.zero_grad(set_to_none=True)
            for emb in interpolated.values():
                if emb.grad is not None:
                    emb.grad.zero_()

            with _GIMHookContext(self.model, self.temperature):
                output = self.model.forward_from_embedding(**forward_inputs)
                logits = output["logit"]
                target_output = self._compute_target_output(logits, target)

                self.model.zero_grad(set_to_none=True)
                target_output.backward(retain_graph=True)

            # ---- accumulate gradients ----
            for key in keys:
                emb = interpolated[key]
                if emb.grad is not None:
                    avg_gradients[key] += emb.grad.detach()

        # ---- average & compute attributions ----
        for key in keys:
            avg_gradients[key] /= steps + 1

        attributions: dict[str, torch.Tensor] = {}
        for key in keys:
            delta = xs[key] - bs[key]
            attr = delta * avg_gradients[key]
            if key in token_keys and attr.dim() >= 3:
                attr = attr.sum(dim=-1)
            attributions[key] = attr

        return attributions

    # ------------------------------------------------------------------
    # Target helpers (shared logic with IG / GIM)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_target(
        logits: torch.Tensor,
        mode: str,
        target_class_idx: Optional[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Convert logits and optional class index into a target tensor."""
        if mode == "binary":
            if target_class_idx is not None:
                return torch.tensor([target_class_idx], device=device)
            return (torch.sigmoid(logits) > 0.5).long()

        if mode == "multiclass":
            if target_class_idx is not None:
                return F.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=logits.shape[-1],
                ).float()
            target = torch.argmax(logits, dim=-1)
            return F.one_hot(target, num_classes=logits.shape[-1]).float()

        if mode == "multilabel":
            if target_class_idx is not None:
                return F.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=logits.shape[-1],
                ).float()
            return (torch.sigmoid(logits) > 0.5).float()

        raise ValueError(f"Unsupported prediction mode: {mode}")

    def _compute_target_output(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar target output for backpropagation."""
        target_f = target.to(logits.device).float()
        mode = self._prediction_mode()

        if mode == "binary":
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(-1)
            target_f = target_f.expand_as(logits)
            signs = 2.0 * target_f - 1.0
            return (signs * logits).sum()
        else:
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(0)
            target_f = target_f.expand_as(logits)
            return (target_f * logits).sum()

    # ------------------------------------------------------------------
    # Baseline generation
    # ------------------------------------------------------------------
    def _generate_baseline(
        self,
        values: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Generate baselines (UNK-token for discrete, near-zero for continuous)."""
        baselines: dict[str, torch.Tensor] = {}
        for k, v in values.items():
            processor = self.model.dataset.input_processors[k]
            if processor.is_token():
                baselines[k] = torch.ones_like(v)
            else:
                baselines[k] = torch.zeros_like(v) + 1e-2
        return baselines

    # ------------------------------------------------------------------
    # Shape mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _map_to_input_shapes(
        attr_values: Dict[str, torch.Tensor],
        input_shapes: dict,
    ) -> Dict[str, torch.Tensor]:
        """Re-shape attribution tensors to match original input shapes."""
        mapped: dict[str, torch.Tensor] = {}
        for key, values in attr_values.items():
            if key not in input_shapes:
                mapped[key] = values
                continue
            orig_shape = input_shapes[key]
            if values.shape == orig_shape:
                mapped[key] = values
                continue
            reshaped = values
            while len(reshaped.shape) < len(orig_shape):
                reshaped = reshaped.unsqueeze(-1)
            if reshaped.shape != orig_shape:
                reshaped = reshaped.expand(orig_shape)
            mapped[key] = reshaped
        return mapped
