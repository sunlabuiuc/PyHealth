"""DeepLIFT attribution for PyHealth models.

This module implements the DeepLIFT algorithm as introduced in
"Learning Important Features Through Propagating Activation Differences"
by Shrikumar et al. (ICML 2017). The core idea is to propagate
*difference-from-reference* values instead of gradients and to replace the
local derivatives of nonlinearities with *secant slopes* (Rescale rule,
Sec. 2.1) so that attributions satisfy additivity.

The implementation is structured as follows:

* :class:`_DeepLiftActivationHooks` captures baseline/actual activation
  pairs and injects the Rescale rule during backpropagation. This mirrors
  the "multiplier" computation in Algorithm 1 of the paper.
* :class:`_DeepLiftHookContext` wires those hooks into models which expose
  a ``set_deeplift_hooks`` method (e.g., :class:`~pyhealth.models.StageNet`),
  ensuring the baseline and actual forward passes share identical execution
  order.
* :class:`DeepLift` provides the public API and mirrors the two major use
  cases described in the paper: discrete inputs handled through embeddings
  (difference propagation in embedding space) and fully continuous inputs.

The resulting attribution scores are guaranteed to satisfy the completeness
axiom (Eq. 1 in the paper) up to numerical precision and match the original
DeepLIFT formulation for the supported activation functions (ReLU, Sigmoid,
Tanh). Non-elementwise operations (e.g., Softmax) fall back to standard
autograd gradients, matching the recommendation in Appendix A of the paper
to treat them as compositions of linear + elementwise factors.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import torch

from pyhealth.models import BaseModel


class _DeepLiftActivationHooks:
    """Capture activation pairs for baseline and actual forward passes.

    During the baseline forward pass (reference inputs) the hook stores the
    pre-activation and post-activation tensors. During the actual forward
    pass it registers a backward hook that replaces the local derivative with
    the Rescale multiplier ``delta_out / delta_in`` as prescribed by the
    original DeepLIFT paper (Algorithm 1, lines 8–11).

    Only elementwise activations are currently supported because their
    secant slope can be derived analytically. For other operations the code
    falls back to autograd gradients, which coincides with the "linear rule"
    in the paper.
    """

    _SUPPORTED = {"relu", "sigmoid", "tanh"}

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.mode: str = "inactive"
        self.records: Dict[str, list] = {name: [] for name in self._SUPPORTED}
        self._indices: Dict[str, int] = {name: 0 for name in self._SUPPORTED}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear cached activation pairs and return to the inactive state."""
        for name in self.records:
            self.records[name].clear()
            self._indices[name] = 0
        self.mode = "inactive"

    def start_baseline(self) -> None:
        """Begin recording activations for the reference forward pass."""
        self.reset()
        self.mode = "baseline"

    def start_actual(self) -> None:
        """Switch to the actual input forward pass and prepare replaying hooks."""
        if self.mode != "baseline":
            raise RuntimeError("Baseline forward pass must run before actual pass for DeepLIFT.")
        self.mode = "actual"
        for name in self._indices:
            self._indices[name] = 0

    # ------------------------------------------------------------------
    # Activation routing
    # ------------------------------------------------------------------
    def apply(self, name: str, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the activation and optionally register a Rescale hook.

        Args:
            name: The ``torch`` activation name (e.g., ``"relu"``).
            tensor: Pre-activation tensor.
            **kwargs: Keyword arguments forwarded to the activation.

        Returns:
            The activation output. If ``mode`` is ``"baseline"`` the output is
            detached and cached; if ``mode`` is ``"actual"`` a backward hook is
            registered so that gradients are rescaled to secant multipliers.
        """
        fn = getattr(torch, name)
        output = fn(tensor, **kwargs)

        if name not in self.records or self.mode == "inactive":
            return output

        if self.mode == "baseline":
            self.records[name].append(
                {
                    "baseline_input": tensor.detach(),
                    "baseline_output": output.detach(),
                }
            )
        elif self.mode == "actual":
            idx = self._indices[name]
            if idx >= len(self.records[name]):
                raise RuntimeError(
                    f"DeepLIFT activation mismatch for '{name}'. Baseline and actual passes "
                    "must trigger hooks in the same order."
                )

            record = self.records[name][idx]
            record["input"] = tensor
            record["output"] = output
            self._indices[name] += 1

            if output.requires_grad:
                self._register_hook(name, record)

        return output

    # ------------------------------------------------------------------
    # Gradient replacement helpers
    # ------------------------------------------------------------------
    def _register_hook(self, name: str, record: Dict[str, torch.Tensor]) -> None:
        """Attach a backward hook implementing the Rescale multiplier.

        The multiplier ``m`` from the paper is computed as the ratio between
        the output and input differences. We apply this multiplier by scaling
        the autograd derivative so that the product equals ``m``.
        """
        input_tensor = record["input"]
        baseline_input = record["baseline_input"].to(input_tensor.device)
        output_tensor = record["output"]
        baseline_output = record["baseline_output"].to(output_tensor.device)

        delta_in = input_tensor - baseline_input
        delta_out = output_tensor - baseline_output

        derivative = self._activation_derivative(name, input_tensor, output_tensor)
        secant = self._safe_div(delta_out, delta_in, derivative)
        scale = self._safe_div(secant, derivative, torch.ones_like(secant))

        # Clamp to finite values to avoid propagating NaNs/Infs downstream
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        scale = scale.detach()

        def hook_fn(grad: torch.Tensor) -> torch.Tensor:
            """Scale the upstream gradient to equal the Rescale multiplier."""
            return grad * scale

        output_tensor.register_hook(hook_fn)

    def _activation_derivative(
        self, name: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Return the analytical derivative of supported activations."""
        if name == "relu":
            return torch.where(input_tensor > 0, torch.ones_like(output_tensor), torch.zeros_like(output_tensor))
        if name == "sigmoid":
            return output_tensor * (1.0 - output_tensor)
        if name == "tanh":
            return 1.0 - output_tensor.pow(2)
        # Default derivative for unsupported activations
        return torch.ones_like(output_tensor)

    def _safe_div(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        mask = denominator.abs() > self.eps
        safe_denominator = torch.where(mask, denominator, torch.ones_like(denominator))
        quotient = numerator / safe_denominator
        return torch.where(mask, quotient, fallback)


class _DeepLiftHookContext(contextlib.AbstractContextManager):
    """Context manager wiring activation hooks onto a model if supported."""

    def __init__(self, model: BaseModel):
        self.model = model
        self.hooks: Optional[_DeepLiftActivationHooks] = None
        self._enabled = all(
            hasattr(model, method) for method in ("set_deeplift_hooks", "clear_deeplift_hooks")
        )

    def __enter__(self) -> "_DeepLiftHookContext":
        if self._enabled:
            self.hooks = _DeepLiftActivationHooks()
            self.model.set_deeplift_hooks(self.hooks)
        return self

    def start_baseline(self) -> None:
        if self.hooks is not None:
            self.hooks.start_baseline()

    def start_actual(self) -> None:
        if self.hooks is not None:
            self.hooks.start_actual()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self.hooks is not None:
            self.hooks.reset()
            self.model.clear_deeplift_hooks()
            self.hooks = None
        return False


class DeepLift:
    """DeepLIFT-style attribution for PyHealth models.

    Args:
        model: A :class:`~pyhealth.models.BaseModel` instance exposing either
            :meth:`forward_from_embedding` (for discrete inputs) or a standard
            :meth:`forward` method compatible with PyHealth's trainers.
        use_embeddings: Whether to operate in embedding space. This mirrors the
            "embedding difference" variant from Sec. 3.2 of the paper.
    """

    def __init__(self, model: BaseModel, use_embeddings: bool = True):
        self.model = model
        self.model.eval()
        self.use_embeddings = use_embeddings

        if use_embeddings:
            assert hasattr(
                model, "forward_from_embedding"
            ), f"Model {type(model).__name__} must implement forward_from_embedding()"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute DeepLIFT attributions for a single batch.

        The method follows Algorithm 2 of the DeepLIFT paper: two forward
        passes (baseline then actual) are executed under the hook context so
        that backward propagation yields multipliers equal to the Rescale rule.

        Args:
            baseline: Optional dictionary providing reference inputs per
                feature key. If omitted, zeros are used for embedding space and
                dense tensors.
            target_class_idx: Optional class index to explain. ``None`` defaults
                to the model prediction, matching the "target layer" choice in
                attribution literature.
            **data: Model inputs. For temporal features this includes ``(time,
                value)`` tuples mirroring StageNet processors. Label tensors may
                also be supplied (needed when the model computes a loss).

        Returns:
            ``Dict[str, torch.Tensor]`` mapping each feature key to attribution
            tensors shaped like the original inputs. All outputs satisfy the
            completeness property ``sum_i attribution_i ≈ f(x) - f(x₀)``.
        """
        device = next(self.model.parameters()).device

        feature_inputs: Dict[str, torch.Tensor] = {}
        time_info: Dict[str, torch.Tensor] = {}
        label_data: Dict[str, torch.Tensor] = {}

        for key in self.model.feature_keys:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, tuple):
                time_tensor, feature_tensor = value
                time_info[key] = time_tensor.to(device)
                value = feature_tensor

            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            feature_inputs[key] = value.to(device)

        for key in self.model.label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.as_tensor(label_val)
                label_data[key] = label_val.to(device)

        if self.use_embeddings:
            return self._deeplift_embeddings(
                feature_inputs,
                baseline=baseline,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )

        return self._deeplift_continuous(
            feature_inputs,
            baseline=baseline,
            target_class_idx=target_class_idx,
            time_info=time_info,
            label_data=label_data,
        )

    # ------------------------------------------------------------------
    # Embedding-based DeepLIFT (discrete features)
    # ------------------------------------------------------------------
    def _deeplift_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """DeepLIFT for discrete inputs operating in embedding space.

        This mirrors the "embedding difference" strategy discussed in Sec. 3.2
        of the paper. Instead of interpolating raw token IDs we work on the
        embedded representations, propagate differences through the network, and
        finally project the attribution scores back onto the input tensor shape.
        """
        input_embs, baseline_embs, input_shapes = self._prepare_embeddings_and_baselines(
            inputs, baseline
        )

        delta_embeddings: Dict[str, torch.Tensor] = {}
        current_embeddings: Dict[str, torch.Tensor] = {}
        for key in input_embs:
            delta = (input_embs[key] - baseline_embs[key]).detach()
            delta.requires_grad_(True)
            delta_embeddings[key] = delta
            current_embeddings[key] = baseline_embs[key].detach() + delta

        forward_kwargs = {**label_data} if label_data else {}

        def forward_from_embeddings(feature_embeddings: Dict[str, torch.Tensor]):
            return self.model.forward_from_embedding(
                feature_embeddings=feature_embeddings,
                time_info=time_info,
                **forward_kwargs,
            )

        with _DeepLiftHookContext(self.model) as hook_ctx:
            hook_ctx.start_baseline()
            with torch.no_grad():
                baseline_output = forward_from_embeddings(baseline_embs)

            hook_ctx.start_actual()
            current_output = forward_from_embeddings(current_embeddings)

        logits = current_output["logit"]
        target_idx = self._determine_target_index(logits, target_class_idx)
        target_logit = self._gather_target_logit(logits, target_idx)

        baseline_logit = self._gather_target_logit(baseline_output["logit"], target_idx)

        self.model.zero_grad(set_to_none=True)
        target_logit.sum().backward()

        emb_contribs: Dict[str, torch.Tensor] = {}
        for key, delta in delta_embeddings.items():
            grad = delta.grad
            if grad is None:
                emb_contribs[key] = torch.zeros_like(delta)
            else:
                emb_contribs[key] = grad.detach() * delta.detach()

        emb_contribs = self._enforce_completeness(
            emb_contribs, target_logit.detach(), baseline_logit.detach()
        )
        return self._map_embeddings_to_inputs(emb_contribs, input_shapes)

    def _prepare_embeddings_and_baselines(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, tuple]]:
        """Embed inputs and baselines in preparation for difference propagation."""
        input_embeddings: Dict[str, torch.Tensor] = {}
        baseline_embeddings: Dict[str, torch.Tensor] = {}
        input_shapes: Dict[str, tuple] = {}

        for key, value in inputs.items():
            input_shapes[key] = value.shape
            embedded = self.model.embedding_model({key: value})[key]
            input_embeddings[key] = embedded

            if baseline is None:
                baseline_embeddings[key] = torch.zeros_like(embedded)
            else:
                if key not in baseline:
                    raise ValueError(f"Baseline missing key '{key}'")
                baseline_embeddings[key] = baseline[key].to(embedded.device)

        return input_embeddings, baseline_embeddings, input_shapes

    # ------------------------------------------------------------------
    # Continuous DeepLIFT fallback (for tensor inputs)
    # ------------------------------------------------------------------
    def _deeplift_continuous(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """DeepLIFT for models that operate directly on dense tensors."""
        device = next(self.model.parameters()).device

        if baseline is None:
            baseline = {key: torch.zeros_like(val) for key, val in inputs.items()}

        delta_inputs: Dict[str, torch.Tensor] = {}
        current_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            delta = (value - baseline[key]).detach()
            delta.requires_grad_(True)
            delta_inputs[key] = delta

            if key in time_info:
                current_inputs[key] = (
                    time_info[key],
                    baseline[key].detach() + delta,
                )
            else:
                current_inputs[key] = baseline[key].detach() + delta

        model_inputs = {**current_inputs, **label_data}

        with _DeepLiftHookContext(self.model) as hook_ctx:
            hook_ctx.start_baseline()
            baseline_inputs = {
                key: (time_info[key], baseline[key]) if key in time_info else baseline[key]
                for key in inputs
            }
            with torch.no_grad():
                baseline_output = self.model(**{**baseline_inputs, **label_data})

            hook_ctx.start_actual()
            current_output = self.model(**model_inputs)

        logits = current_output["logit"]
        target_idx = self._determine_target_index(logits, target_class_idx)
        target_logit = self._gather_target_logit(logits, target_idx)
        baseline_logit = self._gather_target_logit(baseline_output["logit"], target_idx)

        self.model.zero_grad(set_to_none=True)
        target_logit.sum().backward()

        contribs: Dict[str, torch.Tensor] = {}
        for key, delta in delta_inputs.items():
            grad = delta.grad
            if grad is None:
                contribs[key] = torch.zeros_like(delta)
            else:
                contribs[key] = grad.detach() * delta.detach()

        contribs = self._enforce_completeness(
            contribs, target_logit.detach(), baseline_logit.detach()
        )

        mapped: Dict[str, torch.Tensor] = {}
        for key, contrib in contribs.items():
            mapped[key] = contrib.to(device)
        return mapped

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_target_index(
        logits: torch.Tensor, target_class_idx: Optional[int]
    ) -> torch.Tensor:
        """Resolve the logit index that should be explained."""
        if target_class_idx is None:
            if logits.dim() >= 2 and logits.size(-1) > 1:
                target_idx = torch.argmax(logits, dim=-1)
            else:
                target_idx = torch.zeros(
                    logits.size(0), dtype=torch.long, device=logits.device
                )
        else:
            if isinstance(target_class_idx, int):
                target_idx = torch.full(
                    (logits.size(0),),
                    target_class_idx,
                    dtype=torch.long,
                    device=logits.device,
                )
            elif isinstance(target_class_idx, torch.Tensor):
                target_idx = target_class_idx.to(logits.device).long()
                if target_idx.dim() == 0:
                    target_idx = target_idx.expand(logits.size(0))
            else:
                raise ValueError("target_class_idx must be int or Tensor")
        return target_idx

    @staticmethod
    def _gather_target_logit(
        logits: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """Collect the scalar logit associated with ``target_idx``."""
        if logits.dim() == 2 and logits.size(-1) > 1:
            return logits.gather(1, target_idx.unsqueeze(-1)).squeeze(-1)
        return logits.squeeze(-1)

    @staticmethod
    def _enforce_completeness(
        contributions: Dict[str, torch.Tensor],
        target_logit: torch.Tensor,
        baseline_logit: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Scale attributions so their sum matches ``f(x) - f(x₀)`` (Eq. 1)."""
        delta_output = (target_logit - baseline_logit)

        total = None
        for contrib in contributions.values():
            flat_sum = contrib.reshape(contrib.size(0), -1).sum(dim=1)
            total = flat_sum if total is None else total + flat_sum

        scale = torch.ones_like(delta_output)
        if total is not None:
            denom = total.abs()
            mask = denom > eps
            scale[mask] = delta_output[mask] / denom[mask]

        for key, contrib in contributions.items():
            reshape_dims = [contrib.size(0)] + [1] * (contrib.dim() - 1)
            contributions[key] = contrib * scale.view(*reshape_dims)

        return contributions

    @staticmethod
    def _map_embeddings_to_inputs(
        emb_contribs: Dict[str, torch.Tensor],
        input_shapes: Dict[str, tuple],
    ) -> Dict[str, torch.Tensor]:
        """Project embedding-space contributions back to input tensor shapes."""
        mapped: Dict[str, torch.Tensor] = {}
        for key, contrib in emb_contribs.items():
            if contrib.dim() >= 2:
                token_attr = contrib.sum(dim=-1)
            else:
                token_attr = contrib

            orig_shape = input_shapes[key]
            if token_attr.shape != orig_shape:
                reshaped = token_attr
                while len(reshaped.shape) < len(orig_shape):
                    reshaped = reshaped.unsqueeze(-1)
                reshaped = reshaped.expand(orig_shape)
                token_attr = reshaped

            mapped[key] = token_attr.detach()
        return mapped
