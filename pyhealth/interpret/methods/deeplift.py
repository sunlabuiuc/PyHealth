from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from pyhealth.models import BaseModel
from .base_interpreter import BaseInterpreter


def _iter_child_modules(module: torch.nn.Module):
    for name, child in module.named_children():
        yield module, name, child
        yield from _iter_child_modules(child)


class _HookedModule(torch.nn.Module):
    """Wrap an activation module to route through DeepLIFT hooks."""

    def __init__(self, hook_name: str, hooks: "_DeepLiftActivationHooks", forward_kwargs: Optional[Dict] = None):
        super().__init__()
        self.hook_name = hook_name
        self.hooks = hooks
        self.forward_kwargs = forward_kwargs or {}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.hooks.apply(self.hook_name, tensor, **self.forward_kwargs)


class _ActivationSwapContext(contextlib.AbstractContextManager):
    """Temporarily replace activation modules with DeepLIFT-aware wrappers."""

    _TARGETS: Dict[Type[torch.nn.Module], Tuple[str, Dict]] = {
        torch.nn.ReLU: ("relu", {}),
        torch.nn.Sigmoid: ("sigmoid", {}),
        torch.nn.Tanh: ("tanh", {}),
    }

    def __init__(self, model: BaseModel):
        self.model = model
        self.hooks = _DeepLiftActivationHooks()
        self._swapped: List[Tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def __enter__(self) -> "_ActivationSwapContext":
        for parent, name, child in _iter_child_modules(self.model):
            for target_cls, (hook_name, fkwargs) in self._TARGETS.items():
                if isinstance(child, target_cls):
                    wrapper = _HookedModule(hook_name, self.hooks, fkwargs)
                    setattr(parent, name, wrapper)
                    self._swapped.append((parent, name, child))
                    break
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        for parent, name, original in reversed(self._swapped):
            setattr(parent, name, original)
        self._swapped.clear()
        self.hooks.reset()
        return False


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
    """Context manager that swaps activations for DeepLIFT without model hooks."""

    def __init__(self, model: BaseModel):
        self.model = model
        self._swap_ctx = _ActivationSwapContext(model)

    def __enter__(self) -> "_DeepLiftHookContext":
        self._swap_ctx.__enter__()
        return self

    def start_baseline(self) -> None:
        self._swap_ctx.hooks.start_baseline()

    def start_actual(self) -> None:
        self._swap_ctx.hooks.start_actual()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self._swap_ctx.__exit__(exc_type, exc, exc_tb)
        return False


class DeepLift(BaseInterpreter):
    """DeepLIFT attribution for PyHealth models.

    Paper: Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning
    Important Features through Propagating Activation Differences. ICML 2017.

    DeepLIFT propagates difference-from-baseline activations using Rescale
    multipliers so that feature attributions sum to the change in model output.
    The implementation injects secant slopes for supported activations
    (ReLU, Sigmoid, Tanh) via module swapping to mirror the original algorithm
    while falling back to autograd gradients for unsupported operations.

    This method is particularly useful for:
        - EHR feature importance: highlight influential visits, codes, or labs
          when auditing StageNet-style models.
        - Contrastive explanations: compare predictions against a clinically
          meaningful baseline patient trajectory.
        - Mixed-input attribution: handle discrete embedding channels and
          continuous features in a unified call.
        - Model debugging: diagnose activation saturation and verify the
          completeness axiom.

    Key Features:
        - Dual operating modes for embedding-based or continuous inputs.
        - Automatic activation module swapping for DeepLIFT Rescale rule.
        - Completeness enforcement ensuring ``sum(attribution) ~= f(x) - f(x0)``.
        - Batch-friendly API accepting trainer-style dictionaries with
          tuple-based inputs following processor schemas.
        - Target control via ``target_class_idx`` to explain any desired logit.
        - Mixed token/continuous feature support using ``is_token()``
          processor introspection.

    Usage Notes:
        1. Choose a baseline dictionary that reflects a neutral clinical state
           when zeros are not meaningful.
        2. Move inputs, baselines, and the model to the same device before
           calling ``attribute``.
        3. Keep ``use_embeddings=True`` for token indices; set it to ``False``
           to attribute continuous tensors directly.
        4. Call ``model.eval()`` so stochastic layers remain deterministic
           during paired forward passes.

    Args:
        model: A :class:`~pyhealth.models.BaseModel` instance exposing either
            :meth:`forward_from_embedding` (for discrete inputs) or the standard
            :meth:`forward` used by PyHealth trainers.
        use_embeddings: Whether to operate in embedding space. Set to ``True``
            (default) for tokenized inputs or ``False`` to attribute continuous
            tensors directly.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.interpret.methods.deeplift import DeepLift
        >>> from pyhealth.models import MLP
        >>>
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "conditions": ["cond-33", "cond-86", "cond-80"],
        ...      "procedures": [1.0, 2.0, 3.5, 4.0], "label": 1},
        ...     {"patient_id": "p1", "visit_id": "v1",
        ...      "conditions": ["cond-55", "cond-12"],
        ...      "procedures": [5.0, 2.0, 3.5, 4.0], "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = MLP(dataset=dataset, embedding_dim=32, hidden_dim=32)
        >>> model.eval()
        >>> test_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        >>> deeplift = DeepLift(model, use_embeddings=True)
        >>>
        >>> batch = next(iter(test_loader))
        >>> attributions = deeplift.attribute(**batch)
        >>> print({k: v.shape for k, v in attributions.items()})

    Algorithm Details:
        1. Run a baseline forward pass while caching activations for supported
           nonlinearities.
        2. Replay the actual inputs with Rescale hooks that substitute secant
           slopes for local derivatives.
        3. Backpropagate the target logit so gradients equal DeepLIFT
           multipliers.
        4. Multiply input differences by the propagated multipliers and enforce
           completeness.

    References:
        [1] Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning
            Important Features through Propagating Activation Differences.
            Proceedings of the 34th International Conference on Machine
            Learning (ICML), 2017. https://proceedings.mlr.press/v70/shrikumar17a.html
    """

    def __init__(self, model: BaseModel, use_embeddings: bool = True):
        super().__init__(model)
        self.use_embeddings = use_embeddings

        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "DeepLIFT. Set use_embeddings=False to use input-level "
                "gradients (only for continuous features)."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        target_class_idx: Optional[int] = None,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute DeepLIFT attributions for a single batch.

        The method follows Algorithm 2 of the DeepLIFT paper: two forward
        passes (baseline then actual) are executed under the hook context so
        that backward propagation yields multipliers equal to the Rescale rule.

        Args:
            baseline: Optional dictionary providing reference inputs per
                feature key. If omitted, UNK tokens are used for discrete
                features and near-zero values for continuous features.
            target_class_idx: Optional class index to explain. ``None`` defaults
                to the model prediction.
            **kwargs: Input data dictionary from a dataloader batch
                containing:
                - Feature keys (e.g., 'conditions', 'procedures'):
                  Input tensors or tuples of tensors for each modality
                - 'label' (optional): Ground truth label tensor
                - Other metadata keys are ignored

        Returns:
            ``Dict[str, torch.Tensor]`` mapping each feature key to attribution
            tensors shaped like the original inputs. All outputs satisfy the
            completeness property ``sum_i attribution_i ≈ f(x) - f(x₀)``.
        """
        device = next(self.model.parameters()).device

        # Filter kwargs to only include model feature keys and ensure they are tuples
        inputs = {
            k: (v,) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if k in self.model.feature_keys
        }

        # Disassemble inputs to get values and masks
        values: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            schema = self.model.dataset.input_processors[k].schema()
            values[k] = v[schema.index("value")]
            if "mask" in schema:
                masks[k] = v[schema.index("mask")]
            else:
                masks[k] = (v[schema.index("value")] != 0).int()

        # Append input masks to inputs for models that expect them
        for k, v in inputs.items():
            if "mask" not in self.model.dataset.input_processors[k].schema():
                inputs[k] = (*v, masks[k])

        # Determine target class from original input
        with torch.no_grad():
            base_logits = self.model.forward(**inputs)["logit"]

        mode = self._prediction_mode()
        if mode == "binary":
            if target_class_idx is not None:
                target = torch.tensor([target_class_idx], device=device)
            else:
                target = (torch.sigmoid(base_logits) > 0.5).long()
        elif mode == "multiclass":
            if target_class_idx is not None:
                target = F.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=base_logits.shape[-1],
                ).float()
            else:
                target = torch.argmax(base_logits, dim=-1)
                target = F.one_hot(
                    target, num_classes=base_logits.shape[-1]
                ).float()
        elif mode == "multilabel":
            if target_class_idx is not None:
                target = F.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=base_logits.shape[-1],
                ).float()
            else:
                target = (torch.sigmoid(base_logits) > 0.5).float()
        else:
            raise ValueError(
                "Unsupported prediction mode for DeepLIFT attribution."
            )

        # Generate baselines
        if baseline is None:
            baselines = self._generate_baseline(
                values, use_embeddings=self.use_embeddings
            )
        else:
            baselines = {
                k: v.to(device)
                for k, v in baseline.items()
                if k in self.model.feature_keys
            }

        # Save raw shapes before embedding for later mapping
        shapes = {k: v.shape for k, v in values.items()}

        # Split features by type using is_token():
        # - Token features (discrete): embed before DeepLIFT, since
        #   working with raw indices is meaningless. Gradients are computed
        #   w.r.t. embeddings, then summed over the embedding dim.
        # - Continuous features: keep raw so each raw dimension gets its
        #   own attribution. The model's forward() handles embedding internally.
        token_keys: set[str] = set()
        if self.use_embeddings:
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None, (
                "Model must have an embedding model for embedding-based "
                "DeepLIFT."
            )
            token_keys = {
                k for k in values
                if self.model.dataset.input_processors[k].is_token()
            }
            if token_keys:
                # Embed token values
                token_values = {k: values[k] for k in token_keys}
                embedded_tokens = embedding_model(token_values)
                for k in token_keys:
                    values[k] = embedded_tokens[k]
                # Embed token baselines so they live in the same space
                token_baselines = {k: baselines[k] for k in token_keys if k in baselines}
                if token_baselines:
                    embedded_baselines = embedding_model(token_baselines)
                    for k in token_baselines:
                        baselines[k] = embedded_baselines[k]

        # Compute DeepLIFT attributions
        attributions = self._deeplift(
            inputs=inputs,
            xs=values,
            bs=baselines,
            target=target,
            token_keys=token_keys,
        )

        return self._map_to_input_shapes(attributions, shapes)

    # ------------------------------------------------------------------
    # Core DeepLIFT computation
    # ------------------------------------------------------------------
    def _deeplift(
        self,
        inputs: Dict[str, tuple[torch.Tensor, ...]],
        xs: Dict[str, torch.Tensor],
        bs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        token_keys: set[str],
    ) -> Dict[str, torch.Tensor]:
        """Core DeepLIFT computation using the Rescale rule.

        Performs two forward passes (baseline and actual) under the activation
        swap context, then backpropagates to obtain DeepLIFT multipliers.

        Args:
            inputs: Full input tuples keyed by feature name.
            xs: Input values (embedded if token features with use_embeddings).
            bs: Baseline values (embedded if token features with use_embeddings).
            target: Target tensor for computing the scalar output to
                differentiate (one-hot for multiclass, class idx for binary).
            token_keys: Set of feature keys that are token (already embedded).

        Returns:
            Dictionary mapping feature keys to attribution tensors.
        """
        keys = sorted(xs.keys())

        # Create delta tensors with gradients enabled
        delta: dict[str, torch.Tensor] = {}
        current: dict[str, torch.Tensor] = {}
        for key in keys:
            d = (xs[key] - bs[key]).detach()
            d.requires_grad_(True)
            d.retain_grad()
            delta[key] = d
            current[key] = bs[key].detach() + d

        # Build forward inputs with current (baseline + delta) values
        # inserted into the proper position in the input tuples
        def _build_forward_inputs(value_dict: dict[str, torch.Tensor]) -> dict:
            fwd_inputs = inputs.copy()
            for k in fwd_inputs.keys():
                schema = self.model.dataset.input_processors[k].schema()
                val_idx = schema.index("value")
                fwd_inputs[k] = (
                    *fwd_inputs[k][:val_idx],
                    value_dict[k],
                    *fwd_inputs[k][val_idx + 1:],
                )
            return fwd_inputs

        # For continuous features, embed them before forward_from_embedding
        def _maybe_embed_continuous(value_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if not self.use_embeddings:
                return value_dict
            continuous_keys = {k for k in value_dict if k not in token_keys}
            if not continuous_keys:
                return value_dict
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None
            continuous_to_embed = {k: value_dict[k] for k in continuous_keys}
            embedded_continuous = embedding_model(continuous_to_embed)
            return {**value_dict, **embedded_continuous}

        # Two forward passes under the hook context
        with _DeepLiftHookContext(self.model) as hook_ctx:
            # --- Baseline forward pass ---
            hook_ctx.start_baseline()
            baseline_values = _maybe_embed_continuous(
                {k: bs[k].detach() for k in keys}
            )
            baseline_fwd = _build_forward_inputs(baseline_values)
            with torch.no_grad():
                if self.use_embeddings:
                    baseline_output = self.model.forward_from_embedding(**baseline_fwd)
                else:
                    baseline_output = self.model.forward(**baseline_fwd)

            # --- Actual forward pass ---
            hook_ctx.start_actual()
            current_values = _maybe_embed_continuous(current)
            current_fwd = _build_forward_inputs(current_values)
            if self.use_embeddings:
                current_output = self.model.forward_from_embedding(**current_fwd)
            else:
                current_output = self.model.forward(**current_fwd)

        logits = current_output["logit"] # type: ignore[index]
        baseline_logits = baseline_output["logit"] # type: ignore[index]

        # Compute per-sample target outputs
        target_output = self._compute_target_output(logits, target)
        baseline_target_output = self._compute_target_output(
            baseline_logits, target
        )

        self.model.zero_grad(set_to_none=True)
        target_output.sum().backward()

        # Collect attributions: grad * delta
        attributions: dict[str, torch.Tensor] = {}
        for key in keys:
            grad = delta[key].grad
            if grad is None:
                attributions[key] = torch.zeros_like(delta[key])
            else:
                attr = grad.detach() * delta[key].detach()
                # For token features, sum over the embedding dimension
                if self.use_embeddings and key in token_keys and attr.dim() >= 3:
                    attr = attr.sum(dim=-1)
                attributions[key] = attr

        # Enforce completeness: sum of attributions == f(x) - f(x0)
        attributions = self._enforce_completeness(
            attributions,
            target_output.detach(),
            baseline_target_output.detach(),
        )

        return attributions

    # ------------------------------------------------------------------
    # Target output computation
    # ------------------------------------------------------------------
    def _compute_target_output(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample target output.

        Creates a differentiable per-sample scalar from the model logits
        that, when summed and differentiated, gives the gradient of the
        target class logit w.r.t. the input.

        Args:
            logits: Model output logits, shape [batch, num_classes] or
                [batch, 1].
            target: Target tensor. For binary: [batch] or [1] with 0/1
                class indices. For multiclass/multilabel: [batch, num_classes]
                one-hot or multi-hot tensor.

        Returns:
            Per-sample target output tensor, shape [batch].
        """
        target_f = target.to(logits.device).float()
        mode = self._prediction_mode()

        if mode == "binary":
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(-1)
            target_f = target_f.expand_as(logits)
            signs = 2.0 * target_f - 1.0
            # Sum over all dims except batch to get per-sample scalar
            per_sample = (signs * logits)
            if per_sample.dim() > 1:
                per_sample = per_sample.sum(dim=tuple(range(1, per_sample.dim())))
            return per_sample
        else:
            # multiclass or multilabel: target is one-hot/multi-hot
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(0)
            target_f = target_f.expand_as(logits)
            per_sample = (target_f * logits)
            if per_sample.dim() > 1:
                per_sample = per_sample.sum(dim=tuple(range(1, per_sample.dim())))
            return per_sample

    # ------------------------------------------------------------------
    # Completeness enforcement
    # ------------------------------------------------------------------
    @staticmethod
    def _enforce_completeness(
        contributions: Dict[str, torch.Tensor],
        target_output: torch.Tensor,
        baseline_output: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Scale attributions so their sum matches ``f(x) - f(x₀)`` (Eq. 1)."""
        delta_output = (target_output - baseline_output)

        total = None
        for contrib in contributions.values():
            flat_sum = contrib.reshape(contrib.size(0), -1).sum(dim=1)
            total = flat_sum if total is None else total + flat_sum

        scale = torch.ones_like(delta_output)
        if total is not None:
            denom = total
            mask = denom.abs() > eps
            scale[mask] = delta_output[mask] / denom[mask]

        for key, contrib in contributions.items():
            reshape_dims = [contrib.size(0)] + [1] * (contrib.dim() - 1)
            contributions[key] = contrib * scale.view(*reshape_dims)

        return contributions

    # ------------------------------------------------------------------
    # Baseline generation
    # ------------------------------------------------------------------
    def _generate_baseline(
        self,
        values: Dict[str, torch.Tensor],
        use_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate raw baselines for DeepLIFT computation.

        Creates reference samples representing the "absence" of features.
        The strategy depends on the feature type:
        - Discrete (token) features: UNK token index (will be embedded
          later in ``attribute()`` alongside the values)
        - Continuous features: small near-zero neutral values

        Args:
            values: Dictionary of raw input value tensors (before embedding).
            use_embeddings: If True, generate baselines suitable for
                embedding-based DeepLIFT.

        Returns:
            Dictionary mapping feature names to baseline tensors in raw
            (pre-embedding) space. Embedding of token baselines is handled
            by the caller (``attribute()``).
        """
        baselines: dict[str, torch.Tensor] = {}

        for k, v in values.items():
            processor = self.model.dataset.input_processors[k]
            if use_embeddings and processor.is_token():
                # Token features: UNK token index as baseline
                baseline = torch.ones_like(v)
            else:
                # Continuous features (or non-embedding mode): near-zero baseline
                baseline = torch.zeros_like(v) + 1e-2
            baselines[k] = baseline

        return baselines

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_to_input_shapes(
        attr_values: Dict[str, torch.Tensor],
        input_shapes: dict,
    ) -> Dict[str, torch.Tensor]:
        """Map attributions back to original input tensor shapes.

        For embedding-based attributions, the embedding dimension has
        already been summed out. This method handles any remaining
        shape mismatches (e.g., expanding scalar attributions to match
        multi-dimensional inputs).

        Args:
            attr_values: Dictionary of attribution tensors.
            input_shapes: Dictionary of original input shapes.

        Returns:
            Dictionary of attributions reshaped to match original inputs.
        """
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
