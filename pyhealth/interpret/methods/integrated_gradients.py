from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pyhealth.models import BaseModel
from pyhealth.interpret.api import Interpretable
from .base_interpreter import BaseInterpreter


class IntegratedGradients(BaseInterpreter):
    """Integrated Gradients attribution method for PyHealth models.

    This class implements the Integrated Gradients method for computing
    feature attributions in neural networks. The method computes the
    integral of gradients along a straight path from a baseline input
    to the actual input.

    The method is based on the paper:
        Axiomatic Attribution for Deep Networks
        Mukund Sundararajan, Ankur Taly, Qiqi Yan
        ICML 2017
        https://arxiv.org/abs/1703.01365

    Integrated Gradients satisfies two fundamental axioms:
        1. Sensitivity: If an input and a baseline differ in one feature
           but have different predictions, then the differing feature
           should be given non-zero attribution.
        2. Implementation Invariance: The attributions are identical for
           functionally equivalent networks.

    Args:
        model (BaseModel): A trained PyHealth model to interpret. Can be
            any model that inherits from BaseModel (e.g., MLP, StageNet,
            Transformer, RNN).
        use_embeddings (bool): If True, compute gradients with respect to
            embeddings rather than discrete input tokens. This is crucial
            for models with discrete inputs (like ICD codes) where direct
            interpolation of token indices is not meaningful. The model
            must support returning embeddings via an 'embed' parameter.
            Default is True.

    Note:
        **Why use_embeddings=True is recommended:**

        When working with discrete features (e.g., ICD diagnosis codes,
        procedure codes), Integrated Gradients needs to interpolate between
        a baseline and the actual input. However, interpolating discrete
        token indices directly creates invalid intermediate values:

        - Input code index: 245 (e.g., "Diabetes Type 2")
        - Baseline index: 0 (padding token)
        - Interpolation creates: 0 -> 61.25 -> 122.5 -> 183.75 -> 245

        Fractional indices like 61.25 cannot be looked up in an embedding
        table and cause "index out of bounds" errors.

        With ``use_embeddings=True``, the method:
        1. Embeds both baseline and input tokens into continuous vectors
        2. Interpolates in the embedding space (which is valid)
        3. Computes gradients with respect to these embeddings
        4. Maps attributions back to the original input tokens

        This makes IG compatible with models like StageNet, Transformer,
        RNN, and MLP that process discrete medical codes.

        Set ``use_embeddings=False`` only when all inputs are continuous
        (e.g., vital signs, lab values) and no embedding layers are used.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import (
        ...     SampleDataset, split_by_patient, get_dataloader
        ... )
        >>> from pyhealth.models import MLP
        >>> from pyhealth.interpret.methods import IntegratedGradients
        >>> from pyhealth.trainer import Trainer
        >>>
        >>> # Define sample data
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86", "cond-80"],
        ...         "procedures": [1.0, 2.0, 3.5, 4.0],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-55", "cond-12"],
        ...         "procedures": [5.0, 2.0, 3.5, 4.0],
        ...         "label": 0,
        ...     },
        ...     # ... more samples
        ... ]
        >>>
        >>> # Create dataset with schema
        >>> input_schema = {
        ...     "conditions": "sequence",
        ...     "procedures": "tensor"
        ... }
        >>> output_schema = {"label": "binary"}
        >>>
        >>> dataset = SampleDataset(
        ...     samples=samples,
        ...     input_schema=input_schema,
        ...     output_schema=output_schema,
        ...     dataset_name="example"
        ... )
        >>>
        >>> # Initialize MLP model
        >>> model = MLP(
        ...     dataset=dataset,
        ...     embedding_dim=128,
        ...     hidden_dim=128,
        ...     dropout=0.3
        ... )
        >>>
        >>> # Split data and create dataloaders
        >>> train_data, val_data, test_data = split_by_patient(
        ...     dataset, [0.7, 0.15, 0.15]
        ... )
        >>> train_loader = get_dataloader(
        ...     train_data, batch_size=32, shuffle=True
        ... )
        >>> val_loader = get_dataloader(
        ...     val_data, batch_size=32, shuffle=False
        ... )
        >>> test_loader = get_dataloader(
        ...     test_data, batch_size=1, shuffle=False
        ... )
        >>>
        >>> # Train model
        >>> trainer = Trainer(model=model, device="cuda:0")
        >>> trainer.train(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     epochs=10,
        ...     monitor="roc_auc"
        ... )
        >>>
        >>> # Compute attributions for test samples
        >>> ig = IntegratedGradients(model)
        >>> data_batch = next(iter(test_loader))
        >>>
        >>> # Option 1: Use zero baseline (default)
        >>> attributions = ig.attribute(**data_batch, steps=5)
        >>> print(attributions)
        {'conditions': tensor([[0.1234, 0.5678, 0.9012]], device='cuda:0'),
         'procedures': tensor([[0.2345, 0.6789, 0.0123, 0.4567]])}
        >>>
        >>> # Option 2: Specify target class explicitly
        >>> data_batch['target_class_idx'] = 1
        >>> attributions = ig.attribute(**data_batch, steps=5)
        >>>
        >>> # Option 3: Use custom baseline
        >>> custom_baseline = {
        ...     'conditions': torch.zeros_like(data_batch['conditions']),
        ...     'procedures': torch.ones_like(data_batch['procedures']) * 0.5
        ... }
        >>> attributions = ig.attribute(
        ...     **data_batch, baseline=custom_baseline, steps=5
        ... )
    """

    def __init__(self, model: BaseModel, use_embeddings: bool = True, steps: int = 50):
        """Initialize IntegratedGradients interpreter.

        Args:
            model: A trained PyHealth model to interpret.
            use_embeddings: If True, compute gradients with respect to
                embeddings rather than discrete input tokens. Default True.
                This is required for models with discrete inputs like ICD
                codes. Set to False only for fully continuous input models.
                When True, the model must implement forward_from_embedding()
                and have an embedding model accessible via get_embedding_model().
            steps: Default number of interpolation steps for Riemann
                approximation of the path integral. Default is 50.
                Can be overridden in attribute() calls. More steps lead to
                better approximation but slower computation.

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method.
        """
        super().__init__(model)
        if not isinstance(model, Interpretable):
            raise ValueError("Model must implement Interpretable interface")
        self.model = model

        self.use_embeddings = use_embeddings
        self.steps = steps


    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: Optional[int] = None,
        target_class_idx: Optional[int] = None,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute Integrated Gradients attributions for input features.

        This method computes the path integral of gradients from a
        baseline input to the actual input. The integral is approximated
        using Riemann sum with the specified number of steps.

        Args:
            baseline: Baseline input for integration. Can be:
                - None: Uses UNK-token baseline for discrete features or
                  small near-zero baseline for continuous features (default)
                - Dict[str, torch.Tensor]: Custom baseline for each feature
            steps: Number of steps to use in the Riemann approximation of
                the integral. If None, uses self.steps (set during
                initialization). More steps lead to better approximation but
                slower computation.
            target_class_idx: Target class index for attribution
                computation. If None, uses the predicted class (argmax of
                model output).
            **kwargs: Input data dictionary from a dataloader batch
                containing:
                - Feature keys (e.g., 'conditions', 'procedures'):
                  Input tensors or tuples of tensors for each modality
                - 'label' (optional): Ground truth label tensor
                - Other metadata keys are ignored

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping each feature key
                to its attribution tensor. Each tensor has the same shape
                as the input tensor, with values indicating the
                contribution of each input element to the model's
                prediction. Positive values indicate features that
                increase the prediction score, while negative values
                indicate features that decrease it.

        Note:
            - This method requires gradients, so the model should not be
              in torch.no_grad() context.
            - For better interpretability, use batch_size=1 or analyze
              samples individually.
            - The sum of attributions across all features approximates
              the difference between the model's prediction for the input
              and the baseline (completeness axiom).

        Examples:
            >>> from pyhealth.interpret.methods import IntegratedGradients
            >>>
            >>> # Assuming you have a trained model and test data
            >>> ig = IntegratedGradients(trained_model)
            >>> test_batch = next(iter(test_loader))
            >>>
            >>> # Compute attributions with default settings
            >>> attributions = ig.attribute(**test_batch)
            >>> print(f"Feature attributions: {attributions.keys()}")
            >>> print(f"Conditions: {attributions['conditions'].shape}")
            >>>
            >>> # Use more steps for better approximation
            >>> attributions = ig.attribute(**test_batch, steps=100)
            >>>
            >>> # Compute attributions for specific class
            >>> attributions = ig.attribute(
            ...     **test_batch, target_class_idx=0, steps=50
            ... )
            >>>
            >>> # Use custom baseline
            >>> custom_baseline = {
            ...     'conditions': torch.zeros_like(test_batch['conditions']),
            ...     'procedures': torch.zeros_like(test_batch['procedures'])
            ... }
            >>> attributions = ig.attribute(
            ...     **test_batch, baseline=custom_baseline, steps=50
            ... )
            >>>
            >>> # Analyze which features are most important
            >>> condition_attr = attributions['conditions'][0]
            >>> top_k = torch.topk(torch.abs(condition_attr), k=5)
            >>> print(f"Most important features: {top_k.indices}")
        """
        # Use instance default if steps not specified
        if steps is None:
            steps = self.steps

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
                "Unsupported prediction mode for Integrated Gradients attribution."
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
        # - Token features (discrete): embed before interpolation, since
        #   interpolating raw indices is meaningless. Gradients are computed
        #   w.r.t. embeddings, then summed over the embedding dim.
        # - Continuous features: keep raw for interpolation so each raw
        #   dimension gets its own attribution. The model's forward() handles
        #   embedding internally.
        if self.use_embeddings:
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None, (
                "Model must have an embedding model for embedding-based "
                "Integrated Gradients."
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

        # Compute integrated gradients
        attributions = self._integrated_gradients(
            inputs=inputs,
            xs=values,
            bs=baselines,
            steps=steps,
            target=target,
        )

        return self._map_to_input_shapes(attributions, shapes)

    # ------------------------------------------------------------------
    # Core IG computation
    # ------------------------------------------------------------------
    def _integrated_gradients(
        self,
        inputs: Dict[str, tuple[torch.Tensor, ...]],
        xs: Dict[str, torch.Tensor],
        bs: Dict[str, torch.Tensor],
        steps: int,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients via Riemann sum approximation.

        For each interpolation step alpha in [0, 1]:
        1. Creates interpolated values: baseline + alpha * (input - baseline)
        2. Inserts them into the input tuples via the processor schema
        3. Runs forward pass (forward_from_embedding or forward)
        4. Computes gradients w.r.t. the interpolated values
        5. Accumulates gradients using a running sum (memory efficient)

        After all steps, computes the final attribution as:
            (input - baseline) * average_gradient

        Args:
            inputs: Full input tuples keyed by feature name.
            xs: Input values (embedded if use_embeddings=True).
            bs: Baseline values (embedded if use_embeddings=True).
            steps: Number of interpolation steps.
            target: Target tensor for computing the scalar output to
                differentiate (one-hot for multiclass, class idx for binary).

        Returns:
            Dictionary mapping feature keys to attribution tensors.
        """
        keys = sorted(xs.keys())

        # Determine which keys are token (already embedded) vs continuous (raw)
        token_keys = set()
        continuous_keys = set()
        if self.use_embeddings:
            for k in keys:
                if self.model.dataset.input_processors[k].is_token():
                    token_keys.add(k)
                else:
                    continuous_keys.add(k)
        # If not using embeddings, all features are treated as continuous/raw

        # Use running sum instead of storing all gradients (memory efficient)
        avg_gradients = {key: torch.zeros_like(xs[key]) for key in keys}

        for step_idx in range(steps + 1):
            alpha = step_idx / steps

            # Create interpolated values with gradients enabled
            interpolated: dict[str, torch.Tensor] = {}
            for key in keys:
                interp = bs[key] + alpha * (xs[key] - bs[key])
                interp = interp.detach().requires_grad_(True)
                # CRITICAL: retain_grad() needed for non-leaf tensors
                interp.retain_grad()
                interpolated[key] = interp

            # Insert interpolated values back into input tuples
            forward_inputs = inputs.copy()
            for k in forward_inputs.keys():
                schema = self.model.dataset.input_processors[k].schema()
                val_idx = schema.index("value")
                forward_inputs[k] = (
                    *forward_inputs[k][:val_idx],
                    interpolated[k],
                    *forward_inputs[k][val_idx + 1:],
                )

            # Forward pass: use forward_from_embedding for token features
            # (already embedded), but continuous features still need embedding
            # inside the model. We always use forward_from_embedding and let
            # it handle both embedded and raw values.
            if self.use_embeddings:
                # For continuous features, embed them before forward_from_embedding
                if continuous_keys:
                    embedding_model = self.model.get_embedding_model()
                    assert embedding_model is not None, (
                        "Model must have an embedding model for embedding-based "
                        "Integrated Gradients."
                    )
                    continuous_to_embed = {
                        k: interpolated[k] for k in continuous_keys
                    }
                    embedded_continuous = embedding_model(continuous_to_embed)
                    for k in continuous_keys:
                        schema = self.model.dataset.input_processors[k].schema()
                        val_idx = schema.index("value")
                        forward_inputs[k] = (
                            *forward_inputs[k][:val_idx],
                            embedded_continuous[k],
                            *forward_inputs[k][val_idx + 1:],
                        )
                output = self.model.forward_from_embedding(**forward_inputs)
            else:
                output = self.model.forward(**forward_inputs)
            logits = output["logit"]

            # Compute target output and backward pass
            target_output = self._compute_target_output(logits, target)

            self.model.zero_grad()
            target_output.backward(retain_graph=True)

            # Accumulate gradients using running sum
            for key in keys:
                emb = interpolated[key]
                if emb.grad is not None:
                    avg_gradients[key] += emb.grad.detach()

        # Average the accumulated gradients
        for key in keys:
            avg_gradients[key] /= steps + 1

        # Compute final attributions: (input - baseline) * avg_gradient
        attributions: dict[str, torch.Tensor] = {}
        for key in keys:
            delta = xs[key] - bs[key]
            attr = delta * avg_gradients[key]

            # When using embeddings, sum over the embedding dimension
            # to collapse from (batch, ..., emb_dim) to (batch, ...)
            # Only for token features that were embedded before interpolation
            if self.use_embeddings and key in token_keys and attr.dim() >= 3:
                attr = attr.sum(dim=-1)

            attributions[key] = attr

        return attributions

    # ------------------------------------------------------------------
    # Target output computation
    # ------------------------------------------------------------------
    def _compute_target_output(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar target output for backpropagation.

        Creates a differentiable scalar from the model logits that,
        when differentiated, gives the gradient of the target class
        logit w.r.t. the input.

        Args:
            logits: Model output logits, shape [batch, num_classes] or
                [batch, 1].
            target: Target tensor. For binary: [batch] or [1] with 0/1
                class indices. For multiclass/multilabel: [batch, num_classes]
                one-hot or multi-hot tensor.

        Returns:
            Scalar tensor for backpropagation.
        """
        target_f = target.to(logits.device).float()
        mode = self._prediction_mode()

        if mode == "binary":
            # target shape: [1] or [batch, 1] with 0/1 values
            # Convert to signs: 0 -> -1, 1 -> 1
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(-1)
            target_f = target_f.expand_as(logits)
            signs = 2.0 * target_f - 1.0
            return (signs * logits).sum()
        else:
            # multiclass or multilabel: target is one-hot/multi-hot
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
        use_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate raw baselines for IG computation.

        Creates reference samples representing the "absence" of features.
        The strategy depends on the feature type:
        - Discrete (token) features: UNK token index (will be embedded
          later in ``attribute()`` alongside the values)
        - Continuous features: small near-zero neutral values

        Args:
            values: Dictionary of raw input value tensors (before embedding).
            use_embeddings: If True, generate baselines suitable for
                embedding-based IG.

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

            # If shapes already match, no adjustment needed
            if values.shape == orig_shape:
                mapped[key] = values
                continue

            # Expand dimensions to match original input
            reshaped = values
            while len(reshaped.shape) < len(orig_shape):
                reshaped = reshaped.unsqueeze(-1)

            if reshaped.shape != orig_shape:
                reshaped = reshaped.expand(orig_shape)

            mapped[key] = reshaped

        return mapped
