import torch
import torch.nn.functional as F
from typing import Dict, Optional

import torch

from pyhealth.models import BaseModel

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
        - Interpolation creates: 0 → 61.25 → 122.5 → 183.75 → 245

        Fractional indices like 61.25 cannot be looked up in an embedding
        table and cause "index out of bounds" errors.

        With `use_embeddings=True`, the method:
        1. Embeds both baseline and input tokens into continuous vectors
        2. Interpolates in the embedding space (which is valid)
        3. Computes gradients with respect to these embeddings
        4. Maps attributions back to the original input tokens

        This makes IG compatible with models like StageNet, Transformer,
        RNN, and MLP that process discrete medical codes.

        Set `use_embeddings=False` only when all inputs are continuous
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

    def __init__(self, model: BaseModel, use_embeddings: bool = True):
        """Initialize IntegratedGradients interpreter.

        Args:
            model: A trained PyHealth model to interpret.
            use_embeddings: If True, compute gradients with respect to
                embeddings rather than discrete input tokens. Default True.
                This is required for models with discrete inputs like ICD
                codes. Set to False only for fully continuous input models.
                When True, the model must implement forward_from_embedding()
                and have an embedding_model attribute.

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method.
        """
        super().__init__(model)
        self.use_embeddings = use_embeddings

        # Check model supports forward_from_embedding if needed
        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "Integrated Gradients. Set use_embeddings=False to use "
                "input-level gradients (only for continuous features)."
            )

    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: int = 50,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute Integrated Gradients attributions for input features.

        This method computes the path integral of gradients from a
        baseline input to the actual input. The integral is approximated
        using Riemann sum with the specified number of steps.

        Args:
            baseline: Baseline input for integration. Can be:
                - None: Uses small random baseline for all features (default)
                - Dict[str, torch.Tensor]: Custom baseline for each feature
            steps: Number of steps to use in the Riemann approximation of
                the integral. More steps lead to better approximation but
                slower computation. Default is 50.
            target_class_idx: Target class index for attribution
                computation. If None, uses the predicted class (argmax of
                model output).
            **data: Input data dictionary from a dataloader batch
                containing:
                - Feature keys (e.g., 'conditions', 'procedures'):
                  Input tensors for each modality
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
        # Extract feature keys and prepare inputs
        feature_keys = self.model.feature_keys
        inputs = {}
        time_info = {}  # Store time information for StageNet-like models
        label_data = {}  # Store label information

        for key in feature_keys:
            if key in data:
                x = data[key]
                # Handle tuple inputs (e.g., StageNet with (time, values))
                if isinstance(x, tuple):
                    time_info[key] = x[0]  # Store time component
                    x = x[1]  # Use values component for attribution

                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)

                x = x.to(next(self.model.parameters()).device)
                inputs[key] = x

        # Store label data for passing to model
        for key in self.model.label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.tensor(label_val)
                label_val = label_val.to(next(self.model.parameters()).device)
                label_data[key] = label_val

        # Compute integrated gradients with single baseline
        attributions = self._integrated_gradients(
            inputs=inputs,
            baseline=baseline,
            steps=steps,
            target_class_idx=target_class_idx,
            time_info=time_info,
            label_data=label_data,
        )

        return attributions

    def _prepare_embeddings_and_baselines(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        """Prepare input embeddings and baseline embeddings.

        This method embeds the input tokens using the model's embedding layer,
        handles nested sequences (4D tensors), and creates baseline embeddings
        in the embedding space.

        Args:
            inputs: Dictionary of input tensors for each feature.
            baseline: Optional custom baseline tensors. If None, creates
                small random baseline in embedding space.

        Returns:
            Tuple of (input_embeddings, baseline_embeddings, input_shapes):
                - input_embeddings: Dict mapping feature keys to embedded
                  input tensors [batch, seq_len, embedding_dim]
                - baseline_embeddings: Dict mapping feature keys to baseline
                  embeddings with same shape as input_embeddings
                - input_shapes: Dict mapping feature keys to original input
                  tensor shapes for later attribution mapping
        """
        input_embeddings = {}
        baseline_embeddings = {}
        input_shapes = {}

        # Process each feature key individually
        for key in inputs:
            # Store original input shape for later attribution mapping
            input_shapes[key] = inputs[key].shape

            # Embed the input values using model's embedding layer
            embedded = self.model.embedding_model({key: inputs[key]})
            x = embedded[key]

            # DO NOT pool 4D tensors - keep individual token embeddings
            # for proper per-token attribution
            # For nested sequences: [batch, seq_len, tokens, embedding_dim]
            # We need to compute gradients for each token separately

            input_embeddings[key] = x

            # Create baseline directly in embedding space
            if baseline is None:
                # Default: small random values preserving structure
                baseline_embeddings[key] = torch.randn_like(x).abs() * 0.01
            else:
                if key not in baseline:
                    raise ValueError(
                        f"Baseline missing key '{key}'. " f"Expected shape: {x.shape}"
                    )
                baseline_embeddings[key] = baseline[key]

        return input_embeddings, baseline_embeddings, input_shapes

    def _compute_target_output(
        self,
        logits: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute target output scalar for backpropagation.

        This method determines the target class (if not specified), creates
        the appropriate one-hot encoding, and computes the scalar output
        that will be used for computing gradients.

        Args:
            logits: Model output logits [batch, num_classes] or [batch, 1]
            target_class_idx: Optional target class index. If None, uses
                the predicted class (argmax of logits).

        Returns:
            Scalar tensor representing the target output for backprop.
        """
        # Determine task type from model's output schema
        output_schema = self.model.dataset.output_schema
        label_key = list(output_schema.keys())[0]
        task_mode = output_schema[label_key]

        # Check if binary classification
        is_binary = task_mode == "binary" or (
            hasattr(task_mode, "__name__")
            and task_mode.__name__ == "BinaryLabelProcessor"
        )

        # Determine target class
        if target_class_idx is None:
            if is_binary:
                # Binary: if sigmoid(logit) > 0.5, class=1, else class=0
                probs = torch.sigmoid(logits)
                tc_idx = (probs > 0.5).long().squeeze(-1)
            else:
                # Multiclass: argmax over classes
                tc_idx = torch.argmax(logits, dim=-1)
        elif not isinstance(target_class_idx, torch.Tensor):
            tc_idx = torch.tensor(target_class_idx, device=logits.device)
        else:
            tc_idx = target_class_idx

        # Create one-hot encoding for target class
        if is_binary:
            # Binary classification case with [batch, 1] logits
            if isinstance(tc_idx, torch.Tensor):
                if tc_idx.numel() > 1:
                    one_hot = torch.where(
                        tc_idx.unsqueeze(-1) == 1,
                        torch.ones_like(logits),
                        -torch.ones_like(logits),
                    )
                else:
                    tc_val = tc_idx.item()
                    one_hot = (
                        torch.ones_like(logits)
                        if tc_val == 1
                        else -torch.ones_like(logits)
                    )
            else:
                one_hot = (
                    torch.ones_like(logits) if tc_idx == 1 else -torch.ones_like(logits)
                )
        else:
            # Multi-class case
            one_hot = F.one_hot(tc_idx, logits.size(-1)).float()

        # Compute target output (scalar for backprop)
        target_output = torch.sum(one_hot.to(logits.device) * logits)
        return target_output

    def _interpolate_and_compute_gradients(
        self,
        input_embeddings: Dict[str, torch.Tensor],
        baseline_embeddings: Dict[str, torch.Tensor],
        steps: int,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, list]:
        """Interpolate between baseline and input, accumulating gradients.

        This is the core of the Integrated Gradients algorithm. For each
        interpolation step, it:
        1. Creates interpolated embeddings between baseline and input
        2. Runs forward pass through the model
        3. Computes gradients w.r.t. the interpolated embeddings
        4. Collects gradients for later averaging

        Args:
            input_embeddings: Embedded input tensors for each feature.
            baseline_embeddings: Baseline embeddings for each feature.
            steps: Number of interpolation steps for Riemann approximation.
            target_class_idx: Target class for attribution.
            time_info: Optional time information for temporal models.
            label_data: Optional label data to pass to model.

        Returns:
            Dictionary mapping feature keys to lists of gradients, one
            gradient tensor per interpolation step.
        """
        all_gradients = {key: [] for key in input_embeddings}

        for step_idx in range(steps + 1):
            alpha = step_idx / steps

            # Create interpolated embeddings with gradients enabled
            interpolated_embeddings = {}
            for key in input_embeddings:
                interp_emb = baseline_embeddings[key] + alpha * (
                    input_embeddings[key] - baseline_embeddings[key]
                )
                # Enable gradients AND retain them (non-leaf tensors!)
                interp_emb = interp_emb.requires_grad_(True)
                # CRITICAL: Must call retain_grad() for non-leaf tensors
                interp_emb.retain_grad()
                interpolated_embeddings[key] = interp_emb

            # Forward pass through the model
            forward_kwargs = {**label_data} if label_data else {}

            # Pass interpolated embeddings directly to model
            # forward_from_embedding will handle 4D -> 3D pooling
            output = self.model.forward_from_embedding(
                feature_embeddings=interpolated_embeddings,
                time_info=time_info,
                **forward_kwargs,
            )
            logits = output["logit"]

            # Compute target output and backward pass
            target_output = self._compute_target_output(logits, target_class_idx)

            self.model.zero_grad()
            target_output.backward(retain_graph=True)

            # Collect gradients for each feature's embedding
            for key in input_embeddings:
                emb = interpolated_embeddings[key]
                if emb.grad is not None:
                    grad = emb.grad.detach().clone()
                    all_gradients[key].append(grad)
                else:
                    all_gradients[key].append(torch.zeros_like(emb))

        return all_gradients

    def _compute_final_attributions(
        self,
        all_gradients: Dict[str, list],
        input_embeddings: Dict[str, torch.Tensor],
        baseline_embeddings: Dict[str, torch.Tensor],
        input_shapes: Dict[str, tuple],
    ) -> Dict[str, torch.Tensor]:
        """Compute final integrated gradients and map to input shapes.

        This method completes the IG computation by:
        1. Averaging gradients across interpolation steps
        2. Applying the IG formula: (input - baseline) * avg_gradient
        3. Summing over embedding dimension
        4. Mapping attributions back to original input tensor shapes

        Important properties of IG attributions:
        - Can be POSITIVE (feature increases prediction) or NEGATIVE
          (feature decreases prediction)
        - Sum approximately to f(input) - f(baseline), NOT to 1
        - Represent contribution to the difference in model output
        - Negative values indicate features that push prediction away
          from the target class

        Args:
            all_gradients: Dictionary of gradient lists from interpolation.
            input_embeddings: Embedded input tensors.
            baseline_embeddings: Baseline embeddings.
            input_shapes: Original input tensor shapes for mapping.

        Returns:
            Dictionary mapping feature keys to attribution tensors with
            the same shape as the original input tensors.
        """
        integrated_grads = {}

        for key in input_embeddings:
            # Average gradients across interpolation steps (exclude last)
            stacked_grads = torch.stack(all_gradients[key][:-1], dim=0)
            avg_grad = torch.mean(stacked_grads, dim=0)

            # Apply IG formula: (input_emb - baseline_emb) * avg_gradient
            delta_emb = input_embeddings[key] - baseline_embeddings[key]
            emb_attribution = delta_emb * avg_grad

            # Sum over embedding dimension to get per-token attribution
            # Handle both 3D [batch, seq, emb] and 4D [batch, seq, tokens, emb]
            if emb_attribution.dim() == 4:
                # [batch, seq_len, tokens, embedding_dim] -> [batch, seq, tok]
                token_attr = emb_attribution.sum(dim=-1)
            elif emb_attribution.dim() == 3:
                # [batch, seq_len, embedding_dim] -> [batch, seq_len]
                token_attr = emb_attribution.sum(dim=-1)
            elif emb_attribution.dim() == 2:
                token_attr = emb_attribution.sum(dim=-1)
            else:
                # Unexpected dimension, keep as is
                token_attr = emb_attribution

            # Map back to original input shape if needed
            orig_shape = input_shapes[key]

            # Check if shapes already match (e.g., 4D case)
            if token_attr.shape == orig_shape:
                integrated_grads[key] = token_attr
                continue

            # For inputs like [batch, seq_len, tokens], expand attributions
            if len(orig_shape) > len(token_attr.shape):
                # Expand dimensions to match
                while len(token_attr.shape) < len(orig_shape):
                    token_attr = token_attr.unsqueeze(-1)
                # Broadcast to original shape
                token_attr = token_attr.expand(orig_shape)

            integrated_grads[key] = token_attr

        return integrated_grads

    def _integrated_gradients_embedding_based(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: int = 50,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients using embedding-level gradients.

        This method implements IG for models with discrete inputs by:
        1. Embedding inputs into continuous space
        2. Interpolating in embedding space
        3. Computing gradients w.r.t. embeddings
        4. Mapping attributions back to input tokens

        Args:
            inputs: Dictionary of input tensors.
            baseline: Optional baseline tensors.
            steps: Number of interpolation steps.
            target_class_idx: Target class for attribution.
            time_info: Optional time information for temporal models.
            label_data: Optional label data.

        Returns:
            Dictionary of attribution tensors matching input shapes.
        """
        # Step 1: Embed inputs and create baselines in embedding space
        input_embs, baseline_embs, shapes = self._prepare_embeddings_and_baselines(
            inputs, baseline
        )

        # Step 2: Interpolate and accumulate gradients across steps
        all_grads = self._interpolate_and_compute_gradients(
            input_embs,
            baseline_embs,
            steps,
            target_class_idx,
            time_info,
            label_data,
        )

        # Step 3: Integrate gradients and map to input space
        attributions = self._compute_final_attributions(
            all_grads, input_embs, baseline_embs, shapes
        )

        return attributions

    def _integrated_gradients_continuous(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: int = 50,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients for continuous inputs.

        This method implements IG for models with continuous inputs by
        directly interpolating input values (not embeddings).

        Args:
            inputs: Dictionary of input tensors.
            baseline: Optional baseline tensors.
            steps: Number of interpolation steps.
            target_class_idx: Target class for attribution.
            time_info: Optional time information for temporal models.
            label_data: Optional label data.

        Returns:
            Dictionary of attribution tensors matching input shapes.
        """
        # Create baseline if not provided
        if baseline is None:
            baseline = {}
            for key in inputs:
                # Use small non-zero baseline for continuous features
                baseline[key] = torch.ones_like(inputs[key]) * 1e-5

        all_gradients = {key: [] for key in inputs}

        # Interpolation loop
        for step_idx in range(steps + 1):
            alpha = step_idx / steps
            scaled_inputs = {}

            for key in inputs:
                baseline_val = baseline[key]
                input_val = inputs[key]

                # Check for discrete inputs
                is_discrete = input_val.dtype in [
                    torch.int64,
                    torch.int32,
                    torch.long,
                ]
                if is_discrete:
                    raise ValueError(
                        f"Feature '{key}' has discrete integer values "
                        "that cannot be interpolated. "
                        "set use_embeddings=True is not yet supported. "
                        "Consider using gradient-based saliency instead."
                    )

                # Interpolate continuous inputs
                delta = input_val - baseline_val
                scaled_input = baseline_val + alpha * delta
                scaled_input = scaled_input.requires_grad_(True)

                # Reconstruct tuple if needed
                if time_info and key in time_info:
                    scaled_inputs[key] = (time_info[key], scaled_input)
                else:
                    scaled_inputs[key] = scaled_input

            # Add label data
            if label_data:
                for key in label_data:
                    scaled_inputs[key] = label_data[key]

            # Forward pass
            output = self.model(**scaled_inputs)
            logits = output["logit"]

            # Compute target output and backward pass
            target_output = self._compute_target_output(logits, target_class_idx)
            self.model.zero_grad()
            target_output.backward(retain_graph=True)

            # Collect gradients
            for key in inputs:
                if time_info and key in time_info:
                    _, scaled_val = scaled_inputs[key]
                    grad = scaled_val.grad
                else:
                    grad = scaled_inputs[key].grad

                if grad is not None:
                    all_gradients[key].append(grad.detach().clone())
                else:
                    all_gradients[key].append(torch.zeros_like(inputs[key]))

        # Compute average gradients (Riemann sum approximation)
        avg_gradients = {}
        for key in inputs:
            # Average all gradients except the last one
            stacked_grads = torch.stack(all_gradients[key][:-1], dim=0)
            avg_gradients[key] = torch.mean(stacked_grads, dim=0)

        # Compute integrated gradients: (input - baseline) * avg_gradients
        integrated_grads = {}
        for key in inputs:
            delta_input = inputs[key] - baseline[key]
            avg_grad = avg_gradients[key]

            # Ensure shapes are compatible
            if delta_input.shape != avg_grad.shape:
                if len(avg_grad.shape) < len(delta_input.shape):
                    # Expand gradient to match input dimensions
                    while len(avg_grad.shape) < len(delta_input.shape):
                        avg_grad = avg_grad.unsqueeze(-1)
                    avg_grad = avg_grad.expand_as(delta_input)
                elif len(delta_input.shape) < len(avg_grad.shape):
                    # This shouldn't happen, but handle it
                    avg_grad = avg_grad.squeeze()

            integrated_grads[key] = delta_input * avg_grad

        return integrated_grads

    def _integrated_gradients(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: int = 50,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients for a single baseline.

        Args:
            inputs: Dictionary of input tensors.
            baseline: Baseline tensors. If None, uses default baseline.
            steps: Number of integration steps.
            target_class_idx: Target class index.
            time_info: Optional time information for temporal models.
            label_data: Optional label data.

        Returns:
            Dictionary of attribution tensors.
        """
        if self.use_embeddings:
            return self._integrated_gradients_embedding_based(
                inputs,
                baseline,
                steps,
                target_class_idx,
                time_info,
                label_data,
            )
        else:
            return self._integrated_gradients_continuous(
                inputs,
                baseline,
                steps,
                target_class_idx,
                time_info,
                label_data,
            )
