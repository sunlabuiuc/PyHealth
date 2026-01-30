import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal, List, Tuple

from pyhealth.models import BaseModel


class LayerwiseRelevancePropagation:
    """Layer-wise Relevance Propagation attribution method for PyHealth models.

    This class implements the LRP method for computing feature attributions
    in neural networks. The method decomposes the network's prediction into
    relevance scores for each input feature through backward propagation of
    relevance from output to input layers.

    The method is based on the paper:
        Layer-wise Relevance Propagation for Neural Networks with
        Local Renormalization Layers
        Alexander Binder, Gregoire Montavon, Sebastian Bach,
        Klaus-Robert Muller, Wojciech Samek
        arXiv:1604.00825, 2016
        https://arxiv.org/abs/1604.00825

    LRP satisfies the conservation property: relevance is conserved at
    each layer, meaning the sum of relevances at the input layer equals
    the model's output for the target class.

    Key differences from Integrated Gradients:
        - LRP: Single backward pass, no baseline needed, sums to f(x)
        - IG: Multiple forward passes, requires baseline, sums to f(x)-f(baseline)

    Args:
        model (BaseModel): A trained PyHealth model to interpret. Must have
            been trained and should be in evaluation mode.
        rule (str): LRP propagation rule to use:
            - "epsilon": ε-rule for numerical stability (default)
            - "alphabeta": αβ-rule for sharper visualizations
        epsilon (float): Stabilizer for ε-rule. Default 0.01.
            Prevents division by zero in relevance redistribution.
        alpha (float): α parameter for αβ-rule. Default 1.0.
            Controls positive contribution weighting.
        beta (float): β parameter for αβ-rule. Default 0.0.
            Controls negative contribution weighting.
        use_embeddings (bool): If True, compute relevance from embedding
            layer for models with discrete inputs. Default True.
            Required for models with discrete medical codes.

    Note:
        This implementation supports:
        - Linear layers (fully connected)
        - Convolutional layers (Conv2d)
        - ReLU activations
        - Pooling operations (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d)
        - Batch normalization
        - Embedding layers
        - Basic sequential models (MLP, simple RNN)
        - CNN-based models (ResNet, VGG, etc.)

        Future versions will add support for:
        - Attention mechanisms
        - Complex temporal models (StageNet)

    Examples:
        >>> from pyhealth.interpret.methods import LayerWiseRelevancePropagation
        >>> from pyhealth.models import MLP
        >>> from pyhealth.datasets import get_dataloader
        >>>
        >>> # Initialize LRP with trained model
        >>> lrp = LayerWiseRelevancePropagation(
        ...     model=trained_model,
        ...     rule="epsilon",
        ...     epsilon=0.01
        ... )
        >>>
        >>> # Get test data
        >>> test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
        >>> test_batch = next(iter(test_loader))
        >>>
        >>> # Compute attributions
        >>> attributions = lrp.attribute(**test_batch)
        >>>
        >>> # Print results
        >>> for feature_key, relevance in attributions.items():
        ...     print(f"{feature_key}: shape={relevance.shape}")
        ...     print(f"  Sum of relevances: {relevance.sum().item():.4f}")
        ...     print(f"  Top 5 indices: {relevance.flatten().topk(5).indices}")
        >>>
        >>> # Use αβ-rule for sharper heatmaps
        >>> lrp_sharp = LayerWiseRelevancePropagation(
        ...     model=trained_model,
        ...     rule="alphabeta",
        ...     alpha=1.0,
        ...     beta=0.0
        ... )
        >>> sharp_attrs = lrp_sharp.attribute(**test_batch)
    """

    def __init__(
        self,
        model: BaseModel,
        rule: Literal["epsilon", "alphabeta"] = "epsilon",
        epsilon: float = 0.01,
        alpha: float = 1.0,
        beta: float = 0.0,
        use_embeddings: bool = True,
    ):
        """Initialize LRP interpreter.

        Args:
            model: A trained PyHealth model to interpret.
            rule: Propagation rule ("epsilon" or "alphabeta").
            epsilon: Stabilizer for epsilon-rule.
            alpha: Alpha parameter for alphabeta-rule.
            beta: Beta parameter for alphabeta-rule.
            use_embeddings: Whether to start from embedding layer.

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method.
        """
        self.model = model
        self.model.eval()  # Ensure model is in evaluation mode
        self.rule = rule
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.use_embeddings = use_embeddings

        # Storage for activations and hooks
        self.hooks = []
        self.activations = {}

        # Validate model compatibility
        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "LRP. Set use_embeddings=False to use input-level LRP "
                "(only for continuous features)."
            )

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP attributions for input features.

        This method computes relevance scores by:
        1. Performing a forward pass to get the prediction
        2. Initializing output layer relevance
        3. Propagating relevance backward through layers
        4. Mapping relevance to input features

        Args:
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
                to its relevance tensor. Each tensor has the same shape
                as the input tensor, with values indicating the
                contribution of each input element to the model's
                prediction.

                Positive values indicate features that increase the
                prediction score, while negative values indicate features
                that decrease it.

                Important: Unlike Integrated Gradients, LRP relevances
                sum to approximately f(x) (the model's output), not to
                f(x) - f(baseline).

        Note:
            - Relevance conservation: Sum of input relevances should
              approximately equal the model's output for the target class.
            - For better interpretability, use batch_size=1 or analyze
              samples individually.
            - The quality of attributions depends on the chosen rule and
              parameters (epsilon, alpha, beta).

        Examples:
            >>> # Basic usage with default settings
            >>> attributions = lrp.attribute(**test_batch)
            >>> print(f'Total relevance: {sum(r.sum() for r in attributions.values())}')
            >>>
            >>> # Specify target class explicitly
            >>> attributions = lrp.attribute(**test_batch, target_class_idx=1)
            >>>
            >>> # Analyze which features are most important
            >>> condition_relevance = attributions['conditions'][0]
            >>> top_k = torch.topk(condition_relevance.flatten(), k=5)
            >>> print(f'Most relevant features: {top_k.indices}')
            >>> print(f'Relevance values: {top_k.values}')
        """
        # Extract feature keys and prepare inputs
        feature_keys = getattr(self.model, 'feature_keys', list(data.keys()))
        inputs = {}
        time_info = {}  # Store time information for StageNet-like models
        label_data = {}  # Store label information

        # Process input features
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
        label_keys = getattr(self.model, 'label_keys', [])
        for key in label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.tensor(label_val)
                label_val = label_val.to(next(self.model.parameters()).device)
                label_data[key] = label_val

        # Compute LRP attributions
        if self.use_embeddings:
            attributions = self._compute_from_embeddings(
                inputs=inputs,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
        else:
            # Direct input-level LRP (for continuous features like images)
            attributions = self._compute_from_inputs(
                inputs=inputs,
                target_class_idx=target_class_idx,
                label_data=label_data,
            )

        return attributions

    def visualize(
        self,
        plt,
        image: torch.Tensor,
        relevance: torch.Tensor,
        title: Optional[str] = None,
        method: str = 'overlay',
        **kwargs
    ) -> None:
        """Visualize LRP relevance maps using the SaliencyVisualizer.
        
        Convenience method for visualizing LRP attributions with various
        visualization styles.
        
        Args:
            plt: matplotlib.pyplot instance
            image: Input image tensor [C, H, W] or [B, C, H, W]
            relevance: LRP relevance tensor (output from attribute())
            title: Optional title for the plot
            method: Visualization method:
                - 'overlay': Image with relevance overlay (default)
                - 'heatmap': Standalone relevance heatmap
                - 'top_k': Highlight top-k most relevant features
            **kwargs: Additional arguments passed to visualization method
                - alpha: Transparency for overlay (default: 0.3)
                - cmap: Colormap (default: 'hot')
                - k: Number of top features for 'top_k' method
        
        Examples:
            >>> lrp = LayerwiseRelevancePropagation(model)
            >>> attributions = lrp.attribute(**batch)
            >>> 
            >>> # Overlay visualization
            >>> lrp.visualize(plt, batch['image'][0], attributions['image'][0])
            >>> 
            >>> # Heatmap only
            >>> lrp.visualize(plt, batch['image'][0], attributions['image'][0],
            ...               method='heatmap')
            >>> 
            >>> # Top-10 features
            >>> lrp.visualize(plt, batch['image'][0], attributions['image'][0],
            ...               method='top_k', k=10)
        """
        from pyhealth.interpret.methods.saliency_visualization import visualize_attribution
        
        if title is None:
            title = f"LRP Attribution ({self.rule}-rule)"
        
        visualize_attribution(plt, image, relevance, title=title, method=method, **kwargs)

    def _compute_from_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP starting from embedding layer.

        This method:
        1. Embeds discrete inputs into continuous space
        2. Performs forward pass while capturing activations
        3. Initializes relevance at output layer
        4. Propagates relevance backward to embeddings
        5. Maps relevance back to input tokens

        Args:
            inputs: Dictionary of input tensors for each feature.
            target_class_idx: Target class for attribution.
            time_info: Optional time information for temporal models.
            label_data: Optional label data to pass to model.

        Returns:
            Dictionary of relevance scores per feature.
        """
        # Step 1: Embed inputs using model's embedding layer
        input_embeddings = {}
        input_shapes = {}  # Store original shapes for later mapping

        for key in inputs:
            input_shapes[key] = inputs[key].shape
            # Get embeddings from model's embedding layer
            embedded = self.model.embedding_model({key: inputs[key]})
            x = embedded[key]

            # Handle nested sequences (4D tensors) by pooling
            if x.dim() == 4:  # [batch, seq_len, tokens, embedding_dim]
                # Sum pool over inner dimension
                x = x.sum(dim=2)  # [batch, seq_len, embedding_dim]

            input_embeddings[key] = x

        # Step 2: Register hooks to capture activations during forward pass
        self._register_hooks()

        try:
            # Step 3: Forward pass through model
            forward_kwargs = {**label_data} if label_data else {}

            with torch.no_grad():
                output = self.model.forward_from_embedding(
                    feature_embeddings=input_embeddings,
                    time_info=time_info,
                    **forward_kwargs,
                )
            logits = output["logit"]

            # Step 4: Determine target class
            if target_class_idx is None:
                target_class_idx = torch.argmax(logits, dim=-1)
            elif not isinstance(target_class_idx, torch.Tensor):
                target_class_idx = torch.tensor(
                    target_class_idx, device=logits.device
                )

            # Step 5: Initialize output relevance
            # For classification: start with the target class output
            if logits.dim() == 2 and logits.size(-1) > 1:
                # Multi-class: one-hot encoding
                batch_size = logits.size(0)
                output_relevance = torch.zeros_like(logits)
                output_relevance[range(batch_size), target_class_idx] = logits[
                    range(batch_size), target_class_idx
                ]
            else:
                # Binary classification
                output_relevance = logits

            # Step 6: Propagate relevance backward through network
            relevance_at_embeddings = self._propagate_relevance_backward(
                output_relevance, input_embeddings
            )

            # Step 7: Map relevance back to input space
            input_relevances = {}
            for key in input_embeddings:
                rel = relevance_at_embeddings.get(key)
                if rel is not None:
                    # Sum over embedding dimension to get per-token relevance
                    if rel.dim() == 3:  # [batch, seq_len, embedding_dim]
                        input_relevances[key] = rel.sum(dim=-1)  # [batch, seq_len]
                    elif rel.dim() == 2:  # [batch, embedding_dim]
                        input_relevances[key] = rel.sum(dim=-1)  # [batch]
                    else:
                        input_relevances[key] = rel

                    # Expand to match original input shape if needed
                    orig_shape = input_shapes[key]
                    if input_relevances[key].shape != orig_shape:
                        # Handle case where input was 3D but we have 2D relevance
                        if len(orig_shape) == 3 and input_relevances[key].dim() == 2:
                            # Broadcast to match
                            input_relevances[key] = input_relevances[key].unsqueeze(
                                -1
                            ).expand(orig_shape)

        finally:
            # Step 8: Clean up hooks
            self._remove_hooks()

        return input_relevances

    def _compute_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class_idx: Optional[int] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP starting directly from continuous inputs (e.g., images).

        This method is used for CNN models that work directly on continuous data
        without an embedding layer.

        Args:
            inputs: Dictionary of input tensors for each feature (e.g., {'image': tensor}).
            target_class_idx: Target class for attribution.
            label_data: Optional label data to pass to model.

        Returns:
            Dictionary of relevance scores per feature.
        """
        self.model.eval()
        
        # Register hooks to capture activations
        self._register_hooks()

        try:
            # Forward pass through model
            forward_kwargs = {**inputs}
            if label_data:
                forward_kwargs.update(label_data)
            
            with torch.no_grad():
                output = self.model(**forward_kwargs)
            
            logits = output.get("logit", output.get("y_prob", output.get("y_pred")))
            
            # Determine target class
            if target_class_idx is None:
                target_class_idx = torch.argmax(logits, dim=-1)
            elif not isinstance(target_class_idx, torch.Tensor):
                target_class_idx = torch.tensor(
                    target_class_idx, device=logits.device
                )

            # Initialize output relevance
            if logits.dim() == 2 and logits.size(-1) > 1:
                # Multi-class: one-hot encoding
                batch_size = logits.size(0)
                output_relevance = torch.zeros_like(logits)
                output_relevance[range(batch_size), target_class_idx] = logits[
                    range(batch_size), target_class_idx
                ]
            else:
                # Binary classification
                output_relevance = logits

            # Propagate relevance backward through network
            relevance_at_inputs = self._propagate_relevance_backward(
                output_relevance, inputs
            )

            # If direct inputs were used, return them directly
            if not isinstance(relevance_at_inputs, dict):
                # Convert to dict format
                relevance_at_inputs = {list(inputs.keys())[0]: relevance_at_inputs}

        finally:
            # Clean up hooks
            self._remove_hooks()

        return relevance_at_inputs

    def _register_hooks(self):
        """Register forward hooks to capture activations during forward pass.

        Hooks are attached to all relevant layer types to capture both
        inputs and outputs for later relevance propagation.
        
        Also detects branching structure (e.g., ModuleDict with parallel branches).
        """

        def save_activation(name):
            def hook(module, input, output):
                # Store both input and output activations
                # Handle tuple inputs (e.g., from LSTM)
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output

                self.activations[name] = {
                    "input": input_tensor,
                    "output": output_tensor,
                    "module": module,
                }

            return hook

        # Register hooks on layers we can propagate through
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LSTM, nn.GRU, 
                                 nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, 
                                 nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
                handle = module.register_forward_hook(save_activation(name))
                self.hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _match_shapes(
        self,
        relevance: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        """Match relevance tensor shape to target shape."""
        if relevance.shape == target_shape:
            return relevance
        
        batch_size = relevance.shape[0]
        
        # 2D -> 4D: expand to spatial
        if relevance.dim() == 2 and len(target_shape) == 4:
            if relevance.shape[1] == target_shape[1] * target_shape[2] * target_shape[3]:
                return relevance.view(batch_size, *target_shape[1:])
            # Uniform distribution fallback
            return (relevance.sum(dim=1, keepdim=True) / (target_shape[1] * target_shape[2] * target_shape[3])
                   ).view(batch_size, 1, 1, 1).expand(batch_size, *target_shape[1:])
        
        # 4D -> 4D: adjust channels and/or spatial dims
        if relevance.dim() == 4 and len(target_shape) == 4:
            if relevance.shape[1] != target_shape[1]:
                relevance = relevance.mean(dim=1, keepdim=True).expand(-1, target_shape[1], -1, -1)
            if relevance.shape[2:] != target_shape[2:]:
                relevance = F.interpolate(relevance, size=target_shape[2:], mode='bilinear', align_corners=False)
            return relevance
        
        # 3D -> 4D: add channel dimension
        if relevance.dim() == 3 and len(target_shape) == 4:
            relevance = relevance.unsqueeze(1).expand(-1, target_shape[1], -1, -1)
            if relevance.shape[2:] != target_shape[2:]:
                relevance = F.interpolate(relevance, size=target_shape[2:], mode='bilinear', align_corners=False)
            return relevance
        
        return relevance

    def _propagate_relevance_backward(
        self,
        output_relevance: torch.Tensor,
        input_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Propagate relevance from output layer back to input embeddings.

        This is the core LRP algorithm. It iterates through layers in
        reverse order, applying the appropriate LRP rule to redistribute
        relevance from each layer to the previous layer.

        Args:
            output_relevance: Relevance at the output layer.
            input_embeddings: Dictionary of input embeddings for each feature.

        Returns:
            Dictionary of relevance scores at the embedding layer.
        """
        current_relevance = output_relevance
        layer_names = list(reversed(list(self.activations.keys())))

        # For MLP models with parallel feature branches, track relevance per branch
        feature_relevances = {}  # Maps feature keys to their relevance tensors
        concat_detected = False
        
        # Propagate through each layer
        for idx, layer_name in enumerate(layer_names):
            activation_info = self.activations[layer_name]
            module = activation_info["module"]
            output_tensor = activation_info["output"]
            
            # Check if this is a concatenation point (PyHealth MLP pattern)
            # Pattern: fc layer takes concatenated input from multiple feature MLPs
            if (not concat_detected and isinstance(module, nn.Linear) and 
                hasattr(self.model, 'feature_keys') and len(self.model.feature_keys) > 1):
                
                # Check if next layers are feature-specific MLPs
                if idx + 1 < len(layer_names):
                    next_name = layer_names[idx + 1]
                    # Pattern like "mlp.conditions.2" or "mlp.labs.0"
                    if 'mlp.' in next_name and any(f in next_name for f in self.model.feature_keys):
                        # This is the concatenation point - split relevance after processing fc
                        concat_detected = True
                        
            # Ensure shape compatibility before layer processing
            if current_relevance.shape != output_tensor.shape:
                current_relevance = self._match_shapes(current_relevance, output_tensor.shape)
            
            # Apply appropriate LRP rule based on layer type
            if isinstance(module, nn.Linear):
                current_relevance = self._lrp_linear(module, activation_info, current_relevance)
            elif isinstance(module, nn.Conv2d):
                current_relevance = self._lrp_conv2d(module, activation_info, current_relevance)
            elif isinstance(module, nn.ReLU):
                current_relevance = self._lrp_relu(activation_info, current_relevance)
            elif isinstance(module, nn.MaxPool2d):
                current_relevance = self._lrp_maxpool2d(module, activation_info, current_relevance)
            elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                current_relevance = self._lrp_avgpool2d(module, activation_info, current_relevance)
            elif isinstance(module, nn.BatchNorm2d):
                current_relevance = self._lrp_batchnorm2d(module, activation_info, current_relevance)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                current_relevance = self._lrp_rnn(module, activation_info, current_relevance)
            
            # After processing, check if we need to split for parallel branches
            if concat_detected and current_relevance.dim() == 2:
                # Split relevance equally among features
                # Each feature gets embedding_dim dimensions
                n_features = len(self.model.feature_keys)
                feature_dim = current_relevance.size(1) // n_features
                
                for i, feature_key in enumerate(self.model.feature_keys):
                    start_idx = i * feature_dim
                    end_idx = (i + 1) * feature_dim
                    feature_relevances[feature_key] = current_relevance[:, start_idx:end_idx]
                
                # Now process each branch independently
                # Continue with the rest of the layers, routing to appropriate branches
                break

        # If we detected concatenation, process remaining layers per feature
        if concat_detected:
            for feature_key in self.model.feature_keys:
                current_rel = feature_relevances[feature_key]
                
                # Find layers for this feature
                for layer_name in layer_names[idx+1:]:
                    if feature_key not in layer_name:
                        continue
                        
                    activation_info = self.activations[layer_name]
                    module = activation_info["module"]
                    output_tensor = activation_info["output"]
                    
                    if current_rel.shape != output_tensor.shape:
                        current_rel = self._match_shapes(current_rel, output_tensor.shape)
                    
                    if isinstance(module, nn.Linear):
                        current_rel = self._lrp_linear(module, activation_info, current_rel)
                    elif isinstance(module, nn.ReLU):
                        current_rel = self._lrp_relu(activation_info, current_rel)
                
                feature_relevances[feature_key] = current_rel
            
            return self._split_relevance_to_features(feature_relevances, input_embeddings)
        
        return self._split_relevance_to_features(current_relevance, input_embeddings)

    def _lrp_linear(
        self,
        module: nn.Linear,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply LRP to a linear (fully connected) layer.

        Uses either epsilon-rule or alphabeta-rule depending on
        initialization.

        Args:
            module: The linear layer.
            activation_info: Dictionary containing input/output activations.
            relevance_output: Relevance from the next layer.

        Returns:
            Relevance for the previous layer.
        """
        if self.rule == "epsilon":
            return self._lrp_linear_epsilon(module, activation_info, relevance_output)
        elif self.rule == "alphabeta":
            return self._lrp_linear_alphabeta(
                module, activation_info, relevance_output
            )
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

    def _lrp_linear_epsilon(
        self,
        module: nn.Linear,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP epsilon-rule for linear layers.

        Formula: R_i = Σ_j (z_ij / (z_j + ε·sign(z_j))) · R_j
        """
        from pyhealth.interpret.methods.lrp_base import stabilize_denominator
        
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        z = F.linear(x, module.weight, module.bias)
        z = stabilize_denominator(z, self.epsilon, rule="epsilon")
        s = relevance_output / z
        c = torch.einsum('bo,oi->bi', s, module.weight)
        return x * c

    def _lrp_linear_alphabeta(
        self,
        module: nn.Linear,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP alphabeta-rule for linear layers.

        Formula: R_i = Σ_j [(α·z_ij^+ / z_j^+) - (β·z_ij^- / z_j^-)] · R_j
        """
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        W_pos, W_neg = torch.clamp(module.weight, min=0), torch.clamp(module.weight, max=0)
        b_pos = torch.clamp(module.bias, min=0) if module.bias is not None else None
        b_neg = torch.clamp(module.bias, max=0) if module.bias is not None else None

        z_pos = F.linear(x, W_pos, b_pos) + 1e-9
        z_neg = F.linear(x, W_neg, b_neg) - 1e-9

        c_pos = torch.einsum('bo,oi->bi', relevance_output / z_pos, W_pos)
        c_neg = torch.einsum('bo,oi->bi', relevance_output / z_neg, W_neg)

        return x * (self.alpha * c_pos - self.beta * c_neg)

    def _lrp_relu(
        self, activation_info: dict, relevance_output: torch.Tensor
    ) -> torch.Tensor:
        """LRP for ReLU - relevance passes through unchanged."""
        return relevance_output

    def _lrp_conv2d(
        self,
        module: nn.Conv2d,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP for Conv2d layers.

        Applies the chosen LRP rule (epsilon or alphabeta) to convolutional layers.
        """
        if self.rule == "epsilon":
            return self._lrp_conv2d_epsilon(module, activation_info, relevance_output)
        elif self.rule == "alphabeta":
            return self._lrp_conv2d_alphabeta(module, activation_info, relevance_output)
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

    def _compute_conv_output_padding(self, module: nn.Conv2d, z_shape: torch.Size, x_shape: torch.Size) -> tuple:
        """Compute output_padding for conv_transpose2d to match input shape."""
        output_padding = []
        for i in range(2):  # H and W dimensions
            stride = module.stride[i] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[i] if isinstance(module.padding, tuple) else module.padding
            dilation = module.dilation[i] if isinstance(module.dilation, tuple) else module.dilation
            kernel_size = module.weight.shape[2 + i]
            expected = (z_shape[2 + i] - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
            output_padding.append(max(0, x_shape[2 + i] - expected))
        return tuple(output_padding)

    def _adjust_spatial_shape(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Adjust spatial dimensions (H, W) to match target shape."""
        if tensor.shape[2:] == target_shape[2:]:
            return tensor
        # Crop or pad as needed
        if tensor.shape[2] > target_shape[2] or tensor.shape[3] > target_shape[3]:
            return tensor[:, :, :target_shape[2], :target_shape[3]]
        if tensor.shape[2] < target_shape[2] or tensor.shape[3] < target_shape[3]:
            pad_h = target_shape[2] - tensor.shape[2]
            pad_w = target_shape[3] - tensor.shape[3]
            return F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor

    def _lrp_conv2d_epsilon(
        self,
        module: nn.Conv2d,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP epsilon-rule for Conv2d."""
        from pyhealth.interpret.methods.lrp_base import stabilize_denominator
        
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        
        z = F.conv2d(x, module.weight, module.bias, stride=module.stride, padding=module.padding,
                     dilation=module.dilation, groups=module.groups)
        z = stabilize_denominator(z, self.epsilon, rule="epsilon")
        s = relevance_output / z
        
        output_padding = self._compute_conv_output_padding(module, z.shape, x.shape)
        c = F.conv_transpose2d(s, module.weight, stride=module.stride, padding=module.padding,
                               output_padding=output_padding, dilation=module.dilation, groups=module.groups)
        c = self._adjust_spatial_shape(c, x.shape)
        return x * c

    def _lrp_conv2d_alphabeta(
        self,
        module: nn.Conv2d,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP alphabeta-rule for Conv2d."""
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        
        W_pos, W_neg = torch.clamp(module.weight, min=0), torch.clamp(module.weight, max=0)
        b_pos = torch.clamp(module.bias, min=0) if module.bias is not None else None
        b_neg = torch.clamp(module.bias, max=0) if module.bias is not None else None
        
        conv_kwargs = dict(stride=module.stride, padding=module.padding, 
                          dilation=module.dilation, groups=module.groups)
        z_pos = F.conv2d(x, W_pos, b_pos, **conv_kwargs)
        z_neg = F.conv2d(x, W_neg, b_neg, **conv_kwargs)
        z_total = z_pos + z_neg + self.epsilon * torch.sign(z_pos + z_neg)
        
        s = relevance_output / z_total
        output_padding = self._compute_conv_output_padding(module, z_pos.shape, x.shape)
        
        c_pos = F.conv_transpose2d(s, W_pos, stride=module.stride, padding=module.padding,
                                   output_padding=output_padding, **conv_kwargs)
        c_neg = F.conv_transpose2d(s, W_neg, stride=module.stride, padding=module.padding,
                                   output_padding=output_padding, **conv_kwargs)
        
        c_pos = self._adjust_spatial_shape(c_pos, x.shape)
        c_neg = self._adjust_spatial_shape(c_neg, x.shape)
        
        return x * (self.alpha * c_pos + self.beta * c_neg)

    def _lrp_maxpool2d(
        self,
        module: nn.MaxPool2d,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP for MaxPool2d.

        For max pooling, relevance is passed only to the winning (maximum) positions.

        Args:
            module: The MaxPool2d layer.
            activation_info: Stored activations.
            relevance_output: Relevance from next layer.

        Returns:
            Relevance for input to this layer.
        """
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        
        # Get the output and indices from max pooling
        output, indices = F.max_pool2d(
            x,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=True
        )
        
        # Unpool the relevance to input size
        relevance_input = F.max_unpool2d(
            relevance_output,
            indices,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_size=x.size()
        )
        
        return relevance_input

    def _lrp_avgpool2d(
        self,
        module: nn.Module,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP for AvgPool2d and AdaptiveAvgPool2d - distribute relevance uniformly."""
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return F.interpolate(relevance_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Regular AvgPool2d: upsample using transposed convolution with uniform weights
        kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        
        channels = relevance_output.size(1)
        weight = torch.ones(channels, 1, *kernel_size, device=x.device) / (kernel_size[0] * kernel_size[1])
        
        relevance_input = F.conv_transpose2d(relevance_output, weight, stride=stride,
                                              padding=padding, groups=channels)
        return self._adjust_spatial_shape(relevance_input, x.shape)

    def _lrp_batchnorm2d(
        self,
        module: nn.BatchNorm2d,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP for BatchNorm2d - pass through with gamma scaling."""
        gamma = module.weight.view(1, -1, 1, 1) if module.weight is not None else 1.0
        return relevance_output * gamma

    def _lrp_rnn(
        self,
        module: nn.Module,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP for RNN/LSTM/GRU - simplified uniform distribution."""
        input_tensor = activation_info["input"]
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]

        if input_tensor.dim() == 3:
            batch_size, seq_len = input_tensor.shape[:2]
            return relevance_output.unsqueeze(1).expand(batch_size, seq_len, -1)
        return relevance_output

    def _split_relevance_to_features(
        self,
        relevance,  # Can be torch.Tensor or Dict[str, torch.Tensor]
        input_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Split combined relevance back to individual features.

        In PyHealth models, embeddings from different features are
        concatenated before final classification. This method splits
        the relevance back to each feature.
        
        Note: After embeddings pass through the model, sequences are typically
        pooled (mean/sum), so relevance shape is [batch, total_concat_dim] where
        total_concat_dim is the sum of all feature dimensions after pooling.

        Args:
            relevance: Either:
                - Tensor [batch, total_dim] - relevance at concatenated layer
                - Dict mapping feature keys to relevance tensors (already split)
            input_embeddings: Original input embeddings for each feature.

        Returns:
            Dictionary mapping feature keys to their relevance tensors.
        """
        relevance_by_feature = {}
        
        # If relevance is already split per feature, just broadcast to input shapes
        if isinstance(relevance, dict):
            for key, rel_tensor in relevance.items():
                if key not in input_embeddings:
                    continue
                    
                emb_shape = input_embeddings[key].shape
                if len(emb_shape) == 3 and rel_tensor.dim() == 2:
                    # Broadcast: [batch, emb_dim] → [batch, seq_len, emb_dim]
                    rel_tensor = rel_tensor.unsqueeze(1).expand(
                        emb_shape[0], emb_shape[1], emb_shape[2]
                    )
                relevance_by_feature[key] = rel_tensor
            return relevance_by_feature

        # Calculate the actual concatenated size for each feature
        # This must match what the model actually does after pooling
        feature_sizes = {}
        for key, emb in input_embeddings.items():
            if emb.dim() == 3:  # [batch, seq_len, embedding_dim]
                # After pooling (mean/sum over seq), becomes [batch, embedding_dim]
                feature_sizes[key] = emb.size(2)  # Just the embedding dimension
            elif emb.dim() == 2:  # [batch, feature_dim]
                # Stays as-is (e.g., tensor features like labs)
                feature_sizes[key] = emb.size(1)
            else:
                # Fallback
                feature_sizes[key] = emb.numel() // emb.size(0)

        # Verify total matches relevance size
        total_size = sum(feature_sizes.values())
        if relevance.dim() == 2 and relevance.size(1) != total_size:
            # Size mismatch - this can happen if model has additional processing
            # Distribute relevance equally to all features as fallback
            for key in input_embeddings:
                relevance_by_feature[key] = relevance / len(input_embeddings)
            return relevance_by_feature

        # Split relevance according to feature sizes
        # Features are concatenated in the order of feature_keys
        if relevance.dim() == 2:  # [batch, total_dim]
            current_idx = 0
            for key in self.model.feature_keys:
                if key in input_embeddings:
                    size = feature_sizes[key]
                    rel_chunk = relevance[:, current_idx : current_idx + size]

                    # For 3D embeddings (sequences), broadcast relevance across sequence
                    emb_shape = input_embeddings[key].shape
                    if len(emb_shape) == 3:
                        # Broadcast: [batch, emb_dim] → [batch, seq_len, emb_dim]
                        rel_chunk = rel_chunk.unsqueeze(1).expand(
                            emb_shape[0], emb_shape[1], emb_shape[2]
                        )
                    # For 2D embeddings (tensors), shape is already correct

                    relevance_by_feature[key] = rel_chunk
                    current_idx += size
        else:
            # If relevance doesn't match expected shape, distribute equally
            for key in input_embeddings:
                relevance_by_feature[key] = relevance / len(input_embeddings)

        return relevance_by_feature


# ============================================================================
# Unified LRP Implementation
# ============================================================================


class UnifiedLRP:
    """Unified Layer-wise Relevance Propagation for CNNs and embedding-based models.
    
    This class automatically detects layer types and applies appropriate
    LRP rules using a modular handler system. Supports:
    
    - **CNNs**: Conv2d, pooling, batch norm, skip connections
    - **Embedding models**: Linear, LSTM, GRU with embeddings
    - **Mixed models**: Multimodal architectures with both images and codes
    
    The implementation ensures relevance conservation at each layer and
    provides comprehensive debugging tools.
    
    Args:
        model: PyTorch model to interpret (can be any nn.Module)
        rule: LRP propagation rule ('epsilon' or 'alphabeta')
        epsilon: Stabilization parameter for epsilon rule (default: 0.01)
        alpha: Positive contribution weight for alphabeta rule (default: 2.0)
        beta: Negative contribution weight for alphabeta rule (default: 1.0)
        validate_conservation: If True, check conservation at each layer (default: True)
        conservation_tolerance: Maximum allowed conservation error (default: 0.01 = 1%)
        
    Examples:
        >>> # For CNN models (images)
        >>> from pyhealth.models import TorchvisionModel
        >>> model = TorchvisionModel(dataset, model_name="resnet18")
        >>> lrp = UnifiedLRP(model, rule='epsilon', epsilon=0.01)
        >>> 
        >>> # Compute attributions
        >>> attributions = lrp.attribute(
        ...     inputs={'image': chest_xray},
        ...     target_class=0
        ... )
        >>> 
        >>> # For embedding-based models
        >>> from pyhealth.models import RNN
        >>> model = RNN(dataset, feature_keys=['conditions'])
        >>> lrp = UnifiedLRP(model, rule='epsilon')
        >>> 
        >>> attributions = lrp.attribute(
        ...     inputs={'conditions': patient_codes},
        ...     target_class=1
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        rule: str = "epsilon",
        epsilon: float = 0.01,
        alpha: float = 2.0,
        beta: float = 1.0,
        validate_conservation: bool = True,
        conservation_tolerance: float = 0.01,
        custom_registry: Optional = None
    ):
        """Initialize UnifiedLRP.
        
        Args:
            model: Model to interpret
            rule: LRP rule ('epsilon', 'alphabeta')
            epsilon: Stabilization parameter
            alpha: Alpha parameter for alphabeta rule
            beta: Beta parameter for alphabeta rule
            validate_conservation: Whether to validate conservation property
            conservation_tolerance: Maximum allowed conservation error (fraction)
            custom_registry: Optional custom handler registry (uses default if None)
        """
        from .lrp_base import create_default_registry, ConservationValidator, AdditionLRPHandler
        
        self.model = model
        self.model.eval()
        
        self.rule = rule
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        self.registry = custom_registry if custom_registry else create_default_registry()
        self.addition_handler = AdditionLRPHandler()
        
        # Clear all handler caches to ensure clean state
        for handler in self.registry._handlers:
            if hasattr(handler, 'clear_cache'):
                handler.clear_cache()
        
        # Detect ResNet architecture and identify skip connections
        self.skip_connections = self._detect_skip_connections()
        self.block_caches = {}
        
        self.validate_conservation = validate_conservation
        self.validator = ConservationValidator(
            tolerance=conservation_tolerance,
            strict=False
        )
        
        self.hooks = []
        self.layer_order = []
    
    def _detect_skip_connections(self):
        """Detect ResNet BasicBlock/Bottleneck modules with skip connections.
        
        Returns:
            List of (block_name, block_module, has_downsample) tuples
        """
        skip_connections = []
        
        for name, module in self.model.named_modules():
            # Check if it's a ResNet BasicBlock or Bottleneck
            module_name = type(module).__name__
            if module_name in ['BasicBlock', 'Bottleneck']:
                # Check if it has a downsample layer (1x1 conv for dimension matching)
                has_downsample = hasattr(module, 'downsample') and module.downsample is not None
                skip_connections.append((name, module, has_downsample))
        
        return skip_connections
    
    def attribute(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
        return_intermediates: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP attributions for given inputs.
        
        This is the main entry point for computing attributions. The method:
        1. Detects relevant layers and registers hooks
        2. Performs forward pass to capture activations
        3. Initializes relevance at output layer
        4. Propagates relevance backward through layers
        5. Returns relevance at input layer(s)
        
        Args:
            inputs: Dictionary of input tensors, e.g.:
                - {'image': torch.Tensor} for CNNs
                - {'conditions': torch.Tensor} for embedding models
                - Multiple keys for multimodal models
            target_class: Class index to explain (None = predicted class)
            return_intermediates: If True, return relevance at all layers
            **kwargs: Additional arguments passed to model forward
            
        Returns:
            Dictionary mapping input keys to relevance tensors
                
        Raises:
            RuntimeError: If model forward pass fails
            ValueError: If inputs are invalid
        """
        from .lrp_base import check_tensor_validity
        
        if not inputs:
            raise ValueError("inputs dictionary cannot be empty")
        
        for key, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Input '{key}' must be a torch.Tensor")
            check_tensor_validity(tensor, f"input[{key}]")
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if self.validate_conservation:
            self.validator.reset()
        
        try:
            self._register_hooks()
            
            with torch.no_grad():
                outputs = self.model(**inputs, **kwargs)
            
            logits = self._extract_logits(outputs)
            
            if target_class is None:
                target_class = torch.argmax(logits, dim=-1)
            
            output_relevance = self._initialize_output_relevance(
                logits, target_class
            )
            
            input_relevances = self._propagate_backward(
                output_relevance,
                inputs,
                return_intermediates
            )
            
            if self.validate_conservation:
                self.validator.print_summary()
            
            return input_relevances
            
        finally:
            self._remove_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all supported layers."""
        self.layer_order.clear()
        
        # Note: Skip connection hooks disabled for sequential processing
        # BasicBlocks are detected but not hooked
        # Downsample layers (part of skip connections) are excluded from sequential processing
        
        # Register hooks for regular layers
        for name, module in self.model.named_modules():
            # Skip downsample layers - they're part of skip connections
            if 'downsample' in name:
                continue
                
            handler = self.registry.get_handler(module)
            
            if handler is not None:
                def create_hook(handler_ref, module_ref, name_ref):
                    def hook(module, input, output):
                        handler_ref.forward_hook(module, input, output)
                    return hook
                
                handle = module.register_forward_hook(
                    create_hook(handler, module, name)
                )
                self.hooks.append(handle)
                self.layer_order.append((name, module, handler))
    
    def _remove_hooks(self):
        """Remove all registered hooks and clear caches."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Clear caches from ALL handlers in the registry (not just registered ones)
        for handler in self.registry._handlers:
            if hasattr(handler, 'clear_cache'):
                handler.clear_cache()
        
        for _, _, handler in self.layer_order:
            handler.clear_cache()
        
        self.layer_order.clear()
        self.block_caches.clear()
        
        # Clear any pending identity relevance
        if hasattr(self, '_pending_identity_relevance'):
            self._pending_identity_relevance.clear()
    
    def _extract_logits(self, outputs) -> torch.Tensor:
        """Extract logits from model output."""
        if isinstance(outputs, dict):
            if 'logit' in outputs:
                return outputs['logit']
            elif 'y_prob' in outputs:
                return torch.log(outputs['y_prob'] + 1e-10)
            elif 'y_pred' in outputs:
                return outputs['y_pred']
            else:
                raise ValueError(
                    f"Cannot extract logits from output keys: {outputs.keys()}"
                )
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")
    
    def _initialize_output_relevance(
        self,
        logits: torch.Tensor,
        target_class
    ) -> torch.Tensor:
        """Initialize relevance at the output layer."""
        batch_size = logits.size(0)
        
        if logits.dim() == 2 and logits.size(-1) > 1:
            output_relevance = torch.zeros_like(logits)
            
            # Convert target_class to tensor if needed
            if isinstance(target_class, int):
                target_class = torch.tensor([target_class] * batch_size)
            elif isinstance(target_class, torch.Tensor):
                if target_class.dim() == 0:
                    target_class = target_class.unsqueeze(0).expand(batch_size)
            
            for i in range(batch_size):
                output_relevance[i, target_class[i]] = logits[i, target_class[i]]
        else:
            output_relevance = logits
        
        return output_relevance
    
    def _propagate_backward(
        self,
        output_relevance: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Propagate relevance backward through all layers.
        
        For ResNet architectures, uses sequential approximation by processing
        only the residual path layers (downsample layers are excluded during
        hook registration). This is a standard approach in the LRP literature.
        
        Args:
            output_relevance: Relevance at the output layer
            inputs: Original model inputs (for final mapping)
            return_intermediates: If True, return relevance at each layer
            
        Returns:
            Dictionary mapping input keys to their relevance scores
        """
        from .lrp_base import check_tensor_validity
        
        current_relevance = output_relevance
        intermediate_relevances = {}
        
        # Process layers in reverse order (standard LRP backward pass)
        for idx in range(len(self.layer_order) - 1, -1, -1):
            name, module, handler = self.layer_order[idx]
            
            # Backward propagation through this layer
            prev_relevance = handler.backward_relevance(
                layer=module,
                relevance_output=current_relevance,
                rule=self.rule,
                epsilon=self.epsilon,
                alpha=self.alpha,
                beta=self.beta
            )
            
            if self.validate_conservation:
                self.validator.validate(
                    layer_name=name,
                    relevance_input=prev_relevance,
                    relevance_output=current_relevance,
                    layer_type=type(module).__name__
                )
            
            if return_intermediates:
                intermediate_relevances[name] = prev_relevance.detach().clone()
            
            current_relevance = prev_relevance
            check_tensor_validity(current_relevance, f"relevance after {name}")
        
        input_relevances = self._map_to_inputs(current_relevance, inputs)
        
        if return_intermediates:
            input_relevances['_intermediates'] = intermediate_relevances
        
        return input_relevances
    
    def _get_parent_basic_block(self, layer_name: str):
        """Get the ID of the parent BasicBlock if this layer is inside one."""
        # E.g., "layer1.0.conv1" -> check if "layer1.0" is a BasicBlock
        parts = layer_name.split('.')
        for i in range(len(parts), 0, -1):
            parent_name = '.'.join(parts[:i])
            parent_module = dict(self.model.named_modules()).get(parent_name)
            if parent_module is not None and type(parent_module).__name__ in ['BasicBlock', 'Bottleneck']:
                return id(parent_module)
        return None
    
    def _is_block_input_layer(self, layer_name: str, block_id: int, skip_map: dict) -> bool:
        """Check if this is the first convolution layer in a BasicBlock."""
        if block_id not in skip_map:
            return False
        
        block_name, _, _ = skip_map[block_id]
        # The first conv is typically named "block_name.conv1"
        return layer_name == f"{block_name}.conv1"
    
    def _map_to_inputs(
        self,
        relevance: torch.Tensor,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Map final relevance tensor back to input structure."""
        if len(inputs) == 1:
            key = list(inputs.keys())[0]
            return {key: relevance}
        
        # Multi-input case
        return {key: relevance for key in inputs.keys()}
    
    def get_conservation_summary(self) -> Dict:
        """Get conservation validation summary."""
        return self.validator.get_summary()
