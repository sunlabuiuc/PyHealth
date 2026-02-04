from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from pyhealth.models import BaseModel
from .base_interpreter import BaseInterpreter


class ShapExplainer(BaseInterpreter):
    """SHAP (SHapley Additive exPlanations) attribution method for PyHealth models.

    This class implements the SHAP method for computing feature attributions in 
    neural networks. SHAP values represent each feature's contribution to the 
    prediction, based on coalitional game theory principles.

    The method is based on the paper:
        A Unified Approach to Interpreting Model Predictions
        Scott Lundberg, Su-In Lee
        NeurIPS 2017
        https://arxiv.org/abs/1705.07874

    Kernel SHAP Method:
    This implementation uses Kernel SHAP, which combines ideas from LIME (Local 
    Interpretable Model-agnostic Explanations) with Shapley values from game theory. 
    The key steps are:
    1. Generate background samples to establish baseline predictions
    2. Create feature coalitions (subsets of features) using weighted sampling
    3. Compute model predictions for each coalition
    4. Solve a weighted least squares problem to estimate Shapley values
    
    Mathematical Foundation:
    The Shapley value for feature i is computed as:
    φᵢ = Σ (|S|!(n-|S|-1)!/n!) * [f₀(S ∪ {i}) - f₀(S)]
    where:
    - S is a subset of features excluding i
    - n is the total number of features
    - f₀(S) is the model prediction with only features in S
    
    SHAP provides several desirable properties:
    1. Local Accuracy: The sum of feature attributions equals the difference between 
       the model output and the expected output
    2. Missingness: Features with zero impact get zero attribution
    3. Consistency: Changing a model to increase a feature's impact increases its attribution

    Args:
        model: A trained PyHealth model to interpret. Can be any model that
            inherits from BaseModel (e.g., MLP, StageNet, Transformer, RNN).
        use_embeddings: If True, compute SHAP values with respect to
            embeddings rather than discrete input tokens. This is crucial
            for models with discrete inputs (like ICD codes). The model
            must support returning embeddings via an 'embed' parameter.
            Default is True.
        n_background_samples: Number of background samples to use for
            estimating feature contributions. More samples give better
            estimates but increase computation time. Default is 100.
        max_coalitions: Maximum number of feature coalitions to sample for
            Kernel SHAP approximation. Default is 1000.
        regularization: L2 regularization strength for the weighted least
            squares problem. Default is 1e-6.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import SampleDataset, get_dataloader
        >>> from pyhealth.models import MLP
        >>> from pyhealth.interpret.methods import ShapExplainer
        >>> from pyhealth.trainer import Trainer
        >>>
        >>> # Create dataset and model 
        >>> dataset = SampleDataset(...)
        >>> model = MLP(...)
        >>> trainer = Trainer(model=model, device="cuda:0")
        >>> trainer.train(...)
        >>> test_batch = next(iter(test_loader))
        >>>
        >>> # Initialize SHAP explainer
        >>> explainer = ShapExplainer(model, use_embeddings=True)
        >>> shap_values = explainer.attribute(**test_batch)
        >>>
        >>> # With custom baseline
        >>> baseline = {
        ...     'conditions': torch.zeros_like(test_batch['conditions']),
        ...     'procedures': torch.full_like(test_batch['procedures'], 
        ...                                   test_batch['procedures'].mean())
        ... }
        >>> shap_values = explainer.attribute(baseline=baseline, **test_batch)
        >>>
        >>> print(shap_values)
        {'conditions': tensor([[0.1234, 0.5678, 0.9012]], device='cuda:0'),
         'procedures': tensor([[0.2345, 0.6789, 0.0123, 0.4567]])}
    """

    def __init__(
        self,
        model: BaseModel,
        use_embeddings: bool = True,
        n_background_samples: int = 100,
        max_coalitions: int = 1000,
        regularization: float = 1e-6,
        random_seed: Optional[int] = 42,
    ):
        """Initialize SHAP explainer.

        Args:
            model: A trained PyHealth model to interpret.
            use_embeddings: If True, compute SHAP values with respect to
                embeddings rather than discrete input tokens.
            n_background_samples: Number of background samples to use for
                estimating feature contributions.
            max_coalitions: Maximum number of feature coalitions to sample.
            regularization: L2 regularization strength for weighted least squares.
            random_seed: Optional random seed for reproducibility. If provided,
                this seed will be used to initialize the random number generator
                before each attribution computation, ensuring deterministic results.

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method.
        """
        super().__init__(model)
        self.use_embeddings = use_embeddings
        self.n_background_samples = n_background_samples
        self.max_coalitions = max_coalitions
        self.regularization = regularization
        self.random_seed = random_seed

        # Validate model requirements
        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "SHAP values. Set use_embeddings=False to use "
                "input-level attributions (only for continuous features)."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute SHAP attributions for input features.

        This is the main interface for computing feature attributions. It handles:
        1. Input preparation and validation
        2. Background sample generation or validation
        3. Feature attribution computation using Kernel SHAP

        Args:
            baseline: Optional dictionary mapping feature names to background 
                     samples. If None, generates samples automatically using
                     _generate_background_samples(). Shape of each tensor should
                     be (n_background_samples, ..., feature_dim).
            target_class_idx: For multi-class models, specifies which class's 
                            prediction to explain. If None, explains the model's
                            maximum prediction across all classes.
            **data: Input data dictionary from dataloader batch. Should contain:
                   - Feature tensors with shape (batch_size, ..., feature_dim)
                   - Optional time information for temporal models
                   - Optional label data for supervised models

        Returns:
            Dictionary mapping feature names to their SHAP values. Each value
            tensor has the same shape as its corresponding input and contains
            the feature's contribution to the prediction relative to the baseline.
            Positive values indicate features that increased the prediction,
            negative values indicate features that decreased it.

        Example:
            >>> shap_values = explainer.attribute(
            ...     x_continuous=torch.tensor([[1.0, 2.0, 3.0]]),
            ...     x_categorical=torch.tensor([[0, 1, 2]]),
            ...     target_class_idx=1
            ... )
            >>> print(shap_values['x_continuous'])  # Shape: (1, 3)
        """
        # Set random seed for reproducibility if specified
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        device = next(self.model.parameters()).device

        # Extract and prepare inputs
        feature_inputs: Dict[str, torch.Tensor] = {}
        time_info: Dict[str, torch.Tensor] = {}
        label_data: Dict[str, torch.Tensor] = {}

        for key in self.model.feature_keys:
            if key not in data:
                continue
            value = data[key]
            
            # Handle (time, value) tuples for temporal data
            if isinstance(value, tuple):
                time_tensor, feature_tensor = value
                if time_tensor is not None:
                    time_info[key] = time_tensor.to(device)
                value = feature_tensor

            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            feature_inputs[key] = value.to(device)

        # Store label data
        for key in self.model.label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.as_tensor(label_val)
                label_data[key] = label_val.to(device)

        # Fix target class for multiclass if not provided to avoid class flipping
        if target_class_idx is None and self._is_multiclass():
            base_logits = self._compute_base_logits(
                feature_inputs,
                time_info=time_info,
                label_data=label_data,
            )
            target_class_idx = base_logits.argmax(dim=-1)

        # Generate or validate background samples
        if baseline is None:
            background = self._generate_background_samples(feature_inputs)
        else:
            background = {k: v.to(device) for k, v in baseline.items()}

        # Compute SHAP values
        if self.use_embeddings:
            return self._shap_embeddings(
                feature_inputs,
                background=background,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
        else:
            return self._shap_continuous(
                feature_inputs,
                background=background,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )

    # ------------------------------------------------------------------
    # Embedding-based SHAP (discrete features)
    # ------------------------------------------------------------------
    def _shap_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
        background: Dict[str, torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute SHAP values for discrete inputs in embedding space.

        Args:
            inputs: Dictionary of input tensors.
            background: Dictionary of background samples.
            target_class_idx: Target class index for attribution.
            time_info: Temporal information for time-series models.
            label_data: Label information for supervised models.

        Returns:
            Dictionary of SHAP values mapped back to input shapes.
        """
        # Embed inputs and background
        input_embs = self.model.embedding_model(inputs)
        background_embs = self.model.embedding_model(background)

        # Store original input shapes for mapping back
        input_shapes = {key: val.shape for key, val in inputs.items()}

        # Compute SHAP values for each feature
        shap_values = {}
        for key in inputs:
            n_features = self._determine_n_features(key, inputs, input_embs)
            
            shap_matrix = self._compute_kernel_shap(
                key=key,
                input_emb=input_embs,
                background_emb=background_embs,
                n_features=n_features,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
            
            shap_values[key] = shap_matrix

        # Map embedding-space attributions back to input shapes
        return self._map_to_input_shapes(shap_values, input_shapes)

    # ------------------------------------------------------------------
    # Continuous SHAP (for tensor inputs)
    # ------------------------------------------------------------------
    def _shap_continuous(
        self,
        inputs: Dict[str, torch.Tensor],
        background: Dict[str, torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute SHAP values for continuous tensor inputs.

        Args:
            inputs: Dictionary of input tensors.
            background: Dictionary of background samples.
            target_class_idx: Target class index for attribution.
            time_info: Temporal information for time-series models.
            label_data: Label information for supervised models.

        Returns:
            Dictionary of SHAP values with same shapes as inputs.
        """
        shap_values = {}
        
        for key in inputs:
            n_features = self._determine_n_features(key, inputs, inputs)
            
            shap_matrix = self._compute_kernel_shap(
                key=key,
                input_emb=inputs,
                background_emb=background,
                n_features=n_features,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
            
            shap_values[key] = shap_matrix

        return shap_values

    # ------------------------------------------------------------------
    # Core Kernel SHAP computation
    # ------------------------------------------------------------------
    def _compute_kernel_shap(
        self,
        key: str,
        input_emb: Dict[str, torch.Tensor],
        background_emb: Dict[str, torch.Tensor],
        n_features: int,
        target_class_idx: Optional[int],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute SHAP values using the Kernel SHAP approximation method.

        This implements the Kernel SHAP algorithm that approximates Shapley values 
        through a weighted least squares regression. The key steps are:

        1. Feature Coalitions: Generate random subsets of features
        2. Model Evaluation: Evaluate mixed samples (background + coalition)
        3. Weighted Least Squares: Solve for SHAP values using kernel weights

        Args:
            key: Feature key being explained.
            input_emb: Dictionary of input embeddings/tensors.
            background_emb: Dictionary of background embeddings/tensors.
            n_features: Number of features to explain.
            target_class_idx: Target class index for multi-class models.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            torch.Tensor: SHAP values with shape (batch_size, n_features).
        """
        device = input_emb[key].device
        batch_size = input_emb[key].shape[0] if input_emb[key].dim() >= 2 else 1
        n_coalitions = min(2 ** n_features, self.max_coalitions)

        # Storage for coalition sampling
        coalition_vectors = []
        coalition_weights = []
        coalition_preds = []

        # Add edge case coalitions explicitly (empty and full)
        # These are crucial for the local accuracy property of SHAP
        edge_coalitions = [
            torch.zeros(n_features, device=device),  # Empty coalition (baseline)
            torch.ones(n_features, device=device),   # Full coalition (actual input)
        ]
        
        for coalition in edge_coalitions:
            per_input_preds = []
            for b_idx in range(batch_size):
                mixed_emb = self._create_mixed_sample(
                    key, coalition, input_emb, background_emb, b_idx
                )
                
                pred = self._evaluate_coalition(
                    key, mixed_emb, background_emb, 
                    target_class_idx, time_info, label_data
                )
                per_input_preds.append(pred)

            coalition_vectors.append(coalition.float())
            coalition_preds.append(torch.stack(per_input_preds, dim=0))
            coalition_weights.append(
                self._compute_kernel_weight(coalition.sum().item(), n_features)
            )

        # Sample remaining coalitions randomly (excluding edge cases already added)
        n_random_coalitions = max(0, n_coalitions - 2)
        for _ in range(n_random_coalitions):
            coalition = torch.randint(2, (n_features,), device=device)
            
            # Evaluate model for each input sample with this coalition
            per_input_preds = []
            for b_idx in range(batch_size):
                mixed_emb = self._create_mixed_sample(
                    key, coalition, input_emb, background_emb, b_idx
                )
                
                pred = self._evaluate_coalition(
                    key, mixed_emb, background_emb, 
                    target_class_idx, time_info, label_data
                )
                per_input_preds.append(pred)

            # Store coalition information
            coalition_vectors.append(coalition.float())
            coalition_preds.append(torch.stack(per_input_preds, dim=0))
            coalition_weights.append(
                self._compute_kernel_weight(coalition.sum().item(), n_features)
            )

        # Solve weighted least squares
        return self._solve_weighted_least_squares(
            coalition_vectors, coalition_preds, coalition_weights, device
        )

    def _create_mixed_sample(
        self,
        key: str,
        coalition: torch.Tensor,
        input_emb: Dict[str, torch.Tensor],
        background_emb: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Create a mixed sample by combining background and input based on coalition.

        Args:
            key: Feature key.
            coalition: Binary vector indicating which features to use from input.
            input_emb: Input embeddings.
            background_emb: Background embeddings.
            batch_idx: Index of the sample in the batch.

        Returns:
            Mixed sample tensor.
        """
        mixed = background_emb[key].clone()
        
        for i, use_input in enumerate(coalition):
            if not use_input:
                continue
                
            # Handle various embedding shapes
            dim = input_emb[key].dim()
            if dim == 4:  # (batch, seq_len, inner_len, emb)
                mixed[:, i, :, :] = input_emb[key][batch_idx, i, :, :]
            elif dim == 3:  # (batch, seq_len, emb)
                mixed[:, i, :] = input_emb[key][batch_idx, i, :]
            else:  # 2D or other
                mixed[:, i] = input_emb[key][batch_idx, i]

        return mixed

    def _evaluate_coalition(
        self,
        key: str,
        mixed_emb: torch.Tensor,
        background_emb: Dict[str, torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Evaluate model prediction for a coalition.

        Args:
            key: Feature key being explained.
            mixed_emb: Mixed embedding tensor.
            background_emb: Background embeddings for other features.
            target_class_idx: Target class index.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Scalar prediction averaged over background samples.
        """
        if self.use_embeddings:
            logits = self._forward_from_embeddings(
                key, mixed_emb, background_emb, time_info, label_data
            )
        else:
            logits = self._forward_from_inputs(
                key, mixed_emb, background_emb, time_info, label_data
            )

        # Extract target class prediction
        pred_vec = self._extract_target_prediction(logits, target_class_idx)
        
        # Average over background samples
        return pred_vec.detach().mean()

    def _forward_from_embeddings(
        self,
        key: str,
        mixed_emb: torch.Tensor,
        background_emb: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass using embeddings.

        Args:
            key: Feature key being explained.
            mixed_emb: Mixed embedding tensor.
            background_emb: Background embeddings.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Model logits.
        """
        # Build feature embeddings dictionary
        feature_embeddings = {key: mixed_emb}
        for fk in self.model.feature_keys:
            if fk not in feature_embeddings:
                if fk in background_emb:
                    feature_embeddings[fk] = background_emb[fk].clone()
                else:
                    # Zero fallback
                    ref_tensor = next(iter(feature_embeddings.values()))
                    feature_embeddings[fk] = torch.zeros_like(ref_tensor)

        # Prepare time info matching background batch size
        time_info_bg = self._prepare_time_info(
            time_info, feature_embeddings, mixed_emb.shape[0]
        )

        # Forward pass
        with torch.no_grad():
            # Create kwargs with proper label key
            forward_kwargs = {
                "time_info": time_info_bg,
            }
            # Add labels shaped for the model's loss (e.g., 1D long for cross entropy)
            forward_kwargs.update(
                self._build_label_kwargs(
                    batch_size=mixed_emb.shape[0], label_data=label_data
                )
            )
            
            model_output = self.model.forward_from_embedding(
                feature_embeddings,
                **forward_kwargs
            )

        return self._extract_logits(model_output)

    def _forward_from_inputs(
        self,
        key: str,
        mixed_inputs: torch.Tensor,
        background_inputs: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass using raw inputs (continuous features).

        Args:
            key: Feature key being explained.
            mixed_inputs: Mixed input tensor.
            background_inputs: Background inputs.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Model logits.
        """
        model_inputs = {}
        for fk in self.model.feature_keys:
            if fk == key:
                model_inputs[fk] = mixed_inputs
            elif fk in background_inputs:
                model_inputs[fk] = background_inputs[fk].clone()
            else:
                model_inputs[fk] = torch.zeros_like(mixed_inputs)

        # Add labels shaped for the model's loss (e.g., 1D long for cross entropy)
        model_inputs.update(
            self._build_label_kwargs(
                batch_size=mixed_inputs.shape[0], label_data=label_data
            )
        )

        output = self.model(**model_inputs)
        return self._extract_logits(output)

    def _prepare_time_info(
        self,
        time_info: Optional[Dict[str, torch.Tensor]],
        feature_embeddings: Dict[str, torch.Tensor],
        n_background: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare time information to match background batch size.

        Args:
            time_info: Original time information.
            feature_embeddings: Feature embeddings to match sequence lengths.
            n_background: Number of background samples.

        Returns:
            Adjusted time information or None.
        """
        if time_info is None:
            return None

        time_info_bg = {}
        for fk, emb in feature_embeddings.items():
            if fk not in time_info or time_info[fk] is None:
                continue

            seq_len = emb.shape[1]
            t_orig = time_info[fk].to(self.model.device)

            # Normalize to 1D sequence
            t_vec = self._normalize_time_vector(t_orig)

            # Adjust length to match embedding sequence length
            t_adj = self._adjust_time_length(t_vec, seq_len)

            # Expand to background batch size
            time_info_bg[fk] = t_adj.unsqueeze(0).expand(n_background, -1)

        return time_info_bg if time_info_bg else None

    def _compute_base_logits(
        self,
        inputs: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Run a single forward on the original inputs to select target class."""
        batch_size = next(iter(inputs.values())).shape[0]

        with torch.no_grad():
            if self.use_embeddings:
                input_embs = self.model.embedding_model(inputs)
                time_info_adj = self._prepare_time_info(
                    time_info, input_embs, batch_size
                )
                forward_kwargs = {"time_info": time_info_adj}
                forward_kwargs.update(
                    self._build_label_kwargs(
                        batch_size=batch_size, label_data=label_data
                    )
                )
                output = self.model.forward_from_embedding(
                    input_embs, **forward_kwargs
                )
            else:
                model_inputs = {
                    fk: inputs[fk] for fk in self.model.feature_keys if fk in inputs
                }
                model_inputs.update(
                    self._build_label_kwargs(
                        batch_size=batch_size, label_data=label_data
                    )
                )
                output = self.model(**model_inputs)

        return self._extract_logits(output)

    # ------------------------------------------------------------------
    # Label handling helpers
    # ------------------------------------------------------------------
    def _build_label_kwargs(
        self,
        batch_size: int,
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Provide dummy labels shaped for the model loss.

        StageNet computes loss inside `forward_from_embedding`; we feed zeros
        with the correct shape/dtype to satisfy the loss signature without
        influencing logits used for attribution.
        """
        if not getattr(self.model, "label_keys", None):
            return {}

        loss_name = ""
        try:
            loss_fn = self.model.get_loss_function()
            loss_name = getattr(loss_fn, "__name__", "").lower()
        except Exception:
            loss_name = ""

        is_cross_entropy = "cross_entropy" in loss_name
        if is_cross_entropy:
            dummy = torch.zeros(
                (batch_size,), device=self.model.device, dtype=torch.long
            )
        else:
            out_size = 1
            try:
                out_size = self.model.get_output_size()
            except Exception:
                pass
            dummy = torch.zeros(
                (batch_size, out_size), device=self.model.device, dtype=torch.float32
            )

        return {label_key: dummy for label_key in self.model.label_keys}

    def _is_multiclass(self) -> bool:
        """Detect multiclass mode via loss function signature."""
        try:
            loss_fn = self.model.get_loss_function()
            loss_name = getattr(loss_fn, "__name__", "").lower()
            return "cross_entropy" in loss_name
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Weighted least squares solver
    # ------------------------------------------------------------------
    def _solve_weighted_least_squares(
        self,
        coalition_vectors: list,
        coalition_preds: list,
        coalition_weights: list,
        device: torch.device,
    ) -> torch.Tensor:
        """Solve weighted least squares to estimate SHAP values.

        Uses Tikhonov regularization for numerical stability.

        Args:
            coalition_vectors: List of coalition binary vectors.
            coalition_preds: List of prediction tensors per coalition.
            coalition_weights: List of kernel weights per coalition.
            device: Device for computation.

        Returns:
            SHAP values with shape (batch_size, n_features).
        """
        # Stack collected data
        X = torch.stack(coalition_vectors, dim=0).to(device)  # (n_coalitions, n_features)
        Y = torch.stack(coalition_preds, dim=0).to(device)    # (n_coalitions, batch_size)
        W = torch.stack(coalition_weights, dim=0).to(device)  # (n_coalitions,)

        # Apply sqrt weights for weighted least squares
        sqrtW = torch.sqrt(W).unsqueeze(1)  # (n_coalitions, 1)
        Xw = sqrtW * X  # (n_coalitions, n_features)
        Yw = sqrtW * Y  # (n_coalitions, batch_size)

        # Add Tikhonov regularization
        n_features = X.shape[1]
        reg_scale = torch.sqrt(torch.tensor(self.regularization, device=device))
        reg_mat = reg_scale * torch.eye(n_features, device=device)

        # Augment for regularized least squares: [Xw; reg_mat] phi = [Yw; 0]
        Xw_aug = torch.cat([Xw, reg_mat], dim=0)
        Yw_aug = torch.cat(
            [Yw, torch.zeros((n_features, Y.shape[1]), device=device)], dim=0
        )

        # Solve using torch.linalg.lstsq
        res = torch.linalg.lstsq(Xw_aug, Yw_aug)
        phi_sol = getattr(res, 'solution', res[0])  # (n_features, batch_size)

        # Return per-sample attributions: (batch_size, n_features)
        return phi_sol.transpose(0, 1)

    # ------------------------------------------------------------------
    # Background sample generation
    # ------------------------------------------------------------------
    def _generate_background_samples(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate background samples for SHAP computation.

        Creates reference samples to establish baseline predictions. The sampling 
        strategy adapts to the feature type:
        - Discrete features: Sample uniformly from observed unique values
        - Continuous features: Sample uniformly from the range [min, max]

        Args:
            inputs: Dictionary mapping feature names to input tensors.

        Returns:
            Dictionary mapping feature names to background sample tensors.
        """
        background_samples = {}

        for key, x in inputs.items():
            if x.dtype in [torch.int64, torch.int32, torch.long]:
                # Discrete features: sample from unique values
                unique_vals = torch.unique(x)
                samples = unique_vals[
                    torch.randint(
                        len(unique_vals),
                        (self.n_background_samples,) + x.shape[1:],
                    )
                ]
            else:
                # Continuous features: sample from range
                min_val = torch.min(x)
                max_val = torch.max(x)
                samples = torch.rand(
                    (self.n_background_samples,) + x.shape[1:], device=x.device
                ) * (max_val - min_val) + min_val

            background_samples[key] = samples.to(x.device)

        return background_samples

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_n_features(
        key: str,
        inputs: Dict[str, torch.Tensor],
        embeddings: Dict[str, torch.Tensor],
    ) -> int:
        """Determine the number of features to explain for a given key.

        Args:
            key: Feature key.
            inputs: Original input tensors.
            embeddings: Embedding tensors.

        Returns:
            Number of features (typically sequence length or feature dimension).
        """
        # Prefer original input shape
        if key in inputs and inputs[key].dim() >= 2:
            return inputs[key].shape[1]

        # Fallback to embedding shape
        emb = embeddings[key]
        if emb.dim() >= 2:
            return emb.shape[1]
        return emb.shape[-1]

    @staticmethod
    
    def _compute_kernel_weight(coalition_size: int, n_features: int) -> torch.Tensor:
        """Compute Kernel SHAP weight for a coalition.

        Correct formula from Lundberg & Lee (2017):
            weight = (M - 1) / (binom(M, |z|) * |z| * (M - |z|))

        Args:
            coalition_size: Number of present features (|z|).
            n_features: Total number of features (M).

        Returns:
            Scalar tensor with the kernel weight.
        """
        M = n_features
        z = coalition_size

        # Edge cases (empty or full coalition)
        if z == 0 or z == M:
            # Assign infinite weight; we approximate with a large number.
            return torch.tensor(1000, dtype=torch.float32)

        # Compute binomial coefficient C(M, z)
        comb_val = math.comb(M, z)

        # SHAP kernel weight
        weight = (M - 1) / (comb_val * z * (M - z))

        return torch.tensor(weight, dtype=torch.float32)

    @staticmethod
    def _extract_logits(model_output) -> torch.Tensor:
        """Extract logits from model output.

        Args:
            model_output: Model output (dict or tensor).

        Returns:
            Logit tensor.
        """
        if isinstance(model_output, dict) and "logit" in model_output:
            return model_output["logit"]
        return model_output

    @staticmethod
    def _extract_target_prediction(
        logits: torch.Tensor, target_class_idx: Optional[int]
    ) -> torch.Tensor:
        """Extract target class prediction from logits.

        Args:
            logits: Model logits.
            target_class_idx: Target class index (None for max prediction).

        Returns:
            Target prediction tensor.
        """
        if target_class_idx is None:
            return torch.max(logits, dim=-1)[0]

        if logits.dim() > 1 and logits.shape[-1] > 1:
            return logits[..., target_class_idx]
        else:
            # Binary classification with single logit
            sig = torch.sigmoid(logits.squeeze(-1))
            return sig if target_class_idx == 1 else 1.0 - sig

    @staticmethod
    def _normalize_time_vector(time_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize time tensor to 1D vector.

        Args:
            time_tensor: Time information tensor.

        Returns:
            1D time vector.
        """
        if time_tensor.dim() == 2 and time_tensor.shape[0] > 0:
            return time_tensor[0].detach()
        elif time_tensor.dim() == 1:
            return time_tensor.detach()
        else:
            return time_tensor.reshape(-1).detach()

    @staticmethod
    def _adjust_time_length(time_vec: torch.Tensor, target_len: int) -> torch.Tensor:
        """Adjust time vector length to match target length.

        Args:
            time_vec: 1D time vector.
            target_len: Target sequence length.

        Returns:
            Adjusted time vector.
        """
        current_len = time_vec.numel()

        if current_len == target_len:
            return time_vec
        elif current_len < target_len:
            # Pad by repeating last value
            if current_len == 0:
                return torch.zeros(target_len, device=time_vec.device)
            pad_len = target_len - current_len
            pad = time_vec[-1].unsqueeze(0).repeat(pad_len)
            return torch.cat([time_vec, pad], dim=0)
        else:
            # Truncate
            return time_vec[:target_len]

    @staticmethod
    def _map_to_input_shapes(
        shap_values: Dict[str, torch.Tensor],
        input_shapes: Dict[str, tuple],
    ) -> Dict[str, torch.Tensor]:
        """Map SHAP values from embedding space back to input shapes.

        For embedding-based attributions, this projects the attribution scores
        from embedding dimensions back to the original input tensor shapes.

        Args:
            shap_values: Dictionary of SHAP values in embedding space.
            input_shapes: Dictionary of original input shapes.

        Returns:
            Dictionary of SHAP values reshaped to match inputs.
        """
        mapped = {}
        for key, values in shap_values.items():
            if key not in input_shapes:
                mapped[key] = values
                continue

            orig_shape = input_shapes[key]
            
            # If shapes already match, no adjustment needed
            if values.shape == orig_shape:
                mapped[key] = values
                continue

            # Reshape to match original input
            reshaped = values
            while len(reshaped.shape) < len(orig_shape):
                reshaped = reshaped.unsqueeze(-1)
            
            if reshaped.shape != orig_shape:
                reshaped = reshaped.expand(orig_shape)
            
            mapped[key] = reshaped

        return mapped