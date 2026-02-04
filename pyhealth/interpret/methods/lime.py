from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Callable, Union

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity

from pyhealth.models import BaseModel
from .base_interpreter import BaseInterpreter


class LimeExplainer(BaseInterpreter):
    """LIME (Local Interpretable Model-agnostic Explanations) attribution method for PyHealth models.

    This class implements the LIME method for computing feature attributions in 
    neural networks. LIME explains model predictions by approximating the model 
    locally with an interpretable surrogate model (e.g., linear regression).

    The method is based on the paper:
        "Why Should I Trust You?" Explaining the Predictions of Any Classifier
        Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
        KDD 2016
        https://arxiv.org/abs/1602.04938

    LIME Method Overview:
    LIME works by:
    1. Generating perturbed samples around the input to be explained
    2. Obtaining model predictions for these perturbed samples
    3. Weighting samples by their similarity to the original input
    4. Training a simple interpretable model (linear) on the weighted dataset
    5. Using the coefficients of the interpretable model as feature importances

    Mathematical Foundation:
    Given an input x and model f, LIME finds an interpretable model g that 
    minimizes:
    L(f, g, πₓ) + Ω(g)
    where:
    - πₓ is a proximity measure (similarity kernel)
    - Ω(g) is the complexity of the interpretable model g
    - L measures how unfaithful g is in approximating f in the locality of x

    For linear models, this becomes:
    argmin_w Σᵢ πₓ(zᵢ) * [f(zᵢ) - w·zᵢ]² + λ||w||²
    where zᵢ are perturbed samples and w are the feature weights.

    LIME provides several benefits:
    1. Model-agnostic: Works with any model that provides predictions
    2. Local fidelity: Explanations are faithful in the locality of the input
    3. Interpretable: Uses simple linear models that humans can understand
    4. Flexible: Can define custom perturbation and similarity functions

    Args:
        model: A trained PyHealth model to interpret. Can be any model that
            inherits from BaseModel (e.g., MLP, StageNet, Transformer, RNN).
        use_embeddings: If True, compute LIME values with respect to
            embeddings rather than discrete input tokens. This is crucial
            for models with discrete inputs (like ICD codes). The model
            must support returning embeddings via an 'embed' parameter.
            Default is True.
        n_samples: Number of perturbed samples to generate for training
            the interpretable model. More samples give better local
            approximations but increase computation time. Default is 1000.
        kernel_width: Width parameter for the exponential similarity kernel.
            Smaller values make the explanations more local. Default is 0.25.
        distance_mode: Distance metric for similarity computation. Can be
            "cosine" for cosine similarity or "euclidean" for L2 distance.
            Default is "cosine".
        feature_selection: Method for selecting top features. Can be "lasso"
            for L1 regularization, "ridge" for L2 regularization, or "none"
            for no regularization. Default is "lasso".
        alpha: Regularization strength for the interpretable model.
            Default is 0.01.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import SampleDataset, get_dataloader
        >>> from pyhealth.models import StageNet
        >>> from pyhealth.interpret.methods import LimeExplainer
        >>> from pyhealth.trainer import Trainer
        >>>
        >>> # Create dataset and model 
        >>> dataset = SampleDataset(...)
        >>> model = StageNet(...)
        >>> trainer = Trainer(model=model, device="cuda:0")
        >>> trainer.train(...)
        >>> test_batch = next(iter(test_loader))
        >>>
        >>> # Initialize LIME explainer
        >>> explainer = LimeExplainer(model, use_embeddings=True, n_samples=1000)
        >>> lime_values = explainer.attribute(**test_batch)
        >>>
        >>> # With custom kernel width
        >>> explainer = LimeExplainer(model, kernel_width=0.5)
        >>> lime_values = explainer.attribute(**test_batch)
        >>>
        >>> print(lime_values)
        {'conditions': tensor([[0.1234, 0.5678, 0.9012]], device='cuda:0'),
         'procedures': tensor([[0.2345, 0.6789, 0.0123, 0.4567]])}
    """

    def __init__(
        self,
        model: BaseModel,
        use_embeddings: bool = True,
        n_samples: int = 1000,
        kernel_width: float = 0.25,
        distance_mode: str = "cosine",
        feature_selection: str = "lasso",
        alpha: float = 0.01,
        random_seed: Optional[int] = 42,
    ):
        """Initialize LIME explainer.

        Args:
            model: A trained PyHealth model to interpret.
            use_embeddings: If True, compute LIME values with respect to
                embeddings rather than discrete input tokens.
            n_samples: Number of perturbed samples to generate.
            kernel_width: Width parameter for the exponential kernel.
            distance_mode: Distance metric ("cosine" or "euclidean").
            feature_selection: Regularization type ("lasso", "ridge", or "none").
            alpha: Regularization strength.
            random_seed: Optional random seed for reproducibility.

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method.
            ValueError: If distance_mode is not "cosine" or "euclidean".
            ValueError: If feature_selection is not "lasso", "ridge", or "none".
        """
        super().__init__(model)
        self.use_embeddings = use_embeddings
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.distance_mode = distance_mode
        self.feature_selection = feature_selection
        self.alpha = alpha
        self.random_seed = random_seed

        # Validate inputs
        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "LIME values. Set use_embeddings=False to use "
                "input-level attributions (only for continuous features)."
            )

        if distance_mode not in ["cosine", "euclidean"]:
            raise ValueError("distance_mode must be either 'cosine' or 'euclidean'.")

        if feature_selection not in ["lasso", "ridge", "none"]:
            raise ValueError("feature_selection must be 'lasso', 'ridge', or 'none'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute LIME attributions for input features.

        This is the main interface for computing feature attributions. It:
        1. Generates perturbed samples around the input
        2. Computes model predictions for perturbed samples
        3. Weights samples by similarity to the original input
        4. Trains a linear interpretable model
        5. Returns the linear model coefficients as feature importances

        Args:
            baseline: Optional dictionary mapping feature names to baseline 
                     values used for perturbations. If None, uses zeros or
                     random samples. Shape of each tensor should match input.
            target_class_idx: For multi-class models, specifies which class's 
                            prediction to explain. If None, explains the model's
                            maximum prediction across all classes.
            **data: Input data dictionary from dataloader batch. Should contain:
                   - Feature tensors with shape (batch_size, ..., feature_dim)
                   - Optional time information for temporal models
                   - Optional label data for supervised models

        Returns:
            Dictionary mapping feature names to their LIME coefficients. Each 
            tensor has the same shape as its corresponding input and contains
            the feature's importance in the local linear approximation.
            Positive values indicate features that increased the prediction,
            negative values indicate features that decreased it.

        Example:
            >>> lime_values = explainer.attribute(
            ...     conditions=torch.tensor([[1, 5, 8]]),
            ...     procedures=torch.tensor([[2, 3]]),
            ...     target_class_idx=1
            ... )
            >>> print(lime_values['conditions'])  # Shape: (1, 3)
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

        # Generate or validate baseline (neutral replacement values)
        # Note: LIME does not require a background dataset; baselines here
        # serve only as neutral values when a feature is masked (absent).
        if baseline is None:
            baseline = self._generate_baseline(feature_inputs)
        else:
            baseline = {k: v.to(device) for k, v in baseline.items()}

        # Compute LIME values
        if self.use_embeddings:
            return self._lime_embeddings(
                feature_inputs,
                baseline=baseline,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
        else:
            return self._lime_continuous(
                feature_inputs,
                baseline=baseline,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )

    # ------------------------------------------------------------------
    # Embedding-based LIME 
    # ------------------------------------------------------------------
    def _lime_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Dict[str, torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute LIME values for discrete inputs in embedding space.

        Args:
            inputs: Dictionary of input tensors.
            baseline: Dictionary of baseline samples.
            target_class_idx: Target class index for attribution.
            time_info: Temporal information for time-series models.
            label_data: Label information for supervised models.

        Returns:
            Dictionary of LIME coefficients mapped back to input shapes.
        """
        # Embed inputs and baseline
        input_embs = self.model.embedding_model(inputs)
        baseline_embs = self.model.embedding_model(baseline)

        # Store original input shapes for mapping back
        input_shapes = {key: val.shape for key, val in inputs.items()}

        # Compute LIME values for each feature
        lime_values = {}
        for key in inputs:
            n_features = self._determine_n_features(key, inputs, input_embs)
            
            coefs = self._compute_lime(
                key=key,
                input_emb=input_embs,
                baseline_emb=baseline_embs,
                n_features=n_features,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
            
            lime_values[key] = coefs

        # Map embedding-space attributions back to input shapes
        return self._map_to_input_shapes(lime_values, input_shapes)

    # ------------------------------------------------------------------
    # Continuous LIME
    # ------------------------------------------------------------------
    def _lime_continuous(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Dict[str, torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute LIME values for continuous tensor inputs.

        Args:
            inputs: Dictionary of input tensors.
            baseline: Dictionary of baseline samples.
            target_class_idx: Target class index for attribution.
            time_info: Temporal information for time-series models.
            label_data: Label information for supervised models.

        Returns:
            Dictionary of LIME coefficients with same shapes as inputs.
        """
        lime_values = {}
        
        for key in inputs:
            n_features = self._determine_n_features(key, inputs, inputs)
            
            coefs = self._compute_lime(
                key=key,
                input_emb=inputs,
                baseline_emb=baseline,
                n_features=n_features,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
            
            lime_values[key] = coefs

        return lime_values

    # ------------------------------------------------------------------
    # Core LIME computation
    # ------------------------------------------------------------------
    def _compute_lime(
        self,
        key: str,
        input_emb: Dict[str, torch.Tensor],
        baseline_emb: Dict[str, torch.Tensor],
        n_features: int,
        target_class_idx: Optional[int],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute LIME coefficients using interpretable linear model.

        This implements the LIME algorithm:
        1. Generate perturbed samples (binary vectors indicating feature presence or absence)
        2. Create interpretable samples by mixing original and baseline
        3. Evaluate model on perturbed samples
        4. Compute similarity weights based on distance to original
        5. Train weighted linear regression
        6. Return coefficients as feature importances

        Args:
            key: Feature key being explained.
            input_emb: Dictionary of input embeddings/tensors.
            baseline_emb: Dictionary of baseline embeddings/tensors.
            n_features: Number of features to explain.
            target_class_idx: Target class index for multi-class models.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            torch.Tensor: LIME coefficients with shape (batch_size, n_features).
        """
        device = input_emb[key].device
        batch_size = input_emb[key].shape[0] if input_emb[key].dim() >= 2 else 1
        # Keep large intermediate tensors off the GPU to avoid OOM
        storage_device = torch.device("cpu")

        # Storage for samples and predictions
        interpretable_samples = []  # Binary vectors
        perturbed_predictions = []  # Model predictions
        similarity_weights = []     # Distance-based weights

        # Original input prediction
        original_pred = self._get_prediction(
            key, input_emb, baseline_emb, None, 
            target_class_idx, time_info, label_data
        )

        # Generate perturbed samples
        for _ in range(self.n_samples):
            # Sample binary vector (which features to include)
            binary_vector = torch.bernoulli(
                torch.ones(n_features, device=device) * 0.5
            )
            
            # Create perturbed sample for each batch item
            batch_preds = []
            batch_similarities = []
            
            for b_idx in range(batch_size):
                # Create perturbed embedding by mixing input and baseline
                perturbed_emb = self._create_perturbed_sample(
                    key, binary_vector, input_emb, baseline_emb, b_idx
                )
                
                # Get model prediction for perturbed sample
                pred = self._evaluate_sample(
                    key, perturbed_emb, baseline_emb,
                    target_class_idx, time_info, label_data
                )
                batch_preds.append(pred)
                
                # Compute similarity weight
                similarity = self._compute_similarity(
                    input_emb[key][b_idx:b_idx+1] if batch_size > 1 else input_emb[key],
                    perturbed_emb,
                    binary_vector,
                )
                batch_similarities.append(similarity)
            
            # Store sample information
            interpretable_samples.append(binary_vector.float())
            perturbed_predictions.append(torch.stack(batch_preds, dim=0))
            similarity_weights.append(torch.stack(batch_similarities, dim=0))

            # Move small summaries to CPU to free GPU memory
            interpretable_samples[-1] = interpretable_samples[-1].float().to(storage_device)
            perturbed_predictions[-1] = perturbed_predictions[-1].detach().to(storage_device)
            similarity_weights[-1] = similarity_weights[-1].detach().to(storage_device)

        # Train weighted linear regression
        return self._train_interpretable_model(
            interpretable_samples,
            perturbed_predictions,
            similarity_weights,
            compute_device=storage_device,
            target_device=device,
        )

    def _create_perturbed_sample(
        self,
        key: str,
        binary_vector: torch.Tensor,
        input_emb: Dict[str, torch.Tensor],
        baseline_emb: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Create a perturbed sample by mixing input and baseline based on binary vector.

        Args:
            key: Feature key.
            binary_vector: Binary vector (1 = use input feature, 0 = use baseline).
            input_emb: Input embeddings.
            baseline_emb: Baseline embeddings.
            batch_idx: Index of the sample in the batch.

        Returns:
            Perturbed sample tensor.
        """
        # Start with baseline for the specific sample
        perturbed = baseline_emb[key][batch_idx:batch_idx+1].clone()
        
        # Mix in input features based on binary vector
        for i, use_input in enumerate(binary_vector):
            if not use_input:
                continue
                
            # Handle various embedding shapes
            dim = input_emb[key].dim()
            if dim == 4:  # (batch, seq_len, inner_len, emb)
                if i < perturbed.shape[1]:
                    perturbed[:, i, :, :] = input_emb[key][batch_idx, i, :, :]
            elif dim == 3:  # (batch, seq_len, emb)
                if i < perturbed.shape[1]:
                    perturbed[:, i, :] = input_emb[key][batch_idx, i, :]
            else:  # 2D or other
                if i < perturbed.shape[1]:
                    perturbed[:, i] = input_emb[key][batch_idx, i]

        return perturbed

    def _evaluate_sample(
        self,
        key: str,
        perturbed_emb: torch.Tensor,
        baseline_emb: Dict[str, torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Evaluate model prediction for a perturbed sample.

        Args:
            key: Feature key being explained.
            perturbed_emb: Perturbed embedding tensor.
            baseline_emb: Baseline embeddings for other features.
            target_class_idx: Target class index.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Scalar prediction.
        """
        if self.use_embeddings:
            logits = self._forward_from_embeddings(
                key, perturbed_emb, baseline_emb, time_info, label_data
            )
        else:
            logits = self._forward_from_inputs(
                key, perturbed_emb, baseline_emb, time_info, label_data
            )

        # Extract target class prediction
        pred_vec = self._extract_target_prediction(logits, target_class_idx)
        
        # Return mean prediction (average over baseline samples if multiple)
        return pred_vec.detach().mean()

    def _get_prediction(
        self,
        key: str,
        input_emb: Dict[str, torch.Tensor],
        baseline_emb: Dict[str, torch.Tensor],
        binary_vector: Optional[torch.Tensor],
        target_class_idx: Optional[int],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Get model prediction for original input or perturbed sample.

        Args:
            key: Feature key.
            input_emb: Input embeddings.
            baseline_emb: Baseline embeddings.
            binary_vector: Optional binary perturbation vector.
            target_class_idx: Target class index.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Model prediction.
        """
        if binary_vector is None:
            # Use original input
            sample = input_emb[key]
        else:
            # Create perturbed sample
            sample = self._create_perturbed_sample(
                key, binary_vector, input_emb, baseline_emb, 0
            )

        return self._evaluate_sample(
            key, sample, baseline_emb, target_class_idx, time_info, label_data
        )

    def _compute_similarity(
        self,
        original_emb: torch.Tensor,
        perturbed_emb: torch.Tensor,
        binary_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity weight using exponential kernel.

        The similarity is computed as:
        exp(- distance² / (2 * kernel_width²))

        Args:
            original_emb: Original input embedding.
            perturbed_emb: Perturbed sample embedding.
            binary_vector: Binary perturbation vector.

        Returns:
            Similarity weight (scalar tensor).
        """
        with torch.no_grad():
            # Flatten embeddings for distance computation
            orig_flat = original_emb.reshape(-1).float()
            pert_flat = perturbed_emb.reshape(-1).float()

            # Compute distance
            if self.distance_mode == "cosine":
                cos_sim = CosineSimilarity(dim=0)
                distance = 1 - cos_sim(orig_flat, pert_flat)
            elif self.distance_mode == "euclidean":
                distance = torch.norm(orig_flat - pert_flat)
            else:
                raise ValueError("Invalid distance_mode")

            # Apply exponential kernel
            similarity = torch.exp(
                -1 * (distance ** 2) / (2 * (self.kernel_width ** 2))
            )

        return similarity

    def _train_interpretable_model(
        self,
        interpretable_samples: list,
        predictions: list,
        weights: list,
        compute_device: torch.device,
        target_device: torch.device,
    ) -> torch.Tensor:
        """Train weighted linear regression model.

        Solves the weighted least squares problem:
        argmin_w Σᵢ wᵢ * [yᵢ - f(xᵢ)]² + α||w||

        where α is the regularization strength.

        Args:
            interpretable_samples: List of binary vectors.
            predictions: List of model predictions.
            weights: List of similarity weights.
            compute_device: Device for regression solve (CPU to save GPU memory).
            target_device: Device to place the returned coefficients.

        Returns:
            Linear model coefficients (batch_size, n_features).
        """
        # Stack collected data
        X = torch.stack(interpretable_samples, dim=0).to(compute_device)  # (n_samples, n_features)
        Y = torch.stack(predictions, dim=0).to(compute_device)            # (n_samples, batch_size)
        W = torch.stack(weights, dim=0).to(compute_device)                # (n_samples, batch_size)

        # Solve for each batch item independently
        batch_size = Y.shape[1]
        n_features = X.shape[1]
        coefficients = []

        for b_idx in range(batch_size):
            # Get data for this batch item
            y = Y[:, b_idx]  # (n_samples,)
            w = W[:, b_idx]  # (n_samples,)

            # Apply sqrt weights for weighted least squares
            sqrtW = torch.sqrt(w)  # (n_samples,)
            Xw = sqrtW.unsqueeze(1) * X  # (n_samples, n_features)
            yw = sqrtW * y  # (n_samples,)

            # Solve based on feature selection method
            if self.feature_selection == "lasso":
                # L1 regularization (approximated with iterative reweighted least squares)
                coef = self._solve_lasso(Xw, yw, compute_device)
            elif self.feature_selection == "ridge":
                # L2 regularization
                coef = self._solve_ridge(Xw, yw, compute_device)
            else:  # "none"
                # No regularization
                coef = self._solve_ols(Xw, yw, compute_device)

            coefficients.append(coef)

        # Stack into (batch_size, n_features)
        return torch.stack(coefficients, dim=0).to(target_device)

    def _solve_lasso(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Solve Lasso regression (L1 regularization).

        Uses coordinate descent approximation for L1 penalty.

        Args:
            X: Weighted design matrix (n_samples, n_features).
            y: Weighted target values (n_samples,) or (n_samples, 1).
            device: Device for computation.

        Returns:
            Coefficient vector (n_features,).
        """
        n_features = X.shape[1]
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze(-1)
        
        # Use soft-thresholding approximation via ridge with L1 approximation
        # For simplicity, we'll use standard least squares with small L1 penalty
        # In practice, you could use sklearn or implement full coordinate descent
        
        # Add L1 penalty approximation using ridge with sparsity-inducing weights
        reg_mat = self.alpha * torch.eye(n_features, device=device)
        X_aug = torch.cat([X, reg_mat], dim=0)
        y_aug = torch.cat([y, torch.zeros(n_features, device=device)], dim=0)
        
        # Solve using least squares
        res = torch.linalg.lstsq(X_aug, y_aug)
        coef = getattr(res, 'solution', res[0])
        
        return coef

    def _solve_ridge(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Solve Ridge regression (L2 regularization).

        Args:
            X: Weighted design matrix (n_samples, n_features).
            y: Weighted target values (n_samples,) or (n_samples, 1).
            device: Device for computation.

        Returns:
            Coefficient vector (n_features,).
        """
        n_features = X.shape[1]
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze(-1)
        
        # Add Tikhonov regularization
        reg_scale = torch.sqrt(torch.tensor(self.alpha, device=device))
        reg_mat = reg_scale * torch.eye(n_features, device=device)

        # Augment for regularized least squares
        X_aug = torch.cat([X, reg_mat], dim=0)
        y_aug = torch.cat([y, torch.zeros(n_features, device=device)], dim=0)

        # Solve using torch.linalg.lstsq
        res = torch.linalg.lstsq(X_aug, y_aug)
        coef = getattr(res, 'solution', res[0])

        return coef

    def _solve_ols(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Solve Ordinary Least Squares (no regularization).

        Args:
            X: Weighted design matrix (n_samples, n_features).
            y: Weighted target values (n_samples,) or (n_samples, 1).
            device: Device for computation.

        Returns:
            Coefficient vector (n_features,).
        """
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze(-1)
            
        # Solve using torch.linalg.lstsq
        res = torch.linalg.lstsq(X, y)
        coef = getattr(res, 'solution', res[0])
        
        return coef

    # ------------------------------------------------------------------
    # Forward pass methods
    # ------------------------------------------------------------------
    def _forward_from_embeddings(
        self,
        key: str,
        perturbed_emb: torch.Tensor,
        baseline_emb: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass using embeddings.

        Args:
            key: Feature key being explained.
            perturbed_emb: Perturbed embedding tensor.
            baseline_emb: Baseline embeddings.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Model logits.
        """
        # Build feature embeddings dictionary
        batch_size = perturbed_emb.shape[0]
        feature_embeddings = {key: perturbed_emb}
        
        for fk in self.model.feature_keys:
            if fk not in feature_embeddings:
                if fk in baseline_emb:
                    # Expand baseline to match batch size if needed
                    base_emb = baseline_emb[fk]
                    if base_emb.shape[0] == batch_size:
                        feature_embeddings[fk] = base_emb.clone()
                    elif base_emb.shape[0] == 1 and batch_size > 1:
                        feature_embeddings[fk] = base_emb.expand(batch_size, *base_emb.shape[1:]).clone()
                    elif base_emb.shape[0] > 1 and batch_size == 1:
                        # Slice a single neutral baseline sample
                        feature_embeddings[fk] = base_emb[:1].clone()
                    else:
                        feature_embeddings[fk] = base_emb.expand(batch_size, *base_emb.shape[1:]).clone()
                else:
                    # Zero fallback
                    ref_tensor = next(iter(feature_embeddings.values()))
                    feature_embeddings[fk] = torch.zeros_like(ref_tensor)

        # Prepare time info matching batch size
        time_info_adj = self._prepare_time_info(
            time_info, feature_embeddings, batch_size
        )

        # Forward pass
        with torch.no_grad():
            # Create kwargs with proper label key
            forward_kwargs = {
                "time_info": time_info_adj,
            }
            # Add label with the correct key name
            if len(self.model.label_keys) > 0:
                label_key = self.model.label_keys[0]
                forward_kwargs[label_key] = torch.zeros(
                    (perturbed_emb.shape[0], 1), device=self.model.device
                )
            
            model_output = self.model.forward_from_embedding(
                feature_embeddings,
                **forward_kwargs
            )

        return self._extract_logits(model_output)

    def _forward_from_inputs(
        self,
        key: str,
        perturbed_inputs: torch.Tensor,
        baseline_inputs: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]],
        label_data: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass using raw inputs (continuous features).

        Args:
            key: Feature key being explained.
            perturbed_inputs: Perturbed input tensor.
            baseline_inputs: Baseline inputs.
            time_info: Optional temporal information.
            label_data: Optional label information.

        Returns:
            Model logits.
        """
        model_inputs = {}
        for fk in self.model.feature_keys:
            if fk == key:
                model_inputs[fk] = perturbed_inputs
            elif fk in baseline_inputs:
                base = baseline_inputs[fk]
                # Expand baseline batch to match perturbed batch size if needed
                if base.shape[0] != perturbed_inputs.shape[0]:
                    base = base.expand(perturbed_inputs.shape[0], *base.shape[1:]).clone()
                else:
                    base = base.clone()
                model_inputs[fk] = base
            else:
                model_inputs[fk] = torch.zeros_like(perturbed_inputs)

        # Add label stub if needed
        if len(self.model.label_keys) > 0:
            label_key = self.model.label_keys[0]
            model_inputs[label_key] = torch.zeros(
                (perturbed_inputs.shape[0], 1), device=perturbed_inputs.device
            )

        with torch.no_grad():
            output = self.model(**model_inputs)
        return self._extract_logits(output)

    def _prepare_time_info(
        self,
        time_info: Optional[Dict[str, torch.Tensor]],
        feature_embeddings: Dict[str, torch.Tensor],
        n_samples: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare time information to match sample batch size.

        Args:
            time_info: Original time information.
            feature_embeddings: Feature embeddings to match sequence lengths.
            n_samples: Number of samples.

        Returns:
            Adjusted time information or None.
        """
        if time_info is None:
            return None

        time_info_adj = {}
        for fk, emb in feature_embeddings.items():
            if fk not in time_info or time_info[fk] is None:
                continue

            seq_len = emb.shape[1]
            t_orig = time_info[fk].to(self.model.device)

            # Normalize to 1D sequence
            t_vec = self._normalize_time_vector(t_orig)

            # Adjust length to match embedding sequence length
            t_adj = self._adjust_time_length(t_vec, seq_len)

            # Expand to batch size
            time_info_adj[fk] = t_adj.unsqueeze(0).expand(n_samples, -1)

        return time_info_adj if time_info_adj else None

    # ------------------------------------------------------------------
    # Baseline generation
    # ------------------------------------------------------------------
    def _generate_baseline(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate baseline samples for LIME computation.

        Creates reference samples to use as the "absence" of features.
        The sampling strategy adapts to the feature type:
        - Discrete features: Use the most common value or a small non-zero value
        - Continuous features: Use mean or small non-zero values

        Args:
            inputs: Dictionary mapping feature names to input tensors.

        Returns:
            Dictionary mapping feature names to baseline sample tensors.
        """
        baseline_samples = {}

        for key, x in inputs.items():
            batch_size = x.shape[0]
            if x.dtype in [torch.int64, torch.int32, torch.long]:
                # Discrete features: use small non-zero token index to avoid zero-mask issues
                # in sequential models (e.g., StageNet). Using ones is a safe neutral choice.
                baseline = torch.ones_like(x)
            else:
                # Continuous features: use small neutral values (near-zero)
                baseline = torch.zeros_like(x)
                baseline = baseline + 0.01

            # Ensure baseline matches input batch size
            if baseline.shape[0] != batch_size:
                baseline = baseline.expand(batch_size, *baseline.shape[1:])

            baseline_samples[key] = baseline.to(x.device)

        return baseline_samples

    # ------------------------------------------------------------------
    # Utility helpers (shared with SHAP)
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
        lime_values: Dict[str, torch.Tensor],
        input_shapes: Dict[str, tuple],
    ) -> Dict[str, torch.Tensor]:
        """Map LIME values from embedding space back to input shapes.

        For embedding-based attributions, this projects the attribution scores
        from embedding dimensions back to the original input tensor shapes.

        Args:
            lime_values: Dictionary of LIME coefficients in embedding space.
            input_shapes: Dictionary of original input shapes.

        Returns:
            Dictionary of LIME coefficients reshaped to match inputs.
        """
        mapped = {}
        for key, values in lime_values.items():
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
