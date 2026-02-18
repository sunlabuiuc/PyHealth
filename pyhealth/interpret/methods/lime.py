from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Callable, Union, cast

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity

from pyhealth.models import BaseModel
from pyhealth.interpret.api import Interpretable
from .base_interpreter import BaseInterpreter


class Lime(BaseInterpreter):
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
        if not isinstance(model, Interpretable):
            raise ValueError("Model must implement Interpretable interface")
        self.model = model
        
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
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
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
            **kwargs: Input data dictionary from dataloader batch. Should contain:
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
        
        # Filter kwargs to only include model feature keys and ensure they are tuples
        inputs = {
            k : (v, ) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if k in self.model.feature_keys
        }
        
        # disassemble inputs
        values = {}
        masks = {}
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
                
        # Append input masks to inputs for baseline generation and perturbation
        for k, v in inputs.items():
            # If the processor schema does not include a mask, we add it as an additional tensor at the end of the tuple.
            # Models should follow this convention to ensure consistent auxiliary inputs and masks.
            if "mask" not in self.model.dataset.input_processors[k].schema():
                inputs[k] = (*v, masks[k])

        # Extract and prepare inputs
        base_logits = self.model.forward(**inputs)["logit"]
        
        # Enforce target class selection for multi-class models to avoid class flipping
        if self._prediction_mode() == "binary":
            if target_class_idx is not None:
                target = torch.tensor([target_class_idx], device=device)
            else:
                target = (torch.sigmoid(base_logits) > 0.5).long()
        elif self._prediction_mode() == "multiclass":
            if target_class_idx is not None:
                target = torch.nn.functional.one_hot(torch.tensor(target_class_idx, device=device), num_classes=base_logits.shape[-1])
            else:
                target = torch.argmax(base_logits, dim=-1)
                target = torch.nn.functional.one_hot(target, num_classes=base_logits.shape[-1])
        elif self._prediction_mode() == "multilabel":
            if target_class_idx is not None:
                target = torch.nn.functional.one_hot(torch.tensor(target_class_idx, device=device), num_classes=base_logits.shape[-1])
            else:
                target = torch.sigmoid(base_logits) > 0.5
        else:
            raise ValueError("Unsupported prediction mode for LIME attribution.")
        
        if baseline is None:
            baselines = self._generate_baseline(values, use_embeddings=self.use_embeddings)
        else:
            baselines = {k: v.to(device) for k, v in baseline.items() if k in self.model.feature_keys}

        # Compute LIME values for each feature
        n_features = self._determine_n_features(values)
        shapes = {k: v.shape for k, v in values.items()}  # Save raw shapes before embedding

        # Split features by type using is_token():
        # - Token features (discrete): perturb in embedding space (raw indices are meaningless)
        # - Continuous features: perturb in raw space (raw values are interpretable)
        if self.use_embeddings:
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None, (
                "Model must have an embedding model for embedding-based LIME."
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

        out = self._compute_lime(
            inputs=inputs,
            xs=values,
            bs=baselines,
            n_features=n_features,
            target=target,
        )
        
        return self._map_to_input_shapes(out, shapes)

    # ------------------------------------------------------------------
    # Core LIME computation
    # ------------------------------------------------------------------
    def _compute_lime(
        self,
        inputs: Dict[str, tuple[torch.Tensor, ...]],
        xs: Dict[str, torch.Tensor],
        bs: Dict[str, torch.Tensor],
        n_features: dict[str, int],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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
        device = next(iter(xs.values())).device
        batch_size = next(iter(xs.values())).shape[0]
        keys = sorted(xs.keys()) # Ensure consistent key order
        # Keep large intermediate tensors off the GPU to avoid OOM
        storage_device = torch.device("cpu")

        # Storage for samples and predictions
        interpretable_samples = []  # Binary vectors
        perturbed_predictions = []  # Model predictions
        similarity_weights = []     # Distance-based weights


        # Generate perturbed samples
        for _ in range(self.n_samples):
            # Create gates for feature inclusion/exclusion, dict[str, (batch_size, n_features)]
            gates = {
                key: torch.bernoulli(torch.ones((n_features[key]), device=device) * 0.5).expand(batch_size, -1)
                for key in xs.keys()
            }
            
            # Create perturbed embedding by mixing input and baseline, dict[str, (batch_size, ...)]
            perturb = self._create_perturbed_sample(
                xs, bs, gates
            )
            
            # Get model prediction for perturbed sample, shape (batch_size, )
            pred = self._evaluate_sample(
                inputs,
                perturb,
                target,
            )
        
            # Create perturbed sample for each batch item
            batch_preds = []
            batch_similarities = []
            
            for b in range(batch_size):
                batch_preds.append(pred[b])
                
                # Compute similarity weight
                x_flatten = torch.cat([xs[k][b].reshape(-1) for k in keys], dim=0)
                p_flatten = torch.cat([perturb[k][b].reshape(-1) for k in keys], dim=0)
                similarity = self._compute_similarity(
                    x_flatten,
                    p_flatten,
                )
                batch_similarities.append(similarity)
            
            # Since gates are identical across batch, just take the first
            g_flatten = torch.cat([gates[k][0] for k in keys], dim=0)
            
            # Store sample information
            interpretable_samples.append(g_flatten.float())
            perturbed_predictions.append(torch.stack(batch_preds, dim=0))
            similarity_weights.append(torch.stack(batch_similarities, dim=0))

            # Move small summaries to CPU to free GPU memory
            interpretable_samples[-1] = interpretable_samples[-1].float().to(storage_device)
            perturbed_predictions[-1] = perturbed_predictions[-1].detach().to(storage_device)
            similarity_weights[-1] = similarity_weights[-1].detach().to(storage_device)

        # Train weighted linear regression
        coefficients = self._train_interpretable_model(
            interpretable_samples,
            perturbed_predictions,
            similarity_weights,
            compute_device=storage_device,
            target_device=device,
        ) # (batch_size, n_features)
        
        return self.split_feature_keys(coefficients, keys, n_features)
        
    def split_feature_keys(self, tensor: torch.Tensor, keys, n_features: dict[str, int]) -> dict[str, torch.Tensor]:
        """Split concatenated coefficients back into per-feature tensors.
        
        Args:
            tensor: Concatenated coefficient tensor (batch_size, total_n_features).
            keys: List of feature keys in order.
            n_features: Dictionary mapping feature keys to number of features.
        """
        out: dict[str, torch.Tensor] = {}
        start = 0
        total = tensor.shape[1]

        for k in keys:
            nf = int(n_features[k])
            end = start + nf
            if end > total:
                raise ValueError(f"Not enough features to slice key={k}: need {end}, have {total}")
            out[k] = tensor[:, start:end]   # (batch_size, nf)
            start = end

        if start != total:
            raise ValueError(f"Unused trailing features: used {start}, total {total}")

        return out        

    def _create_perturbed_sample(
        self,
        xs: dict[str, torch.Tensor],
        bs: dict[str, torch.Tensor],
        gates: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Create a perturbed sample by mixing input and baseline based on gate tensor.

        Args:
            key: Feature key.
            xs: Original input tensors.
            bs: Baseline tensors.
            gates: Binary gate tensors indicating which features to include (1) or exclude (0).

        Returns:
            Perturbed sample tensor.
        """
        perurb = {}
        for key, gate in gates.items():
            # Expand gate dims to broadcast over trailing dimensions (e.g. embed_dim)
            g = gate
            while g.dim() < xs[key].dim():
                g = g.unsqueeze(-1)
            perurb[key] = torch.where(g == 1, xs[key], bs[key])
            
        return perurb

    def _evaluate_sample(
        self,
        inputs: dict[str, tuple[torch.Tensor, ...]],
        perturb: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate model prediction for a perturbed sample.

        Args:
            inputs: Original input tuples (used for non-value fields like time/mask).
            perturb: Perturbed sample tensors. Token features are already
                embedded; continuous features are still in raw space.
            target: Target class tensor.

        Returns:
            Model prediction for the perturbed sample, shape (batch_size, ).
        """
        inputs = inputs.copy()

        if self.use_embeddings:
            # Embed continuous (non-token) perturbed features that are still raw
            # so forward_from_embedding receives proper embeddings.
            # Token features were already embedded before perturbation.
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None, (
                "Model must have an embedding model for embedding-based LIME."
            )
            continuous_keys = {
                k for k in perturb
                if not self.model.dataset.input_processors[k].is_token()
            }
            if continuous_keys:
                continuous_values = {k: perturb[k] for k in continuous_keys}
                embedded_continuous = embedding_model(continuous_values)
                perturb = {
                    k: (embedded_continuous[k] if k in continuous_keys else v)
                    for k, v in perturb.items()
                }

        for k in inputs.keys():
            # Insert perturbed value tensor back into input tuple
            schema = self.model.dataset.input_processors[k].schema()
            inputs[k] = (*inputs[k][:schema.index("value")], perturb[k], *inputs[k][schema.index("value")+1:])

        if self.use_embeddings:
            # Values are already embedded; bypass the model's own embedding.
            logits = self.model.forward_from_embedding(**inputs)["logit"]
        else:
            # Values are raw (token IDs / continuous floats); let the
            # model's regular forward pass handle embedding internally.
            logits = self.model.forward(**inputs)["logit"]
        
        # Reduce to [batch_size, ] by taking absolute difference from target class logit
        return (target - logits).abs().mean(dim=tuple(range(1, logits.ndim)))

    def _compute_similarity(
        self,
        original_emb: torch.Tensor,
        perturbed_emb: torch.Tensor,
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
    # Baseline generation
    # ------------------------------------------------------------------
    def _generate_baseline(
        self, 
        values: Dict[str, torch.Tensor],
        use_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate raw baselines for LIME perturbation.

        Creates reference samples to use as the "absence" of features.
        The sampling strategy adapts to the feature type:
        - Discrete (token) features: UNK token index (will be embedded later
          in ``attribute()`` alongside the values)
        - Continuous features: small neutral value (near-zero)

        Args:
            values: Dictionary of raw input tensors (pre-embedding).
            use_embeddings: If True, generate baselines suitable for
                embedding-based LIME.

        Returns:
            Dictionary mapping feature names to baseline tensors in raw
            (pre-embedding) space.  Embedding of token baselines is handled
            by the caller (``attribute()``).
        """
        baselines = {}

        for k, v in values.items():
            processor = self.model.dataset.input_processors[k]
            if processor.is_token():
                # Token features: UNK token (index 1) as baseline.
                # When use_embeddings=True, embedding happens later in
                # attribute(); when use_embeddings=False, the UNK token
                # IDs are used directly as the perturbed replacement.
                baseline = torch.ones_like(v)
            else:
                # Continuous features: use small neutral values (near-zero)
                baseline = torch.zeros_like(v) + 1e-2
            baselines[k] = baseline

        return baselines

    # ------------------------------------------------------------------
    # Utility helpers (shared with SHAP)
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_n_features(
        raw_x: Dict[str, torch.Tensor],
    ) -> dict[str, int]:
        """Determine the number of features to explain for a given key.

        Args:
            raw_x: Original input tensors, it is usually the last tensor in tuple.
                It should be the raw input, meaning it does not go through any embedding layer yet.

        Returns:
            Number of features (typically sequence length or feature dimension).
        """
        out = {}
        for key, value in raw_x.items():
            if value.dim() >= 2:
                out[key] = value.shape[1]
            else:
                out[key] = value.shape[-1]
        return out

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

LimeExplainer = Lime  # Alias for backward compatibility