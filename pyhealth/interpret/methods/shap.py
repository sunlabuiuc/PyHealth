from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from pyhealth.models import BaseModel
from pyhealth.interpret.api import Interpretable
from .base_interpreter import BaseInterpreter


class Shap(BaseInterpreter):
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
        self.model = model
        self.use_embeddings = use_embeddings
        self.n_background_samples = n_background_samples
        self.max_coalitions = max_coalitions
        self.regularization = regularization
        self.random_seed = random_seed

        # Validate model requirements
        if use_embeddings and not isinstance(model, Interpretable):
            raise ValueError("Model must implement Interpretable interface or use_embeddings must be False.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        target_class_idx: Optional[int] = None,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
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
            **kwargs: Input data dictionary from dataloader batch. Should contain:
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

        # Filter kwargs to only include model feature keys and ensure they are tuples
        inputs = {
            k: (v,) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if k in self.model.feature_keys
        }

        # Disassemble inputs into values and masks
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
                target = torch.nn.functional.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=base_logits.shape[-1],
                )
            else:
                target = torch.argmax(base_logits, dim=-1)
                target = torch.nn.functional.one_hot(
                    target, num_classes=base_logits.shape[-1]
                )
        elif self._prediction_mode() == "multilabel":
            if target_class_idx is not None:
                target = torch.nn.functional.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=base_logits.shape[-1],
                )
            else:
                target = torch.sigmoid(base_logits) > 0.5
        else:
            raise ValueError("Unsupported prediction mode for SHAP attribution.")

        if baseline is None:
            baselines = self._generate_background_samples(
                values, use_embeddings=self.use_embeddings
            )
        else:
            baselines = {
                k: v.to(device)
                for k, v in baseline.items()
                if k in self.model.feature_keys
            }

        # Compute SHAP values
        n_features = self._determine_n_features(values)
        shapes = {k: v.shape for k, v in values.items()}

        # Embed values when using embedding-based SHAP.
        # Split by is_token(): token features are embedded before perturbation
        # (raw indices are meaningless for interpolation), while continuous
        # features stay raw so each raw dimension gets its own SHAP value.
        # Continuous features will be embedded inside _evaluate_sample().
        if self.use_embeddings and isinstance(self.model, Interpretable):
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None, (
                "Model must have an embedding model for embedding-based SHAP."
            )
            token_keys = {
                k for k in values
                if self.model.dataset.input_processors[k].is_token()
            }
            if token_keys:
                token_embedded = embedding_model(
                    {k: values[k] for k in token_keys}
                )
                values = {**values, **token_embedded}
            # Embed token baselines too so xs and bs are in the same space
            token_baselines = {
                k: v for k, v in baselines.items() if k in token_keys
            }
            if token_baselines:
                embedded_baselines = embedding_model(token_baselines)
                baselines = {**baselines, **embedded_baselines}

        out = self._compute_kernel_shap(
            inputs=inputs,
            xs=values,
            bs=baselines,
            n_features=n_features,
            target=target,
        )

        return self._map_to_input_shapes(out, shapes)

    # ------------------------------------------------------------------
    # Core Kernel SHAP computation
    # ------------------------------------------------------------------
    def _compute_kernel_shap(
        self,
        inputs: Dict[str, tuple[torch.Tensor, ...]],
        xs: Dict[str, torch.Tensor],
        bs: Dict[str, torch.Tensor],
        n_features: dict[str, int],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute SHAP values using the Kernel SHAP approximation method.

        This implements the Kernel SHAP algorithm that approximates Shapley values
        through a weighted least squares regression. The key steps are:

        1. Feature Coalitions: Generate random subsets of features
        2. Model Evaluation: Evaluate mixed samples (background + coalition)
        3. Weighted Least Squares: Solve for SHAP values using kernel weights

        Args:
            inputs: Dictionary of input tuples from the dataloader.
            xs: Dictionary of input values (or embeddings).
            bs: Dictionary of baseline values (or embeddings).
            n_features: Dictionary mapping feature keys to feature counts.
            target: Target tensor for prediction comparison.

        Returns:
            Dictionary mapping feature keys to SHAP value tensors.
        """
        device = next(iter(xs.values())).device
        batch_size = next(iter(xs.values())).shape[0]
        keys = sorted(xs.keys())  # Ensure consistent key order
        total_features = sum(n_features[k] for k in keys)
        n_coalitions = min(2 ** total_features, self.max_coalitions)

        # Storage for coalition sampling
        coalition_vectors = []
        coalition_weights = []
        coalition_preds = []

        # Add edge case coalitions explicitly (empty and full)
        # These are crucial for the local accuracy property of SHAP
        edge_coalitions = [
            torch.zeros(total_features, device=device),  # Empty coalition (baseline)
            torch.ones(total_features, device=device),   # Full coalition (actual input)
        ]

        for coalition in edge_coalitions:
            gates = self._split_coalition_to_gates(
                coalition, keys, n_features, batch_size
            )
            perturb = self._create_perturbed_sample(xs, bs, gates)
            pred = self._evaluate_sample(inputs, perturb, target)

            coalition_vectors.append(coalition.float())
            coalition_preds.append(pred.detach())
            coalition_weights.append(
                self._compute_kernel_weight(
                    int(coalition.sum().item()), total_features
                )
            )

        # Sample remaining coalitions randomly (excluding edge cases already added)
        n_random_coalitions = max(0, n_coalitions - 2)
        for _ in range(n_random_coalitions):
            coalition = torch.randint(
                2, (total_features,), device=device
            ).float()

            gates = self._split_coalition_to_gates(
                coalition, keys, n_features, batch_size
            )
            perturb = self._create_perturbed_sample(xs, bs, gates)
            pred = self._evaluate_sample(inputs, perturb, target)

            coalition_vectors.append(coalition.float())
            coalition_preds.append(pred.detach())
            coalition_weights.append(
                self._compute_kernel_weight(
                    int(coalition.sum().item()), total_features
                )
            )

        # Solve weighted least squares
        shap_values = self._solve_weighted_least_squares(
            coalition_vectors, coalition_preds, coalition_weights, device
        )

        return self.split_feature_keys(shap_values, keys, n_features)

    def _split_coalition_to_gates(
        self,
        coalition: torch.Tensor,
        keys: list,
        n_features: dict[str, int],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Split a flat coalition vector into per-feature-key gate tensors.

        Args:
            coalition: Flat binary vector of length total_features.
            keys: Sorted list of feature keys.
            n_features: Dictionary mapping feature keys to feature counts.
            batch_size: Batch size to expand gates to.

        Returns:
            Dictionary mapping feature keys to gate tensors (batch_size, n_features_k).
        """
        gates = {}
        start = 0
        for k in keys:
            nf = n_features[k]
            gates[k] = coalition[start : start + nf].expand(batch_size, -1)
            start += nf
        return gates

    def split_feature_keys(
        self,
        tensor: torch.Tensor,
        keys: list,
        n_features: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """Split concatenated SHAP values back into per-feature tensors.

        Args:
            tensor: Concatenated SHAP value tensor (batch_size, total_n_features).
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
                raise ValueError(
                    f"Not enough features to slice key={k}: need {end}, have {total}"
                )
            out[k] = tensor[:, start:end]  # (batch_size, nf)
            start = end

        if start != total:
            raise ValueError(
                f"Unused trailing features: used {start}, total {total}"
            )

        return out

    def _create_perturbed_sample(
        self,
        xs: dict[str, torch.Tensor],
        bs: dict[str, torch.Tensor],
        gates: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Create a perturbed sample by mixing input and baseline based on gate tensor.

        Args:
            xs: Original input tensors.
            bs: Baseline tensors.
            gates: Binary gate tensors indicating which features to include (1) or exclude (0).

        Returns:
            Dictionary of perturbed sample tensors.
        """
        perturb = {}
        for key, gate in gates.items():
            # Expand gate dims to broadcast over trailing dimensions (e.g. embed_dim)
            g = gate
            while g.dim() < xs[key].dim():
                g = g.unsqueeze(-1)
            perturb[key] = torch.where(g == 1, xs[key], bs[key])
        return perturb

    def _evaluate_sample(
        self,
        inputs: dict[str, tuple[torch.Tensor, ...]],
        perturb: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate model prediction for a perturbed sample.

        Unlike LIME (which uses distance-from-target), Kernel SHAP requires the
        **actual model prediction** for the target class so the weighted least
        squares correctly decomposes f(x) - E[f(x)] into per-feature Shapley
        values.

        Args:
            inputs: Original input tuples from the dataloader.
            perturb: Dictionary of perturbed value tensors.
            target: Target tensor used to select which class prediction to
                return.  For binary this is a 0/1 scalar or (batch,1) tensor;
                for multiclass/multilabel it is a one-hot vector.

        Returns:
            Target-class prediction scalar per batch item, shape (batch_size,).
        """
        inputs = inputs.copy()

        if self.use_embeddings and isinstance(self.model, Interpretable):
            # For continuous (non-token) features, embed through
            # embedding_model so forward_from_embedding receives proper
            # embeddings.  Token features were already embedded before
            # perturbation.
            embedding_model = self.model.get_embedding_model()
            assert embedding_model is not None, (
                "Model must have an embedding model for embedding-based SHAP."
            )
            continuous_keys = {
                k for k in perturb
                if not self.model.dataset.input_processors[k].is_token()
            }
            if continuous_keys:
                continuous_embedded = embedding_model(
                    {k: perturb[k] for k in continuous_keys}
                )
                perturb = {**perturb, **continuous_embedded}

        for k in inputs.keys():
            # Insert perturbed value tensor back into input tuple
            schema = self.model.dataset.input_processors[k].schema()
            inputs[k] = (
                *inputs[k][: schema.index("value")],
                perturb[k],
                *inputs[k][schema.index("value") + 1 :],
            )

        if self.use_embeddings and isinstance(self.model, Interpretable):
            # Values are already embedded; bypass the model's own embedding.
            logits = self.model.forward_from_embedding(**inputs)["logit"]
        else:
            # Values are raw (token IDs / continuous floats); let the
            # model's regular forward pass handle embedding internally.
            logits = self.model.forward(**inputs)["logit"]

        return self._extract_target_prediction(logits, target)

    def _extract_target_prediction(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the model's prediction for the target class.

        Kernel SHAP decomposes f(x) ≈ φ₀ + Σ φᵢ zᵢ via weighted least squares.
        Using **raw logits** (unbounded) rather than probabilities (bounded
        [0, 1]) is critical: sigmoid compression squashes coalition differences
        in the saturated regions, producing uniformly small SHAP values and
        degraded feature rankings.

        Args:
            logits: Raw model logits, shape (batch_size, n_classes) or
                (batch_size, 1).
            target: Target indicator.  Binary: scalar/tensor with 0 or 1.
                Multiclass: one-hot tensor.  Multilabel: multi-hot tensor.

        Returns:
            Scalar prediction per batch item, shape (batch_size,).
        """
        mode = self._prediction_mode()

        if mode == "binary":
            # Use raw logit — not sigmoid probability — to preserve the
            # dynamic range that Kernel SHAP's linear decomposition needs.
            logit = logits.squeeze(-1)  # (batch,)
            t = target.float()
            if t.dim() > 1:
                t = t.squeeze(-1)
            # target=1  →  logit   (higher logit ⇒ more positive class)
            # target=0  → −logit   (higher value ⇒ more negative class)
            return t * logit + (1 - t) * (-logit)

        elif mode == "multiclass":
            # target is one-hot; dot-product extracts the target-class logit
            return (target.float() * logits).sum(dim=-1)  # (batch,)

        elif mode == "multilabel":
            # target is multi-hot; average logits over active labels
            t = target.float()
            n_active = t.sum(dim=-1).clamp(min=1)  # avoid div-by-zero
            return (t * logits).sum(dim=-1) / n_active  # (batch,)

        else:
            # regression or unknown — just return the logit
            return logits.squeeze(-1)

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
        self,
        values: Dict[str, torch.Tensor],
        use_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate background samples for SHAP computation.

        Creates reference samples to establish baseline predictions.
        The sampling strategy adapts to the feature type:
        - Discrete features: Embed UNK token as the baseline
        - Continuous features: Use a small neutral value (e.g., near-zero)

        Args:
            values: The dictionary of input tensors for which we plan to perturb.
                It should be the raw input, meaning it does not go through any
                embedding layer yet.
            use_embeddings: If True, generate baselines suitable for
                embedding-based SHAP.

        Returns:
            Dictionary mapping feature names to background sample tensors.
        """
        baselines = {}

        for k, v in values.items():
            if self.model.dataset.input_processors[k].is_token():
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
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_n_features(
        raw_x: Dict[str, torch.Tensor],
    ) -> dict[str, int]:
        """Determine the number of features to explain for each key.

        Args:
            raw_x: Original input tensors. It should be the raw input,
                meaning it does not go through any embedding layer yet.

        Returns:
            Dictionary mapping feature keys to feature counts.
        """
        out = {}
        for key, value in raw_x.items():
            if value.dim() >= 2:
                out[key] = value.shape[1]
            else:
                out[key] = value.shape[-1]
        return out

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
    
ShapExplainer = Shap  # Alias for backward compatibility