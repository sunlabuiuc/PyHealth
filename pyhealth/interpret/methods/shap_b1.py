import torch
import numpy as np
from typing import Dict, Optional, List, Union, Tuple

from pyhealth.models import BaseModel


class ShapExplainer:
    """SHAP (SHapley Additive exPlanations) attribution method for PyHealth models.

    This class implements the SHAP method for computing feature attributions in 
    neural networks. SHAP values represent each feature's contribution to the 
    prediction, based on coalitional game theory principles.

    The method is based on the papers:
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
    φᵢ = Σ (|S|!(n-|S|-1)!/n!) * [fₓ(S ∪ {i}) - fₓ(S)]
    where:
    - S is a subset of features excluding i
    - n is the total number of features
    - fₓ(S) is the model prediction with only features in S
    
    SHAP combines game theory with local explanations, providing several desirable properties:
    1. Local Accuracy: The sum of feature attributions equals the difference between 
       the model output and the expected output
    2. Missingness: Features with zero impact get zero attribution
    3. Consistency: Changing a model to increase a feature's impact increases its attribution

    Args:
        model (BaseModel): A trained PyHealth model to interpret. Can be
            any model that inherits from BaseModel (e.g., MLP, StageNet,
            Transformer, RNN).
        use_embeddings (bool): If True, compute SHAP values with respect to
            embeddings rather than discrete input tokens. This is crucial
            for models with discrete inputs (like ICD codes). The model
            must support returning embeddings via an 'embed' parameter.
            Default is True.
        n_background_samples (int): Number of background samples to use for
            estimating feature contributions. More samples give better
            estimates but increase computation time. Default is 100.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import (
        ...     SampleDataset, split_by_patient, get_dataloader
        ... )
        >>> from pyhealth.models import MLP
        >>> from pyhealth.interpret.methods import ShapExplainer
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
        ...     # ... more samples
        ... ]
        >>>
        >>> # Create dataset and model 
        >>> dataset = SampleDataset(...)
        >>> model = MLP(...)
        >>> trainer = Trainer(model=model, device="cuda:0")
        >>> trainer.train(...)
        >>> test_batch = next(iter(test_loader))
        >>>
        >>> # Initialize SHAP explainer with different methods
        >>> # 1. Auto method (uses exact for small feature sets, kernel for large)
        >>> explainer_auto = ShapExplainer(model, method='auto')
        >>> shap_auto = explainer_auto.attribute(**test_batch)
        >>>
        >>> # 2. Exact computation (for small feature sets)
        >>> explainer_exact = ShapExplainer(model, method='exact')
        >>> shap_exact = explainer_exact.attribute(**test_batch)
        >>>
        >>> # 3. Kernel SHAP (efficient for high-dimensional features)
        >>> explainer_kernel = ShapExplainer(model, method='kernel')
        >>> shap_kernel = explainer_kernel.attribute(**test_batch)
        >>>
        >>> # 4. DeepSHAP (optimized for neural networks)
        >>> explainer_deep = ShapExplainer(model, method='deep')
        >>> shap_deep = explainer_deep.attribute(**test_batch)
        >>>
        >>> # All methods return the same format of SHAP values
        >>> print(shap_auto)  # Same structure for all methods
        {'conditions': tensor([[0.1234, 0.5678, 0.9012]], device='cuda:0'),
         'procedures': tensor([[0.2345, 0.6789, 0.0123, 0.4567]])}
    """

    def __init__(
        self, 
        model: BaseModel, 
        method: str = 'auto',
        use_embeddings: bool = True,
        n_background_samples: int = 100,
        exact_threshold: int = 15
    ):
        """Initialize SHAP explainer.

        This implementation supports three methods for computing SHAP values:
        1. Classic Shapley (Exact): Used when feature count <= exact_threshold and method='exact'
           - Computes exact Shapley values by evaluating all possible feature coalitions
           - Provides exact results but computationally expensive for high dimensions
        
        2. Kernel SHAP (Approximate): Used when feature count > exact_threshold or method='kernel'
           - Approximates Shapley values using weighted least squares regression
           - More efficient for high-dimensional features but provides estimates
           
        3. DeepSHAP (Deep Learning): Used when method='deep'
           - Combines DeepLIFT's backpropagation-based rules with Shapley values
           - Specifically optimized for deep neural networks
           - Provides fast approximation by exploiting network architecture
           - Requires model to support gradient computation

        Args:
            model: A trained PyHealth model to interpret. Can be any model that
                inherits from BaseModel (e.g., MLP, StageNet, Transformer, RNN).
            method: Method to use for SHAP computation. Options:
                - 'auto': Automatically select based on feature count
                - 'exact': Use classic Shapley (exact computation)
                - 'kernel': Use Kernel SHAP (model-agnostic approximation)
                - 'deep': Use DeepSHAP (neural network specific approximation)
                Default is 'auto'.
            use_embeddings: If True, compute SHAP values with respect to
                embeddings rather than discrete input tokens. This is crucial
                for models with discrete inputs (like ICD codes).
            n_background_samples: Number of background samples to use for
                estimating feature contributions. More samples give better
                estimates but increase computation time.
            exact_threshold: Maximum number of features for using exact Shapley
                computation in 'auto' mode. Above this, switches to Kernel SHAP
                approximation. Default is 15 (2^15 = 32,768 possible coalitions).

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method, or if method='deep'
                but model does not support gradient computation.
            ValueError: If method is not one of ['auto', 'exact', 'kernel', 'deep'].
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.use_embeddings = use_embeddings
        self.n_background_samples = n_background_samples
        self.exact_threshold = exact_threshold
        
        # Validate and store computation method
        valid_methods = ['auto', 'exact', 'kernel', 'deep']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        self.method = method

        # Validate model requirements
        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "SHAP values. Set use_embeddings=False to use "
                "input-level gradients (only for continuous features)."
            )
            
        # Additional validation for DeepSHAP
        if method == 'deep':
            assert hasattr(model, "parameters") and next(model.parameters(), None) is not None, (
                f"Model {type(model).__name__} must be a neural network with "
                "parameters that support gradient computation to use DeepSHAP method."
            )
        
    def _generate_background_samples(
        self, 
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate background samples for SHAP computation.

        Creates reference samples to establish baseline predictions for SHAP value
        computation. The sampling strategy adapts to the feature type:

        For discrete features:
        - Samples uniformly from the set of unique values observed in the input
        - Preserves the discrete nature of categorical variables
        - Maintains valid values from the training distribution

        For continuous features:
        - Samples uniformly from the range [min(x), max(x)]
        - Captures the full span of possible values
        - Ensures diverse background distribution

        The number of samples is controlled by self.n_background_samples, with
        more samples providing better estimates at the cost of computation time.

        Args:
            inputs: Dictionary mapping feature names to input tensors. Each tensor
                   should have shape (batch_size, ..., feature_dim) where feature_dim
                   is the dimensionality of each feature.

        Returns:
            Dictionary mapping feature names to background sample tensors. Each
            tensor has shape (n_background_samples, ..., feature_dim) and matches
            the device of the input tensor.

        Note:
            Background samples are crucial for SHAP value computation as they
            establish the baseline against which feature contributions are measured.
            Poor background sample selection can lead to misleading attributions.
        """
        background_samples = {}
        
        for key, x in inputs.items():
            # Handle discrete vs continuous features
            if x.dtype in [torch.int64, torch.int32, torch.long]:
                # Discrete features: sample uniformly from observed values
                unique_vals = torch.unique(x)
                samples = unique_vals[torch.randint(
                    len(unique_vals), 
                    (self.n_background_samples,) + x.shape[1:]
                )]
            else:
                # Continuous features: sample uniformly from range
                min_val = torch.min(x)
                max_val = torch.max(x)
                samples = torch.rand(
                    (self.n_background_samples,) + x.shape[1:],
                    device=x.device
                ) * (max_val - min_val) + min_val
                
            background_samples[key] = samples.to(x.device)
            
        return background_samples

    def _compute_classic_shapley(
        self,
        key: str,
        input_emb: Dict[str, torch.Tensor],
        background_emb: Dict[str, torch.Tensor],
        n_features: int,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute exact Shapley values by evaluating all possible feature coalitions.

        This method implements the classic Shapley value computation, providing
        exact attribution values by exhaustively evaluating all possible feature
        combinations. Suitable for small feature sets (n_features ≤ exact_threshold).

        Algorithm Steps:
        1. Feature Enumeration:
           - Generate all possible feature coalitions (2^n combinations)
           - For each feature i, consider coalitions with and without i
        
        2. Value Computation:
           - For each coalition S and feature i:
             * Compute f(S ∪ {i}) - f(S)
             * Weight by |S|!(n-|S|-1)!/n!
        
        3. Aggregation:
           - Sum weighted marginal contributions
           - Normalize by number of coalitions

        Theoretical Properties:
        - Exactness: Provides true Shapley values, not approximations
        - Uniqueness: Only attribution method satisfying efficiency,
          symmetry, dummy, and additivity axioms
        - Computational Complexity: O(2^n) where n is number of features

        Args:
            key: Feature key being analyzed in the input dictionary
            input_emb: Dictionary mapping feature keys to their embeddings/values
                      Shape: (batch_size, ..., feature_dim)
            background_emb: Dictionary of baseline/background embeddings
                          Shape: (n_background, ..., feature_dim)
            n_features: Total number of features to analyze
            target_class_idx: For multi-class models, specifies which class's
                            prediction to explain. If None, explains the model's
                            maximum prediction.
            time_info: Optional temporal information for time-series models
            label_data: Optional label information for supervised models

        Returns:
            torch.Tensor: Exact Shapley values for each feature. Shape matches
                         the feature dimension of the input, with each value
                         representing that feature's exact contribution to the
                         prediction difference from baseline.

        Note:
            This method is computationally intensive for large feature sets.
            Use only when n_features ≤ exact_threshold (default 15).
        """
        import itertools
        
        device = input_emb[key].device
        shap_values = torch.zeros(n_features, device=device)
        
        # Generate all possible coalitions (except empty set)
        all_features = set(range(n_features))
        n_players = n_features
        
        # For each feature
        for i in range(n_features):
            marginal_contributions = []
            
            # For each possible coalition size
            for size in range(n_players):
                # Generate all coalitions of this size that exclude feature i
                other_features = list(all_features - {i})
                for coalition in itertools.combinations(other_features, size):
                    coalition = set(coalition)
                    
                    # Create mixed samples for coalition and coalition+i
                    mixed_without_i = background_emb[key].clone()
                    mixed_with_i = background_emb[key].clone()
                    
                    # Set coalition features
                    for j in coalition:
                        mixed_without_i[..., j] = input_emb[key][..., j]
                        mixed_with_i[..., j] = input_emb[key][..., j]
                    
                    # Add feature i to second coalition
                    mixed_with_i[..., i] = input_emb[key][..., i]
                    
                    # Compute model outputs
                    if self.use_embeddings:
                        output_without_i = self.model.forward_from_embedding(
                            {key: mixed_without_i},
                            time_info=time_info,
                            **(label_data or {})
                        )
                        output_with_i = self.model.forward_from_embedding(
                            {key: mixed_with_i},
                            time_info=time_info,
                            **(label_data or {})
                        )
                    else:
                        output_without_i = self.model(
                            **{key: mixed_without_i},
                            **(time_info or {}),
                            **(label_data or {})
                        )
                        output_with_i = self.model(
                            **{key: mixed_with_i},
                            **(time_info or {}),
                            **(label_data or {})
                        )
                    
                    # Get predictions
                    logits_without_i = output_without_i["logit"]
                    logits_with_i = output_with_i["logit"]
                    
                    if target_class_idx is None:
                        pred_without_i = torch.max(logits_without_i, dim=-1)[0]
                        pred_with_i = torch.max(logits_with_i, dim=-1)[0]
                    else:
                        pred_without_i = logits_without_i[..., target_class_idx]
                        pred_with_i = logits_with_i[..., target_class_idx]
                    
                    # Calculate marginal contribution
                    marginal = pred_with_i - pred_without_i
                    weight = (
                        torch.factorial(torch.tensor(size)) * 
                        torch.factorial(torch.tensor(n_players - size - 1))
                    ) / torch.factorial(torch.tensor(n_players))
                    
                    marginal_contributions.append(marginal.detach() * weight)
            
            # Average marginal contributions
            shap_values[i] = torch.stack(marginal_contributions).mean()
        
        return shap_values

    def _compute_deep_shap(
        self,
        key: str,
        input_emb: Dict[str, torch.Tensor],
        background_emb: Dict[str, torch.Tensor],
        n_features: int,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute SHAP values using the DeepSHAP algorithm.

        DeepSHAP combines ideas from DeepLIFT and Shapley values to provide
        computationally efficient feature attribution for deep neural networks.
        It propagates attribution from the output to input layer by layer using
        modified backpropagation rules.

        Key Features:
        1. Computational Efficiency:
           - Uses backpropagation instead of model evaluations
           - Linear complexity in terms of feature count
           - Particularly efficient for deep networks

        2. Attribution Rules:
           - Multiplier rule for linear operations
           - Chain rule for composed functions
           - Special handling of non-linearities (ReLU, etc.)

        3. Theoretical Properties:
           - Satisfies completeness (attributions sum to output delta)
           - Preserves implementation invariance
           - Maintains linear composition

        Args:
            key: Feature key being analyzed
            input_emb: Dictionary of input embeddings/features
            background_emb: Dictionary of background embeddings/features
            n_features: Number of features
            target_class_idx: Target class for attribution
            time_info: Optional temporal information
            label_data: Optional label information

        Returns:
            torch.Tensor: SHAP values computed using DeepSHAP method
        """
        device = input_emb[key].device
        requires_grad = True

        # Enable gradient computation
        input_tensor = input_emb[key].clone().detach().requires_grad_(True)
        background_tensor = background_emb[key].mean(0).detach()  # Use mean of background

        # Forward pass
        if self.use_embeddings:


            output = self.model.forward_from_embedding(
                {key: input_tensor},
                time_info=time_info,
                **(label_data or {})
            )
            baseline_output = self.model.forward_from_embedding(
                {key: background_tensor},
                time_info=time_info,
                **(label_data or {})
            )
        else:
            output = self.model(
                **{key: input_tensor},
                **(time_info or {}),
                **(label_data or {})
            )
            baseline_output = self.model(
                **{key: background_tensor},
                **(time_info or {}),
                **(label_data or {})
            )

        # Get predictions
        logits = output["logit"]
        baseline_logits = baseline_output["logit"]

        if target_class_idx is None:
            pred = torch.max(logits, dim=-1)[0]
            baseline_pred = torch.max(baseline_logits, dim=-1)[0]
        else:
            pred = logits[..., target_class_idx]
            baseline_pred = baseline_logits[..., target_class_idx]

        # Compute gradients
        diff = (pred - baseline_pred).sum()
        grad = torch.autograd.grad(diff, input_tensor)[0]

        # Scale gradients by input difference from reference
        input_diff = input_tensor - background_tensor
        shap_values = grad * input_diff

        return shap_values.detach()


    def _compute_kernel_shap_matrix(
        self,
        key: str,
        input_emb: Dict[str, torch.Tensor],
        background_emb: Dict[str, torch.Tensor],
        n_features: int,
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute SHAP values using the Kernel SHAP approximation method.

        This implements the Kernel SHAP algorithm that approximates Shapley values 
        through a weighted least squares regression. The key steps are:

        1. Feature Coalitions:
           - Generates random subsets of features
           - Each coalition represents a possible combination of features
           - Uses efficient sampling to cover the feature space

        2. Model Evaluation:
           - For each coalition, creates a mixed sample using background values
           - Replaces subset of features with actual input values
           - Computes model prediction for this mixed sample

        3. Weighted Least Squares:
           - Uses kernel weights based on coalition sizes
           - Weights emphasize coalitions that help estimate Shapley values
           - Solves regression to find feature contributions
        
        Args:
            inputs: Dictionary of input tensors containing the feature values
                   to explain.
            background: Dictionary of background samples used to establish
                       baseline predictions.
            target_class_idx: Optional index of target class for multi-class
                            models. If None, uses maximum prediction.
            time_info: Optional temporal information for time-series data.
            label_data: Optional label information for supervised models.

        Returns:
            torch.Tensor: Approximated SHAP values for each feature
        """
        n_coalitions = min(2 ** n_features, 1000)  # Cap number of coalitions
        coalition_weights = []
        coalition_values = []
        
        for _ in range(n_coalitions):
            # Random coalition
            coalition = torch.randint(2, (n_features,), device=input_emb[key].device)
            
            # Create mixed sample
            mixed = background_emb[key].clone()
            for i, use_input in enumerate(coalition):
                if use_input:
                    mixed[..., i] = input_emb[key][..., i]
            
            # Forward pass
            """
            if self.use_embeddings:
                output = self.model.forward_from_embedding(
                    {key: mixed},
                    time_info=time_info,
                    **(label_data or {})
                )
            """
            if self.use_embeddings:
                # --- SAFETY PATCH: ensure all model feature embeddings exist ---
                feature_embeddings = {key: mixed}
                for fk in self.model.feature_keys:
                    if fk not in feature_embeddings:
                        # Create zero tensor shaped like existing embedding
                        ref_tensor = next(iter(feature_embeddings.values()))
                        feature_embeddings[fk] = torch.zeros_like(ref_tensor).to(self.model.device)
                # ---------------------------------------------------------------

                output = self.model.forward_from_embedding(
                    feature_embeddings,
                    time_info=time_info,
                    **(label_data or {})
                )
            else:
                output = self.model(
                    **{key: mixed},
                    **(time_info or {}),
                    **(label_data or {})
                )
            
            logits = output["logit"]
            
            # Get target class prediction
            if target_class_idx is None:
                pred = torch.max(logits, dim=-1)[0]
            else:
                pred = logits[..., target_class_idx]
            
            coalition_values.append(pred.detach())
            coalition_size = torch.sum(coalition).item()
            
            # Compute kernel SHAP weight
            # The kernel SHAP weight is designed to approximate Shapley values efficiently.
            # For a coalition of size |z| in a set of M features, the weight is:
            # weight = (M-1) / (binom(M-1,|z|-1) * |z| * (M-|z|))
            #
            # Special cases:
            # - Empty coalition (|z|=0) or full coalition (|z|=M): weight=1000.0
            #   These edge cases are crucial for baseline and full feature effects
            #
            # The weights ensure:
            # 1. Local accuracy: Sum of SHAP values equals model output difference
            # 2. Consistency: Increased feature impact leads to higher attribution
            # 3. Efficiency: Reduces computation from O(2^M) to O(M³)
            if coalition_size == 0 or coalition_size == n_features:
                weight = torch.tensor(1000.0)  # Large weight for edge cases
            else:
                weight = (n_features - 1) / (
                    coalition_size * (n_features - coalition_size) * 
                    torch.special.comb(n_features - 1, coalition_size - 1)
                )
                weight = torch.tensor(weight, dtype=torch.float32)
            
            coalition_weights.append(weight)
        
        # Convert to tensors
        coalition_weights = torch.stack(coalition_weights)
        coalition_values = torch.stack(coalition_values)
        
        # Solve weighted least squares
        weighted_values = coalition_values * coalition_weights.unsqueeze(-1)
        return torch.linalg.lstsq(
            weighted_values, 
            coalition_weights * coalition_values
        )[0]

    def _compute_shapley_values(
        self,
        inputs: Dict[str, torch.Tensor],
        background: Dict[str, torch.Tensor],
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute SHAP values using the selected attribution method.

        This is the main orchestrator for SHAP value computation. It automatically
        selects and applies the appropriate method based on feature count and
        user settings:

        1. Classic Shapley (method='exact' or auto with few features):
           - Exact computation using all possible feature coalitions
           - Provides true Shapley values
           - Suitable for n_features ≤ exact_threshold

        2. Kernel SHAP (method='kernel' or auto with many features):
           - Efficient approximation using weighted least squares
           - Model-agnostic approach
           - Suitable for high-dimensional features

        3. DeepSHAP (method='deep'):
           - Neural network model specific implementation
           - Uses backpropagation-based attribution
           - Most efficient for deep learning models

        Args:
            inputs: Dictionary of input tensors to explain
            background: Dictionary of background/baseline samples
            target_class_idx: Specific class to explain (None for max class)
            time_info: Optional temporal information for time-series models
            label_data: Optional label information for supervised models

        Returns:
            Dictionary mapping feature names to their SHAP values. Values
            represent each feature's contribution to the difference between
            the model's prediction and the baseline prediction.
        """

        shap_values = {}
        
        # Convert inputs to embedding space if needed
        if self.use_embeddings:
            input_emb = self.model.embedding_model(inputs)
            #background_emb = {
            #    k: self.model.embedding_model({k: v})[k] 
            #    for k, v in background.items()
            #}
            background_emb = self.model.embedding_model(background)
        else:
            input_emb = inputs
            background_emb = background

        print("Input_emb keys:", input_emb.keys())
        print("Background_emb keys:", background_emb.keys())


        # Compute SHAP values for each feature
        for key in inputs:
            # Get dimensions
            if self.use_embeddings:
                feature_dim = input_emb[key].shape[-1]
            else:
                feature_dim = 1 if input_emb[key].dim() == 2 else input_emb[key].shape[-1]

            # Get dimensions and determine computation method
            n_features = feature_dim
            
            # Choose computation method based on settings and feature count
            computation_method = self.method
            if computation_method == 'auto':
                computation_method = 'exact' if n_features <= self.exact_threshold else 'kernel'

            if computation_method == 'exact':
                # Use classic Shapley for exact computation
                shap_matrix = self._compute_classic_shapley(
                    key=key,
                    input_emb=input_emb,
                    background_emb=background_emb,
                    n_features=n_features,
                    target_class_idx=target_class_idx,
                    time_info=time_info,
                    label_data=label_data
                )
            elif computation_method == 'deep':
                # Use DeepSHAP for neural network specific computation
                shap_matrix = self._compute_deep_shap(
                    key=key,
                    input_emb=input_emb,
                    background_emb=background_emb,
                    n_features=n_features,
                    target_class_idx=target_class_idx,
                    time_info=time_info,
                    label_data=label_data
                )
            else:
                # Use Kernel SHAP for approximate computation
                shap_matrix = self._compute_kernel_shap_matrix(
                    key=key,
                    input_emb=input_emb,
                    background_emb=background_emb,
                    n_features=n_features,
                    target_class_idx=target_class_idx,
                    time_info=time_info,
                    label_data=label_data
                )
            
            shap_values[key] = shap_matrix

        return shap_values

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
        3. Feature attribution computation using either exact or approximate methods
        4. Device management and tensor type conversion

        The method automatically chooses between:
        - Classic Shapley (exact) for feature_count ≤ exact_threshold
        - Kernel SHAP (approximate) for feature_count > exact_threshold

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
            >>> # Single sample attribution
            >>> shap_values = explainer.attribute(
            ...     x_continuous=torch.tensor([[1.0, 2.0, 3.0]]),
            ...     x_categorical=torch.tensor([[0, 1, 2]]),
            ...     target_class_idx=1
            ... )
            >>> print(shap_values['x_continuous'])  # Shape: (1, 3)
        """
        # Extract feature keys and prepare inputs
        feature_keys = self.model.feature_keys
        inputs = {}
        time_info = {}
        label_data = {}

        for key in feature_keys:
            if key in data:
                x = data[key]
                if isinstance(x, tuple):
                    time_info[key] = x[0]
                    x = x[1]

                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)

                x = x.to(next(self.model.parameters()).device)
                inputs[key] = x

        # Store label data
        for key in self.model.label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.tensor(label_val)
                label_val = label_val.to(next(self.model.parameters()).device)
                label_data[key] = label_val

        # Generate or use provided background samples
        if baseline is None:
            background = self._generate_background_samples(inputs)
        else:
            background = baseline

        # Compute SHAP values
        attributions = self._compute_shapley_values(
            inputs=inputs,
            background=background,
            target_class_idx=target_class_idx,
            time_info=time_info,
            label_data=label_data,
        )

        return attributions