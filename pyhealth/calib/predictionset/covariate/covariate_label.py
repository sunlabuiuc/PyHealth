"""
Covariate Shift Adaptive Conformal Prediction.

This module implements conformal prediction with covariate shift correction
using likelihood ratio weighting. The implementation supports both:
1. KDE-based density estimation for automatic weight computation
2. User-provided custom weights for flexibility

The KDE-based correction approach is based on the CoDrug method, which uses
energy-based models and kernel density estimation to assess molecular densities
and construct weighted conformal prediction sets.

Papers:
    Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and
    Aaditya Ramdas. "Conformal prediction under covariate shift."
    Advances in neural information processing systems 32 (2019).
    https://arxiv.org/abs/1904.06019
    
    Laghuvarapu, Siddhartha, Zhen Lin, and Jimeng Sun.
    "Conformal Drug Property Prediction with Density Estimation under 
    Covariate Shift." NeurIPS 2023.
    https://arxiv.org/abs/2310.12033
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.calibration.kcal.kde import RBFKernelMean
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel

__all__ = ["CovariateLabel", "fit_kde"]


def fit_kde(
    cal_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    bandwidth: Optional[Union[float, str]] = "scott",
    kernel: str = "rbf",
) -> tuple[Callable, Callable]:
    """Fit KDEs on calibration and test embeddings using PyHealth's KDE.

    This implements the KDE-based density estimation approach from the CoDrug
    paper (Laghuvarapu et al., NeurIPS 2023) for computing likelihood ratios
    under covariate shift. The method uses kernel density estimation on both
    calibration and test embeddings to estimate p_test(x) / p_cal(x).

    This uses the PyHealth torch-based RBF kernel density estimator
    which is more efficient than sklearn for GPU computation.

    Reference:
        Laghuvarapu, S., Lin, Z., & Sun, J. (2023). Conformal Drug Property
        Prediction with Density Estimation under Covariate Shift. NeurIPS 2023.
        https://arxiv.org/abs/2310.12033

    Args:
        cal_embeddings: Calibration embeddings as numpy array of shape
            (n_cal_samples, embedding_dim)
        test_embeddings: Test embeddings as numpy array of shape
            (n_test_samples, embedding_dim)
        bandwidth: Bandwidth for KDE. Can be:
            - "scott": Use Scott's rule (default)
            - float: Use specified bandwidth
        kernel: Kernel type. Currently only "rbf" is supported.

    Returns:
        Tuple of (kde_cal, kde_test) where each is a callable that takes
        embeddings and returns density estimates.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.covariate import fit_kde
        >>>
        >>> # Extract embeddings from your model
        >>> cal_embeddings = np.random.randn(100, 64)
        >>> test_embeddings = np.random.randn(50, 64)
        >>>
        >>> # Fit KDEs
        >>> kde_cal, kde_test = fit_kde(cal_embeddings, test_embeddings)
        >>>
        >>> # Use in CovariateLabel
        >>> from pyhealth.calib.predictionset.covariate import (
        ...     CovariateLabel)
        >>> predictor = CovariateLabel(
        ...     model=model,
        ...     alpha=0.1,
        ...     kde_cal=kde_cal,
        ...     kde_test=kde_test
        ... )
    """
    if kernel != "rbf":
        raise ValueError(f"Only 'rbf' kernel supported, got {kernel}")

    # Calculate bandwidth if needed (embeddings may be 1D or 3D; flatten to (n_samples, n_features))
    def get_bandwidth(embeddings, bw):
        if isinstance(bw, str):
            emb = np.asarray(embeddings)
            if emb.ndim == 1:
                emb = emb.reshape(-1, 1)
            else:
                emb = emb.reshape(emb.shape[0], -1)
            n_samples, n_features = emb.shape
            if bw == "scott":
                return n_samples ** (-1.0 / (n_features + 4))
            else:
                raise ValueError(f"Unknown bandwidth method: {bw}")
        return bw

    # Convert to torch tensors; flatten to (n_samples, n_features) for KDE
    def _flatten_emb(emb):
        emb = np.asarray(emb)
        if emb.ndim == 1:
            return emb.reshape(-1, 1)
        return emb.reshape(emb.shape[0], -1)

    cal_emb_2d = _flatten_emb(cal_embeddings)
    test_emb_2d = _flatten_emb(test_embeddings)
    n_cal, n_test = cal_emb_2d.shape[0], test_emb_2d.shape[0]
    print(f"  Calibration embeddings: {n_cal} x {cal_emb_2d.shape[1]}, test: {n_test} x {test_emb_2d.shape[1]}")
    cal_emb_torch = torch.from_numpy(cal_emb_2d).float()
    test_emb_torch = torch.from_numpy(test_emb_2d).float()

    # Fit KDE on calibration embeddings
    print("  Computing bandwidth and building calibration KDE...")
    cal_bw = get_bandwidth(cal_embeddings, bandwidth)
    kern_cal = RBFKernelMean(h=cal_bw)

    # Fit KDE on test embeddings
    print("  Computing bandwidth and building test KDE...")
    test_bw = get_bandwidth(test_embeddings, bandwidth)
    kern_test = RBFKernelMean(h=test_bw)

    # Create callable functions that compute density (flatten to 2D so 3D embeddings work)
    def _to_2d_tensor(data):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        else:
            data = data.reshape(data.shape[0], -1)
        return torch.from_numpy(data).float()

    def kde_cal(data):
        """Compute density using calibration KDE."""
        data = _to_2d_tensor(data)
        with torch.no_grad():
            K = kern_cal(data, cal_emb_torch)  # (n_query, n_cal)
            density = K.mean(dim=1)  # Average over calibration points
        return density.numpy()

    def kde_test(data):
        """Compute density using test KDE."""
        data = _to_2d_tensor(data)
        with torch.no_grad():
            K = kern_test(data, test_emb_torch)  # (n_query, n_test)
            density = K.mean(dim=1)  # Average over test points
        return density.numpy()

    return kde_cal, kde_test


def _compute_likelihood_ratio(
    kde_test: Callable, kde_cal: Callable, data: np.ndarray
) -> np.ndarray:
    """Compute likelihood ratio for covariate shift correction.

    Args:
        kde_test: Density estimator fitted on test distribution
        kde_cal: Density estimator fitted on calibration distribution
        data: Input data to compute likelihood ratio for

    Returns:
        Likelihood ratios (test density / calibration density)
    """
    test_density = kde_test(data)
    cal_density = kde_cal(data)
    # Add small epsilon to avoid division by zero
    return test_density / (cal_density + 1e-10)


def _query_weighted_quantile(
    scores: np.ndarray, alpha: float, weights: np.ndarray
) -> float:
    """Compute weighted quantile of scores.

    Args:
        scores: Array of conformity scores
        alpha: Quantile level (between 0 and 1)
        weights: Weights for each score

    Returns:
        The weighted alpha-quantile of scores
    """
    # Sort scores and corresponding weights
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    # Find the index where cumulative weight exceeds alpha
    idx = np.searchsorted(cum_weights, alpha, side="left")

    # Handle edge cases
    if idx >= len(sorted_scores):
        idx = len(sorted_scores) - 1

    return sorted_scores[idx]


class CovariateLabel(SetPredictor):
    """Covariate shift adaptive conformal prediction for multiclass.

    This extends the LABEL method to handle covariate shift between
    calibration and test distributions using likelihood ratio weighting.
    The method maintains coverage guarantees under covariate shift by
    reweighting calibration examples according to the likelihood ratio
    between test and calibration densities.

    The default KDE-based approach follows the CoDrug method (Laghuvarapu et al.,
    NeurIPS 2023), which uses kernel density estimation on embeddings to compute
    likelihood ratios. Alternatively, users can provide custom weights directly
    for more flexibility (e.g., from importance sampling, propensity scores, or
    domain-specific methods).

    Papers:
        Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and
        Aaditya Ramdas. "Conformal prediction under covariate shift."
        Advances in neural information processing systems 32 (2019).
        https://arxiv.org/abs/1904.06019
        
        Laghuvarapu, Siddhartha, Zhen Lin, and Jimeng Sun.
        "Conformal Drug Property Prediction with Density Estimation under 
        Covariate Shift." NeurIPS 2023.
        https://arxiv.org/abs/2310.12033

    Args:
        model: A trained base model
        alpha: Target mis-coverage rate(s). Can be:
            - float: marginal coverage P(Y not in C(X)) <= alpha
            - array: class-conditional P(Y not in C(X) | Y=k) <= alpha[k]
        kde_test: Optional density estimator fitted on test distribution.
            Should be a callable that takes embeddings (numpy array) and
            returns density estimates. Can be obtained via fit_kde().
            Used for KDE-based likelihood ratio weighting (CoDrug approach).
        kde_cal: Optional density estimator fitted on calibration
            distribution. Should be a callable that takes embeddings
            (numpy array) and returns density estimates.
            Used for KDE-based likelihood ratio weighting (CoDrug approach).
        debug: Whether to use debug mode (processes fewer samples for
            faster iteration)

    Examples:
        **Example 1: KDE-based approach (CoDrug method)**
        
        >>> from pyhealth.datasets import ISRUCDataset
        >>> from pyhealth.datasets import split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset.covariate import (
        ...     CovariateLabel, fit_kde)
        >>> import numpy as np
        >>>
        >>> # Prepare data
        >>> sleep_ds = ISRUCDataset("/data/ISRUC-I").set_task(
        ...     sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(
        ...     sleep_ds, [0.6, 0.2, 0.2])
        >>>
        >>> # Train model
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...                  label_key="label", mode="multiclass")
        >>> # ... training code ...
        >>>
        >>> # Extract embeddings (example - adjust for your model)
        >>> def extract_embeddings(model, dataset):
        ...     loader = get_dataloader(dataset, batch_size=32,
        ...         shuffle=False)
        ...     all_embs = []
        ...     for batch in loader:
        ...         batch['embed'] = True
        ...         output = model(**batch)
        ...         all_embs.append(output['embed'].cpu().numpy())
        ...     return np.concatenate(all_embs, axis=0)
        >>>
        >>> cal_embs = extract_embeddings(model, val_data)
        >>> test_embs = extract_embeddings(model, test_data)
        >>>
        >>> # KDE-based approach: automatically compute weights
        >>> cal_model = CovariateLabel(model, alpha=0.1)
        >>> cal_model.calibrate(cal_dataset=val_data,
        ...     cal_embeddings=cal_embs, test_embeddings=test_embs)
        >>>
        >>> # Evaluate
        >>> test_dl = get_dataloader(test_data, batch_size=32,
        ...     shuffle=False)
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(
        ...     model=cal_model).inference(test_dl,
        ...     additional_outputs=['y_predset'])
        >>> print(get_metrics_fn(cal_model.mode)(
        ...     y_true_all, y_prob_all, metrics=['accuracy', 'miscoverage_ps'],
        ...     y_predset=extra_output['y_predset']))
        
        **Example 2: Custom weights approach**
        
        >>> # If you have your own covariate shift correction method
        >>> # (e.g., importance sampling, propensity scores, etc.)
        >>> def compute_custom_weights(cal_data, test_data):
        ...     # Your custom weight computation
        ...     # Should return weights proportional to p_test(x) / p_cal(x)
        ...     return weights  # shape: (n_cal,)
        >>>
        >>> custom_weights = compute_custom_weights(val_data, test_data)
        >>> cal_model = CovariateLabel(model, alpha=0.1)
        >>> cal_model.calibrate(cal_dataset=val_data, cal_weights=custom_weights)
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: Union[float, np.ndarray],
        kde_test: Optional[Callable] = None,
        kde_cal: Optional[Callable] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if model.mode not in ("multiclass", "binary"):
            raise NotImplementedError(
                "CovariateLabel only supports multiclass or binary classification"
            )

        self.mode = self.model.mode

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.device = model.device
        self.debug = debug

        # Store alpha
        if not isinstance(alpha, float):
            alpha = np.asarray(alpha)
        self.alpha = alpha

        # Store density estimators
        if kde_test is not None and kde_cal is not None:
            self.kde_test = kde_test
            self.kde_cal = kde_cal
        else:
            self.kde_test = None
            self.kde_cal = None

        # Will be set during calibration
        self.t = None
        self._sum_cal_weights = None

    def calibrate(
        self,
        cal_dataset: IterableDataset,
        cal_embeddings: Optional[np.ndarray] = None,
        test_embeddings: Optional[np.ndarray] = None,
        cal_weights: Optional[np.ndarray] = None,
    ):
        """Calibrate the thresholds with covariate shift correction.

        This method supports three approaches for handling covariate shift:
        
        1. **KDE-based (CoDrug approach)**: Provide cal_embeddings and 
           test_embeddings (and optionally kde_test/kde_cal). The method will
           use kernel density estimation to compute likelihood ratios.
           
        2. **Custom weights**: Directly provide cal_weights computed from your
           own covariate shift correction method (e.g., importance sampling,
           propensity scores, discriminator-based methods, etc.).
           
        3. **Pre-fitted KDEs**: Provide kde_test and kde_cal during initialization
           along with cal_embeddings here.

        Args:
            cal_dataset: Calibration set
            cal_embeddings: Optional pre-computed calibration embeddings
                of shape (n_cal, embedding_dim). If provided along with
                test_embeddings and KDEs are not set, will be used to
                compute likelihood ratios via KDE (CoDrug approach).
            test_embeddings: Optional pre-computed test embeddings
                of shape (n_test, embedding_dim). Used with cal_embeddings
                for KDE-based likelihood ratio computation.
            cal_weights: Optional custom weights for calibration samples
                of shape (n_cal,). If provided, these weights will be used
                directly instead of computing likelihood ratios via KDE.
                Weights should represent importance weights or likelihood ratios
                p_test(x) / p_cal(x). These will be normalized internally.

        Note:
            You must provide ONE of:
            1. cal_weights (custom weights), OR
            2. kde_test and kde_cal during initialization, OR
            3. cal_embeddings and test_embeddings here

        Examples:
            >>> # Approach 1: KDE-based (CoDrug)
            >>> model.calibrate(cal_dataset, cal_embeddings, test_embeddings)
            >>> 
            >>> # Approach 2: Custom weights (e.g., from importance sampling)
            >>> custom_weights = compute_importance_weights(cal_data, test_data)
            >>> model.calibrate(cal_dataset, cal_weights=custom_weights)
        """
        # Get predictions and true labels first
        cal_dataset_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            debug=self.debug,
        )

        y_prob = cal_dataset_dict["y_prob"]
        y_true = cal_dataset_dict["y_true"]
        N, K = y_prob.shape

        # Binary: model outputs (N, 1) for positive class; treat as K=2
        if K == 1:
            y_true = np.asarray(y_true).ravel().astype(np.intp)
            p1 = np.asarray(y_prob[:, 0], dtype=np.float64).ravel()
            conformity_scores = np.where(y_true == 1, p1, 1.0 - p1)
            K = 2
        else:
            y_true = np.asarray(y_true).ravel().astype(np.intp)
            conformity_scores = y_prob[np.arange(N), y_true]

        # Determine weights: either custom or KDE-based
        if cal_weights is not None:
            # Use custom weights provided by user
            if len(cal_weights) != N:
                raise ValueError(
                    f"cal_weights must have length {N} (size of calibration set), "
                    f"got {len(cal_weights)}"
                )
            likelihood_ratios = np.asarray(cal_weights, dtype=np.float64)
            print("Using custom calibration weights")
        else:
            # Use KDE-based approach (CoDrug method)
            # Check if we have KDEs
            if self.kde_test is None or self.kde_cal is None:
                if cal_embeddings is None or test_embeddings is None:
                    raise ValueError(
                        "Must provide ONE of:\n"
                        "  1. cal_weights (custom weights), OR\n"
                        "  2. kde_test and kde_cal during __init__, OR\n"
                        "  3. cal_embeddings and test_embeddings during calibrate()"
                    )

                # Fit KDEs if embeddings provided
                print("Fitting KDEs on provided embeddings (CoDrug approach)...")
                self.kde_cal, self.kde_test = fit_kde(cal_embeddings, test_embeddings)

            # Use provided embeddings or extract from calibration data
            if cal_embeddings is not None:
                X = cal_embeddings
            else:
                # KDEs should already be provided in this case
                # We just need to get the embeddings for likelihood ratio
                # This assumes the model outputs embeddings
                raise NotImplementedError(
                    "Automatic embedding extraction not yet supported. "
                    "Please provide cal_embeddings and test_embeddings."
                )

            # Compute likelihood ratios using KDE (can be slow for large N)
            n_cal = X.shape[0] if hasattr(X, "shape") else len(X)
            print(f"Computing likelihood ratios via KDE (evaluating on {n_cal} calibration points)...")
            likelihood_ratios = _compute_likelihood_ratio(
                self.kde_test, self.kde_cal, X
            )
            print("  Likelihood ratios computed.")

        # Normalize weights
        weights = likelihood_ratios / np.sum(likelihood_ratios)
        self._sum_cal_weights = np.sum(likelihood_ratios)

        # Conformity scores already set above (with binary handling)

        # Compute weighted quantile thresholds
        if isinstance(self.alpha, float):
            # Marginal coverage: single threshold
            t = _query_weighted_quantile(conformity_scores, self.alpha, weights)
        else:
            # Class-conditional coverage: one threshold per class
            t = []
            for k in range(K):
                mask = y_true == k
                if np.sum(mask) > 0:
                    class_scores = conformity_scores[mask]
                    class_weights = weights[mask]
                    # Renormalize class weights
                    class_weights = class_weights / np.sum(class_weights)
                    t_k = _query_weighted_quantile(
                        class_scores, self.alpha[k], class_weights
                    )
                else:
                    # If no calibration examples, use -inf (include all)
                    t_k = -np.inf
                t.append(t_k)

        self.t = torch.tensor(t, device=self.device)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation with prediction set construction.

        Returns:
            Dictionary with all results from base model, plus:
                - y_predset: Boolean tensor indicating which classes
                    are in the prediction set
        """
        pred = self.model(**kwargs)
        y_prob = pred["y_prob"]

        # Binary: expand (batch, 1) to (batch, 2) only for set construction; keep pred["y_prob"] as-is for metrics
        if y_prob.shape[-1] == 1:
            p1 = y_prob.squeeze(-1).clamp(0.0, 1.0)
            y_prob_2 = torch.stack([1.0 - p1, p1], dim=-1)
        else:
            y_prob_2 = y_prob

        # Broadcast threshold for (batch, K)
        th = self.t.to(device=y_prob_2.device, dtype=y_prob_2.dtype)
        if th.dim() >= 1 and th.numel() > 1:
            th = th.view(1, -1)
        pred["y_predset"] = y_prob_2 > th
        return pred


if __name__ == "__main__":
    """
    Demonstration of three approaches for covariate shift correction:
    1. Embeddings approach: Automatic KDE computation (CoDrug method)
    2. Pre-fitted KDEs approach: User provides KDE estimators
    3. Custom weights approach: User provides custom importance weights
    """
    from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn
    from pyhealth.trainer import Trainer, get_metrics_fn

    # Setup data and model
    sleep_ds = ISRUCDataset("/srv/local/data/trash", dev=True).set_task(
        sleep_staging_isruc_fn
    )
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])

    model = SparcNet(
        dataset=sleep_ds, feature_keys=["signal"], label_key="label", mode="multiclass"
    )
    # ... Train the model here ...

    # Helper function to extract embeddings (mock implementation)
    def extract_embeddings(model, dataset):
        """Extract embeddings from model for a dataset."""
        # In practice, you would do:
        # loader = get_dataloader(dataset, batch_size=32, shuffle=False)
        # all_embs = []
        # for batch in loader:
        #     batch['embed'] = True
        #     output = model(**batch)
        #     all_embs.append(output['embed'].cpu().numpy())
        # return np.concatenate(all_embs, axis=0)
        
        # For demo, return random embeddings
        n_samples = len(dataset)
        embedding_dim = 64
        return np.random.randn(n_samples, embedding_dim)

    print("=" * 80)
    print("APPROACH 1: Embeddings (Automatic KDE - CoDrug Method)")
    print("=" * 80)
    print("This approach automatically computes KDEs from embeddings.")
    print("Best for: When you have model embeddings and want automatic density estimation.\n")

    # Extract embeddings from calibration and test sets
    cal_embeddings = extract_embeddings(model, val_data)
    test_embeddings = extract_embeddings(model, test_data)

    # Create model and calibrate with embeddings
    cal_model_1 = CovariateLabel(model, alpha=0.1)
    cal_model_1.calibrate(
        cal_dataset=val_data,
        cal_embeddings=cal_embeddings,
        test_embeddings=test_embeddings
    )

    # Evaluate
    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true, y_prob, _, extra = Trainer(model=cal_model_1).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    metrics_1 = get_metrics_fn(cal_model_1.mode)(
        y_true, y_prob,
        metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra["y_predset"]
    )
    print(f"Results: {metrics_1}\n")

    print("=" * 80)
    print("APPROACH 2: Pre-fitted KDEs")
    print("=" * 80)
    print("This approach uses pre-computed KDE estimators.")
    print("Best for: When you want control over KDE parameters or reuse KDEs.\n")

    # Fit KDEs separately with custom parameters
    kde_cal, kde_test = fit_kde(
        cal_embeddings,
        test_embeddings,
        bandwidth=0.5,  # Custom bandwidth
        kernel="rbf"
    )

    # Create model with pre-fitted KDEs
    cal_model_2 = CovariateLabel(
        model,
        alpha=0.1,
        kde_test=kde_test,
        kde_cal=kde_cal
    )
    cal_model_2.calibrate(
        cal_dataset=val_data,
        cal_embeddings=cal_embeddings  # Still need embeddings for likelihood ratio computation
    )

    # Evaluate
    y_true, y_prob, _, extra = Trainer(model=cal_model_2).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    metrics_2 = get_metrics_fn(cal_model_2.mode)(
        y_true, y_prob,
        metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra["y_predset"]
    )
    print(f"Results: {metrics_2}\n")

    print("=" * 80)
    print("APPROACH 3: Custom Weights")
    print("=" * 80)
    print("This approach uses user-provided importance weights.")
    print("Best for: Alternative covariate shift methods (importance sampling,")
    print("          propensity scores, discriminator-based, domain-specific).\n")

    # Compute custom weights using your own method
    # Examples of custom weight computation:
    
    # Option A: Uniform weights (no covariate shift correction)
    custom_weights = np.ones(len(val_data))
    
    # Option B: Importance sampling weights (mock example)
    # In practice, you might use:
    # - Discriminator-based methods
    # - Propensity score matching
    # - Domain adaptation techniques
    # - Energy-based models
    # custom_weights = compute_importance_weights(val_data, test_data)
    
    # Option C: Exponential weights based on distance (mock example)
    # distances = compute_distribution_distances(val_data, test_data)
    # custom_weights = np.exp(-distances)
    
    print(f"Using custom weights (shape: {custom_weights.shape})")

    # Create model and calibrate with custom weights
    cal_model_3 = CovariateLabel(model, alpha=0.1)
    cal_model_3.calibrate(
        cal_dataset=val_data,
        cal_weights=custom_weights  # Provide weights directly
    )

    # Evaluate
    y_true, y_prob, _, extra = Trainer(model=cal_model_3).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    metrics_3 = get_metrics_fn(cal_model_3.mode)(
        y_true, y_prob,
        metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra["y_predset"]
    )
    print(f"Results: {metrics_3}\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Approach 1 (Embeddings):     ", metrics_1)
    print("Approach 2 (Pre-fitted KDEs):", metrics_2)
    print("Approach 3 (Custom Weights): ", metrics_3)
    print("\nAll three approaches are valid and can be chosen based on your needs!")
    print("- Use Approach 1 for simplicity with embeddings (CoDrug method)")
    print("- Use Approach 2 for fine-grained control over KDE parameters")
    print("- Use Approach 3 for alternative covariate shift correction methods")
