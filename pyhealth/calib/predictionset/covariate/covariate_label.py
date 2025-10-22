"""
Covariate Shift Adaptive Conformal Prediction.

This module implements conformal prediction with covariate shift correction
using likelihood ratio weighting based on density estimation.

Paper:
    Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and
    Aaditya Ramdas. "Conformal prediction under covariate shift."
    Advances in neural information processing systems 32 (2019).
"""

from typing import Callable, Dict, Union

import numpy as np
import torch
from torch.utils.data import Subset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["CovariateLabel"]


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

    Paper:
        Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and
        Aaditya Ramdas. "Conformal prediction under covariate shift."
        Advances in neural information processing systems 32 (2019).

    Args:
        model: A trained base model
        alpha: Target mis-coverage rate(s). Can be:
            - float: marginal coverage P(Y not in C(X)) <= alpha
            - array: class-conditional P(Y not in C(X) | Y=k) <= alpha[k]
        kde_test: Density estimator fitted on test distribution. Should
            be a callable that takes data and returns density estimates.
        kde_cal: Density estimator fitted on calibration distribution.
            Should be a callable that takes data and returns estimates.
        data_key: Key to extract input data for density estimation
            (default: "signal")
        debug: Whether to use debug mode (processes fewer samples for
            faster iteration)

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset
        >>> from pyhealth.datasets import split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset.covariate import (
        ...     CovariateLabel)
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
        >>> # Fit density estimators (you need to provide these)
        >>> # kde_cal = fit_kde(val_data)
        >>> # kde_test = fit_kde(test_data)
        >>>
        >>> # Create covariate-adaptive set predictor
        >>> cal_model = CovariateLabel(model, alpha=0.1,
        ...     kde_test=kde_test, kde_cal=kde_cal)
        >>> cal_model.calibrate(cal_dataset=val_data)
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
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: Union[float, np.ndarray],
        kde_test: Callable,
        kde_cal: Callable,
        data_key: str = "signal",
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if model.mode != "multiclass":
            raise NotImplementedError(
                "CovariateLabel only supports multiclass classification"
            )

        self.mode = self.model.mode

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.device = model.device
        self.debug = debug
        self.data_key = data_key

        # Store alpha
        if not isinstance(alpha, float):
            alpha = np.asarray(alpha)
        self.alpha = alpha

        # Store density estimators
        self.kde_test = kde_test
        self.kde_cal = kde_cal

        # Will be set during calibration
        self.t = None
        self._sum_cal_weights = None

    def calibrate(self, cal_dataset: Subset):
        """Calibrate the thresholds with covariate shift correction.

        Args:
            cal_dataset: Calibration set
        """
        # Get predictions and true labels
        cal_dataset_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            incl_data_keys=[self.data_key],
            debug=self.debug,
        )

        y_prob = cal_dataset_dict["y_prob"]
        y_true = cal_dataset_dict["y_true"]
        X = cal_dataset_dict[self.data_key]

        N, K = y_prob.shape

        # Compute likelihood ratios for covariate shift correction
        likelihood_ratios = _compute_likelihood_ratio(self.kde_test, self.kde_cal, X)

        # Normalize weights
        weights = likelihood_ratios / np.sum(likelihood_ratios)
        self._sum_cal_weights = np.sum(likelihood_ratios)

        # Extract conformity scores (probabilities of true class)
        conformity_scores = y_prob[np.arange(N), y_true]

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

        # Construct prediction set by thresholding probabilities
        pred["y_predset"] = pred["y_prob"] > self.t

        return pred


if __name__ == "__main__":
    # Example usage (requires actual density estimators)
    from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn

    sleep_ds = ISRUCDataset("/srv/local/data/trash", dev=True).set_task(
        sleep_staging_isruc_fn
    )
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])

    model = SparcNet(
        dataset=sleep_ds, feature_keys=["signal"], label_key="label", mode="multiclass"
    )

    # Note: In practice, you would fit proper KDE estimators here
    # For demonstration, using dummy estimators
    def dummy_kde(data):
        return np.ones(len(data))

    cal_model = CovariateLabel(model, alpha=0.1, kde_test=dummy_kde, kde_cal=dummy_kde)
    cal_model.calibrate(cal_dataset=val_data)

    # Evaluate
    from pyhealth.trainer import Trainer, get_metrics_fn

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    print(
        get_metrics_fn(cal_model.mode)(
            y_true_all,
            y_prob_all,
            metrics=["accuracy", "miscoverage_ps"],
            y_predset=extra_output["y_predset"],
        )
    )
