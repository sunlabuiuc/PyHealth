"""
Base Conformal Prediction (Split Conformal)

Standard split conformal prediction for multiclass classification without
covariate shift correction. 

This method constructs prediction sets with coverage guarantees by calibrating
score thresholds on a held-out calibration set.

Paper:
    Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer.
    "Algorithmic learning in a random world." Springer, 2005.
    
    Papadopoulos, Harris, Kostas Proedrou, Volodya Vovk, and Alex Gammerman.
    "Inductive confidence machines for regression." ECML 2002.
"""

from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["BaseConformal"]


def _query_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the alpha-quantile of scores for conformal prediction.

    Args:
        scores: Array of conformity scores
        alpha: Quantile level (between 0 and 1), typically the miscoverage rate

    Returns:
        The alpha-quantile of scores
    """
    scores = np.sort(scores)
    N = len(scores)
    # Use ceiling to get conservative coverage
    loc = int(np.ceil(alpha * (N + 1))) - 1
    return -np.inf if loc == -1 else scores[loc]


def _query_weighted_quantile(
    scores: np.ndarray, alpha: float, weights: np.ndarray
) -> float:
    """Compute weighted quantile of scores (e.g. for NCP or covariate shift).

    Args:
        scores: Array of conformity scores
        alpha: Quantile level (between 0 and 1)
        weights: Weights for each score (same length as scores)

    Returns:
        The weighted alpha-quantile of scores
    """
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    w_sum = np.sum(sorted_weights)
    if w_sum <= 0:
        return -np.inf
    cum_weights = np.cumsum(sorted_weights) / w_sum
    idx = np.searchsorted(cum_weights, alpha, side="left")
    if idx >= len(sorted_scores):
        idx = len(sorted_scores) - 1
    return float(sorted_scores[idx])


class BaseConformal(SetPredictor):
    """Base Conformal Prediction for multiclass classification.

    This implements standard split conformal prediction, which constructs
    prediction sets with distribution-free coverage guarantees. The method
    calibrates thresholds on a calibration set and uses them to construct
    prediction sets on test data.

    The method guarantees that:
    - For marginal coverage (alpha is float): P(Y not in C(X)) <= alpha
    - For class-conditional coverage (alpha is array): P(Y not in C(X) | Y=k) <= alpha[k]

    where C(X) denotes the prediction set for input X.

    Papers:
        Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer.
        "Algorithmic learning in a random world." Springer, 2005.

        Lei, Jing, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani,
        and Larry Wasserman. "Distribution-free predictive inference for
        regression." Journal of the American Statistical Association (2018).

    Args:
        model: A trained base model that outputs predicted probabilities
        alpha: Target miscoverage rate(s). Can be:
            - float: marginal coverage P(Y not in C(X)) <= alpha
            - array: class-conditional P(Y not in C(X) | Y=k) <= alpha[k]
        score_type: Type of conformity score to use. Options:
            - "aps": Adaptive Prediction Sets (default, uses probability scores)
            - "threshold": Simple threshold on probabilities
        debug: Whether to use debug mode (processes fewer samples)

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset.base_conformal import BaseConformal
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
        >>> # Create conformal predictor with marginal coverage
        >>> conformal_model = BaseConformal(model, alpha=0.1)
        >>> conformal_model.calibrate(cal_dataset=val_data)
        >>>
        >>> # Evaluate
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(
        ...     model=conformal_model).inference(test_dl,
        ...     additional_outputs=['y_predset'])
        >>> print(get_metrics_fn(conformal_model.mode)(
        ...     y_true_all, y_prob_all,
        ...     metrics=['accuracy', 'miscoverage_ps', 'avg_set_size'],
        ...     y_predset=extra_output['y_predset']))
        {'accuracy': 0.71, 'miscoverage_ps': 0.095, 'avg_set_size': 1.8}
        >>>
        >>> # Class-conditional coverage
        >>> conformal_model_cc = BaseConformal(
        ...     model, alpha=[0.1, 0.15, 0.1, 0.1, 0.1])
        >>> conformal_model_cc.calibrate(cal_dataset=val_data)
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: Union[float, np.ndarray],
        score_type: str = "aps",
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if model.mode not in ("multiclass", "binary"):
            raise NotImplementedError(
                "BaseConformal only supports multiclass or binary classification"
            )

        self.mode = self.model.mode

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.device = model.device
        self.debug = debug
        self.score_type = score_type

        # Store alpha
        if not isinstance(alpha, float):
            alpha = np.asarray(alpha)
        self.alpha = alpha

        # Will be set during calibration
        self.t = None

    def _compute_conformity_scores(
        self, y_prob: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """Compute conformity scores from predictions and true labels.

        Args:
            y_prob: Predicted probabilities of shape (N, K)
            y_true: True class labels of shape (N,)

        Returns:
            Conformity scores of shape (N,)
        """
        N = len(y_true)
        # Ensure integer indices (y_true can be float e.g. 0.0/1.0 from binary tasks)
        y_true = np.asarray(y_true, dtype=np.int64)
        if self.score_type == "aps" or self.score_type == "threshold":
            # Use probability of true class as conformity score
            # Higher score = more conforming (better prediction)
            scores = y_prob[np.arange(N), y_true]
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

        return scores

    def calibrate(self, cal_dataset: IterableDataset):
        """Calibrate the thresholds for prediction set construction.

        Args:
            cal_dataset: Calibration set (held-out validation data)
        """
        # Get predictions and true labels
        cal_dataset_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            debug=self.debug,
        )

        y_prob = cal_dataset_dict["y_prob"]
        y_true = cal_dataset_dict["y_true"]
        N, K = y_prob.shape

        # Compute conformity scores
        conformity_scores = self._compute_conformity_scores(y_prob, y_true)

        # Compute quantile thresholds
        if isinstance(self.alpha, float):
            # Marginal coverage: single threshold
            t = _query_quantile(conformity_scores, self.alpha)
        else:
            # Class-conditional coverage: one threshold per class
            if len(self.alpha) != K:
                raise ValueError(
                    f"alpha must have length {K} for class-conditional "
                    f"coverage, got {len(self.alpha)}"
                )
            t = []
            for k in range(K):
                mask = y_true == k
                if np.sum(mask) > 0:
                    class_scores = conformity_scores[mask]
                    t_k = _query_quantile(class_scores, self.alpha[k])
                else:
                    # If no calibration examples, use -inf (include all)
                    print(
                        f"Warning: No calibration examples for class {k}, "
                        "using -inf threshold"
                    )
                    t_k = -np.inf
                t.append(t_k)

        self.t = torch.tensor(t, device=self.device)

        if self.debug:
            print(f"Calibrated thresholds: {self.t}")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation with prediction set construction.

        Returns:
            Dictionary with all results from base model, plus:
                - y_predset: Boolean tensor indicating which classes
                    are in the prediction set
        """
        if self.t is None:
            raise RuntimeError(
                "Model must be calibrated before inference. "
                "Call calibrate() first."
            )

        pred = self.model(**kwargs)

        # Construct prediction set by thresholding probabilities
        # Include classes with probability >= threshold
        pred["y_predset"] = pred["y_prob"] >= self.t

        return pred


if __name__ == "__main__":
    # Example usage
    from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn

    sleep_ds = ISRUCDataset("/srv/local/data/trash", dev=True).set_task(
        sleep_staging_isruc_fn
    )
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])

    model = SparcNet(
        dataset=sleep_ds,
        feature_keys=["signal"],
        label_key="label",
        mode="multiclass",
    )

    # Marginal coverage
    conformal_model = BaseConformal(model, alpha=0.1)
    conformal_model.calibrate(cal_dataset=val_data)

    # Evaluate
    from pyhealth.trainer import Trainer, get_metrics_fn

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true_all, y_prob_all, _, extra_output = Trainer(
        model=conformal_model
    ).inference(test_dl, additional_outputs=["y_predset"])

    print(
        get_metrics_fn(conformal_model.mode)(
            y_true_all,
            y_prob_all,
            metrics=["accuracy", "miscoverage_ps"],
            y_predset=extra_output["y_predset"],
        )
    )

