"""
Provides an implementation of the Random Forest model that is compatible with PyHealth
pipelines.
"""
import logging
from itertools import product
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch import nn
from torch.utils.data import DataLoader

from pyhealth.datasets import SampleEHRDataset
from pyhealth.metrics import binary_metrics_fn, regression_metrics_fn
from pyhealth.models import BaseModel
from pyhealth.models.utils import DataLoaderToNumpy

logger = logging.getLogger(__name__)


class RandomForest(BaseModel):
    """Random Forest model.

    Wraps sklearn's RandomForestClassifier for classification tasks and
    sklearn's RandomForestRegressor for regression tasks.

    Args:
        dataset: dataset to train the model.
        pad_batches: whether to pad batches such that their dimensions are equal.
            Default is True. If False, a ValueError will be raised if padding is
            needed and dimensions are not equal.
        f1_average: averaging strategy for F1 in evaluate(). Can be "macro",
            "weighted", or "binary". Default is "macro".
        n_estimators: number of trees in the forest. Default is 100.
        criterion: function to measure the quality of a split. For classification
            tasks this can be either: "gini", "entropy", "log_loss". For regression,
            this can be either: "friedman_mse", "squared_error", "absolute_error",
            "poisson". If None, criterion is set to "gini" for classification tasks
            and "friedman_mse" for regression tasks.
        max_depth: maximum depth of the tree. If None, nodes are expanded until all
            leaves are pure or contain fewer than min_samples_split samples. Default
            is None.
        min_samples_split: minimum number of samples required to split an internal node.
            Default is 2.
        min_samples_leaf: minimum number of samples required to be at a leaf node.
            Default is 1.
        min_weight_fraction_leaf: minimum weighted fraction of the sum total of weights
            required to be at a leaf node. Default is 0.0.
        max_features: number of features to consider when looking for the best split.
            Can be {"sqrt", "log2", None}, int or float. Default is "sqrt".
        max_leaf_nodes: grow trees with max_leaf_nodes in best-first fashion.
            If None, then unlimited number of leaf nodes. Default is None.
        min_impurity_decrease: A node will be split if this split induces a decrease of
            the impurity greater than or equal to this value. Default is 0.0.
        bootstrap: whether bootstrap samples are used when building trees.
            Default is True.
        oob_score: whether to use out-of-bag samples to estimate the generalization
            score. Default is False.
        n_jobs: number of jobs to run in parallel. None means 1 unless in a joblib
            parallel backend context. -1 uses all converters Default is None.
        random_state: controls randomness. Default is 42.
        verbose: controls verbosity when fitting and predicting. Default is 0.
        warm_start: reuse the solution of the previous call to fit and add more
            estimators to the ensemble. Default is False.
        class_weight: weights associated with classes. Can be {“balanced”,
            “balanced_subsample”}, dict or list of dicts,
            Default is None.
        ccp_alpha: complexity parameter used for Minimal Cost-Complexity Pruning.
            Default is 0.0.
        max_samples: If bootstrap is True, the number of samples to draw from X to
            train each base estimator. Default = None.
        monotonic_cst: Indicates the monotonicity constraint to enforce on each
            feature. 1: monotonic increase, 0: no constraint, -1: monotonic decrease.
            If monotonic_cst is None, no constraints are applied. Default is None.
        **kwargs: Additional arguments passed to pass to a deeper layer

    Raises:
         ValueError, TypeError: If invalid model parameters are found.
    """
    # Criterion Constants
    GINI = "gini"
    FRIEDMAN_MSE = "friedman_mse"
    CLASSIFICATION_CRITERIA = {"gini", "entropy", "log_loss"}
    REGRESSION_CRITERIA = {"squared_error", "friedman_mse", "poisson",
                           "absolute_error"}

    # F1 Avg Constants
    MACRO = "macro"
    WEIGHTED = "weighted"
    BINARY = "binary"
    VALID_F1_AVG = {MACRO, WEIGHTED, BINARY}

    # Key Constants
    KEY_Y_TRUE = "y_true"
    KEY_Y_PROB = "y_prob"
    KEY_LOGIT = "logit"
    KEY_LOSS = "loss"
    METRIC_ACCURACY = "accuracy"
    METRIC_F1 = "f1"
    METRIC_AUROC = "roc_auc"
    METRIC_MSE = "mse"
    METRIC_MAE = "mae"

    def __init__(
            self,
            dataset: SampleEHRDataset,
            pad_batches: bool = True,
            f1_average: str = "macro",
            n_estimators: int = 100,
            criterion: str = None,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: Union[str, int, float, None] = "sqrt",
            max_leaf_nodes: Optional[int] = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: bool = False,
            n_jobs: Optional[int] = None,
            random_state: Optional[int] = 42,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Optional[str] = None,
            ccp_alpha: float = 0.0,
            max_samples: Optional[int] = None,
            **kwargs,
    ):
        # Call base constructor
        super(RandomForest, self).__init__(dataset = dataset)

        # Save off inputs
        self.pad_batches = pad_batches
        self._is_fitted = False
        self.feature_dim = None
        self.f1_average = f1_average

        # Validate label keys
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        # Verify f1_average is valid
        if not isinstance(self.f1_average, str):
            raise TypeError("Input f1_average must be a string.")
        if self.f1_average.lower() not in self.VALID_F1_AVG:
            raise ValueError("Input f1_average unsupported.")

        # If no criterion is given, set the criterion based on the task whether it is
        # binary or regression
        if criterion is None:
            if self.mode != "regression":
                criterion = self.GINI
            else:
                criterion = self.FRIEDMAN_MSE

        # Validate criterion
        if self.mode != "regression":
            valid_criterion = RandomForest.CLASSIFICATION_CRITERIA
        else:
            valid_criterion = RandomForest.REGRESSION_CRITERIA

        if criterion not in valid_criterion:
            raise ValueError("Input criterion unsupported.")

        # Create a numpy converter since sklearn expects numpy arrays
        self.converter = DataLoaderToNumpy(
            feature_keys = self.feature_keys,
            label_key = self.label_key,
            pad_batches = pad_batches,
        )

        # Store parameters into a dictionary that we can pass through whether we
        # instantiate a classifier or regressor sklearn model based on the mode
        hyperparams = dict(
            n_estimators = n_estimators,
            criterion = criterion,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features,
            max_leaf_nodes = max_leaf_nodes,
            min_impurity_decrease = min_impurity_decrease,
            bootstrap = bootstrap,
            oob_score = oob_score,
            n_jobs = n_jobs,
            random_state = random_state,
            verbose = verbose,
            warm_start = warm_start,
            ccp_alpha = ccp_alpha,
            max_samples = max_samples,
        )

        # Create internal models
        try:
            if self.mode == "regression":
                self.model = RandomForestRegressor(**hyperparams)
            else:
                # RandomForestClassifier also includes class weight unlike regressor
                self.model = RandomForestClassifier(
                    class_weight = class_weight,
                    **hyperparams)
        except TypeError:
            raise TypeError("Invalid model parameters")

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward propagation (inference).

        Args:
            **kwargs: A variable number of keyword arguments representing input
             features. Each keyword argument is a tensor or a tuple of tensors of
             shape (batch_size, ...).

        Returns:
             a dictionary containing keys loss, y_prob, y_true, logit

        Raises:
            RuntimeError: if fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted.")

        x, y_np = self.converter.transform([kwargs])

        return self._generate_inference_output(self._predict_numpy(x), y_np)

    def fit(self, dataloader: DataLoader):
        """Fit the Random Forest model to the dataloader

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            self
        """
        # Get numpy matrices for use with sklearn
        x, y = self.converter.transform(dataloader)

        # Fit the model and flag that the model has been fitted
        self.model.fit(x, y)

        # Track that the model is fitted
        self._is_fitted = True

        return self

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Optional[float]]:
        """Evaluates the model and returns calculated metrics.

        Args:
            dataloader: PyTorch DataLoader for evaluation (test dataset)

        Returns:
            Dictionary of metrics with different keys depending on the mode:
                - classification: 'accuracy', 'f1', 'auroc'. auroc may be None.
                - regression: 'mse', 'mae'
        Raises:
            RuntimeError: if model has not been fitted.
            ValueError: Error during metric computation.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted.")

        # Get numpy matrices for use with sklearn
        x, y_true = self.converter.transform(dataloader)
        y_prob_np = self._predict_numpy(x)

        results = {}
        try:
            if self.mode != "regression":
                # use pyhealth metric utils
                results = binary_metrics_fn(
                    y_true = y_true,
                    y_prob = y_prob_np[:, 1],
                    metrics = ["roc_auc", "f1", "accuracy"],
                )
            else:
                # use pyhealth metric utils
                results = regression_metrics_fn(
                    x = y_true.copy(),
                    x_rec = y_prob_np.view(-1).reshape(-1),
                    metrics = ["mse", "mae"]
                )
        except ValueError:
            logger.warning("Failed to compute metrics.")

        return results

    def get_params(self) -> Dict[str, Any]:
        """Returns the model parameters. Wraps sklearn's get_params() and updates the
        dictionary with additional parameters used by this wrapper class. This is
        useful to verify the model has been initialized correctly.

        Returns:
             Dictionary of model parameters
        """
        params = self.model.get_params()
        params.update({
            "pad_batches": self.pad_batches,
            "f1_average": self.f1_average
        })

        return params

    def _calculate_loss(self, logit, y_true) -> torch.Tensor:
        """Compute loss between predictions and labels.

        Args:
            logit: Tensor outputs
            y_true: Tensor labels

        Returns:
            : loss tensor
        """

        if self.mode != "regression":
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logit, y_true.long().view(-1))
        else:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logit, y_true.float().view(-1, 1))

        return loss

    def _generate_inference_output(
            self,
            y_prob_np: np.ndarray,
            y_np: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Generates the inference output in the format expected by PyHealth

        Args:
            y_prob_np: numpy array of shape (n_samples, n_classes)
            y_np: numpy array of shape (n_samples,)

        Returns:
            Dictionary with keys loss, y_prob, y_true, logit
        """
        y_prob = torch.from_numpy(y_prob_np)
        y_true = torch.from_numpy(y_np)

        return {
            self.KEY_LOSS: self._calculate_loss(y_prob, y_true),
            self.KEY_Y_PROB: y_prob,
            self.KEY_Y_TRUE: y_true,
            self.KEY_LOGIT: y_prob,
        }

    def _predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """Run prediction using the internal sklearn model.

        Args:
            x: numpy array of shape (n_samples, n_features).

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        if isinstance(self.model, RandomForestClassifier):
            prediction_results = self.model.predict_proba(x).astype(np.float32)
        else:
            prediction_results = self.model.predict(x).astype(np.float32).reshape(-1, 1)

        return prediction_results

    @staticmethod
    def tune(
            dataset: SampleEHRDataset,
            train_loader: DataLoader,
            val_loader: DataLoader,
            param_grid: Dict[Any, Any],
            task: str = "classification",
            fixed_params: Optional[Dict[Any, Any]] = None,
            metric: str = "roc_auc",
            maximize: bool = True,
            return_all: bool = False,
    ) -> Union[
        tuple[Optional[dict[Any, Any]], Union[float, Any]],
        tuple[Optional[dict[Any, Any]], Union[float, Any], list],
    ]:
        """
        Performs a "grid search" over the given dictionary of model parameters and
        returns the best combination of parameters. Best meaning, the parameters that
        either maximized or minimized (according to the maximize parameters) the given
        matric.

        Args:
            dataset: PyHealth dataset object
            train_loader: Training dataloader
            val_loader: Validation dataloader
            param_grid: Dict of hyperparameters to search
            task: task type, either "classification" or "regression". Default is
                  "classification".
            fixed_params: Params that stay constant
            metric: Metric to optimize (e.g., 'roc_auc', 'mae')
                    maximize: 'max' or 'min'
            maximize: True to maximize the metric, False otherwise
            return_all: True to get back all parameter combination and metric results
                or False to receive only the best performing combination of parameters.
                Defaults to False.

        Returns:
            best_params: Dictionary of the best combination of parameters
            best_score:  Best metric
            results: list of all the parameter combinations and their metric result
            if return_all is enabled
        Raises:
            ValueError: if task or metric is not a string, or unsupported.
        """
        # Validate that the task is supported
        if not isinstance(task, str):
            raise TypeError("Input task is expected to be a string.")
        if task.lower() not in ["classification", "regression"]:
            raise ValueError("Input task unsupported.")

        # Define what metrics are supported based on the given task
        if task == "classification":
            valid_metrics = [RandomForest.METRIC_F1, RandomForest.METRIC_ACCURACY,
                             RandomForest.METRIC_AUROC]
            if not isinstance(metric, str) or metric not in valid_metrics:
                raise ValueError("Input metric unsupported.")
        else:
            valid_metrics = [RandomForest.METRIC_MSE, RandomForest.METRIC_MAE]
            if not isinstance(metric, str) or metric not in valid_metrics:
                raise ValueError("Input metric unsupported.")
        fixed_params = fixed_params or {}

        # Determine if we want to see the highest or lowest score form this metric
        if maximize:
            best_score = -float("inf")
        else:
            best_score = float("inf")
        best_params = None

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        results = []
        for parameter_combination in product(*values):
            params = dict(zip(keys, parameter_combination))

            # Gather parameters
            all_params = {**fixed_params, **params}

            # Create the model with the given params
            model = RandomForest(
                dataset = dataset,
                **all_params
            )

            # Fit the model to the training data
            model.fit(train_loader)

            # Calculate metrics
            metrics = model.evaluate(val_loader)

            # Get the specific metric of interest
            score = metrics[metric]

            results.append({
                "params": all_params,
                "score": score
            })

            # Keep track of the best score and params noting that we may either me
            # maximizing or minimizing
            if (score is not None
                    and ((maximize and score > best_score) or
                         (not maximize and score < best_score))):
                best_score = score
                best_params = all_params

        if return_all:
            tuning_results = best_params, best_score, results
        else:
            tuning_results = best_params, best_score

        return tuning_results
