from typing import Dict, List, Optional, Tuple 
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel


class CaliForest(BaseModel):
    """ 
    CaliForest: A random forest model, calibrated using out-of-bag samples for healthcare prediction tasks.

    Paper: Y. Park and J. C. Ho. CaliForest: Calibrated Random Forest for Health Data. 
    ACM Conference on Health, Inference, and Learning 2020.

    Args:
        dataset: Dataset to train the model. 
        mode: One of "binary", "multiclass", or "multilabel".
        n_estimators: Number of trees in the forest. Default is 300.
        criterion: Function to measure the quality of a split. Default is "gini".
        max_depth: Maximum depth of the tree. Default is 5.
        min_samples_split: Minimum number of samples required to split a node. Default is 2.
        min_samples_leaf: Minimum number of samples required at a leaf node. Default is 1.
        ctype: Calibration type, one of "isotonic" or "logistic". Default is "isotonic".
        alpha0: Prior alpha parameter for variance estimation. Default is 100.
        beta0: Prior beta parameter for variance estimation. Default is 25.
    """
    
    def __init__(
        self,
        dataset: SampleEHRDataset,
        n_estimators: int = 300,
        criterion: str = "gini",
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        ctype: str = "isotonic",
        alpha0: float = 100,
        beta0: float = 25,
        **kwargs
    ):
        super(CaliForest, self).__init__(
            dataset=dataset,
            # TODO:
            # does this still need feature_keys, label_keys etc?
            # or is that moved to the dataset now
            # saw some other models still have it ..?
        )
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0
        
        # initialize model components
        self.estimators = []
        self.calibrator = None
        self.is_fitted_ = False
        
        # does this class need a dummy parameter to register with PyTorch (?)
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def _prepare_input(self, **kwargs) -> Tuple[np.ndarray, torch.Tensor]:
        """Prepare input features and labels from batch data.
        
        This method processes the batch data and formats it for scikit-learn models.
        
        Args:
            **kwargs: Batch data.
                
        Returns:
            Tuple of (features, labels), where features is a numpy array of shape (batch_size, feature_dim),
            and labels is a tensor of shape (batch_size, num_classes) or (batch_size,).
        """
        features = []
        
        # process each feature key
        for key in self.feature_keys:
            # get the feature values from the batch
            values = kwargs.get(key, None)
            
            if values is None:
                raise ValueError(f"Feature key {key} not found in input data")
            
            # handle different feature types based on their structure
            if isinstance(values, torch.Tensor):
                # for tensors, flatten all but the first dimension
                batch_size = values.size(0)
                values = values.view(batch_size, -1).float()
                features.append(values.detach().cpu().numpy())
            elif isinstance(values, list):
                # for lists, convert to numpy arrays
                values_np = np.array(values)
                if values_np.ndim == 1:
                    values_np = values_np.reshape(-1, 1)
                features.append(values_np)
            else:
                raise ValueError(f"Unsupported feature type for {key}: {type(values)}")
        
        # concatenate all features into a single numpy array
        if len(features) == 1:
            X = features[0]
        else:
            # ensure all feature arrays have the same first dimension
            batch_sizes = [f.shape[0] for f in features]
            if len(set(batch_sizes)) > 1:
                raise ValueError(f"Inconsistent batch sizes across features: {batch_sizes}")
            
            # concatenate along the feature dimension
            X = np.concatenate(features, axis=1)
        
        # process labels
        labels = kwargs.get(self.label_keys, None)
        if labels is None:
            raise ValueError(f"Label key {self.label_keys} not found in input data")
        
        if isinstance(labels, torch.Tensor):
            y = labels.detach().cpu()
        else:
            y = torch.tensor(labels)
        
        return X, y
        
    def _fit(self, X, y):
        """Fit the CaliForest model.
        
        Args:
            X: Input features as numpy array.
            y: Target labels as numpy array.
            
        Returns:
            self: Fitted model.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # initialize model components
        self.estimators = []
        for i in range(self.n_estimators):
            self.estimators.append(DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features="auto"
            ))
            
        # initialize calibrator based on the chosen type
        if self.ctype == "logistic":
            self.calibrator = LogisticRegression(
                penalty="none",
                solver="saga",
                max_iter=5000
            )
        elif self.ctype == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0,
                y_max=1,
                out_of_bounds="clip"
            )
        else:
            raise ValueError(f"Calibration type '{self.ctype}' not supported. Use 'logistic' or 'isotonic'.")
        
        # prepare matrices for out-of-bag sampling
        n, m = X.shape
        Y_oob = np.full((n, self.n_estimators), np.nan)
        n_oob = np.zeros(n)
        IB = np.zeros((n, self.n_estimators), dtype=int)
        OOB = np.full((n, self.n_estimators), True)
        
        # generate bootstrap indices for each estimator
        for eid in range(self.n_estimators):
            IB[:, eid] = np.random.choice(n, n)
            OOB[IB[:, eid], eid] = False
        
        # fit each estimator and collect out-of-bag predictions
        for eid, est in enumerate(self.estimators):
            ib_idx = IB[:, eid]
            oob_idx = OOB[:, eid]
            est.fit(X[ib_idx, :], y[ib_idx])
            if np.any(oob_idx):
                Y_oob[oob_idx, eid] = est.predict_proba(X[oob_idx, :])[:, 1]
                n_oob[oob_idx] += 1
        
        # use out-of-bag predictions to calibrate
        oob_idx = n_oob > 1
        if np.any(oob_idx):
            Y_oob_ = Y_oob[oob_idx, :]
            n_oob_ = n_oob[oob_idx]
            z_hat = np.nanmean(Y_oob_, axis=1)
            z_true = y[oob_idx]

            # compute weights for calibration
            beta = self.beta0 + np.nanvar(Y_oob_, axis=1) * n_oob_ / 2
            alpha = self.alpha0 + n_oob_ / 2
            z_weight = alpha / beta

            # fit calibrator
            if self.ctype == "logistic":
                self.calibrator.fit(z_hat[:, np.newaxis], z_true, z_weight)
            elif self.ctype == "isotonic":
                self.calibrator.fit(z_hat, z_true, z_weight)
            
        self.is_fitted_ = True
        return self
    
    def _predict_proba(self, X):
        """Predict class probabilities for input features.
        
        Args:
            X: Input features as numpy array.
            
        Returns:
            Predicted class probabilities.
        """
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        
        n, m = X.shape
        n_est = len(self.estimators)
        z = np.zeros(n)
        y_mat = np.zeros((n, 2))
        
        # average predictions from all estimators
        for eid, est in enumerate(self.estimators):
            z += est.predict_proba(X)[:, 1]
        z /= n_est
        
        # apply calibration
        if self.ctype == "logistic":
            y_mat[:, 1] = self.calibrator.predict_proba(z[:, np.newaxis])[:, 1]
        elif self.ctype == "isotonic":
            y_mat[:, 1] = self.calibrator.predict(z)
            
        y_mat[:, 0] = 1 - y_mat[:, 1]
        return y_mat
    
    def _predict(self, X):
        """Predict class labels for input features.
        
        Args:
            X: Input features as numpy array.
            
        Returns:
            Predicted class labels.
        """
        proba = self._predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        This method processes the input data, performs model training or inference,
        and returns the results.
        
        Args:
            **kwargs: Input data containing features and labels.
            
        Returns:
            Dict with keys:
                loss: The loss value as a tensor.
                y_prob: Predicted probabilities.
                y_true: True labels.
                logit: Raw model outputs before activation.
        """
        # prepare input features and labels
        X, y_true = self._prepare_input(**kwargs)
        
        # if in training mode and not yet fitted, fit the model
        if self.training and not self.is_fitted_:
            self._fit(X, y_true.numpy().ravel())
        
        # make predictions
        if not self.is_fitted_:
            # return default values if model is not fitted yet
            batch_size = X.shape[0]
            
            if self.mode == "binary":
                y_prob = torch.ones((batch_size, 1)) * 0.5
                logits = torch.zeros((batch_size, 1))
            else:  # multiclass
                num_classes = len(torch.unique(y_true))
                y_prob = torch.ones((batch_size, num_classes)) / num_classes
                logits = torch.zeros((batch_size, num_classes))
                
        else:
            # get predictions from fitted model
            if self.mode == "binary":
                y_prob_np = self._predict_proba(X)[:, 1:2]  # only positive class
                logits = torch.tensor(np.log(y_prob_np / (1 - y_prob_np + 1e-10))).float()  # convert to logits with small epsilon
                y_prob = torch.tensor(y_prob_np).float()
            else:  # multiclass
                y_prob_np = self._predict_proba(X)
                logits = torch.tensor(np.log(y_prob_np + 1e-10)).float()  # add small epsilon to avoid log(0)
                y_prob = torch.tensor(y_prob_np).float()
        
        # calculate loss
        if self.mode == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y_true.float())
        elif self.mode == "multilabel":
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y_true.float())
        else:  # multiclass
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y_true.long().squeeze())
        
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
