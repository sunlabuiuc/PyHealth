from typing import Dict, List, Optional, Tuple

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
import numpy as np
from pyhealth.datasets.splitter import split_by_visit
import lightgbm as lgb
from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, auc, precision_recall_curve, roc_auc_score, cohen_kappa_score
import torch


class LightGBMClassifier(BaseModel):
    """A LightGBM classifier implementation for PyHealth framework.
    
    This model handles classification tasks using LightGBM with support for:
    - Binary and multiclass classification
    - Custom train/val/test splits
    - Comprehensive evaluation metrics
    - PyHealth dataset integration

    Args:
        dataset: The PyHealth dataset containing samples with features and labels.
        feature_keys: List of keys to extract features from samples (e.g., ["signal"]).
        label_key: Key containing the label in each sample (e.g., "label").
        mode: Classification mode, either "binary" or "multiclass".
        train_test_split: Optional list of ratios for train/val/test split (e.g., [0.7, 0.1, 0.2]).
        **kwargs: Additional LightGBM parameters (e.g., num_leaves, learning_rate).

    Attributes:
        feature_keys: List of feature keys used for model input.
        label_key: Key used for labels.
        train_test_split: Ratios for dataset splitting.
        params: LightGBM parameters including objective function.
        model: The trained LightGBM classifier instance.
        train_samples: Training samples after split.
        val_samples: Validation samples after split.
        test_samples: Test samples after split.

    Examples:
        >>> from pyhealth.models import LightGBMClassifier
        >>> model = LightGBMClassifier(
        ...     dataset=dataset,
        ...     feature_keys=["signal"],
        ...     label_key="label",
        ...     mode="binary",
        ...     train_test_split=[0.7, 0.1, 0.2],
        ...     num_leaves=31,
        ...     learning_rate=0.05
        ... )
        >>> model.fit()
        >>> results = model.evaluate()
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        train_test_split: List[float] = None,
        **kwargs
    ):
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.train_test_split = train_test_split
        self.train_samples, self.val_samples, self.test_samples = split_by_visit(dataset, train_test_split)

        super().__init__(
            dataset=dataset,
        )

        self.params = {
            "objective": "binary" if mode == "binary" else "multiclass",
            "verbose": -1,
            **kwargs
        }
        self.model = None

    def fit(self):
        """Trains the LightGBM classifier using the training samples.
        
        If validation samples are provided, they will be used for early stopping.
        Automatically handles conversion of PyHealth samples to numpy arrays.
        """
        X_train, y_train = self._samples_to_arrays(self.train_samples)
        print(X_train.shape)
        print(y_train.shape)

        eval_set = None
        if self.val_samples:
            X_val, y_val = self._samples_to_arrays(self.val_samples)
            eval_set = [(X_val, y_val)]

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train, eval_set=eval_set)

    def predict(self):
        """Generates predictions for the test samples.
        
        Returns:
            numpy.ndarray: Array of predicted class labels.
        """
        X, _ = self._samples_to_arrays(self.test_samples)
        return self.model.predict(X)

    def predict_proba(self):
        """Generates class probability estimates for the test samples.
        
        Returns:
            numpy.ndarray: Array of shape (n_samples, n_classes) with probabilities.
        """
        X, _ = self._samples_to_arrays(self.test_samples)
        return self.model.predict_proba(X)

    def evaluate(self, model_name="LightGBM"):
        """Performs comprehensive evaluation of the model.
        
        Computes multiple metrics and generates classification reports:
        - Accuracy, Precision, Recall, F1 Score
        - AUROC and AUPRC
        - Cohen's Kappa
        - Classification report
        - Confusion matrix visualization

        Args:
            model_name: Optional name for the model in results output.

        Returns:
            List[Dict]: A list containing a dictionary of evaluation metrics.
        """
        X_test, y_test = self._samples_to_arrays(self.test_samples)
        y_pred = self.predict()
        y_pred_proba = self.predict_proba()

        # Metrics
        results = []
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            labels=[1],
        )
        recall = recall_score(
            y_test,
            y_pred,
            labels=[1],
        )  # Recall is the same as sensitivity
        f1 = f1_score(
            y_test,
            y_pred,
            labels=[1],
        )
        auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])
        precision_recall_auc = auc(recalls, precisions)
        y_pred_kappa = np.argmax(y_pred_proba, axis=1)
        cohen_kappa = cohen_kappa_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        print(report)
        results.append(
            {
                "Model": model_name,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUROC": auroc,
                'AUPRC': precision_recall_auc,
                "Accuracy": accuracy,
                "Cohen's Kappa": cohen_kappa,
            }
        )

        return results

    def _samples_to_arrays(self, samples):
        """Converts PyHealth samples to numpy arrays for model input.
        
        Handles:
        - Multiple feature keys concatenation
        - Tensor to numpy conversion
        - Feature flattening
        - Label extraction

        Args:
            samples: List of PyHealth samples containing features and labels.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: 
                - Features array (n_samples, n_features)
                - Labels array (n_samples,)
        """
        X = []
        y = []

        for sample in samples:
            features_list = []
            for feature_key in self.feature_keys:
                feature = sample[feature_key]
                if isinstance(feature, torch.Tensor):
                    feature = feature.numpy()
                # Flatten if needed
                if hasattr(feature, 'flatten'):
                    feature = feature.flatten()
                features_list.append(feature)

            combined_features = np.concatenate(features_list)
            X.append(combined_features)

            label = sample[self.label_key]
            if isinstance(label, torch.Tensor):
                label = label.item()
            y.append(sample[self.label_key])

        return np.array(X), np.array(y)
    

