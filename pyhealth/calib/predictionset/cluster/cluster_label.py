"""
Cluster-Based Conformal Prediction.

This module implements conformal prediction with cluster-specific calibration
thresholds using K-means clustering on patient embeddings. The method groups
similar patients into clusters and computes separate calibration thresholds
for each cluster, enabling cluster-aware prediction sets.

This serves as a baseline approach for future personalized/dynamic conformal
prediction methods that use patient similarity for calibration set construction.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import IterableDataset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.utils import extract_embeddings, prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["ClusterLabel"]


class ClusterLabel(SetPredictor):
    """Cluster-based conformal prediction for multiclass classification.

    This method uses K-means clustering on patient embeddings to group similar
    patients into clusters. Each cluster gets its own calibration threshold,
    computed from the conformity scores of calibration samples in that cluster.
    At inference time, test samples are assigned to their nearest cluster and
    use the cluster-specific threshold.

    This approach is simpler than KDE-based methods and serves as a baseline
    for more advanced personalized conformal prediction approaches.

    Args:
        model: A trained base model that supports embedding extraction
            (must support `embed=True` in forward pass)
        alpha: Target miscoverage rate(s). Can be:
            - float: marginal coverage P(Y not in C(X)) <= alpha
            - array: class-conditional P(Y not in C(X) | Y=k) <= alpha[k]
        n_clusters: Number of K-means clusters. Default is 5.
        random_state: Random seed for K-means clustering. Default is 42.
        debug: Whether to use debug mode (processes fewer samples for
            faster iteration)

    Examples:
        >>> from pyhealth.datasets import TUEVDataset, split_by_sample_conformal
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.models import ContraWR
        >>> from pyhealth.tasks import EEGEventsTUEV
        >>> from pyhealth.calib.predictionset.cluster import ClusterLabel
        >>> from pyhealth.calib.utils import extract_embeddings
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>>
        >>> # Prepare data
        >>> dataset = TUEVDataset(root="path/to/tuev")
        >>> sample_dataset = dataset.set_task(EEGEventsTUEV())
        >>> train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal(
        ...     sample_dataset, ratios=[0.6, 0.1, 0.15, 0.15], seed=42
        ... )
        >>>
        >>> # Train model
        >>> model = ContraWR(dataset=sample_dataset)
        >>> # ... training code ...
        >>>
        >>> # Extract embeddings for clustering
        >>> train_embeddings = extract_embeddings(model, train_ds, batch_size=32)
        >>> cal_embeddings = extract_embeddings(model, cal_ds, batch_size=32)
        >>>
        >>> # Create and calibrate cluster-based predictor
        >>> cluster_predictor = ClusterLabel(model=model, alpha=0.1, n_clusters=5)
        >>> cluster_predictor.calibrate(
        ...     cal_dataset=cal_ds,
        ...     train_embeddings=train_embeddings,
        ...     cal_embeddings=cal_embeddings,
        ... )
        >>>
        >>> # Evaluate
        >>> test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
        >>> y_true, y_prob, _, extra = Trainer(model=cluster_predictor).inference(
        ...     test_loader, additional_outputs=["y_predset"]
        ... )
        >>> metrics = get_metrics_fn(cluster_predictor.mode)(
        ...     y_true, y_prob, metrics=["accuracy", "miscoverage_ps"],
        ...     y_predset=extra["y_predset"]
        ... )
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: Union[float, np.ndarray],
        n_clusters: int = 5,
        random_state: int = 42,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if model.mode != "multiclass":
            raise NotImplementedError(
                "ClusterLabel only supports multiclass classification"
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

        # Store clustering parameters
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Will be set during calibration
        self.kmeans_model = None
        self.cluster_thresholds = None  # Dict mapping cluster_id -> threshold(s)

    def calibrate(
        self,
        cal_dataset: IterableDataset,
        train_embeddings: Optional[np.ndarray] = None,
        cal_embeddings: Optional[np.ndarray] = None,
    ):
        """Calibrate cluster-specific thresholds.

        This method:
        1. Combines train and calibration embeddings for clustering
        2. Fits K-means on the combined embeddings
        3. Assigns calibration samples to clusters
        4. Computes cluster-specific calibration thresholds

        Args:
            cal_dataset: Calibration set
            train_embeddings: Optional pre-computed training embeddings
                of shape (n_train, embedding_dim). If not provided, will
                be extracted from the model (requires train_dataset parameter).
            cal_embeddings: Optional pre-computed calibration embeddings
                of shape (n_cal, embedding_dim). If not provided, will be
                extracted from cal_dataset.

        Note:
            Either provide embeddings directly or ensure the model supports
            embedding extraction via `embed=True` flag.
        """
        # Get predictions and true labels from calibration set
        cal_dataset_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            debug=self.debug,
        )

        y_prob = cal_dataset_dict["y_prob"]
        y_true = cal_dataset_dict["y_true"]
        N, K = y_prob.shape

        # Extract embeddings if not provided
        if cal_embeddings is None:
            print("Extracting embeddings from calibration set...")
            cal_embeddings = extract_embeddings(
                self.model, cal_dataset, batch_size=32, device=self.device
            )
        else:
            cal_embeddings = np.asarray(cal_embeddings)

        if train_embeddings is None:
            raise ValueError(
                "train_embeddings must be provided. "
                "Extract embeddings from training set using extract_embeddings()."
            )
        else:
            train_embeddings = np.asarray(train_embeddings)

        # Combine embeddings for clustering
        print(f"Combining embeddings: train={train_embeddings.shape}, cal={cal_embeddings.shape}")
        all_embeddings = np.concatenate([train_embeddings, cal_embeddings], axis=0)
        print(f"Total embeddings for clustering: {all_embeddings.shape}")

        # Fit K-means on combined embeddings
        print(f"Fitting K-means with {self.n_clusters} clusters...")
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_model.fit(all_embeddings)

        # Assign calibration samples to clusters
        # Note: cal_embeddings start at index len(train_embeddings) in all_embeddings
        cal_start_idx = len(train_embeddings)
        cal_cluster_labels = self.kmeans_model.labels_[cal_start_idx:]

        print(f"Cluster assignments: {np.bincount(cal_cluster_labels)}")

        # Compute conformity scores (probabilities of true class)
        conformity_scores = y_prob[np.arange(N), y_true]

        # Compute cluster-specific thresholds
        self.cluster_thresholds = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = cal_cluster_labels == cluster_id
            cluster_scores = conformity_scores[cluster_mask]

            if len(cluster_scores) == 0:
                print(
                    f"Warning: No calibration samples in cluster {cluster_id}, "
                    "using -inf threshold (include all classes)"
                )
                if isinstance(self.alpha, float):
                    self.cluster_thresholds[cluster_id] = -np.inf
                else:
                    self.cluster_thresholds[cluster_id] = np.array(
                        [-np.inf] * K
                    )
            else:
                if isinstance(self.alpha, float):
                    # Marginal coverage: single threshold per cluster
                    t = _query_quantile(cluster_scores, self.alpha)
                    self.cluster_thresholds[cluster_id] = t
                else:
                    # Class-conditional coverage: one threshold per class per cluster
                    if len(self.alpha) != K:
                        raise ValueError(
                            f"alpha must have length {K} for class-conditional "
                            f"coverage, got {len(self.alpha)}"
                        )
                    t = []
                    for k in range(K):
                        class_mask = (y_true[cluster_mask] == k)
                        if np.sum(class_mask) > 0:
                            class_scores = cluster_scores[class_mask]
                            t_k = _query_quantile(class_scores, self.alpha[k])
                        else:
                            # If no calibration examples for this class in this cluster
                            print(
                                f"Warning: No calibration examples for class {k} "
                                f"in cluster {cluster_id}, using -inf threshold"
                            )
                            t_k = -np.inf
                        t.append(t_k)
                    self.cluster_thresholds[cluster_id] = np.array(t)

        if self.debug:
            print(f"Cluster thresholds: {self.cluster_thresholds}")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation with cluster-specific prediction set construction.

        Returns:
            Dictionary with all results from base model, plus:
                - y_predset: Boolean tensor indicating which classes
                    are in the prediction set
        """
        if self.kmeans_model is None or self.cluster_thresholds is None:
            raise RuntimeError(
                "Model must be calibrated before inference. "
                "Call calibrate() first."
            )

        # Get base model prediction
        pred = self.model(**kwargs)

        # Extract embedding for this sample to assign to cluster
        embed_kwargs = {**kwargs, "embed": True}
        embed_output = self.model(**embed_kwargs)
        if "embed" not in embed_output:
            raise ValueError(
                f"Model {type(self.model).__name__} does not return "
                "embeddings. Make sure the model supports the "
                "embed=True flag in its forward() method."
            )

        # Get embedding and assign to cluster
        sample_embedding = embed_output["embed"].detach().cpu().numpy()
        if sample_embedding.ndim == 1:
            sample_embedding = sample_embedding.reshape(1, -1)

        cluster_id = self.kmeans_model.predict(sample_embedding)[0]

        # Get cluster-specific threshold
        cluster_threshold = self.cluster_thresholds[cluster_id]

        # Convert to tensor if needed
        if isinstance(cluster_threshold, np.ndarray):
            cluster_threshold = torch.tensor(
                cluster_threshold, device=self.device, dtype=pred["y_prob"].dtype
            )
        else:
            cluster_threshold = torch.tensor(
                cluster_threshold, device=self.device, dtype=pred["y_prob"].dtype
            )

        # Construct prediction set using cluster-specific threshold
        pred["y_predset"] = pred["y_prob"] >= cluster_threshold

        return pred


if __name__ == "__main__":
    """
    Demonstration of cluster-based conformal prediction.
    """
    from pyhealth.datasets import TUEVDataset, split_by_sample_conformal, get_dataloader
    from pyhealth.models import ContraWR
    from pyhealth.tasks import EEGEventsTUEV
    from pyhealth.calib.predictionset.cluster import ClusterLabel
    from pyhealth.calib.utils import extract_embeddings
    from pyhealth.trainer import Trainer, get_metrics_fn

    # Setup data and model
    dataset = TUEVDataset(root="downloads/tuev/v2.0.1/edf", subset="both")
    sample_dataset = dataset.set_task(EEGEventsTUEV())
    train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal(
        sample_dataset, ratios=[0.6, 0.1, 0.15, 0.15], seed=42
    )

    model = ContraWR(dataset=sample_dataset)
    # ... Train the model here ...

    # Extract embeddings
    train_embeddings = extract_embeddings(model, train_ds, batch_size=32)
    cal_embeddings = extract_embeddings(model, cal_ds, batch_size=32)

    # Create and calibrate cluster-based predictor
    cluster_predictor = ClusterLabel(model=model, alpha=0.1, n_clusters=5)
    cluster_predictor.calibrate(
        cal_dataset=cal_ds,
        train_embeddings=train_embeddings,
        cal_embeddings=cal_embeddings,
    )

    # Evaluate
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
    y_true, y_prob, _, extra = Trainer(model=cluster_predictor).inference(
        test_loader, additional_outputs=["y_predset"]
    )
    metrics = get_metrics_fn(cluster_predictor.mode)(
        y_true, y_prob, metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra["y_predset"]
    )
    print(f"Results: {metrics}")
