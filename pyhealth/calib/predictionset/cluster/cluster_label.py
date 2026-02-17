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

        if model.mode not in ("multiclass", "binary"):
            raise NotImplementedError(
                "ClusterLabel only supports multiclass or binary classification"
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
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError(
                f"n_clusters must be a positive integer, got {n_clusters!r}"
            )
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
        batch_size: int = 32,
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
            batch_size: Batch size for embedding extraction when
                cal_embeddings is not provided. Default is 32.

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

        # Binary: model outputs (N, 1); treat as K=2 for conformity and thresholds
        if K == 1:
            y_true = np.asarray(y_true).ravel().astype(np.intp)
            p1 = np.asarray(y_prob[:, 0], dtype=np.float64).ravel()
            conformity_scores = np.where(y_true == 1, p1, 1.0 - p1)
            K = 2
        else:
            y_true = np.asarray(y_true).ravel().astype(np.intp)
            conformity_scores = y_prob[np.arange(N), y_true]

        # Extract embeddings if not provided
        if cal_embeddings is None:
            print("Extracting embeddings from calibration set...")
            cal_embeddings = extract_embeddings(
                self.model, cal_dataset, batch_size=batch_size, device=self.device
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

        # Flatten to 2D (n_samples, n_features) so KMeans works with 3D embeddings (e.g. TFM)
        def _flatten_emb(emb):
            emb = np.asarray(emb)
            if emb.ndim <= 2:
                return emb.reshape(emb.shape[0], -1) if emb.ndim == 2 else emb.reshape(-1, 1)
            return emb.reshape(emb.shape[0], -1)

        train_embeddings = _flatten_emb(train_embeddings)
        cal_embeddings = _flatten_emb(cal_embeddings)

        # Combine embeddings for clustering
        print(f"Combining embeddings: train={train_embeddings.shape}, cal={cal_embeddings.shape}")
        all_embeddings = np.concatenate([train_embeddings, cal_embeddings], axis=0)
        print(f"Total embeddings for clustering: {all_embeddings.shape}")

        # Fit K-means on combined embeddings (verbose=1 so long runs show progress)
        print(f"Fitting K-means with {self.n_clusters} clusters (n_init=10, may take a while for large N)...")
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            verbose=1,
        )
        self.kmeans_model.fit(all_embeddings)
        print("  K-means fit done.")

        # Assign calibration samples to clusters
        # Note: cal_embeddings start at index len(train_embeddings) in all_embeddings
        cal_start_idx = len(train_embeddings)
        cal_cluster_labels = self.kmeans_model.labels_[cal_start_idx:]

        print(f"Cluster assignments: {np.bincount(cal_cluster_labels)}")

        # Conformity scores already set above (with binary handling)

        # Compute cluster-specific thresholds
        print(f"Computing cluster-specific thresholds for {self.n_clusters} clusters...")
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

        print("  Cluster thresholds computed.")

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

        # Single forward pass with embed=True to get both predictions and
        # embeddings (avoids double compute)
        pred = self.model(**{**kwargs, "embed": True})
        if "embed" not in pred:
            raise ValueError(
                f"Model {type(self.model).__name__} does not return "
                "embeddings. Make sure the model supports the "
                "embed=True flag in its forward() method."
            )

        # Flatten to 2D (batch_size, n_features) so KMeans.predict works with 3D embeddings
        sample_embedding = pred["embed"].detach().cpu().numpy()
        if sample_embedding.ndim == 1:
            sample_embedding = sample_embedding.reshape(1, -1)
        elif sample_embedding.ndim > 2:
            sample_embedding = sample_embedding.reshape(sample_embedding.shape[0], -1)

        # Predict cluster for each sample in the batch
        cluster_ids = self.kmeans_model.predict(sample_embedding)

        # Get cluster-specific threshold for each sample
        cluster_thresholds = np.array(
            [self.cluster_thresholds[cid] for cid in cluster_ids]
        )
        y_prob = pred["y_prob"]

        # Binary: expand (batch, 1) to (batch, 2) only for set construction; keep pred["y_prob"] as-is
        if y_prob.shape[-1] == 1:
            p1 = y_prob.squeeze(-1).clamp(0.0, 1.0)
            y_prob_2 = torch.stack([1.0 - p1, p1], dim=-1)
        else:
            y_prob_2 = y_prob

        cluster_thresholds = torch.as_tensor(
            cluster_thresholds, device=self.device, dtype=y_prob_2.dtype
        )

        # Broadcast thresholds to match y_prob shape (batch_size, n_classes).
        if y_prob_2.ndim > 1 and cluster_thresholds.ndim == 1:
            view_shape = (cluster_thresholds.shape[0],) + (1,) * (
                y_prob_2.ndim - 1
            )
            cluster_thresholds = cluster_thresholds.view(view_shape)

        pred["y_predset"] = y_prob_2 >= cluster_thresholds
        pred.pop("embed", None)  # do not expose internal embedding to caller
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
