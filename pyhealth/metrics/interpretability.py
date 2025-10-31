"""Interpretability metrics for evaluating feature attribution methods.

This module provides metrics to evaluate the faithfulness of interpretability
methods, specifically comprehensiveness and sufficiency. These metrics are
computed at the dataset level by iterating through batches.

Key Design:
-----------
- Works with PyHealth's dataloader and batch dictionary format
- Iterates through batches to compute dataset-level metrics
- Supports per-feature evaluation (each feature_key evaluated independently)
- Importance scores must match input structure (same keys and shapes)
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader


def comprehensiveness(
    model: Callable,
    dataloader: DataLoader,
    importance_scores_dict: Dict[str, torch.Tensor],
    percentiles: List[int] = [1, 5, 10, 20, 50],
    feature_keys: Optional[List[str]] = None,
    ablation_value: Union[float, str] = 0.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute comprehensiveness metric across a dataset.

    Comprehensiveness measures how much the prediction changes when you REMOVE
    (ablate) the most important tokens/features. Higher values indicate more
    faithful interpretations.

    The metric is computed as:
        COMP = (1/|B|) × Σ[p_c(x)(x) - p_c(x)(x \\ x:q%)]
                        q∈B

    Where:
        - x is the original input
        - x:q% are the top q% most important features
        - x \\ x:q% is input with top q% features removed (ablated)
        - p_c(x)(·) is predicted probability for original predicted class
        - B is the set of percentiles (default: {1, 5, 10, 20, 50})

    Args:
        model: PyHealth model that accepts **batch and returns dict with
            'y_prob' or 'logit'.
        dataloader: DataLoader providing batches as dicts with feature_keys.
        importance_scores_dict: Dictionary mapping feature_key to importance
            scores tensor. Shape must match the corresponding feature in batch.
            Format: {feature_key: tensor of shape matching batch[feature_key]}
        percentiles: Percentiles to evaluate. Default: [1, 5, 10, 20, 50].
        feature_keys: Specific features to evaluate. If None, evaluates all
            features in importance_scores_dict.
        ablation_value: Ablation strategy:
            - float: Use this value (e.g., 0.0)
            - "mean": Use per-feature mean
            - "random": Random values from normal distribution
        device: Device for computation (e.g., "cpu", "cuda:0").

    Returns:
        Dictionary mapping feature_key to comprehensiveness score.
        Format: {feature_key: comp_score}

    Examples:
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.models import MLP
        >>>
        >>> # Assume we have importance scores from an attribution method
        >>> importance_dict = {
        ...     "conditions": torch.randn(len(dataset), 50),  # (N, seq_len)
        ...     "procedures": torch.randn(len(dataset), 30),
        ... }
        >>>
        >>> comp_scores = comprehensiveness(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     importance_scores_dict=importance_dict,
        ...     feature_keys=["conditions", "procedures"],
        ...     device="cuda:0"
        ... )
        >>> print(comp_scores)
        {'conditions': 0.234, 'procedures': 0.189}
    """
    model.eval()
    model = model.to(device)

    # Determine which features to evaluate
    if feature_keys is None:
        feature_keys = list(importance_scores_dict.keys())

    # Initialize results storage
    results = {key: [] for key in feature_keys}

    # Track sample index across batches
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch[list(batch.keys())[0]].shape[0]

            # Move batch to device
            batch_device = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_device[key] = val.to(device)
                else:
                    batch_device[key] = val

            # Get original predictions for this batch
            outputs = model(**batch_device)
            y_prob = outputs["y_prob"]

            # Get predicted classes
            if y_prob.dim() == 1:
                # Binary case
                y_prob = y_prob.unsqueeze(1)
                y_prob = torch.cat([1 - y_prob, y_prob], dim=1)

            pred_classes = torch.argmax(y_prob, dim=1)

            # Evaluate each feature independently
            for feat_key in feature_keys:
                if feat_key not in batch_device:
                    continue
                if feat_key not in importance_scores_dict:
                    continue

                x_feat = batch_device[feat_key]

                # Get importance scores for this batch
                importance_batch = importance_scores_dict[feat_key][
                    sample_idx : sample_idx + batch_size
                ].to(device)

                # Assert shape compatibility
                assert x_feat.shape == importance_batch.shape, (
                    f"Shape mismatch for {feat_key}: "
                    f"input {x_feat.shape} vs "
                    f"importance {importance_batch.shape}"
                )

                # Compute comprehensiveness for each sample in batch
                for i in range(batch_size):
                    x_sample = x_feat[i : i + 1]
                    importance_sample = importance_batch[i : i + 1]
                    original_prob = y_prob[i, pred_classes[i]].item()
                    class_idx = pred_classes[i].item()

                    # Compute comp for this sample and feature
                    comp_score = _comprehensiveness_single_feature(
                        model=model,
                        batch_template=batch_device,
                        feature_key=feat_key,
                        x_feature=x_sample,
                        importance_feature=importance_sample,
                        original_prob=original_prob,
                        class_idx=class_idx,
                        percentiles=percentiles,
                        ablation_value=ablation_value,
                    )

                    results[feat_key].append(comp_score)

            sample_idx += batch_size

    # Average across all samples for each feature
    final_results = {key: float(np.mean(scores)) for key, scores in results.items()}

    return final_results


def _comprehensiveness_single_feature(
    model: Callable,
    batch_template: Dict[str, Any],
    feature_key: str,
    x_feature: torch.Tensor,
    importance_feature: torch.Tensor,
    original_prob: float,
    class_idx: int,
    percentiles: List[int],
    ablation_value: Union[float, str],
) -> float:
    """Compute comprehensiveness for a single sample and single feature.

    Args:
        model: PyHealth model
        batch_template: Template batch dict (used to construct modified batch)
        feature_key: Name of feature being evaluated
        x_feature: Feature tensor (1, *)
        importance_feature: Importance scores (1, *)
        original_prob: Original prediction probability
        class_idx: Target class index
        percentiles: List of percentiles to evaluate
        ablation_value: Ablation strategy

    Returns:
        Average comprehensiveness score across percentiles
    """
    # Flatten for easier indexing
    x_flat = x_feature.flatten()
    importance_flat = importance_feature.flatten()

    # Determine ablation value
    if ablation_value == "mean":
        abl_val = x_flat.mean().item()
    elif ablation_value == "random":
        abl_val = None  # Will sample per iteration
    else:
        abl_val = float(ablation_value)

    comp_scores = []

    for q in percentiles:
        # Calculate number of features to ablate (top q%)
        num_features = len(importance_flat)
        num_to_ablate = max(1, int(num_features * q / 100.0))

        # Get top-q% indices using torch.topk
        _, top_indices = torch.topk(importance_flat, num_to_ablate, largest=True)

        # Create ablated version
        x_ablated = x_flat.clone()
        if ablation_value == "random":
            x_ablated[top_indices] = torch.randn(num_to_ablate, device=x_flat.device)
        else:
            x_ablated[top_indices] = abl_val

        # Reshape back to original shape
        x_ablated = x_ablated.reshape(x_feature.shape)

        # Create modified batch
        batch_modified = batch_template.copy()
        batch_modified[feature_key] = x_ablated

        # Get prediction on ablated input
        with torch.no_grad():
            outputs_ablated = model(**batch_modified)

        y_prob_ablated = outputs_ablated["y_prob"]
        if y_prob_ablated.dim() == 1:
            y_prob_ablated = y_prob_ablated.unsqueeze(1)
            y_prob_ablated = torch.cat([1 - y_prob_ablated, y_prob_ablated], dim=1)

        prob_ablated = y_prob_ablated[0, class_idx].item()

        # Comprehensiveness: drop in probability
        comp_score = original_prob - prob_ablated
        comp_scores.append(comp_score)

    return float(np.mean(comp_scores))


def sufficiency(
    model: Callable,
    dataloader: DataLoader,
    importance_scores_dict: Dict[str, torch.Tensor],
    percentiles: List[int] = [1, 5, 10, 20, 50],
    feature_keys: Optional[List[str]] = None,
    ablation_value: Union[float, str] = 0.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute sufficiency metric across a dataset.

    Sufficiency measures how much the prediction changes when you KEEP ONLY
    the most important tokens/features (remove everything else). Lower values
    indicate more faithful interpretations.

    The metric is computed as:
        SUFF = (1/|B|) × Σ[p_c(x)(x) - p_c(x)(x:q%)]
                        q∈B

    Where:
        - x is the original input
        - x:q% are the top q% most important features (all others removed)
        - p_c(x)(·) is predicted probability for original predicted class
        - B is the set of percentiles (default: {1, 5, 10, 20, 50})

    Args:
        model: PyHealth model that accepts **batch and returns dict with
            'y_prob' or 'logit'.
        dataloader: DataLoader providing batches as dicts with feature_keys.
        importance_scores_dict: Dictionary mapping feature_key to importance
            scores tensor. Shape must match the corresponding feature in batch.
        percentiles: Percentiles to evaluate. Default: [1, 5, 10, 20, 50].
        feature_keys: Specific features to evaluate. If None, evaluates all
            features in importance_scores_dict.
        ablation_value: Ablation strategy for non-important features.
        device: Device for computation.

    Returns:
        Dictionary mapping feature_key to sufficiency score.

    Examples:
        >>> suff_scores = sufficiency(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     importance_scores_dict=importance_dict,
        ...     feature_keys=["conditions"],
        ...     device="cuda:0"
        ... )
        >>> print(suff_scores)
        {'conditions': 0.089}
    """
    model.eval()
    model = model.to(device)

    # Determine which features to evaluate
    if feature_keys is None:
        feature_keys = list(importance_scores_dict.keys())

    # Initialize results storage
    results = {key: [] for key in feature_keys}

    # Track sample index across batches
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch[list(batch.keys())[0]].shape[0]

            # Move batch to device
            batch_device = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_device[key] = val.to(device)
                else:
                    batch_device[key] = val

            # Get original predictions for this batch
            outputs = model(**batch_device)
            y_prob = outputs["y_prob"]

            # Get predicted classes
            if y_prob.dim() == 1:
                y_prob = y_prob.unsqueeze(1)
                y_prob = torch.cat([1 - y_prob, y_prob], dim=1)

            pred_classes = torch.argmax(y_prob, dim=1)

            # Evaluate each feature independently
            for feat_key in feature_keys:
                if feat_key not in batch_device:
                    continue
                if feat_key not in importance_scores_dict:
                    continue

                x_feat = batch_device[feat_key]

                # Get importance scores for this batch
                importance_batch = importance_scores_dict[feat_key][
                    sample_idx : sample_idx + batch_size
                ].to(device)

                # Assert shape compatibility
                assert x_feat.shape == importance_batch.shape, (
                    f"Shape mismatch for {feat_key}: "
                    f"input {x_feat.shape} vs "
                    f"importance {importance_batch.shape}"
                )

                # Compute sufficiency for each sample in batch
                for i in range(batch_size):
                    x_sample = x_feat[i : i + 1]
                    importance_sample = importance_batch[i : i + 1]
                    original_prob = y_prob[i, pred_classes[i]].item()
                    class_idx = pred_classes[i].item()

                    # Compute suff for this sample and feature
                    suff_score = _sufficiency_single_feature(
                        model=model,
                        batch_template=batch_device,
                        feature_key=feat_key,
                        x_feature=x_sample,
                        importance_feature=importance_sample,
                        original_prob=original_prob,
                        class_idx=class_idx,
                        percentiles=percentiles,
                        ablation_value=ablation_value,
                    )

                    results[feat_key].append(suff_score)

            sample_idx += batch_size

    # Average across all samples for each feature
    final_results = {key: float(np.mean(scores)) for key, scores in results.items()}

    return final_results


def _sufficiency_single_feature(
    model: Callable,
    batch_template: Dict[str, Any],
    feature_key: str,
    x_feature: torch.Tensor,
    importance_feature: torch.Tensor,
    original_prob: float,
    class_idx: int,
    percentiles: List[int],
    ablation_value: Union[float, str],
) -> float:
    """Compute sufficiency for a single sample and single feature.

    Args:
        model: PyHealth model
        batch_template: Template batch dict
        feature_key: Name of feature being evaluated
        x_feature: Feature tensor (1, *)
        importance_feature: Importance scores (1, *)
        original_prob: Original prediction probability
        class_idx: Target class index
        percentiles: List of percentiles to evaluate
        ablation_value: Ablation strategy

    Returns:
        Average sufficiency score across percentiles
    """
    # Flatten for easier indexing
    x_flat = x_feature.flatten()
    importance_flat = importance_feature.flatten()

    # Determine ablation value
    if ablation_value == "mean":
        abl_val = x_flat.mean().item()
    elif ablation_value == "random":
        abl_val = None
    else:
        abl_val = float(ablation_value)

    suff_scores = []

    for q in percentiles:
        # Calculate number of features to keep (top q%)
        num_features = len(importance_flat)
        num_to_keep = max(1, int(num_features * q / 100.0))

        # Get top-q% indices
        _, top_indices = torch.topk(importance_flat, num_to_keep, largest=True)

        # Create sufficient version (keep only top q%)
        if ablation_value == "random":
            x_sufficient = torch.randn_like(x_flat)
        else:
            x_sufficient = torch.full_like(x_flat, abl_val)

        x_sufficient[top_indices] = x_flat[top_indices]

        # Reshape back
        x_sufficient = x_sufficient.reshape(x_feature.shape)

        # Create modified batch
        batch_modified = batch_template.copy()
        batch_modified[feature_key] = x_sufficient

        # Get prediction
        with torch.no_grad():
            outputs_sufficient = model(**batch_modified)

        y_prob_sufficient = outputs_sufficient["y_prob"]
        if y_prob_sufficient.dim() == 1:
            y_prob_sufficient = y_prob_sufficient.unsqueeze(1)
            y_prob_sufficient = torch.cat(
                [1 - y_prob_sufficient, y_prob_sufficient], dim=1
            )

        prob_sufficient = y_prob_sufficient[0, class_idx].item()

        # Sufficiency: drop in probability
        suff_score = original_prob - prob_sufficient
        suff_scores.append(suff_score)

    return float(np.mean(suff_scores))


def interpretability_metrics_fn(
    model: Callable,
    dataloader: DataLoader,
    importance_scores_dict: Dict[str, torch.Tensor],
    metrics: Optional[List[str]] = None,
    percentiles: List[int] = [1, 5, 10, 20, 50],
    feature_keys: Optional[List[str]] = None,
    ablation_value: Union[float, str] = 0.0,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Compute interpretability metrics across a dataset.

    This function evaluates the faithfulness of interpretability methods using
    comprehensiveness and sufficiency metrics over an entire dataset.

    Args:
        model: PyHealth model.
        dataloader: DataLoader providing batches.
        importance_scores_dict: Dict mapping feature_key to importance scores.
        metrics: List of metrics to compute. Options: ["comprehensiveness",
            "sufficiency", "comp", "suff"]. Default: both.
        percentiles: Percentiles to evaluate.
        feature_keys: Specific features to evaluate. If None, evaluates all.
        ablation_value: Ablation strategy.
        device: Device for computation.

    Returns:
        Nested dictionary: {metric_name: {feature_key: score}}

    Examples:
        >>> from pyhealth.metrics import interpretability_metrics_fn
        >>>
        >>> # Assume we have a trained model and importance scores
        >>> results = interpretability_metrics_fn(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     importance_scores_dict={
        ...         "conditions": conditions_importance,
        ...         "procedures": procedures_importance,
        ...     },
        ...     metrics=["comprehensiveness", "sufficiency"],
        ...     device="cuda:0"
        ... )
        >>>
        >>> print(results)
        {
            'comprehensiveness': {
                'conditions': 0.234,
                'procedures': 0.189
            },
            'sufficiency': {
                'conditions': 0.089,
                'procedures': 0.067
            }
        }
    """
    if metrics is None:
        metrics = ["comprehensiveness", "sufficiency"]

    results = {}

    for metric in metrics:
        metric_lower = metric.lower()

        if metric_lower in ["comprehensiveness", "comp"]:
            comp_scores = comprehensiveness(
                model=model,
                dataloader=dataloader,
                importance_scores_dict=importance_scores_dict,
                percentiles=percentiles,
                feature_keys=feature_keys,
                ablation_value=ablation_value,
                device=device,
            )
            results["comprehensiveness"] = comp_scores

        elif metric_lower in ["sufficiency", "suff"]:
            suff_scores = sufficiency(
                model=model,
                dataloader=dataloader,
                importance_scores_dict=importance_scores_dict,
                percentiles=percentiles,
                feature_keys=feature_keys,
                ablation_value=ablation_value,
                device=device,
            )
            results["sufficiency"] = suff_scores

        else:
            raise ValueError(f"Unknown interpretability metric: {metric}")

    return results


if __name__ == "__main__":
    # Example usage with mock data
    import torch
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("Testing Interpretability Metrics")
    print("=" * 80)

    # Create a simple mock PyHealth-style model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_cond = torch.nn.Linear(5, 16)
            self.fc_proc = torch.nn.Linear(3, 16)
            self.fc_out = torch.nn.Linear(32, 2)
            self.feature_keys = ["conditions", "procedures"]

        def forward(self, conditions, procedures, **kwargs):
            h1 = torch.relu(self.fc_cond(conditions))
            h2 = torch.relu(self.fc_proc(procedures))
            h = torch.cat([h1, h2], dim=1)
            logits = self.fc_out(h)
            y_prob = torch.softmax(logits, dim=-1)
            return {"y_prob": y_prob, "logit": logits}

    # Create mock dataset
    n_samples = 20
    conditions_data = torch.randn(n_samples, 5)
    procedures_data = torch.randn(n_samples, 3)
    labels = torch.randint(0, 2, (n_samples,))

    # Create importance scores (mock - in reality from attribution method)
    importance_conditions = torch.abs(torch.randn(n_samples, 5))
    importance_procedures = torch.abs(torch.randn(n_samples, 3))

    # Create custom dataset that returns dicts
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, conditions, procedures, labels):
            self.conditions = conditions
            self.procedures = procedures
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "conditions": self.conditions[idx],
                "procedures": self.procedures[idx],
                "label": self.labels[idx],
            }

    dataset = DictDataset(conditions_data, procedures_data, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize model
    model = MockModel()

    # Prepare importance scores dict
    importance_dict = {
        "conditions": importance_conditions,
        "procedures": importance_procedures,
    }

    print("\nComputing interpretability metrics...")
    print("-" * 80)

    # Compute metrics
    results = interpretability_metrics_fn(
        model=model,
        dataloader=dataloader,
        importance_scores_dict=importance_dict,
        metrics=["comprehensiveness", "sufficiency"],
        percentiles=[10, 20, 50],
        device="cpu",
    )

    print("\nResults:")
    print("=" * 80)
    for metric_name, feature_scores in results.items():
        print(f"\n{metric_name.upper()}:")
        for feature, score in feature_scores.items():
            print(f"  {feature}: {score:.4f}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
