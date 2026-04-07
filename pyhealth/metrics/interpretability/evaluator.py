"""Evaluator for interpretability metrics.

This module provides high-level interfaces for evaluating attribution methods
using removal-based metrics like Comprehensiveness and Sufficiency.
"""

from typing import Dict, List, Optional
import warnings

import torch

from pyhealth.models import BaseModel

from .comprehensiveness import ComprehensivenessMetric
from .sufficiency import SufficiencyMetric
from .utils import SampleClass, SampleFilterFn, threshold_sample_filter


class Evaluator:
    """High-level interface for evaluating interpretations.

    This class provides a convenient API for computing multiple
    interpretability metrics at once, both on individual batches and
    across entire datasets.

    Args:
        model: PyHealth BaseModel to evaluate
        percentages: List of percentages to evaluate at.
            Default: [1, 5, 10, 20, 50].
        ablation_strategy: How to ablate features. Options:
            - 'zero': Set ablated features to 0
            - 'mean': Set ablated features to feature mean across batch
            - 'noise': Add Gaussian noise to ablated features
            Default: 'zero'.
        sample_filter: A callable that classifies each sample for evaluation.
            Signature: (class_probs, classifier_type) -> sample_classes
            where class_probs has shape (batch_size,) and contains the
            probability for the predicted class (sigmoid/softmax output
            with target class already applied), and sample_classes is a
            tensor of SampleClass values:
            - SampleClass.POSITIVE: evaluate with attributions as-is
            - SampleClass.NEGATIVE: evaluate with negated attributions
            - SampleClass.IGNORE: exclude from evaluation
            If None, uses default_sample_filter.
        positive_threshold: .. deprecated::
            This parameter is deprecated and will be removed in a future
            release. Use ``sample_filter`` with
            :func:`threshold_sample_filter` instead.
            Threshold for positive class in binary classification.
            Default: None.

    Examples:
        >>> from pyhealth.models import StageNet
        >>> from pyhealth.metrics.interpretability import Evaluator
        >>> from pyhealth.metrics.interpretability.utils import (
        ...     SampleClass,
        ...     threshold_sample_filter,
        ... )
        >>>
        >>> # Initialize evaluator with default filter
        >>> evaluator = Evaluator(model)
        >>>
        >>> # Initialize with custom filter that ignores low-confidence
        >>> def confident_filter(class_probs, classifier_type):
        ...     batch_size = class_probs.shape[0]
        ...     result = torch.full(
        ...         (batch_size,), SampleClass.POSITIVE,
        ...         dtype=torch.long, device=class_probs.device,
        ...     )
        ...     result[class_probs < 0.6] = SampleClass.IGNORE
        ...     return result
        >>> evaluator = Evaluator(model, sample_filter=confident_filter)
        >>>
        >>> # Evaluate on a single batch
        >>> inputs = {'conditions': torch.randn(32, 50)}
        >>> attributions = {'conditions': torch.randn(32, 50)}
        >>> batch_results = evaluator.evaluate(inputs, attributions)
        >>> for metric, (scores, mask) in batch_results.items():
        >>>     print(f"{metric}: {scores[mask].mean():.4f}")
        >>>
        >>> # Evaluate across entire dataset with an attribution method
        >>> from pyhealth.interpret.methods import IntegratedGradients
        >>> ig = IntegratedGradients(model)
        >>> results = evaluator.evaluate_attribution(test_loader, ig)
        >>> print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
        >>> print(f"Sufficiency: {results['sufficiency']:.4f}")
    """

    def __init__(
        self,
        model: BaseModel,
        percentages: List[float] = [1, 5, 10, 20, 50],
        ablation_strategy: str = "zero",
        sample_filter: Optional[SampleFilterFn] = None,
        positive_threshold: Optional[float] = None,
    ):
        self.model = model
        self.percentages = percentages
        self.ablation_strategy = ablation_strategy
        self.positive_threshold = positive_threshold

        # Resolve the effective sample filter:
        #   1. explicit sample_filter wins
        #   2. positive_threshold → threshold_sample_filter(positive_threshold)
        #   3. fallback → default (threshold_sample_filter(0.5))
        if sample_filter is not None:
            if positive_threshold is not None:
                warnings.warn(
                    "Both sample_filter and positive_threshold were given. "
                    "sample_filter takes precedence; positive_threshold is "
                    "ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            resolved_filter = sample_filter
        elif positive_threshold is not None:
            warnings.warn(
                "positive_threshold is deprecated and will be removed in a "
                "future release. Use sample_filter with "
                "threshold_sample_filter() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_filter = threshold_sample_filter(positive_threshold)
        else:
            resolved_filter = threshold_sample_filter(0.5)

        self.sample_filter = resolved_filter
        self.metrics = {
            "comprehensiveness": ComprehensivenessMetric(
                model,
                percentages=percentages,
                ablation_strategy=ablation_strategy,
                sample_filter=resolved_filter,
            ),
            "sufficiency": SufficiencyMetric(
                model,
                percentages=percentages,
                ablation_strategy=ablation_strategy,
                sample_filter=resolved_filter,
            ),
        }

    def evaluate(
        self,
        inputs: Dict[str, torch.Tensor],
        attributions: Dict[str, torch.Tensor],
        metrics: List[str] = ["comprehensiveness", "sufficiency"],
        return_per_percentage: bool = False,
        debug: bool = False,
    ):
        """Evaluate multiple metrics at once.

        Args:
            inputs: Model inputs (batch)
            attributions: Attribution scores (batch)
            metrics: List of metrics to compute. Options:
                ["comprehensiveness", "sufficiency"]
            return_per_percentage: If True, return per-percentage breakdown.
                If False (default), return averaged scores.
            debug: If True, print detailed debug information (only used
                when return_per_percentage=True)

        Returns:
            If return_per_percentage=False (default):
                Dictionary mapping metric name -> (scores, valid_mask)
                - scores: Raw metric scores for all samples
                - valid_mask: Binary mask indicating valid samples to average

            If return_per_percentage=True:
                Dictionary mapping metric name -> Dict[percentage -> scores]
                Example: {'comprehensiveness': {10: tensor(...), 20: ...}}

        Note:
            For binary classifiers, all samples are evaluated
            (both positive and negative predictions).
            Use: scores[valid_mask].mean()

        Examples:
            >>> # Default: averaged scores
            >>> results = evaluator.evaluate(inputs, attributions)
            >>> comp_scores, valid_mask = results['comprehensiveness']
            >>> print(f"Mean: {comp_scores[valid_mask].mean():.4f}")
            >>>
            >>> # Per-percentage breakdown
            >>> detailed = evaluator.evaluate(
            ...     inputs, attributions, return_per_percentage=True
            ... )
            >>> comp_10_pct = detailed['comprehensiveness'][10]
            >>> suff_20_pct = detailed['sufficiency'][20]
        """
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                result = self.metrics[metric_name].compute(
                    inputs,
                    attributions,
                    return_per_percentage=return_per_percentage,
                    debug=debug,
                )
                results[metric_name] = result
        return results

    def evaluate_attribution(
        self,
        dataloader,
        method,
        metrics: List[str] = ["comprehensiveness", "sufficiency"],
    ) -> Dict[str, float]:
        """Evaluate an attribution method across an entire dataset.

        This method computes the average faithfulness metrics for an
        attribution approach across all batches in a dataloader. It
        automatically computes attributions using the provided method
        and evaluates them.

        Args:
            dataloader: PyTorch DataLoader with test/validation data.
                Should yield batches compatible with the model.
            method: Attribution method instance (e.g., IntegratedGradients).
                Must implement the BaseInterpreter interface with an
                attribute(**data) method.
            metrics: List of metrics to compute. Options:
                ["comprehensiveness", "sufficiency"]
                Default: both metrics.

        Returns:
            Dictionary mapping metric names to their average scores
            across the entire dataset. For binary classifiers, only
            positive class (predicted class=1) samples are included
            in the average.

            Example: {'comprehensiveness': 0.345, 'sufficiency': 0.123}

        Note:
            For binary classifiers, negative class (predicted class=0)
            samples are excluded from the average, as ablation metrics
            are not meaningful for the default/null class.

        Examples:
            >>> from pyhealth.interpret.methods import IntegratedGradients
            >>> from pyhealth.metrics.interpretability import Evaluator
            >>>
            >>> # Initialize evaluator and attribution method
            >>> evaluator = Evaluator(model)
            >>> ig = IntegratedGradients(model, use_embeddings=True)
            >>>
            >>> # Evaluate across test set
            >>> results = evaluator.evaluate_attribution(
            ...     test_loader, ig, metrics=["comprehensiveness"]
            ... )
            >>> print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
            >>>
            >>> # Compare multiple methods
            >>> from pyhealth.interpret.methods import CheferRelevance
            >>> chefer = CheferRelevance(model)
            >>> ig_results = evaluator.evaluate_attribution(test_loader, ig)
            >>> chefer_results = evaluator.evaluate_attribution(
            ...     test_loader, chefer
            ... )
            >>> print("Method Comparison:")
            >>> print(f"  IG Comp: {ig_results['comprehensiveness']:.4f}")
            >>> print(
            ...     f"  Chefer Comp: "
            ...     f"{chefer_results['comprehensiveness']:.4f}"
            ... )
        """
        # Get model device
        model_device = next(self.model.parameters()).device

        # Tracking for statistics and debug output
        batch_count = 0
        total_samples = 0
        total_valid = {metric_name: 0 for metric_name in metrics}
        running_sum = {metric_name: 0.0 for metric_name in metrics}

        # Process each batch
        for batch in dataloader:
            batch_count += 1
            # Move batch to model device
            batch_on_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_on_device[key] = value.to(model_device)
                elif isinstance(value, tuple) and len(value) >= 2:
                    # Handle (time, values) tuples
                    time_part = value[0]
                    if time_part is not None and isinstance(time_part, torch.Tensor):
                        time_part = time_part.to(model_device)

                    values_part = value[1]
                    if isinstance(values_part, torch.Tensor):
                        values_part = values_part.to(model_device)

                    batch_on_device[key] = (time_part, values_part) + value[2:]
                else:
                    batch_on_device[key] = value

            # Compute attributions for this batch
            attributions = method.attribute(**batch_on_device)

            # Evaluate metrics on this batch (returns scores and masks)
            batch_results = self.evaluate(
                batch_on_device, attributions, metrics=metrics
            )

            # Accumulate statistics incrementally (no tensor storage)
            first_metric = metrics[0]
            batch_size = len(batch_results[first_metric][0])
            total_samples += batch_size

            for metric_name in metrics:
                scores, valid_mask = batch_results[metric_name]

                # Track statistics efficiently
                num_valid = valid_mask.sum().item()
                total_valid[metric_name] += num_valid

                # Update running sum (valid scores only)
                valid_scores_batch = (scores * valid_mask).sum().item()
                running_sum[metric_name] += valid_scores_batch

            # Debug output every 10 batches
            if batch_count % 1 == 0:
                print(f"\n[Batch {batch_count}] Progress update:")
                print(f"  Total samples processed: {total_samples}")

                # Compute running averages from accumulated statistics
                for metric_name in metrics:
                    num_valid_so_far = total_valid[metric_name]
                    if num_valid_so_far > 0:
                        running_avg = running_sum[metric_name] / num_valid_so_far
                        print(
                            f"  {metric_name}: {running_avg:.6f} "
                            f"({num_valid_so_far}/{total_samples} valid)"
                        )
                    else:
                        print(f"  {metric_name}: N/A " f"(no valid samples yet)")

        # Compute final averages from accumulated statistics
        results = {}
        for metric_name in metrics:
            if total_valid[metric_name] > 0:
                # Average = running_sum / total_valid
                results[metric_name] = (
                    running_sum[metric_name] / total_valid[metric_name]
                )
            else:
                # No valid samples
                results[metric_name] = float("nan")

        # Final summary
        print(f"\n{'='*70}")
        print("[FINAL] Dataset evaluation complete:")
        print(f"  Total batches: {batch_count}")
        print(f"  Total samples: {total_samples}")
        for metric_name in metrics:
            num_valid_final = total_valid[metric_name]
            if metric_name in results:
                score = results[metric_name]
                if score == score:  # Not NaN
                    print(
                        f"  {metric_name}: {score:.6f} "
                        f"({num_valid_final}/{total_samples} valid)"
                    )
                else:
                    print(f"  {metric_name}: NaN " f"(no valid samples)")

        # Sanity check warnings
        if "comprehensiveness" in results and "sufficiency" in results:
            comp = results["comprehensiveness"]
            suff = results["sufficiency"]
            if comp == comp and suff == suff:  # Both not NaN
                if comp < 0:
                    print("\n⚠ WARNING: Negative comprehensiveness detected!")
                    print("  - Removing 'important' features INCREASED " "confidence")
                    print("  - Possible causes:")
                    print("    * Attribution scores may be inverted/wrong")
                    print("    * Features with negative attributions")
                    print("    * Model predictions unstable")

                if suff > comp:
                    print("\n⚠ WARNING: Sufficiency > Comprehensiveness!")
                    print("  - Keeping top features worse than removing them")
                    print("  - This suggests:")
                    print("    * Attribution quality is poor")
                    print("    * Important features not correctly identified")
                    print("    * Consider checking attribution method")

        valid_ratio = sum(total_valid.values()) / (len(metrics) * total_samples) if total_samples > 0 else 0
        if valid_ratio < 0.1 and total_samples > 0:
            print(f"\n⚠ WARNING: Only {valid_ratio*100:.1f}% valid samples")
            print("  - Most predictions are negative class")
            print("  - Consider:")
            print("    * Checking model predictions distribution")
            print("    * Adjusting sample_filter to include more samples")
            print("    * Using balanced test set")

        print(f"{'='*70}\n")

        return results

# Functional API (wraps Evaluator for convenience)
def evaluate_attribution(
    model: BaseModel,
    dataloader,
    method,
    metrics: List[str] = ["comprehensiveness", "sufficiency"],
    percentages: List[float] = [1, 5, 10, 20, 50],
    ablation_strategy: str = "zero",
    sample_filter: Optional[SampleFilterFn] = None,
    positive_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate an attribution method across a dataset (functional API).

    This is a convenience function that wraps the Evaluator class for
    simple one-off evaluations. For multiple evaluations with the same
    configuration, consider using the Evaluator class directly for
    better efficiency.

    Args:
        model: PyHealth BaseModel to evaluate
        dataloader: PyTorch DataLoader with test/validation data
        method: Attribution method instance (e.g., IntegratedGradients).
            Must implement the BaseInterpreter interface.
        metrics: List of metrics to compute. Options:
            ["comprehensiveness", "sufficiency"]
            Default: both metrics.
        percentages: List of percentages to evaluate at.
            Default: [1, 5, 10, 20, 50].
        ablation_strategy: How to ablate features. Options:
            - 'zero': Set ablated features to 0
            - 'mean': Set ablated features to feature mean across batch
            - 'noise': Add Gaussian noise to ablated features
            Default: 'zero'.
        sample_filter: A callable that classifies each sample for
            evaluation. Signature:
            (class_probs, classifier_type) -> sample_classes
            where class_probs has shape (batch_size,) and contains the
            probability for the predicted class (sigmoid/softmax output
            with target class already applied), and sample_classes is a
            tensor of SampleClass values:
            - SampleClass.POSITIVE: evaluate with attributions as-is
            - SampleClass.NEGATIVE: evaluate with negated attributions
            - SampleClass.IGNORE: exclude from evaluation
            If None, uses default_sample_filter.
        positive_threshold: .. deprecated::
            This parameter is deprecated and will be removed in a future
            release. Use ``sample_filter`` with
            :func:`threshold_sample_filter` instead.
            Threshold for positive class in binary classification.
            Default: None.

    Returns:
        Dictionary mapping metric names to their average scores
        across the entire dataset. Averaging uses mask-based filtering
        to exclude IGNORE samples.

        Example: {'comprehensiveness': 0.345, 'sufficiency': 0.123}

    Examples:
        >>> from pyhealth.interpret.methods import IntegratedGradients
        >>> from pyhealth.metrics.interpretability import (
        ...     evaluate_attribution
        ... )
        >>> from pyhealth.metrics.interpretability.utils import SampleClass
        >>>
        >>> # Simple one-off evaluation
        >>> ig = IntegratedGradients(model, use_embeddings=True)
        >>> results = evaluate_attribution(
        ...     model, test_loader, ig,
        ...     metrics=["comprehensiveness"],
        ...     percentages=[10, 20, 50]
        ... )
        >>> print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
        >>>
        >>> # Custom filter to ignore uncertain predictions
        >>> def ignore_uncertain(class_probs, classifier_type):
        ...     batch_size = class_probs.shape[0]
        ...     result = torch.full(
        ...         (batch_size,), SampleClass.POSITIVE,
        ...         dtype=torch.long, device=class_probs.device,
        ...     )
        ...     result[class_probs < 0.7] = SampleClass.IGNORE
        ...     return result
        >>> results = evaluate_attribution(
        ...     model, test_loader, ig,
        ...     sample_filter=ignore_uncertain,
        ... )
        >>>
        >>> # For comparing multiple methods efficiently, use Evaluator:
        >>> from pyhealth.metrics.interpretability import Evaluator
        >>> evaluator = Evaluator(model, percentages=[10, 20, 50])
        >>> ig_results = evaluator.evaluate_attribution(test_loader, ig)
        >>> chefer_results = evaluator.evaluate_attribution(
        ...     test_loader, chefer
        ... )
    """
    evaluator = Evaluator(
        model,
        percentages=percentages,
        ablation_strategy=ablation_strategy,
        sample_filter=sample_filter,
        positive_threshold=positive_threshold,
    )
    return evaluator.evaluate_attribution(dataloader, method, metrics=metrics)
