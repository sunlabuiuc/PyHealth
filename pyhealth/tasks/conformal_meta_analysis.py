"""Conformal Meta-Analysis Task for PyHealth.

Defines a regression task that extracts feature vectors and true
effect values from trial records. Also carries along the extra
fields needed for conformal meta-analysis (observed effect, variance,
prior mean) when they are present in the data.

This corresponds to Task 1 in Kaul & Gordon (2024): predicting the
unobserved true effect u given trial features x.

Each sample contains:
    - ``features``: trial feature vector X
    - ``true_effect``: target true effect U (the regression label)
    - ``observed_effect``: noisy observation Y (optional, for CMA)
    - ``variance``: within-trial variance V (optional, for CMA)
    - ``prior_mean``: untrusted prior M(X) (optional, for CMA)

Plain regression models only use ``features`` and ``true_effect``.
The CMA model additionally uses ``observed_effect``, ``variance``,
and ``prior_mean``.

Reference:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    In Proceedings of Machine Learning Research, volume 259, 563-593.

Examples:
    >>> from pyhealth.datasets import PMLBMetaAnalysisDataset
    >>> from pyhealth.tasks import ConformalMetaAnalysisTask
    >>> dataset = PMLBMetaAnalysisDataset(
    ...     root="./data/pmlb",
    ...     pmlb_dataset_name="1196_BNG_pharynx",
    ...     synthesize_noise=True,
    ...     dev=True,
    ... )
    >>> task = ConformalMetaAnalysisTask()
    >>> samples = dataset.set_task(task)
    >>> print(samples[0])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask

# Columns excluded from the feature vector by default.
DEFAULT_NON_FEATURE_COLUMNS = [
    "visit_id",
    "true_effect",
    "observed_effect",
    "variance",
    "prior_mean",
]


@dataclass
class ConformalMetaAnalysisTask(BaseTask):
    """Regression task for conformal meta-analysis.

    Extracts the feature vector X and true effect U from each trial.
    If the data also contains observed_effect, variance, or
    prior_mean columns, they are carried through on each sample so
    the CMA model can use them.

    Args:
        task_name: Task identifier. Defaults to
            "ConformalMetaAnalysisTask".
        target_column: Column holding the true effect. Defaults
            to "true_effect".
        feature_columns: Optional explicit list of feature column
            names. If None, auto-detects all numeric columns not
            in ``exclude_columns``.
        exclude_columns: Columns to exclude when auto-detecting
            features. Defaults to the standard non-feature columns.

    Attributes:
        input_schema: Schema describing input features.
        output_schema: Schema describing the prediction target.
    """

    task_name: str = "ConformalMetaAnalysisTask"
    target_column: str = "true_effect"
    feature_columns: Optional[List[str]] = None
    exclude_columns: List[str] = field(
        default_factory=lambda: list(DEFAULT_NON_FEATURE_COLUMNS)
    )

    split_column: Optional[str] = None  # e.g., "split"
    split_value: Optional[str] = None  # e.g., "trusted"

    # Source column names in the dataset's attribute dict. Set to
    # None to disable the corresponding sample key entirely (useful
    # for plain regression without synthetic meta-analysis noise).
    observed_column: Optional[str] = "observed_effect"
    variance_column: Optional[str] = "variance"
    prior_column: Optional[str] = "prior_mean"

    # Populated in __post_init__ based on which source columns are
    # configured, so the SampleDataset schema matches what __call__
    # actually emits.
    input_schema: Dict[str, str] = field(
        default_factory=lambda: {"features": "tensor"}
    )
    output_schema: Dict[str, str] = field(
        default_factory=lambda: {"true_effect": "regression"}
    )

    def __post_init__(self) -> None:
        """Build input_schema from the configured source columns.

        Ensures that ``input_schema`` contains exactly the keys that
        ``__call__`` will emit on each sample. Without this, declaring
        (for example) ``observed_effect`` as a required tensor input
        while never emitting it (because the source column is missing)
        would cause downstream ``SampleDataset`` validation to fail.
        """
        if self.observed_column:
            self.input_schema["observed_effect"] = "tensor"
        if self.variance_column:
            self.input_schema["variance"] = "tensor"
        if self.prior_column:
            self.input_schema["prior_mean"] = "tensor"

    def __call__(self, patient) -> List[Dict]:
        """Process one trial into a sample dict.

        Args:
            patient: Patient object from the dataset.

        Returns:
            List with one sample dict, or [] if the trial is missing
            a valid target value.
        """
        events = patient.get_events()
        if not events:
            return []

        attr = events[0].attr_dict

        # Optional split filter
        if self.split_column and self.split_value:
            if attr.get(self.split_column) != self.split_value:
                return []

        # Target (always emitted as "true_effect")
        if self.target_column not in attr:
            return []
        try:
            true_effect = float(attr[self.target_column])
        except (TypeError, ValueError):
            return []

        # Build feature list
        if self.feature_columns is not None:
            feature_keys = self.feature_columns
        else:
            feature_keys = sorted(
                k for k in attr.keys() if k not in set(self.exclude_columns)
            )
        features: List[float] = []
        for k in feature_keys:
            val = attr.get(k)
            try:
                features.append(float(val))
            except (TypeError, ValueError):
                features.append(0.0)

        sample: Dict = {
            "patient_id": patient.patient_id,
            "visit_id": attr.get("visit_id", patient.patient_id),
            "features": features,
            "true_effect": true_effect,
        }

        # Map configured source columns -> fixed sample keys.
        # If a source column is configured but missing or non-numeric
        # for this trial, fall back to 0.0 so the sample's keys match
        # the SampleDataset input_schema built in __post_init__.
        mapping = [
            ("observed_effect", self.observed_column),
            ("variance", self.variance_column),
            ("prior_mean", self.prior_column),
        ]
        for sample_key, source_col in mapping:
            if source_col is None:
                continue
            try:
                sample[sample_key] = float(attr.get(source_col, 0.0))
            except (TypeError, ValueError):
                sample[sample_key] = 0.0

        return [sample]