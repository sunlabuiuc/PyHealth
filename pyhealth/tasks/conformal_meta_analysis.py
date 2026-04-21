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
from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask

# Columns excluded from the feature vector by default.
DEFAULT_NON_FEATURE_COLUMNS = [
    "visit_id",
    "true_effect",
    "observed_effect",
    "variance",
    "prior_mean",
]


# @dataclass is used here (deviating from the class-attribute pattern
# in COVID19CXRClassification) because this task exposes ten
# configurable knobs and needs per-instance mutable defaults for
# ``input_schema``/``output_schema``. The reference pattern with
# bare class attributes would either require a verbose hand-written
# ``__init__`` or share a single mutable schema dict across all
# instances (a subtle footgun once ``__post_init__`` mutates it).
@dataclass
class ConformalMetaAnalysisTask(BaseTask):
    """Regression task for conformal meta-analysis.

    Extracts the feature vector X and true effect U from each trial.
    If the data also contains observed_effect, variance, or
    prior_mean columns, they are carried through on each sample so
    the CMA model can use them.

    This task is dataset-agnostic: it works with both
    :class:`AmiodaroneTrialDataset` (one event per trial, split into
    trusted/untrusted) and :class:`PMLBMetaAnalysisDataset` (one
    event per simulated trial, with or without synthetic noise).

    Args:
        task_name: Task identifier. Defaults to
            ``"ConformalMetaAnalysisTask"``.
        target_column: Name of the attribute holding the true
            effect U. Defaults to ``"true_effect"``.
        feature_columns: Optional explicit list of feature column
            names. If None, auto-detects all attribute keys not in
            ``exclude_columns`` (sorted for determinism).
        exclude_columns: Columns to exclude when auto-detecting
            features. Defaults to :data:`DEFAULT_NON_FEATURE_COLUMNS`
            (``visit_id``, the target, and the three CMA columns).
        split_column: Optional attribute name used to partition
            trials (e.g. ``"split"`` in the amiodarone dataset).
            When set together with ``split_value``, only trials
            whose ``split_column`` matches are emitted as samples.
        split_value: Value of ``split_column`` to keep (e.g.
            ``"trusted"`` for the 10 placebo-controlled amiodarone
            trials). Ignored unless ``split_column`` is also set.
        event_type: Optional event-type filter passed through to
            ``patient.get_events()``. Defaults to None, which
            accepts all events; set to ``"amiodarone_trials"`` or
            ``"pmlb_meta_analysis"`` to match the reference pattern
            of filtering by event type.
        observed_column: Source column for the noisy observation Y.
            Set to None to disable the ``observed_effect`` sample
            key entirely — used for plain regression without the
            CMA noise columns.
        variance_column: Source column for the within-trial
            variance V. Set to None to disable the ``variance``
            sample key entirely.
        prior_column: Source column for the untrusted prior mean
            M(X). Set to None to disable the ``prior_mean`` sample
            key entirely — used before the prior encoder has been
            trained.

    Attributes:
        input_schema: Dict mapping sample keys to processor types.
            Always contains ``{"features": "tensor"}``; each of
            ``"observed_effect"``, ``"variance"``, and
            ``"prior_mean"`` is added by :meth:`__post_init__` iff
            the corresponding ``*_column`` field is not None.
        output_schema: Dict mapping the regression target. Fixed at
            ``{"true_effect": "regression"}``.
    """

    task_name: str = "ConformalMetaAnalysisTask"
    target_column: str = "true_effect"
    feature_columns: Optional[List[str]] = None
    exclude_columns: List[str] = field(
        default_factory=lambda: list(DEFAULT_NON_FEATURE_COLUMNS)
    )

    split_column: Optional[str] = None  # e.g., "split"
    split_value: Optional[str] = None  # e.g., "trusted"

    # Event-type filter for patient.get_events(). Optional because a
    # single instance of this task may be reused across datasets with
    # different event types (amiodarone_trials vs pmlb_meta_analysis);
    # COVID19CXRClassification hard-codes event_type="covid19_cxr"
    # only because it's bound to a single dataset.
    event_type: Optional[str] = None

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

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one trial into a single sample dict.

        The method iterates the patient's events (one per trial in
        both the amiodarone and PMLB datasets), applies the optional
        split filter, extracts the target value, builds the feature
        vector, and finally attaches any configured CMA columns
        (observed_effect, variance, prior_mean).

        Missing or non-numeric values are handled defensively rather
        than with an exception:

            - Missing or unparseable ``target_column``: the trial
              is skipped (returns ``[]``) because a regression
              sample without a label is unusable.
            - Missing or unparseable feature values: silently
              imputed as ``0.0``. This is safe because the
              dataset's ``prepare_metadata`` already rescales every
              feature to a zero-centered range ([-1, 1] for
              continuous, {-1, 0, 1} for booleans, [0, 1] with 0.5
              NA-imputation for percentages), so 0.0 is a
              reasonable neutral value.
            - Missing or unparseable CMA source column values:
              silently imputed as ``0.0`` so the sample's keys
              stay in sync with ``input_schema``.

        Args:
            patient: A Patient object from the dataset, typed as
                ``Any`` to match the established task pattern.

        Returns:
            A list with either one sample dict or zero dicts. When
            a sample is emitted, its keys are:

                - ``patient_id``: trial identifier
                - ``visit_id``: single visit identifier for the trial
                - ``features``: list of floats (the feature vector X)
                - ``true_effect``: target float (the true effect U)
                - ``observed_effect``: float, present iff
                  ``observed_column`` is not None
                - ``variance``: float, present iff
                  ``variance_column`` is not None
                - ``prior_mean``: float, present iff
                  ``prior_column`` is not None

            Returns ``[]`` when the patient has no events or the
            target value is missing, unparseable, or filtered out
            by ``split_column``/``split_value``.
        """
        # Pass event_type through only when configured, so the task
        # remains dataset-agnostic (amiodarone_trials vs
        # pmlb_meta_analysis) but can still match the reference
        # pattern of filtering by event type when set.
        if self.event_type is not None:
            events = patient.get_events(event_type=self.event_type)
        else:
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

        sample: Dict[str, Any] = {
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