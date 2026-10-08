"""Evaluation metrics for synthetic (generative) EHR data.

This subpackage provides metrics for assessing synthetic electronic health
record (EHR) data along three axes:

    - **Privacy** (:mod:`pyhealth.metrics.generative.privacy`): NNAAR,
      membership inference, and discriminator-based adversarial accuracy.
    - **Utility / fidelity** (:mod:`pyhealth.metrics.generative.utility`):
      machine learning efficacy (TRTR vs TSTR) and code-prevalence similarity.

The convenience function :func:`evaluate_synthetic_ehr` runs the full suite
and returns a single merged dictionary of ``{metric_name: (mean, std)}``.

Input format:
    Every metric consumes plain pandas dataframes in *flat / long* format --
    **one row per (patient, visit, code) event** -- so the logic stays easy to
    inspect. By default each dataframe has four columns
    ``[id, time, visit_codes, labels]`` (override the names via the
    ``subject_col`` / ``visit_col`` / ``code_col`` / ``label_col`` arguments):

        - ``id`` (``subject_col``): patient identifier. Any hashable value;
          commonly ``str`` or ``int``.
        - ``time`` (``visit_col``): visit index / timestep. Sortable, usually
          ``int``; visits are ordered per patient by this column.
        - ``visit_codes`` (``code_col``): a **single** medical code for this
          row (``str`` or ``int``). One code per row -- a visit containing *k*
          codes spans *k* rows. Cells are scalars, **not** lists or arrays.
        - ``labels`` (``label_col``): per-patient binary label (0/1, ``int``).

    The real (``train_ehr``, ``test_ehr``) and synthetic (``syn_ehr``)
    dataframes must all share this same schema. ``labels`` is ignored by the
    privacy metrics and is overwritten internally by the utility metrics, but
    is required so every dataframe has a uniform schema.

Why dataframes (and not plain ``List[...]``)?
    The flat dataframe is purely the *interchange* format -- a single, uniform
    interface shared by every metric and produced once by
    :func:`pyhealth.tasks.to_evaluation_dataframe`. Internally each family uses
    whatever representation is most natural:

        - **Privacy** metrics immediately reduce the frame to a nested
          ``List[List[set]]`` (sequence of per-visit code sets) via
          ``convert_visits_to_sets`` and do all distance work on plain Python
          lists -- no pandas in the hot loop.
        - **Utility / fidelity** metrics genuinely benefit from the dataframe:
          code-prevalence uses ``groupby(...).nunique()``, MLE builds the
          next-visit-prediction supervision with grouped per-patient label
          assignment, and the discriminator metric concatenates / filters /
          relabels real-vs-synthetic rows. Re-implementing these on raw lists
          would be more code for no gain.

    So the long-form frame keeps the public API consistent and the heavy
    transforms readable, while the per-metric internals are free to drop down
    to lists where that is simpler.

Note:
    The MLE (utility) component is currently hard-coded to next-visit
    prediction and is therefore only meaningful for sequential generators
    (HALO, GPT2, PromptEHR). It will be expanded to support pluggable
    downstream tasks so that bag-of-codes generators (MedGAN, CorGAN) can
    be evaluated with a static-label task (e.g. mortality, readmission).
    Until then, prefer the privacy and prevalence metrics when evaluating
    MedGAN/CorGAN output.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from .privacy import (
    calc_membership_inference,
    calc_nnaar,
    compute_discriminator_privacy,
)
from .utility import compute_mle, compute_prevalence_metrics
from .utils import train_lstm_model, train_sklearn_model

logger = logging.getLogger(__name__)

__all__ = [
    "calc_nnaar",
    "calc_membership_inference",
    "compute_discriminator_privacy",
    "compute_mle",
    "compute_prevalence_metrics",
    "evaluate_synthetic_ehr",
]


def evaluate_synthetic_ehr(
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    sample_size: int = 1000,
    mode: str = "lstm",
    metrics: str = "all",
    lstm_params: Optional[Dict] = None,
    sklearn_params: Optional[Dict] = None,
    n_bootstraps: int = 100,
    n_runs: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Runs the full synthetic-EHR evaluation suite.

    Computes privacy and/or utility metrics comparing synthetic EHR data
    against real train/test data, and returns a single merged dictionary.

    All three dataframes are flat / long-format (one row per
    ``(patient, visit, code)`` event) and must share the same schema. See the
    module docstring (:mod:`pyhealth.metrics.generative`) for the full column
    contract, and the example below for how to build them.

    Args:
        train_ehr: Real training EHR dataframe, flat
            ``[id, time, visit_codes, labels]`` format.
        test_ehr: Real held-out test EHR dataframe; same schema as ``train_ehr``.
        syn_ehr: Synthetic EHR dataframe; same schema as ``train_ehr``.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes (one code per row).
        label_col: Column name for the per-patient binary label.
        sample_size: Number of patients sampled per dataset for the
            privacy metrics.
        mode: Predictive backbone for the utility metrics; ``"lstm"`` uses the
            built-in LSTM classifier, ``"rf"`` uses a random forest.
        metrics: Which metric group to compute: ``"all"``, ``"privacy"`` or
            ``"utility"``.
        lstm_params: Optional overrides for the LSTM (``embed_dim``,
            ``hidden_dim``, ``batch_size``, ``epochs``).
        sklearn_params: Optional overrides for the sklearn model (``model``).
        n_bootstraps: Number of bootstrap resamples for the utility metrics.
        n_runs: Number of sampling runs for the privacy metrics.

    Returns:
        Dictionary mapping each metric name to a ``(mean, std)`` tuple.

    Raises:
        ValueError: If ``metrics`` or ``mode`` is not a recognized value.

    Examples:
        The inputs are flat / long-format dataframes -- one row per
        ``(patient, visit, code)`` event -- with four columns by default:

            - ``id``: patient identifier (any hashable; ``str`` or ``int``).
            - ``time``: visit index / timestep (sortable; usually ``int``).
            - ``visit_codes``: a single medical code for this row (``str`` or
              ``int``). One code per row -- a visit with *k* codes spans *k*
              rows; cells are scalars, not lists/arrays.
            - ``labels``: per-patient binary label (0/1, ``int``).

        ``train_ehr``, ``test_ehr`` and ``syn_ehr`` must all share this schema.

        >>> import pandas as pd
        >>> from pyhealth.metrics.generative import evaluate_synthetic_ehr
        >>>
        >>> # One row per (patient, visit, code). Patient "p0" has two visits
        >>> # (time 0 with two codes, time 1 with one code); "p1" has one visit.
        >>> rows = [
        ...     {"id": "p0", "time": 0, "visit_codes": "428.0", "labels": 0},
        ...     {"id": "p0", "time": 0, "visit_codes": "250.00", "labels": 0},
        ...     {"id": "p0", "time": 1, "visit_codes": "401.9", "labels": 0},
        ...     {"id": "p1", "time": 0, "visit_codes": "428.0", "labels": 0},
        ... ]
        >>> train_ehr = pd.DataFrame(rows)
        >>> test_ehr = train_ehr.copy()  # same schema; real held-out patients
        >>> syn_ehr = train_ehr.copy()   # same schema; generator output
        >>>
        >>> results = evaluate_synthetic_ehr(
        ...     train_ehr, test_ehr, syn_ehr, metrics="privacy", sample_size=2
        ... )
        >>> nnaar_mean, nnaar_std = results["nnaar"]
        >>>
        >>> # Custom column names: pass *_col to match your dataframe.
        >>> results = evaluate_synthetic_ehr(
        ...     train_ehr, test_ehr, syn_ehr,
        ...     subject_col="id", visit_col="time",
        ...     code_col="visit_codes", label_col="labels",
        ... )
    """
    if metrics not in ("all", "privacy", "utility"):
        raise ValueError(
            f"Unknown metrics group: {metrics!r}. "
            "Expected 'all', 'privacy' or 'utility'."
        )
    if mode not in ("lstm", "rf"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'lstm' or 'rf'.")

    lstm_params = lstm_params or {}
    sklearn_params = sklearn_params or {}
    final_output: Dict[str, Tuple[float, float]] = {}

    if metrics in ("all", "privacy"):
        final_output.update(
            calc_nnaar(
                train_ehr,
                test_ehr,
                syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                sample_size=sample_size,
                n_runs=n_runs,
            )
        )
        final_output.update(
            calc_membership_inference(
                train_ehr,
                test_ehr,
                syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                num_attack_samples=sample_size,
                n_runs=n_runs,
            )
        )

    if metrics in ("all", "utility"):
        if mode == "lstm":
            train_fn = train_lstm_model
            train_kwargs = {
                "embed_dim": lstm_params.get("embed_dim", 32),
                "hidden_dim": lstm_params.get("hidden_dim", 32),
                "batch_size": lstm_params.get("batch_size", 32),
                "epochs": lstm_params.get("epochs", 5),
                "verbose": False,
            }
        else:
            train_fn = train_sklearn_model
            train_kwargs = {"model": sklearn_params.get("model", "rf")}

        final_output.update(
            compute_mle(
                train_fn=train_fn,
                train_ehr=train_ehr,
                test_ehr=test_ehr,
                syn_ehr=syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                n_bootstraps=n_bootstraps,
                **train_kwargs,
            )
        )
        final_output.update(
            compute_discriminator_privacy(
                train_fn=train_fn,
                train_ehr=train_ehr,
                test_ehr=test_ehr,
                syn_ehr=syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                n_bootstraps=n_bootstraps,
                **train_kwargs,
            )
        )
        final_output.update(
            compute_prevalence_metrics(
                train_ehr,
                syn_ehr,
                subject_col=subject_col,
                code_col=code_col,
                n_bootstraps=n_bootstraps,
            )
        )

    return final_output
