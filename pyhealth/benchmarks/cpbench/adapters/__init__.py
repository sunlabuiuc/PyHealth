"""Select the evaluation procedure for each explicitly supported method."""

from pathlib import Path

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.calib.predictionset import (
    LABEL,
    RC3P,
    SCRIB,
    BaseConformal,
    ClusterLabel,
    CovariateLabel,
    FavMac,
    LabelShift,
    NeighborhoodLabel,
)

from . import cluster_label, covariate_label, favmac, fixed_calibration

ADAPTERS = {
    BaseConformal: fixed_calibration,
    LABEL: fixed_calibration,
    RC3P: fixed_calibration,
    SCRIB: fixed_calibration,
    NeighborhoodLabel: fixed_calibration,
    LabelShift: fixed_calibration,
    FavMac: favmac,
    ClusterLabel: cluster_label,
    CovariateLabel: covariate_label,
}


def run(evaluation: Evaluation, cache_dir: Path | None) -> Result:
    """Dispatch by exact method identity, without guessing from its base classes.

    Examples:
        >>> run(Evaluation({"method": {"class": dict}}), cache_dir=None)
        Traceback (most recent call last):
            ...
        ValueError: No CPBench adapter for <class 'dict'>
    """
    method = evaluation.params["method"]["class"]

    try:
        adapter = ADAPTERS[method]

    except KeyError:
        raise ValueError(f"No CPBench adapter for {method!r}") from None

    return adapter.run(evaluation, cache_dir)


def required_model_outputs(evaluation):
    """Return an adapter's split-to-field requirements for base-model inference.

    Runners can union these requirements across methods before collecting once.

    Examples:
        >>> ev = Evaluation({"method": {"class": FavMac}})
        >>> required_model_outputs(ev)
        {'calibration': ('logit',), 'test': ('logit', 'y_prob')}
    """
    method = evaluation.params["method"]["class"]

    requirement = getattr(ADAPTERS[method], "required_model_outputs", None)
    if requirement is None:
        raise NotImplementedError("adapter does not expose base-output requirements")

    return requirement(evaluation)
