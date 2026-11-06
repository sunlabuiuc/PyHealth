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
    BaseConformal: fixed_calibration.run,
    LABEL: fixed_calibration.run,
    RC3P: fixed_calibration.run,
    SCRIB: fixed_calibration.run,
    NeighborhoodLabel: fixed_calibration.run,
    LabelShift: fixed_calibration.run,
    FavMac: favmac.run,
    ClusterLabel: cluster_label.run,
    CovariateLabel: covariate_label.run,
}


def run(
    evaluation: Evaluation, cache_dir: Path | None, output_dir: Path | None
) -> Result:
    """Dispatch by exact method identity, without guessing from its base classes."""
    method = evaluation.params["method"]["class"]

    try:
        adapter = ADAPTERS[method]

    except KeyError:
        raise ValueError(f"No CPBench adapter for {method!r}") from None

    return adapter(evaluation, cache_dir, output_dir)
