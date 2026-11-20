"""
PyHealth task for autism classification from ABIDE functional connectivity.

Dataset link:
    http://preprocessed-connectomes-project.org/abide/

Dataset paper: (please cite if you use this dataset)
    A. Di Martino, C.-G. Yan, Q. Li, et al. "The autism brain imaging data
    exchange: towards a large-scale evaluation of the intrinsic brain
    architecture in autism." Molecular Psychiatry 19, 659-667 (2014).

Description:
    Implements ``ABIDEAutismClassification`` (binary autism vs. typical
    control from a Pearson-correlation connectivity matrix, BrainGB style)
    and the helper ``abide_connectivity``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

#: ABIDE ``DX_GROUP`` codes.
ABIDE_DX_AUTISM = 1
ABIDE_DX_CONTROL = 2


def abide_connectivity(
    timeseries: np.ndarray,
    fisher_z: bool = False,
    vectorize: bool = False,
) -> np.ndarray:
    """Pearson-correlation connectivity of a ``(time, roi)`` ROI series.

    ROIs with zero variance (no signal in the scan) get zero correlation with
    every other ROI and 1 on the diagonal, instead of NaN.

    Args:
        timeseries (np.ndarray): ROI time series, shape ``(time, n_roi)``.
        fisher_z (bool): Apply ``arctanh`` to the off-diagonal values (the
            diagonal is set to 0). Default ``False``.
        vectorize (bool): Return the strict upper triangle as a vector of
            length ``n_roi * (n_roi - 1) / 2`` instead of the matrix.

    Returns:
        np.ndarray: ``(n_roi, n_roi)`` matrix or upper-triangle vector,
        float32.

    Examples:
        >>> ts = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, 5.0], [3.0, 7.0, 5.0]])
        >>> abide_connectivity(ts).round(2)
        array([[1.  , 0.99, 0.  ],
               [0.99, 1.  , 0.  ],
               [0.  , 0.  , 1.  ]], dtype=float32)
        >>> abide_connectivity(ts, vectorize=True).shape
        (3,)
    """
    x = np.asarray(timeseries, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0)
    valid = std > 0
    z = np.zeros_like(x)
    z[:, valid] = x[:, valid] / std[valid]
    corr = (z.T @ z) / x.shape[0]
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    if fisher_z:
        off = ~np.eye(corr.shape[0], dtype=bool)
        corr[off] = np.arctanh(np.clip(corr[off], -0.999999, 0.999999))
        np.fill_diagonal(corr, 0.0)
    if vectorize:
        corr = corr[np.triu_indices(corr.shape[0], k=1)]
    return corr.astype(np.float32)


class ABIDEAutismClassification(BaseTask):
    """Binary autism vs. typical-control classification on ABIDE I.

    Works with :class:`~pyhealth.datasets.ABIDEDataset`.

    One sample per scan. The input ``connectivity`` is the Pearson
    correlation matrix of the scan's ROI time series (``(200, 200)`` for the
    default cc200 atlas), as in BrainGB, fed through the ``tensor`` processor.
    The label ``autism`` is 1 for ``DX_GROUP == 1`` (autism) and 0 for
    ``DX_GROUP == 2`` (typical control). Samples also carry ``site``,
    ``age`` and ``sex``; ``age`` is ``-1.0`` when the phenotypic file has no
    ``AGE_AT_SCAN``, so filter or impute it before using it as a feature.

    Args:
        fisher_z (bool): Fisher z-transform the correlations. Default
            ``False`` (BrainGB uses raw correlations).
        vectorize (bool): Emit the upper triangle as a flat vector (19,900
            values for cc200), handy for linear or MLP baselines. Default
            ``False``.
        min_timepoints (int): Skip scans with fewer time points. Default 10.

    Examples:
        >>> task = ABIDEAutismClassification()
        >>> task.output_schema
        {'autism': 'binary'}
        >>> from pyhealth.datasets import ABIDEDataset
        >>> ds = ABIDEDataset(root="/data/abide")  # doctest: +SKIP
        >>> samples = ds.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "ABIDEAutismClassification"
    input_schema: ClassVar[dict[str, str]] = {"connectivity": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"autism": "binary"}

    def __init__(
        self,
        fisher_z: bool = False,
        vectorize: bool = False,
        min_timepoints: int = 10,
    ) -> None:
        self.fisher_z = fisher_z
        self.vectorize = vectorize
        self.min_timepoints = int(min_timepoints)
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per ABIDE scan of the patient."""
        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="scans"):
            dx = int(event.dx_group)
            if dx not in (ABIDE_DX_AUTISM, ABIDE_DX_CONTROL):
                continue
            ts = np.loadtxt(str(event.roi_file), ndmin=2)
            if ts.shape[0] < self.min_timepoints or not np.isfinite(ts).all():
                logger.warning("Skipping %s: unusable ROI series", event.record_id)
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "site": str(event.site),
                    "age": float(event.age),
                    "sex": str(event.sex),
                    "connectivity": abide_connectivity(
                        ts, fisher_z=self.fisher_z, vectorize=self.vectorize
                    ),
                    "autism": int(dx == ABIDE_DX_AUTISM),
                }
            )
        return samples
