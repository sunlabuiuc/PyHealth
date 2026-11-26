"""
PyHealth task for left- vs right-hand motor imagery on BCI Competition IV 2a.

Dataset link:
    https://bnci-horizon-2020.eu/database/data-sets (entry 001-2014)

Description:
    Implements ``BCIIV2aLeftRightImagery``: the binary left-hand vs
    right-hand subset of the four-class motor imagery task, a common
    benchmark on this dataset (for example MOABB's ``LeftRightImagery``
    paradigm on ``BNCI2014_001``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from pyhealth.tasks.bciiv2a_motor_imagery import BCIIV2aMotorImageryClassification


class BCIIV2aLeftRightImagery(BCIIV2aMotorImageryClassification):
    """Binary left- vs right-hand motor imagery on BCI Competition IV 2a.

    Works with :class:`~pyhealth.datasets.BCICompetitionIV2aDataset`.

    Keeps only the left-hand and right-hand trials of each session (about
    half of the 288 trials) and labels them ``0`` (left hand) and ``1``
    (right hand). Feet and tongue trials are skipped. The epoch window,
    artifact handling, EOG option and band-pass are the same as in
    :class:`~pyhealth.tasks.BCIIV2aMotorImageryClassification`, and each
    sample carries the same ``session``, ``split`` and ``session_fold``
    fields for the official T-to-E split.

    Two-class left/right-hand imagery is the most common reduced benchmark
    on this dataset (for example MOABB's ``LeftRightImagery`` paradigm on
    ``BNCI2014_001``). The competition itself scored the four-class task.

    Args:
        tmin (float): Window start, seconds after trial start. Default 2.0.
        tmax (float): Window end, seconds after trial start. Default 6.0.
        drop_artifacts (bool): Skip artifact-marked trials. Default ``True``.
        include_eog (bool): Append the 3 EOG channels. Default ``False``.
        l_freq (float | None): Optional band-pass low edge (Hz).
        h_freq (float | None): Optional band-pass high edge (Hz).

    Examples:
        >>> from pyhealth.tasks import BCIIV2aLeftRightImagery
        >>> task = BCIIV2aLeftRightImagery()
        >>> task.output_schema
        {'label': 'binary'}
        >>> from pyhealth.datasets import BCICompetitionIV2aDataset
        >>> ds = BCICompetitionIV2aDataset(root="/data/bnci/001-2014")  # doctest: +SKIP
        >>> samples = ds.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "BCIIV2aLeftRightImagery"
    output_schema: ClassVar[dict[str, Any]] = {"label": "binary"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per left- or right-hand trial of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``sessions`` events.

        Returns:
            Samples as in the four-class task, with ``label`` 0 (left hand)
            or 1 (right hand).
        """
        return [s for s in super().__call__(patient) if s["label"] in (0, 1)]
