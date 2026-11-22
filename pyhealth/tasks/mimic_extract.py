"""Binary ICU prediction tasks on MIMIC-Extract.

Reproduces the four prediction tasks used to evaluate CaliForest:

    Park, Yubin, and Joyce C. Ho. "CaliForest: Calibrated Random Forest for
    Health Data." ACM Conference on Health, Inference, and Learning (CHIL),
    2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC8299436/

on data produced by MIMIC-Extract (Wang et al., CHIL 2020):

    - in-hospital mortality  -> :class:`MIMICExtractMortalityHospital`
    - in-ICU mortality       -> :class:`MIMICExtractMortalityICU`
    - ICU length of stay > 3 -> :class:`MIMICExtractLengthOfStay` (threshold 3)
    - ICU length of stay > 7 -> :class:`MIMICExtractLengthOfStay` (threshold 7)

Each ICU stay becomes one sample: hourly vitals/labs from the first
``window_hours`` of the stay, flattened to a fixed-length feature vector, and
a binary label. Stays shorter than ``window_hours + gap_hours`` are excluded,
so no data from the gap before the prediction time is used.

Expected event tables (names are set in the constants below):

    ``patients``: one event per ICU stay, with ``icustay_id``, ``mort_hosp``,
    ``mort_icu`` and ``los_icu`` (ICU length of stay in days).

    ``vitals_labs``: one event per (stay, hour, variable), with
    ``icustay_id``, ``hours_in`` (hours since ICU admission), ``variable``,
    and one column per aggregate (``mean``, ``count``, ``std``).
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from pyhealth.data import Patient

from .base_task import BaseTask

# Table and column names this task reads. Keep them here so they can be
# matched to MIMICExtractDataset's config in one place.
STAY_TABLE = "patients"
HOURLY_TABLE = "vitals_labs"
STAY_ID = "icustay_id"
HOUR = "hours_in"
VARIABLE = "variable"
AGGREGATES = ("mean", "count", "std")


class MIMICExtractBinaryTask(BaseTask):
    """Base class for binary per-ICU-stay tasks on MIMIC-Extract.

    Handles the cohort filter, the observation window, and feature
    construction. Subclasses only define :meth:`label`.

    Features for one stay form a ``window_hours x len(variables) x
    len(aggregates)`` array, flattened to one vector. Missing hourly means are
    carried forward within the window and then set to ``fill_value``; missing
    counts are 0; missing standard deviations are ``fill_value``.

    Args:
        variables: Ordered names of the hourly variables to use as features.
            The order fixes the feature layout, so it must be the same for
            every stay. Names are matched case-insensitively (stored lower
            case, as ``MIMICExtractDataset`` stores them), so MIMIC-Extract's
            own names such as ``"Heart Rate"`` work.
        window_hours: Hours from ICU admission used as input. Default 24.
        gap_hours: Hours between the end of the window and the prediction
            time. Stays shorter than ``window_hours + gap_hours`` are
            excluded. Default 6.
        aggregates: Which hourly aggregates to use, any of ``"mean"``,
            ``"count"``, ``"std"``. Default: all three.
        fill_value: Value for means and standard deviations that are still
            missing after carrying forward. Default 0.0.
        code_mapping: Passed through to :class:`BaseTask`.

    Raises:
        ValueError: If ``variables`` is empty or has names that are equal
            ignoring case, ``window_hours`` is not positive, ``gap_hours`` is
            negative, or ``aggregates`` is empty or contains an unknown name.

    Examples:
        Subclasses set ``task_name`` and implement :meth:`label`:

        >>> from pyhealth.tasks import MIMICExtractBinaryTask
        >>> class ProlongedICUStay(MIMICExtractBinaryTask):
        ...     task_name = "ProlongedICUStay"
        ...     def label(self, stay):
        ...         return int(float(stay["los_icu"]) > 14)
        >>> task = ProlongedICUStay(variables=["heart rate"], window_hours=12)
        >>> task.window_hours, task.output_schema
        (12, {'label': 'binary'})
    """

    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        variables: Sequence[str],
        window_hours: int = 24,
        gap_hours: int = 6,
        aggregates: Sequence[str] = AGGREGATES,
        fill_value: float = 0.0,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(code_mapping=code_mapping)
        if len(variables) == 0:
            raise ValueError("variables must not be empty")
        if window_hours <= 0:
            raise ValueError(f"window_hours must be positive, got {window_hours}")
        if gap_hours < 0:
            raise ValueError(f"gap_hours must be >= 0, got {gap_hours}")
        unknown = set(aggregates) - set(AGGREGATES)
        if len(aggregates) == 0 or unknown:
            raise ValueError(
                f"aggregates must be a non-empty subset of {AGGREGATES}, "
                f"got {list(aggregates)}"
            )
        names = [str(v).strip().lower() for v in variables]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"variables must be unique ignoring case, got duplicates of "
                f"{duplicates}"
            )
        self.variables = names
        self.window_hours = window_hours
        self.gap_hours = gap_hours
        self.aggregates = [a for a in AGGREGATES if a in aggregates]
        self.fill_value = fill_value

    def label(self, stay: Any) -> int:
        """Returns the binary label for one ICU stay.

        Args:
            stay: The stay's event from the ``patients`` table.

        Returns:
            0 or 1.
        """
        raise NotImplementedError

    def _features(self, patient: Patient, stay_id: Any) -> list[float]:
        """Builds the flattened feature vector for one ICU stay.

        Args:
            patient: The patient the stay belongs to.
            stay_id: The ``icustay_id`` of the stay.

        Returns:
            A list of ``window_hours * len(variables) * len(aggregates)``
            floats.
        """
        n_hours, n_vars = self.window_hours, len(self.variables)
        var_index = {name: i for i, name in enumerate(self.variables)}
        grid = {
            agg: np.full((n_hours, n_vars), np.nan) for agg in self.aggregates
        }

        hourly = patient.get_events(
            event_type=HOURLY_TABLE,
            filters=[(STAY_ID, "==", stay_id), (HOUR, "<", n_hours)],
        )
        for event in hourly:
            col = var_index.get(str(event[VARIABLE]).strip().lower())
            if col is None:
                continue
            row = int(event[HOUR])
            for agg in self.aggregates:
                value = event[agg]
                if value is not None:
                    grid[agg][row, col] = float(value)

        if "mean" in grid:
            means = grid["mean"]
            for row in range(1, n_hours):  # carry the last mean forward
                gaps = np.isnan(means[row])
                means[row, gaps] = means[row - 1, gaps]
        for agg, values in grid.items():
            missing = 0.0 if agg == "count" else self.fill_value
            values[np.isnan(values)] = missing

        stacked = np.stack([grid[agg] for agg in self.aggregates], axis=-1)
        return stacked.ravel().tolist()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Creates one sample per eligible ICU stay of ``patient``.

        Args:
            patient: A patient from ``MIMICExtractDataset``.

        Returns:
            A list of samples with ``patient_id``, ``visit_id`` (the
            ``icustay_id``), ``features``, and ``label``.
        """
        min_stay_days = (self.window_hours + self.gap_hours) / 24
        samples = []
        for stay in patient.get_events(event_type=STAY_TABLE):
            if float(stay["los_icu"]) < min_stay_days:
                continue
            stay_id = stay[STAY_ID]
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(stay_id),
                    "features": self._features(patient, stay_id),
                    "label": self.label(stay),
                }
            )
        return samples


class MIMICExtractMortalityHospital(MIMICExtractBinaryTask):
    """In-hospital mortality from the first hours of an ICU stay.

    Label is MIMIC-Extract's ``mort_hosp``: 1 if the patient died during the
    hospital admission.

    Examples:
        >>> from pyhealth.tasks import MIMICExtractMortalityHospital
        >>> task = MIMICExtractMortalityHospital(variables=["heart rate"])
        >>> task.task_name
        'MIMICExtractMortalityHospital'
    """

    task_name: str = "MIMICExtractMortalityHospital"

    def label(self, stay: Any) -> int:
        """Returns ``mort_hosp`` as 0 or 1."""
        return int(stay["mort_hosp"])


class MIMICExtractMortalityICU(MIMICExtractBinaryTask):
    """In-ICU mortality from the first hours of an ICU stay.

    Label is MIMIC-Extract's ``mort_icu``: 1 if the patient died in the ICU.

    Examples:
        >>> from pyhealth.tasks import MIMICExtractMortalityICU
        >>> task = MIMICExtractMortalityICU(variables=["heart rate"])
        >>> task.task_name
        'MIMICExtractMortalityICU'
    """

    task_name: str = "MIMICExtractMortalityICU"

    def label(self, stay: Any) -> int:
        """Returns ``mort_icu`` as 0 or 1."""
        return int(stay["mort_icu"])


class MIMICExtractLengthOfStay(MIMICExtractBinaryTask):
    """Prolonged ICU stay from the first hours of an ICU stay.

    Label is 1 if ICU length of stay (``los_icu``, in days) is greater than
    ``threshold_days``. The paper uses 3 and 7.

    Args:
        variables: See :class:`MIMICExtractBinaryTask`.
        threshold_days: Length of stay, in days, above which the label is 1.
            Default 3.
        **kwargs: Other arguments of :class:`MIMICExtractBinaryTask`.

    Raises:
        ValueError: If ``threshold_days`` is not positive.

    Examples:
        >>> from pyhealth.tasks import MIMICExtractLengthOfStay
        >>> task = MIMICExtractLengthOfStay(
        ...     variables=["heart rate"], threshold_days=7
        ... )
        >>> task.threshold_days
        7
    """

    task_name: str = "MIMICExtractLengthOfStay"

    def __init__(
        self, variables: Sequence[str], threshold_days: float = 3, **kwargs: Any
    ) -> None:
        super().__init__(variables=variables, **kwargs)
        if threshold_days <= 0:
            raise ValueError(
                f"threshold_days must be positive, got {threshold_days}"
            )
        self.threshold_days = threshold_days

    def label(self, stay: Any) -> int:
        """Returns 1 if ``los_icu`` is greater than ``threshold_days``."""
        return int(float(stay["los_icu"]) > self.threshold_days)
