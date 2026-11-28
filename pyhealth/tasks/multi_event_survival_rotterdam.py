# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
"""Recurrence and death times/indicators for the Rotterdam cohort."""

from typing import ClassVar

from pyhealth.data import Patient

from .base_task import BaseTask

ROTTERDAM_FEATURES: tuple[str, ...] = (
    "year",
    "age",
    "meno",
    "grade",
    "nodes",
    "pgr",
    "er",
    "hormon",
    "chemo",
    "size_mm",
)
ROTTERDAM_TREATMENT_FEATURES: tuple[str, ...] = ("hormon", "chemo")
ROTTERDAM_EVENTS: tuple[str, ...] = ("recurrence", "death")
# Representative tumor diameters (mm) for the source size categories, as in
# the MENSA authors' Rotterdam preprocessing.
ROTTERDAM_SIZE_MM: dict[str, float] = {"<=20": 10.0, "20-50": 35.0, ">50": 75.0}


class MultiEventSurvivalRotterdam(BaseTask):
    """Baseline features with recurrence and death times and indicators.

    Works with :class:`~pyhealth.datasets.RotterdamDataset`.

    Each patient's single ``rotterdam`` record is validated and converted
    from strings by :func:`~pyhealth.datasets.rotterdam.parse_rotterdam_record`.
    The task emits ``features: [10]`` in :attr:`feature_names` order, with
    tumor size coded as 10, 35 or 75 mm. With ``use_treatment=False``, the
    hormonal therapy and chemotherapy indicators (``hormon``, ``chemo``) are
    removed, leaving ``features: [8]`` in the same relative order. Labels are
    the same either way. It also emits
    ``event_times: [rtime, dtime]`` and ``event_indicators: [recur, death]``,
    where 1 is an observed event and 0 is censoring at that time.

    Times stay in source days and are never shifted, scaled or replaced. A
    recurrence censored before a later death keeps its own ``rtime``, and
    tied event times are kept. Features are not standardized here; fit any
    standardization on training patients only, after splitting.

    Like PyHealth's other duration/event survival tasks, the labels use the
    ``"tensor"`` processor, so models consuming them set ``mode=None`` and
    supply their own censored likelihood to the standard Trainer.

    Args:
        use_treatment: Whether to include ``hormon`` and ``chemo``. Default
            True (ten features); False gives eight features.

    Attributes:
        use_treatment: Whether the treatment indicators are included.
        feature_names: The ten or eight feature names, in output order.
        event_names: Names of the two events, in label order.

    Examples:
        >>> from pyhealth.tasks import MultiEventSurvivalRotterdam
        >>> task = MultiEventSurvivalRotterdam()
        >>> task.output_schema
        {'event_times': 'tensor', 'event_indicators': 'tensor'}
        >>> task.feature_names[-1]
        'size_mm'
        >>> len(MultiEventSurvivalRotterdam(use_treatment=False).feature_names)
        8
        >>> from pyhealth.datasets import RotterdamDataset
        >>> samples = RotterdamDataset(root="../data").set_task()  # doctest: +SKIP
    """

    task_name: str = "MultiEventSurvivalRotterdam"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {
        "event_times": "tensor",
        "event_indicators": "tensor",
    }
    event_names: ClassVar[tuple[str, ...]] = ROTTERDAM_EVENTS

    def __init__(self, use_treatment: bool = True) -> None:
        """Choose the feature set.

        Args:
            use_treatment: Whether to include ``hormon`` and ``chemo``.

        Raises:
            TypeError: If ``use_treatment`` is not a bool.
        """
        super().__init__()
        if not isinstance(use_treatment, bool):
            raise TypeError("use_treatment must be True or False")
        self.use_treatment = use_treatment
        # Stored per instance so each variant gets its own set_task cache.
        self.feature_names: tuple[str, ...] = tuple(
            name
            for name in ROTTERDAM_FEATURES
            if use_treatment or name not in ROTTERDAM_TREATMENT_FEATURES
        )

    def __call__(self, patient: Patient) -> list[dict]:
        """Return one sample, rejecting missing or multiple source records.

        Args:
            patient: Patient containing exactly one ``rotterdam`` event whose
                attributes are the 14 non-ID source columns.

        Returns:
            One sample with the patient ID, features, event times and event
            indicators.

        Raises:
            ValueError: If the patient does not have exactly one record or a
                value fails validation. Messages name the patient.
        """
        # Imported here because pyhealth.datasets imports pyhealth.tasks on load.
        from pyhealth.datasets.rotterdam import parse_rotterdam_record

        events = patient.get_events("rotterdam")
        if len(events) != 1:
            raise ValueError(
                f"Rotterdam pid {patient.patient_id!r}: expected exactly one "
                f"rotterdam event, got {len(events)}"
            )
        values = parse_rotterdam_record(events[0].attr_dict, patient.patient_id)
        values["size_mm"] = ROTTERDAM_SIZE_MM[values["size"]]
        return [
            {
                "patient_id": patient.patient_id,
                "features": [float(values[name]) for name in self.feature_names],
                "event_times": [values["rtime"], values["dtime"]],
                "event_indicators": [values["recur"], values["death"]],
            }
        ]
