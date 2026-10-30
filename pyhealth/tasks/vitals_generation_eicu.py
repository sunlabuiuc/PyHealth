"""Static vital-sign vectors per eICU ICU stay, for tabular generators.

Each patient contributes one sample: their first ICU stay, summarized as age
plus the mean of four vital signs over the stay's first 24 hours. The task
only extracts and cleans values; generators such as
:class:`~pyhealth.models.GAN` scale them, fitting the scale on the training
split.
"""

from typing import Any, ClassVar

from pyhealth.data.data import Patient

from .base_task import BaseTask


def _to_float(value: Any) -> float | None:
    """Parses one eICU value; missing or non-numeric values become None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_age(value: Any) -> float | None:
    """Parses eICU's age, which is a string and reads "> 89" for 90 and over."""
    if isinstance(value, str) and value.strip() == "> 89":
        return 90.0
    return _to_float(value)


class CARTVitalsGenerationEICU(BaseTask):
    """One vital-sign vector per eICU patient, for generating new patients.

    The features are those of the Cardiac Arrest Risk Triage (CART) score
    (Churpek et al., 2012) as used for eICU by Zhang et al. (2024): age,
    respiratory rate, heart rate and diastolic blood pressure, plus mean
    blood pressure. Unlike that work, which models vitals over time, each
    sample here is one static snapshot of a stay.

    Takes each patient's first ICU stay (earliest discharge year, then
    health-system stay, then unit visit number) and averages each vital over
    the stay's first 24 hours (``0 <= observationoffset < 1440`` minutes).
    Readings outside plausible ranges are treated as charting errors and
    dropped before averaging. Stays without a heart rate, respiration rate or
    non-invasive blood pressure in the window, or without an age, are skipped.

    The sample's ``vitals`` is ``[age, heart rate, respiration rate,
    diastolic BP, mean BP]``, in the units eICU records (years, beats/min,
    breaths/min, mmHg, mmHg); see :attr:`feature_names`.

    References:
        Churpek, M. M., et al. (2012). Derivation of a cardiac arrest
        prediction model using ward vital signs. *Critical Care Medicine*,
        40(7), 2102-2108.

        Zhang, X., Pu, Y., et al. (2024). Trajectory Flow Matching with
        Applications to Clinical Time Series Modeling. NeurIPS 2024.
        https://arxiv.org/abs/2410.21154

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import CARTVitalsGenerationEICU
        >>> dataset = eICUDataset(  # doctest: +SKIP
        ...     root="https://physionet.org/files/eicu-crd-demo/2.0.1",
        ...     tables=["vitalperiodic", "vitalaperiodic"],
        ... )
        >>> samples = dataset.set_task(CARTVitalsGenerationEICU())  # doctest: +SKIP
        >>> samples[0]["vitals"].shape  # doctest: +SKIP
        torch.Size([5])
    """

    task_name: str = "CARTVitalsGenerationEICU"
    input_schema: ClassVar[dict[str, str]] = {"vitals": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {}

    feature_names: ClassVar[tuple[str, ...]] = (
        "age",
        "heartrate",
        "respiration",
        "noninvasivediastolic",
        "noninvasivemean",
    )
    # (table, attribute) -> plausible (low, high) range.
    vitals: ClassVar[dict[tuple[str, str], tuple[float, float]]] = {
        ("vitalperiodic", "heartrate"): (20, 250),
        ("vitalperiodic", "respiration"): (4, 60),
        ("vitalaperiodic", "noninvasivediastolic"): (20, 150),
        ("vitalaperiodic", "noninvasivemean"): (30, 200),
    }
    window_minutes: int = 24 * 60

    @staticmethod
    def _stay_order(stay: Any) -> tuple[float, float, float]:
        keys = ("hospitaldischargeyear", "patienthealthsystemstayid", "unitvisitnumber")
        return tuple(
            value if (value := _to_float(getattr(stay, key, None))) is not None
            else float("inf")
            for key in keys
        )

    def _window_mean(
        self, events: list, attr: str, low: float, high: float
    ) -> float | None:
        """Means one vital's plausible readings within the first window."""
        values = []
        for event in events:
            offset = _to_float(getattr(event, "observationoffset", None))
            value = _to_float(getattr(event, attr, None))
            if offset is None or value is None:
                continue
            if 0 <= offset < self.window_minutes and low <= value <= high:
                values.append(value)
        return sum(values) / len(values) if values else None

    def __call__(self, patient: Patient) -> list[dict]:
        """Summarize the patient's first ICU stay as one vital-sign vector."""
        stays = patient.get_events(event_type="patient")
        if not stays:
            return []
        stay = min(stays, key=self._stay_order)
        age = _parse_age(getattr(stay, "age", None))
        if age is None:
            return []

        stay_id = str(getattr(stay, "patientunitstayid", ""))
        events = {
            table: patient.get_events(
                event_type=table, filters=[("patientunitstayid", "==", stay_id)]
            )
            for table in {table for table, _ in self.vitals}
        }
        means = [
            self._window_mean(events[table], attr, low, high)
            for (table, attr), (low, high) in self.vitals.items()
        ]
        if any(mean is None for mean in means):
            return []
        return [
            {
                "patient_id": patient.patient_id,
                "visit_id": stay_id,
                "vitals": [age, *means],
            }
        ]
