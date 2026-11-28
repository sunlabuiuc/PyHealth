"""MIMIC-III 24-hour decompensation task for dynamic survival analysis.

Contributors:
    Zhipeng Zhou
    Xinyue Zuo

Related paper:
    Dynamic Survival Analysis for Early Event Prediction
    https://proceedings.mlr.press/v248/yeche24a.html

This module implements hourly MIMIC-III decompensation sample generation and
survival targets for DSA-style early event prediction.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask


class DecompensationMIMIC3(BaseTask):
    """Hourly decompensation prediction for MIMIC-III ICU stays.

    Each sample represents one hourly prediction time during an ICU stay.
    Clinical measurements are mapped to the 17 MIMIC-III benchmark channels
    and encoded as 12 continuous features, 47 categorical one-hot features,
    and 17 observation-mask features, for 76 dynamic features per timestep.

    The prediction target is represented as a discrete-time survival target
    over ``horizon_hours`` future hourly bins. ``hazard_label`` identifies the
    event bin, ``hazard_mask`` identifies observed future bins, and ``y_true``
    indicates whether death occurs strictly within the prediction horizon.

    The cohort follows the MIMIC-III benchmark convention of adult ICU stays
    and excludes transfers between wards or care units within the stay.

    Args:
        horizon_hours: Number of future hourly hazard bins. Defaults to 24.
        min_observation_hours: Earliest ICU hour at which a prediction sample
            may be generated. Defaults to 5.
        survtls_lengthscale: Optional survTLS Gaussian smoothing lengthscale.
            When provided, event-time survival targets are smoothed using the
            paper's released-code formulation. For MIMIC-III decompensation,
            the paper uses a lengthscale of 8. Defaults to None.

    Examples:
        >>> from pyhealth.tasks import DecompensationMIMIC3
        >>> task = DecompensationMIMIC3(horizon_hours=24, survtls_lengthscale=8.0)
        >>> task.horizon_hours
        24
        >>> sorted(task.output_schema)
        ['hazard_label', 'hazard_mask', 'y_true']

    Apply the configured task to a MIMIC-III patient to generate hourly
    samples, or pass it to the dataset's task-processing interface.

    Returns:
        Samples containing dynamic continuous, categorical, and observation
        mask sequences together with discrete-time survival targets.
    """

    task_name: str = "DecompensationMIMIC3"

    CHANNELS: ClassVar[tuple[str, ...]] = (
        "Capillary refill rate",
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
        "Glucose",
        "Heart Rate",
        "Height",
        "Mean blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "Weight",
        "pH",
    )

    CATEGORICAL_VALUES: ClassVar[dict[str, tuple[str, ...]]] = {
        "Capillary refill rate": ("0.0", "1.0"),
        "Glascow coma scale eye opening": (
            "To Pain",
            "3 To speech",
            "1 No Response",
            "4 Spontaneously",
            "None",
            "To Speech",
            "Spontaneously",
            "2 To pain",
        ),
        "Glascow coma scale motor response": (
            "1 No Response",
            "3 Abnorm flexion",
            "Abnormal extension",
            "No response",
            "4 Flex-withdraws",
            "Localizes Pain",
            "Flex-withdraws",
            "Obeys Commands",
            "Abnormal Flexion",
            "6 Obeys Commands",
            "5 Localizes Pain",
            "2 Abnorm extensn",
        ),
        "Glascow coma scale total": (
            "11", "10", "13", "12", "15", "14",
            "3", "5", "4", "7", "6", "9", "8",
        ),
        "Glascow coma scale verbal response": (
            "1 No Response",
            "No Response",
            "Confused",
            "Inappropriate Words",
            "Oriented",
            "No Response-ETT",
            "5 Oriented",
            "Incomprehensible sounds",
            "1.0 ET/Trach",
            "4 Confused",
            "2 Incomp sounds",
            "3 Inapprop words",
        ),
    }

    CHARTEVENT_ITEMIDS: ClassVar[dict[int, str]] = {
        115: "Capillary refill rate",
        3348: "Capillary refill rate",
        8377: "Capillary refill rate",
        8368: "Diastolic blood pressure",
        8440: "Diastolic blood pressure",
        8441: "Diastolic blood pressure",
        8502: "Diastolic blood pressure",
        8503: "Diastolic blood pressure",
        8504: "Diastolic blood pressure",
        8507: "Diastolic blood pressure",
        8508: "Diastolic blood pressure",
        8555: "Diastolic blood pressure",
        220051: "Diastolic blood pressure",
        220180: "Diastolic blood pressure",
        224643: "Diastolic blood pressure",
        225310: "Diastolic blood pressure",
        189: "Fraction inspired oxygen",
        727: "Fraction inspired oxygen",
        3420: "Fraction inspired oxygen",
        3422: "Fraction inspired oxygen",
        223835: "Fraction inspired oxygen",
        184: "Glascow coma scale eye opening",
        220739: "Glascow coma scale eye opening",
        454: "Glascow coma scale motor response",
        223901: "Glascow coma scale motor response",
        198: "Glascow coma scale total",
        723: "Glascow coma scale verbal response",
        223900: "Glascow coma scale verbal response",
        807: "Glucose",
        811: "Glucose",
        1529: "Glucose",
        3745: "Glucose",
        220621: "Glucose",
        225664: "Glucose",
        226537: "Glucose",
        211: "Heart Rate",
        220045: "Heart Rate",
        1394: "Height",
        226707: "Height",
        226730: "Height",
        52: "Mean blood pressure",
        224: "Mean blood pressure",
        456: "Mean blood pressure",
        6702: "Mean blood pressure",
        6927: "Mean blood pressure",
        220052: "Mean blood pressure",
        220181: "Mean blood pressure",
        224322: "Mean blood pressure",
        225312: "Mean blood pressure",
        646: "Oxygen saturation",
        834: "Oxygen saturation",
        8498: "Oxygen saturation",
        220227: "Oxygen saturation",
        220277: "Oxygen saturation",
        614: "Respiratory rate",
        615: "Respiratory rate",
        618: "Respiratory rate",
        651: "Respiratory rate",
        3603: "Respiratory rate",
        220210: "Respiratory rate",
        224422: "Respiratory rate",
        224689: "Respiratory rate",
        224690: "Respiratory rate",
        51: "Systolic blood pressure",
        442: "Systolic blood pressure",
        455: "Systolic blood pressure",
        6701: "Systolic blood pressure",
        220050: "Systolic blood pressure",
        220179: "Systolic blood pressure",
        224167: "Systolic blood pressure",
        225309: "Systolic blood pressure",
        227243: "Systolic blood pressure",
        676: "Temperature",
        677: "Temperature",
        678: "Temperature",
        679: "Temperature",
        3654: "Temperature",
        3655: "Temperature",
        223761: "Temperature",
        223762: "Temperature",
        763: "Weight",
        3580: "Weight",
        3581: "Weight",
        3582: "Weight",
        3693: "Weight",
        224639: "Weight",
        226512: "Weight",
        226531: "Weight",
        780: "pH",
        860: "pH",
        1126: "pH",
        1673: "pH",
        3839: "pH",
        4202: "pH",
        4753: "pH",
        220274: "pH",
        223830: "pH",
    }

    LABEVENT_ITEMIDS: ClassVar[dict[int, str]] = {
        50809: "Glucose",
        50931: "Glucose",
        51478: "Glucose",
        50817: "Oxygen saturation",
        50820: "pH",
        50831: "pH",
        51094: "pH",
        51491: "pH",
    }

    N_CONTINUOUS: ClassVar[int] = 12
    N_CATEGORICAL: ClassVar[int] = 47
    N_VALUE_FEATURES: ClassVar[int] = 59
    N_MASKS: ClassVar[int] = 17
    N_DYNAMIC_FEATURES: ClassVar[int] = 76

    NORMAL_VALUES: ClassVar[dict[str, float | str]] = {
        "Capillary refill rate": "0.0",
        "Diastolic blood pressure": 59.0,
        "Fraction inspired oxygen": 0.21,
        "Glascow coma scale eye opening": "4 Spontaneously",
        "Glascow coma scale motor response": "6 Obeys Commands",
        "Glascow coma scale total": "15",
        "Glascow coma scale verbal response": "5 Oriented",
        "Glucose": 128.0,
        "Heart Rate": 86.0,
        "Height": 170.0,
        "Mean blood pressure": 77.0,
        "Oxygen saturation": 98.0,
        "Respiratory rate": 19.0,
        "Systolic blood pressure": 118.0,
        "Temperature": 36.6,
        "Weight": 81.0,
        "pH": 7.4,
    }

    CONTINUOUS_MEANS: ClassVar[tuple[float, ...]] = (
        61.73746731237323,
        0.2941986945120962,
        140.38699662380586,
        85.53853938464474,
        169.80350234321443,
        79.59640838796953,
        104.98108350952208,
        19.867925156740334,
        122.3975667186341,
        36.94399135355236,
        83.76208973419455,
        7.207267791681239,
    )

    CONTINUOUS_STDS: ClassVar[tuple[float, ...]] = (
        221.89647947315552,
        0.1714931950435367,
        1766.7412133808477,
        57.11693013941011,
        6.222631784123813,
        148.53013549514117,
        5135.238732591619,
        7.3213287278950965,
        141.3988560835634,
        5.277914736088037,
        27.233083363220278,
        1.6568268628593597,
    )

    input_schema: ClassVar[dict[str, Any]] = {
    "dynamic_continuous": (
        "jagged",
        {
            "levels": [
                {
                    "ordering": "series",
                    "values": [N_CONTINUOUS],
                }
            ]
        },
    ),
    "dynamic_categorical": (
        "jagged",
        {
            "levels": [
                {
                    "ordering": "series",
                    "values": [N_CATEGORICAL],
                }
            ]
        },
    ),
    "dynamic_mask": (
        "jagged",
        {
            "levels": [
                {
                    "ordering": "series",
                    "values": [N_MASKS],
                }
            ]
        },
    ),
}
    output_schema: ClassVar[dict[str, Any]] = {
        "hazard_label": "tensor",
        "hazard_mask": "tensor",
        "y_true": "binary",
    }

    @staticmethod
    def _parse_float(value: Any) -> float | None:
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "error"}:
            return None

        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _clean_value(
        cls,
        channel: str,
        value: Any,
        valueuom: Any = None,
        label: Any = None,
    ) -> float | str | None:
        text = "" if value is None else str(value).strip()
        unit = "" if valueuom is None else str(valueuom).lower()
        item_label = "" if label is None else str(label).lower()

        if channel == "Capillary refill rate":
            if text in {"Normal <3 secs", "Brisk"}:
                return "0.0"
            if text in {"Abnormal >3 secs", "Delayed"}:
                return "1.0"
            return None

        if channel == "Systolic blood pressure":
            if "/" in text:
                parts = text.split("/")
                if len(parts) == 2:
                    return cls._parse_float(parts[0])
            return cls._parse_float(text)

        if channel == "Diastolic blood pressure":
            if "/" in text:
                parts = text.split("/")
                if len(parts) == 2:
                    return cls._parse_float(parts[1])
            return cls._parse_float(text)

        if channel in {"Glucose", "pH"}:
            return cls._parse_float(text)

        if channel == "Oxygen saturation":
            numeric = cls._parse_float(text)
            if numeric is None:
                return None
            if numeric <= 1:
                numeric *= 100.0
            return numeric

        if channel == "Fraction inspired oxygen":
            numeric = cls._parse_float(text)
            if numeric is None:
                return None

            if "torr" not in unit:
                numeric /= 100.0

            return numeric

        if channel == "Temperature":
            numeric = cls._parse_float(text)
            if numeric is None:
                return None

            if numeric >= 79:
                numeric = (numeric - 32.0) * 5.0 / 9.0

            return numeric

        if channel == "Weight":
            numeric = cls._parse_float(text)
            if numeric is None:
                return None

            if "oz" in unit or "oz" in item_label:
                numeric /= 16.0
                numeric *= 0.453592
            elif "lb" in unit or "lb" in item_label:
                numeric *= 0.453592

            return numeric

        if channel == "Height":
            numeric = cls._parse_float(text)
            if numeric is None:
                return None

            if "in" in unit or "in" in item_label:
                numeric = round(numeric * 2.54)

            return numeric

        if channel in cls.CATEGORICAL_VALUES:
            return text if text in cls.CATEGORICAL_VALUES[channel] else None

        return cls._parse_float(text)

    @classmethod
    def _hourly_observations(
        cls,
        patient: Any,
        stay: Any,
        intime: datetime,
        outtime: datetime,
    ) -> dict[int, dict[str, float | str]]:
        hourly: dict[int, dict[str, float | str]] = {}
        observations: list[tuple[datetime, str, float | str]] = []

        icustay_id = getattr(stay, "icustay_id", None)
        hadm_id = getattr(stay, "hadm_id", None)

        for event in patient.get_events(event_type="chartevents"):
            event_stay_id = getattr(event, "icustay_id", None)
            if (
                icustay_id is not None
                and event_stay_id is not None
                and str(event_stay_id) != str(icustay_id)
            ):
                continue

            timestamp = cls._to_datetime(getattr(event, "timestamp", None))
            if timestamp is None or timestamp < intime or timestamp > outtime:
                continue

            try:
                itemid = int(getattr(event, "itemid", None))
            except (TypeError, ValueError):
                continue

            channel = cls.CHARTEVENT_ITEMIDS.get(itemid)
            if channel is None:
                continue

            value = cls._clean_value(
                channel,
                getattr(event, "value", None),
                getattr(event, "valueuom", None),
                getattr(event, "label", None),
            )
            if value is not None:
                observations.append((timestamp, channel, value))

        for event in patient.get_events(event_type="labevents"):
            event_hadm_id = getattr(event, "hadm_id", None)
            if (
                hadm_id is not None
                and event_hadm_id is not None
                and str(event_hadm_id) != str(hadm_id)
            ):
                continue

            timestamp = cls._to_datetime(getattr(event, "timestamp", None))
            if timestamp is None or timestamp < intime or timestamp > outtime:
                continue

            try:
                itemid = int(getattr(event, "itemid", None))
            except (TypeError, ValueError):
                continue

            channel = cls.LABEVENT_ITEMIDS.get(itemid)
            if channel is None:
                continue

            value = cls._clean_value(
                channel,
                getattr(event, "value", None),
                getattr(event, "valueuom", None),
                getattr(event, "label", None),
            )
            if value is not None:
                observations.append((timestamp, channel, value))

        observations.sort(key=lambda x: x[0])

        for timestamp, channel, value in observations:
            hours = (timestamp - intime).total_seconds() / 3600.0
            bin_id = int(hours - 1e-6)
            bin_id = max(bin_id, 0)
            hourly.setdefault(bin_id, {})[channel] = value

        return hourly

    @classmethod
    def _value_feature_names(cls) -> list[str]:
        names: list[str] = []

        for channel in cls.CHANNELS:
            if channel in cls.CATEGORICAL_VALUES:
                for value in cls.CATEGORICAL_VALUES[channel]:
                    names.append(f"{channel}->{value}")
            else:
                names.append(channel)

        return names

    @classmethod
    def _build_dynamic(
        cls,
        hourly: dict[int, dict[str, float | str]],
        end_hour: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if end_hour < 1:
            raise ValueError("end_hour must be at least 1")

        continuous_channels = [
            channel
            for channel in cls.CHANNELS
            if channel not in cls.CATEGORICAL_VALUES
        ]

        continuous_index = {
            channel: i
            for i, channel in enumerate(continuous_channels)
        }

        categorical_names: list[str] = []
        for channel in cls.CHANNELS:
            if channel in cls.CATEGORICAL_VALUES:
                for value in cls.CATEGORICAL_VALUES[channel]:
                    categorical_names.append(f"{channel}->{value}")

        categorical_index = {
            name: i
            for i, name in enumerate(categorical_names)
        }

        continuous = np.zeros(
            (end_hour, cls.N_CONTINUOUS),
            dtype=float,
        )
        categorical = np.zeros(
            (end_hour, cls.N_CATEGORICAL),
            dtype=float,
        )
        masks = np.zeros(
            (end_hour, cls.N_MASKS),
            dtype=float,
        )

        previous: dict[str, float | str] = {}

        for hour in range(end_hour):
            observed = hourly.get(hour, {})

            for channel_id, channel in enumerate(cls.CHANNELS):
                is_observed = channel in observed

                if is_observed:
                    current = observed[channel]
                    previous[channel] = current
                elif channel in previous:
                    current = previous[channel]
                else:
                    current = cls.NORMAL_VALUES[channel]

                if channel in cls.CATEGORICAL_VALUES:
                    feature_name = f"{channel}->{current}"
                    feature_id = categorical_index.get(feature_name)

                    if feature_id is None:
                        normal = cls.NORMAL_VALUES[channel]
                        feature_name = f"{channel}->{normal}"
                        feature_id = categorical_index[feature_name]

                    categorical[hour, feature_id] = 1.0
                else:
                    feature_id = continuous_index[channel]
                    value = float(current)
                    continuous[hour, feature_id] = (
                        value - cls.CONTINUOUS_MEANS[feature_id]
                    ) / cls.CONTINUOUS_STDS[feature_id]

                masks[hour, channel_id] = float(is_observed)

        time = [float(hour + 1) for hour in range(end_hour)]

        return (
            {"time": time, "values": continuous},
            {"time": time, "values": categorical},
            {"time": time, "values": masks},
        )

    def __init__(
        self,
        horizon_hours: int = 24,
        min_observation_hours: int = 5,
        survtls_lengthscale: float | None = None,
    ) -> None:
        super().__init__()
        if horizon_hours < 1:
            raise ValueError("horizon_hours must be positive")
        if min_observation_hours < 0:
            raise ValueError("min_observation_hours must be non-negative")
        if survtls_lengthscale is not None and survtls_lengthscale <= 0:
            raise ValueError("survtls_lengthscale must be positive")

        self.horizon_hours = int(horizon_hours)
        self.min_observation_hours = int(min_observation_hours)
        self.survtls_lengthscale = (
            float(survtls_lengthscale)
            if survtls_lengthscale is not None
            else None
        )

    @staticmethod
    def _normal_cdf(x: float, mean: float, std: float) -> float:
        z = (x - mean) / (std * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))

    def _smooth_survival_label(self, event_bin: int) -> list[float]:
        if self.survtls_lengthscale is None:
            raise ValueError("survtls_lengthscale is not configured")

        mean = float(event_bin)
        std = mean / self.survtls_lengthscale
        pmf = [0.0] * self.horizon_hours

        for k in range(1, self.horizon_hours + 1):
            lower = k - 0.5
            upper = k + 0.5
            pmf[k - 1] = (
                self._normal_cdf(upper, mean, std)
                - self._normal_cdf(lower, mean, std)
            )

        pmf[0] = self._normal_cdf(1.5, mean, std)
        pmf[-1] = 1.0 - self._normal_cdf(
            self.horizon_hours - 0.5,
            mean,
            std,
        )

        return pmf

    @staticmethod
    def _to_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value

        value = str(value).strip()
        if not value or value.lower() in {"nan", "nat", "none"}:
            return None

        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _in_interval(
        value: datetime | None,
        start: datetime | None,
        end: datetime | None,
    ) -> bool:
        return (
            value is not None
            and start is not None
            and end is not None
            and start <= value <= end
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generate hourly decompensation samples for one patient.

        Args:
            patient: PyHealth patient containing MIMIC-III admissions, ICU
                stays, chart events, lab events, and patient demographics.

        Returns:
            List of hourly prediction samples. Patients or stays that do not
            satisfy cohort or minimum-observation requirements return no
            samples.
        """
        stays = patient.get_events(event_type="icustays")
        admissions = patient.get_events(event_type="admissions")
        patient_rows = patient.get_events(event_type="patients")

        if not stays or not admissions:
            return []

        admissions_by_hadm = {
            str(event.hadm_id): event
            for event in admissions
            if getattr(event, "hadm_id", None) is not None
        }

        dob = None
        dod = None
        if patient_rows:
            dob = self._to_datetime(getattr(patient_rows[0], "dob", None))
            dod = self._to_datetime(getattr(patient_rows[0], "dod", None))

        samples: list[dict[str, Any]] = []

        for stay in stays:
            hadm_id = getattr(stay, "hadm_id", None)
            if hadm_id is None:
                continue

            hadm_id = str(hadm_id)
            admission = admissions_by_hadm.get(hadm_id)
            if admission is None:
                continue

            first_wardid = getattr(stay, "first_wardid", None)
            last_wardid = getattr(stay, "last_wardid", None)
            first_careunit = getattr(stay, "first_careunit", None)
            last_careunit = getattr(stay, "last_careunit", None)

            if (
                first_wardid != last_wardid
                or first_careunit != last_careunit
            ):
                continue

            intime = self._to_datetime(getattr(stay, "timestamp", None))
            outtime = self._to_datetime(getattr(stay, "outtime", None))
            admittime = self._to_datetime(getattr(admission, "timestamp", None))
            dischtime = self._to_datetime(getattr(admission, "dischtime", None))
            deathtime = self._to_datetime(getattr(admission, "deathtime", None))

            if intime is None or outtime is None:
                continue
            if outtime <= intime:
                continue

            if dob is not None:
                age_years = (intime - dob).total_seconds() / (365.0 * 86400.0)
                if age_years < 0:
                    age_years = 90
                if age_years < 18:
                    continue

            mortality = (
                self._in_interval(dod, admittime, dischtime)
                or self._in_interval(deathtime, admittime, dischtime)
            )

            death_time = deathtime

            observed_end = outtime
            if death_time is not None and death_time < observed_end:
                observed_end = death_time

            observed_hours = (
                observed_end - intime
            ).total_seconds() / 3600.0

            if observed_hours < self.min_observation_hours:
                continue

            hourly = self._hourly_observations(
                patient,
                stay,
                intime,
                outtime,
            )

            last_prediction_hour = math.floor(observed_hours)

            for t in range(
                self.min_observation_hours,
                last_prediction_hour + 1,
            ):
                if not any(hour < t for hour in hourly):
                    continue
                remaining_observed = max(0.0, observed_hours - t)

                observed_bins = min(
                    self.horizon_hours,
                    math.ceil(remaining_observed),
                )

                hazard_label = [0.0] * self.horizon_hours
                hazard_mask = [0.0] * self.horizon_hours

                for k in range(observed_bins):
                    hazard_mask[k] = 1.0

                y_true = 0

                if mortality and death_time is not None:
                    delta = (
                        death_time - intime
                    ).total_seconds() / 3600.0 - t

                    # MIMIC benchmark uses death within strictly < 24 hours.
                    if 0.0 < delta < self.horizon_hours:
                        y_true = 1
                        event_bin = math.ceil(delta)
                        event_index = event_bin - 1

                        if self.survtls_lengthscale is not None:
                            hazard_label = self._smooth_survival_label(event_bin)
                        else:
                            hazard_label[event_index] = 1.0

                        for k in range(event_bin, self.horizon_hours):
                            hazard_mask[k] = 0.0

                (
                    dynamic_continuous,
                    dynamic_categorical,
                    dynamic_mask,
                ) = self._build_dynamic(hourly, t)

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "hadm_id": hadm_id,
                        "icustay_id": getattr(stay, "icustay_id", None),
                        "prediction_time": float(t),
                        "mortality": int(mortality),
                        "dynamic_continuous": dynamic_continuous,
                        "dynamic_categorical": dynamic_categorical,
                        "dynamic_mask": dynamic_mask,
                        "hazard_label": hazard_label,
                        "hazard_mask": hazard_mask,
                        "y_true": y_true,
                    }
                )

        return samples
