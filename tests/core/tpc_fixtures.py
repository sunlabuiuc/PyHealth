"""
Author: Manling Tan, Yining Jin, Xiao Luo
NetID: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483
"""


"""Small synthetic eICU records; adapted from YAIB regression fixtures."""

from pathlib import Path
import polars as pl
import yaml
from pyhealth.data import Patient

# Every column of the eICU config, as in the dataset's event dataframe.
_CONFIG = yaml.safe_load(
    (
        Path(__file__).parents[2] / "pyhealth" / "datasets" / "configs" / "eicu.yaml"
    ).read_text()
)
COLUMNS = [
    f"{table}/{column}"
    for table, config in _CONFIG["tables"].items()
    for column in config["attributes"]
]


def _labels(records):
    """The label values of a ``timeseries_*`` label series."""
    return [value[0] for value in records["values"]]


def _event(event_type, **attrs):
    # Attributes are strings in the dataset's event dataframe.
    row = {"event_type": event_type, "timestamp": None}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _frame(rows, patient_id="p1"):
    df = pl.from_dicts(rows, infer_schema_length=None)
    return df.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(
            (pl.col(c) if c in df.columns else pl.lit(None)).cast(pl.String).alias(c)
            for c in COLUMNS
        ),
    )


def _stay_rows(
    stay_id,
    hours,
    age="60",
    gender="Male",
    height=180,
    weight=80,
    status="Alive",
    hospital_id=1,
    hr_hours=None,
):
    """A stay and one heart rate per hour, so it passes the base cohort."""
    rows = [
        _event(
            "patient",
            patientunitstayid=stay_id,
            unitdischargeoffset=int(hours * 60),
            gender=gender,
            age=age,
            admissionheight=height,
            admissionweight=weight,
            unitdischargestatus=status,
            hospitalid=hospital_id,
        )
    ]
    for hour in range(int(hours)) if hr_hours is None else hr_hours:
        rows.append(
            _event(
                "vitalperiodic",
                patientunitstayid=stay_id,
                observationoffset=hour * 60 + 5,
                heartrate=80 + hour,
            )
        )
    return rows


def _patient(stay_id=10, hours: float = 48, extra=(), patient_id="p1", **stay):
    rows = _stay_rows(stay_id, hours, **stay) + list(extra)
    return Patient(patient_id=patient_id, data_source=_frame(rows, patient_id))


def _lab(minute, name, value, unit="mg/dL", stay_id=10):
    return _event(
        "lab",
        patientunitstayid=stay_id,
        labresultoffset=minute,
        labname=name,
        labresult=value,
        labmeasurenameinterface=unit,
    )
