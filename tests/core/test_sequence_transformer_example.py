"""Tests that later height/weight values do not change the demo inputs."""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SequenceTransformerNetwork
from pyhealth.processors import BinaryLabelProcessor
from pyhealth.tasks import YAIBMortalityMIMIC3


def _patient(future_weight: float, future_height: float) -> Patient:
    """One synthetic stay; only the later height/weight values change."""
    start = datetime(2150, 1, 1, 8)
    rows = []

    def event(event_type, timestamp, **attributes):
        row = {"event_type": event_type, "timestamp": timestamp}
        for key, value in attributes.items():
            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            row[f"{event_type}/{key}"] = None if value is None else str(value)
        rows.append(row)

    event("patients", None, dob=start - timedelta(days=60 * 365), gender="M")
    event(
        "admissions",
        start - timedelta(hours=10),
        hadm_id=1,
        dischtime=start + timedelta(hours=80),
        deathtime=None,
        hospital_expire_flag=0,
    )
    event(
        "icustays",
        start,
        icustay_id=10,
        hadm_id=1,
        first_careunit="MICU",
        outtime=start + timedelta(hours=72),
    )
    for hour in range(72):
        event(
            "chartevents",
            start + timedelta(hours=hour, minutes=5),
            icustay_id=10,
            hadm_id=1,
            itemid=211,
            value=80 + hour,
            valuenum=80 + hour,
            valueuom="BPM",
        )
    for hour in [2, 36, 48]:
        for item, value, unit in [
            (762, 80.0 if hour == 2 else future_weight, "kg"),
            (920, 70.0 if hour == 2 else future_height, "Inch"),
        ]:
            event(
                "chartevents",
                start + timedelta(hours=hour),
                icustay_id=10,
                hadm_id=1,
                itemid=item,
                value=value,
                valuenum=value,
                valueuom=unit,
            )
    frame = pl.from_dicts(rows, infer_schema_length=None)
    frame = frame.with_columns(
        pl.lit("synthetic-patient").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(column).cast(pl.String) for column in frame.columns if "/" in column),
    )
    return Patient(patient_id="synthetic-patient", data_source=frame)


def test_late_height_and_weight_do_not_change_dynamic_only_model_input():
    """Later measurements should not change the model input or prediction.

    The stay is included in both cases. This does not test or change YAIB's
    rules for selecting stays.
    """
    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples/mimic3_mortality_sequence_transformer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "stn_example_under_test", example_path
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    task = example.DynamicOnlyMortality()
    patients = [_patient(80.0, 70.0), _patient(120.0, 80.0)]
    stock = [YAIBMortalityMIMIC3()(patient)[0] for patient in patients]
    assert stock[0]["static"]["values"][0][2:] != stock[1]["static"]["values"][0][2:]
    samples = [task(patient)[0] for patient in patients]
    np.testing.assert_allclose(
        samples[0]["dynamic"]["values"],
        samples[1]["dynamic"]["values"],
        equal_nan=True,
    )
    assert set(task.input_schema) == {"dynamic"}
    # Fix the binary vocabulary independently of this same-outcome pair.
    labels = BinaryLabelProcessor()
    labels.fit([{"mortality": 0}, {"mortality": 1}], "mortality")
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        output_processors={"mortality": labels},
        in_memory=True,
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2)))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = SequenceTransformerNetwork(dataset, seq_len=example.SEQ_LEN).eval()
        assert model.feature_keys == ["dynamic"]
        series = model.series(**batch)
        torch.testing.assert_close(series[0], series[1], rtol=0, atol=0)
        with torch.no_grad():
            probability = model(**batch)["y_prob"]
        torch.testing.assert_close(probability[0], probability[1], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
