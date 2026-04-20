import pandas as pd
import polars as pl
from pyhealth.data import Patient
from pyhealth.tasks import PhysioNetMortalityTask

def test_physionet_mortality_task():
    df = pd.DataFrame({
        "patient_id": ["1"] * 8,
        "event_type": ["outcomes"] + ["events"] * 7,
        "timestamp":[
            pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01 10:00:00"), 
            pd.Timestamp("2012-01-01 12:00:00"), pd.Timestamp("2012-01-01 12:00:00"),
            pd.Timestamp("2012-01-01 12:00:00"), pd.Timestamp("2012-01-01 12:00:00"),
            pd.Timestamp("2012-01-01 12:00:00"), pd.Timestamp("2012-01-02 10:00:00")
        ],
        "outcomes/in-hospital_death":[1] + [None] * 7,
        "events/parameter":[None, "HR", "Age", "Gender", "Height", "Weight", "ICUType", "HR"],
        "events/value":[None, 80.0, 55.0, 1.0, 180.0, 75.0, 3.0, 85.0]
    })
    patient = Patient(patient_id="1", data_source=pl.DataFrame(df))
    
    task = PhysioNetMortalityTask(n_timesteps=16)
    samples = task(patient)
    
    assert len(samples) == 1
    sample = samples[0]
    assert sample["label"] == 1
    assert sample["x_ts"][0].shape == (16, 36)
    assert sample["x_static"][0] == 55.0

def test_physionet_mortality_task_edge_cases():
    task = PhysioNetMortalityTask()
    
    df1 = pd.DataFrame({"patient_id": ["2"], "event_type": ["events"], "timestamp":[pd.Timestamp("2012-01-01")], "events/parameter":["HR"], "events/value":[80.0]})
    assert len(task(Patient(patient_id="2", data_source=pl.DataFrame(df1)))) == 0
    
    df2 = pd.DataFrame({"patient_id": ["3"], "event_type": ["outcomes"], "timestamp":[pd.Timestamp("2012-01-01")], "outcomes/in-hospital_death":[1]})
    assert len(task(Patient(patient_id="3", data_source=pl.DataFrame(df2)))) == 0

    df3 = pd.DataFrame({
        "patient_id": ["4", "4"], "event_type": ["outcomes", "events"],
        "timestamp":[pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01")],
        "outcomes/in-hospital_death":[1, None], "events/parameter": [None, "HR"], "events/value":[None, 80.0]
    })
    assert len(task(Patient(patient_id="4", data_source=pl.DataFrame(df3)))) == 1