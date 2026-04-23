from __future__ import annotations

"""
Minimal example: TPC remaining ICU LoS (MIMIC-IV).

This is the paper-style setting:
  - remaining ICU length-of-stay regression
  - hourly predictions starting at hour 5
  - MSLE loss
"""

import os
import sys

# Put cache inside repo by default (avoids sandbox permission errors).
os.environ.setdefault("PYHEALTH_CACHE_PATH", os.path.join(os.path.dirname(__file__), "..", ".pyhealth_cache"))

# Ensure we import the *local* repo `pyhealth/` rather than any site-packages install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyhealth.datasets import MIMIC4EHRDataset, split_by_patient, get_dataloader
from pyhealth.tasks import RemainingLengthOfStayTPC_MIMIC4
from pyhealth.models import TPC
from pyhealth.trainer import Trainer

# Using Labebents and chartevents itemids reported 
# in table 17 of the paper by Rocheteau et al. (2021)
LABEVENTS_ITEMIDS = [
    50861,   # Alanine Aminotransferase (ALT)
    50863,   # Alkaline Phosphatase
    50868,   # Anion Gap
    50878,   # Asparate Aminotransferase (AST)
    50882,   # Bicarbonate
    50885,   # Bilirubin, Total
    50893,   # Calcium, Total
    50804,   # Calculated Total CO2
    50902,   # Chloride        
    50912,   # Creatinine       
    50808,   # Free Calcium
    50931,   # Glucose         
    51221,   # Hematocrit
    50810,   # Hematocrit, Calculated
    51222,   # Hemoglobin
    51237,   # INR(PT)
    50813,   # Lactate
    51248,   # MCH
    51249,   # MCHC
    51250,   # MCV
    50960,   # Magnesium
    50817,   # Oxygen Saturation
    51274,   # PT
    51275,   # PTT
    50970,   # Phosphate
    51265,   # Platelet Count
    50971,   # Potassium
    51277,   # RDW
    52172,   # RDW-SD
    51279,   # Red Blood Cells
    50983,   # Sodium
    51006,   # Urea Nitrogen
    51301,   # White Blood Cells
    50818,   # pCO2
    50820,   # pH
    50821,   # pO2
]

CHARTEVENTS_ITEMIDS = [
    229319,  # Activity / Mobility (JH-HLM)
    223876,  # Apnea Interval
    220058,  # Arterial Blood Pressure Alarm - High
    220056,  # Arterial Blood Pressure Alarm - Low
    220051,  # Arterial Blood Pressure diastolic
    220052,  # Arterial Blood Pressure mean
    220050,  # Arterial Blood Pressure systolic
    229323,  # Current Dyspnea Assessment
    224639,  # Daily Weight
    226871,  # Expiratory Ratio
    223875,  # Fspn High
    220739,  # GCS - Eye Opening
    223901,  # GCS - Motor Response
    223900,  # GCS - Verbal Response
    225664,  # Glucose finger stick (range 70-100)
    220045,  # Heart Rate
    220047,  # Heart Rate Alarm - Low
    220046,  # Heart rate Alarm - High
    223835,  # Inspired O2 Fraction
    224697,  # Mean Airway Pressure
    224687,  # Minute Volume
    220293,  # Minute Volume Alarm - High
    220292,  # Minute Volume Alarm - Low
    220180,  # Non Invasive Blood Pressure diastolic
    220181,  # Non Invasive Blood Pressure mean
    220179,  # Non Invasive Blood Pressure systolic
    223751,  # Non-Invasive Blood Pressure Alarm - High
    223752,  # Non-Invasive Blood Pressure Alarm - Low
    223834,  # O2 Flow
    223770,  # O2 Saturation Pulseoxymetry Alarm - Low
    220277,  # O2 saturation pulseoxymetry
    220339,  # PEEP set
    224701,  # PSV Level
    223791,  # Pain Level
    224409,  # Pain Level Response
    223873,  # Paw High
    224695,  # Peak Insp. Pressure
    225677,  # Phosphorous
    224696,  # Plateau Pressure
    224161,  # Resp Alarm - High
    224162,  # Resp Alarm - Low
    220210,  # Respiratory Rate
    224688,  # Respiratory Rate (Set)
    224690,  # Respiratory Rate (Total)
    224689,  # Respiratory Rate (spontaneous)
    228096,  # Richmond-RAS Scale
    228409,  # Strength L Arm
    228410,  # Strength L Leg
    228412,  # Strength R Arm
    228411,  # Strength R Leg
    223761,  # Temperature Fahrenheit
    224685,  # Tidal Volume (observed)
    224684,  # Tidal Volume (set)
    224686,  # Tidal Volume (spontaneous)
    224700,  # Total PEEP Level
    223849,  # Ventilator Mode
    223874,  # Vti High
]


def main():
    # Adjust these paths for your environment.
    ehr_root = "./datasets/mimic-iv-demo/2.2"
    cache_dir = os.path.join(_REPO_ROOT, ".pyhealth_dataset_cache")

    dataset = MIMIC4EHRDataset(
        root=ehr_root,
        tables=["patients", "admissions", "icustays", "labevents", "chartevents"],
        dev=True,
        num_workers=1,
        cache_dir=cache_dir,
    )

    task = RemainingLengthOfStayTPC_MIMIC4(
        labevent_itemids=LABEVENTS_ITEMIDS,
        chartevent_itemids=CHARTEVENTS_ITEMIDS,
    )
    sample_dataset = dataset.set_task(task)

    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, ratios=[0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)

    model = TPC(dataset=sample_dataset)
    trainer = Trainer(model, metrics=["mae", "mse"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=5,
        monitor="mae",
        monitor_criterion="min",
    )


if __name__ == "__main__":
    main()

