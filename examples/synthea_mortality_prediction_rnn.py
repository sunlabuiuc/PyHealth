"""Ablation study for Synthea post-discharge mortality prediction.

Author: Justin Xu

Paper: Raphael Poulain, Mehak Gupta, and Rahmatollah Beheshti.
    "CEHR-GAN-BERT: Incorporating Temporal Information from Structured EHR
    Data to Improve Prediction Tasks." MLHC 2022, Section A.2 (Mortality-Disch).
    https://proceedings.mlr.press/v182/poulain22a.html

Description: Runs an ablation study over feature sets and prediction windows
    for the Synthea mortality prediction task.  For each configuration, trains
    an RNN model and reports cohort size, positive rate, AUROC, and F1 score.

    Feature sets:
        - conditions_only: only diagnosis codes
        - conditions+medications: diagnoses + medication orders
        - full: diagnoses + medications + procedures

    Prediction windows: 180, 365, 730 days

Usage:
    # Use built-in demo data (20 synthetic patients):
    python examples/synthea_mortality_prediction_rnn.py --demo

    # Use real Synthea data:
    python examples/synthea_mortality_prediction_rnn.py --root /path/to/synthea/csv
"""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import SyntheaDataset
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionSynthea
from pyhealth.trainer import Trainer


# ======================================================================
# Demo data generator
# ======================================================================

def _generate_demo_data(root: str) -> None:
    """Create 20 synthetic patients (4 deceased) with Synthea CSV format.

    Patient layout:
        - p001-p004: deceased patients (various death timings)
        - p005-p020: alive patients
    Each patient has 2-3 inpatient encounters with conditions,
    medications, and procedures linked to those encounters.
    """
    os.makedirs(root, exist_ok=True)

    # ---- patients.csv ------------------------------------------------
    patients_rows = [
        ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY"],
        # Deceased patients
        ["p001", "1950-01-15", "2021-06-20", "M", "white", "nonhispanic"],
        ["p002", "1942-08-03", "2022-03-10", "F", "black", "nonhispanic"],
        ["p003", "1938-12-25", "2023-08-15", "M", "white", "hispanic"],
        ["p004", "1960-04-18", "2021-03-10", "F", "asian", "nonhispanic"],
        # Alive patients
        ["p005", "1980-05-05", "", "M", "white", "nonhispanic"],
        ["p006", "1975-11-22", "", "F", "black", "hispanic"],
        ["p007", "1990-02-14", "", "M", "asian", "nonhispanic"],
        ["p008", "1955-07-30", "", "F", "white", "nonhispanic"],
        ["p009", "1968-09-12", "", "M", "white", "hispanic"],
        ["p010", "1985-01-01", "", "F", "black", "nonhispanic"],
        ["p011", "1972-06-18", "", "M", "white", "nonhispanic"],
        ["p012", "1948-03-09", "", "F", "asian", "hispanic"],
        ["p013", "1995-10-27", "", "M", "white", "nonhispanic"],
        ["p014", "1982-12-05", "", "F", "black", "nonhispanic"],
        ["p015", "1963-04-20", "", "M", "white", "hispanic"],
        ["p016", "1970-08-15", "", "F", "white", "nonhispanic"],
        ["p017", "1988-11-11", "", "M", "asian", "nonhispanic"],
        ["p018", "1953-02-28", "", "F", "black", "hispanic"],
        ["p019", "1977-07-07", "", "M", "white", "nonhispanic"],
        ["p020", "1992-09-30", "", "F", "white", "nonhispanic"],
    ]

    # ---- encounters.csv ----------------------------------------------
    # Each patient gets 2-3 inpatient encounters
    encounters_rows = [
        ["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS",
         "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
        # p001: 2 encounters, dies ~100 days after last discharge
        ["e001", "2020-01-10T08:00:00Z", "2020-01-15T16:00:00Z",
         "p001", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e002", "2021-03-01T09:00:00Z", "2021-03-12T14:00:00Z",
         "p001", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        # p002: 2 encounters, dies ~160 days after last discharge
        ["e003", "2021-04-10T10:00:00Z", "2021-04-18T12:00:00Z",
         "p002", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e004", "2021-10-01T08:00:00Z", "2021-10-05T17:00:00Z",
         "p002", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        # p003: 2 encounters, dies ~431 days after last discharge
        ["e005", "2022-01-10T07:00:00Z", "2022-01-20T15:00:00Z",
         "p003", "inpatient", "185347001", "Encounter for problem",
         "22298006", "MI"],
        ["e006", "2022-06-01T09:00:00Z", "2022-06-10T16:00:00Z",
         "p003", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        # p004: 1 encounter, dies DURING encounter -> excluded
        ["e007", "2021-01-05T08:00:00Z", "2021-03-10T12:00:00Z",
         "p004", "inpatient", "185347001", "Encounter for problem",
         "254637007", "Cancer"],
        # p005-p020: alive patients, 2-3 encounters each
        ["e008", "2019-04-01T10:00:00Z", "2019-04-05T14:00:00Z",
         "p005", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e009", "2020-08-15T08:00:00Z", "2020-08-20T16:00:00Z",
         "p005", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e010", "2021-11-01T07:00:00Z", "2021-11-10T15:00:00Z",
         "p005", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e011", "2020-02-10T08:00:00Z", "2020-02-15T16:00:00Z",
         "p006", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e012", "2021-05-20T09:00:00Z", "2021-05-28T14:00:00Z",
         "p006", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e013", "2020-06-15T10:00:00Z", "2020-06-20T12:00:00Z",
         "p007", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e014", "2021-09-01T08:00:00Z", "2021-09-07T17:00:00Z",
         "p007", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e015", "2019-11-10T07:00:00Z", "2019-11-18T15:00:00Z",
         "p008", "inpatient", "185347001", "Encounter for problem",
         "22298006", "MI"],
        ["e016", "2020-07-01T09:00:00Z", "2020-07-08T16:00:00Z",
         "p008", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e017", "2021-01-15T08:00:00Z", "2021-01-22T12:00:00Z",
         "p009", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e018", "2022-03-10T10:00:00Z", "2022-03-18T14:00:00Z",
         "p009", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e019", "2020-05-05T08:00:00Z", "2020-05-12T16:00:00Z",
         "p010", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e020", "2021-08-20T09:00:00Z", "2021-08-28T14:00:00Z",
         "p010", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e021", "2019-09-01T10:00:00Z", "2019-09-08T12:00:00Z",
         "p011", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e022", "2020-12-15T08:00:00Z", "2020-12-22T17:00:00Z",
         "p011", "inpatient", "185347001", "Encounter for problem",
         "22298006", "MI"],
        ["e023", "2021-07-01T07:00:00Z", "2021-07-10T15:00:00Z",
         "p011", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e024", "2020-03-10T09:00:00Z", "2020-03-17T16:00:00Z",
         "p012", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e025", "2021-06-15T08:00:00Z", "2021-06-22T12:00:00Z",
         "p012", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e026", "2021-02-01T10:00:00Z", "2021-02-08T14:00:00Z",
         "p013", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e027", "2022-05-20T08:00:00Z", "2022-05-28T16:00:00Z",
         "p013", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e028", "2020-10-01T09:00:00Z", "2020-10-08T14:00:00Z",
         "p014", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e029", "2021-12-10T08:00:00Z", "2021-12-18T17:00:00Z",
         "p014", "inpatient", "185347001", "Encounter for problem",
         "22298006", "MI"],
        ["e030", "2019-07-15T10:00:00Z", "2019-07-22T12:00:00Z",
         "p015", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e031", "2020-11-01T08:00:00Z", "2020-11-08T16:00:00Z",
         "p015", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e032", "2021-04-15T09:00:00Z", "2021-04-22T14:00:00Z",
         "p016", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e033", "2022-01-05T08:00:00Z", "2022-01-12T17:00:00Z",
         "p016", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e034", "2020-08-01T10:00:00Z", "2020-08-08T12:00:00Z",
         "p017", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e035", "2021-10-20T08:00:00Z", "2021-10-28T16:00:00Z",
         "p017", "inpatient", "185347001", "Encounter for problem",
         "22298006", "MI"],
        ["e036", "2019-12-01T09:00:00Z", "2019-12-08T14:00:00Z",
         "p018", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e037", "2021-03-15T08:00:00Z", "2021-03-22T17:00:00Z",
         "p018", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e038", "2020-04-10T10:00:00Z", "2020-04-17T12:00:00Z",
         "p019", "inpatient", "185347001", "Encounter for problem",
         "195662009", "Pneumonia"],
        ["e039", "2021-07-20T08:00:00Z", "2021-07-28T16:00:00Z",
         "p019", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
        ["e040", "2022-02-01T09:00:00Z", "2022-02-08T14:00:00Z",
         "p019", "inpatient", "185347001", "Encounter for problem",
         "44054006", "Diabetes"],
        ["e041", "2021-05-10T08:00:00Z", "2021-05-17T17:00:00Z",
         "p020", "inpatient", "185347001", "Encounter for problem",
         "22298006", "MI"],
        ["e042", "2022-08-15T10:00:00Z", "2022-08-22T12:00:00Z",
         "p020", "inpatient", "185347001", "Encounter for problem",
         "38341003", "Hypertension"],
    ]

    # ---- conditions.csv ----------------------------------------------
    # One condition per encounter
    conditions_rows = [
        ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
        ["2020-01-10T08:00:00Z", "p001", "e001", "38341003", "Hypertension"],
        ["2021-03-01T09:00:00Z", "p001", "e002", "44054006", "Diabetes mellitus type 2"],
        ["2021-04-10T10:00:00Z", "p002", "e003", "195662009", "Acute viral pharyngitis"],
        ["2021-10-01T08:00:00Z", "p002", "e004", "38341003", "Hypertension"],
        ["2022-01-10T07:00:00Z", "p003", "e005", "22298006", "Myocardial infarction"],
        ["2022-06-01T09:00:00Z", "p003", "e006", "38341003", "Hypertension"],
        ["2021-01-05T08:00:00Z", "p004", "e007", "254637007", "Non-small cell lung cancer"],
        ["2019-04-01T10:00:00Z", "p005", "e008", "195662009", "Acute viral pharyngitis"],
        ["2020-08-15T08:00:00Z", "p005", "e009", "38341003", "Hypertension"],
        ["2021-11-01T07:00:00Z", "p005", "e010", "44054006", "Diabetes mellitus type 2"],
        ["2020-02-10T08:00:00Z", "p006", "e011", "38341003", "Hypertension"],
        ["2021-05-20T09:00:00Z", "p006", "e012", "44054006", "Diabetes mellitus type 2"],
        ["2020-06-15T10:00:00Z", "p007", "e013", "195662009", "Acute viral pharyngitis"],
        ["2021-09-01T08:00:00Z", "p007", "e014", "38341003", "Hypertension"],
        ["2019-11-10T07:00:00Z", "p008", "e015", "22298006", "Myocardial infarction"],
        ["2020-07-01T09:00:00Z", "p008", "e016", "38341003", "Hypertension"],
        ["2021-01-15T08:00:00Z", "p009", "e017", "44054006", "Diabetes mellitus type 2"],
        ["2022-03-10T10:00:00Z", "p009", "e018", "195662009", "Acute viral pharyngitis"],
        ["2020-05-05T08:00:00Z", "p010", "e019", "38341003", "Hypertension"],
        ["2021-08-20T09:00:00Z", "p010", "e020", "44054006", "Diabetes mellitus type 2"],
        ["2019-09-01T10:00:00Z", "p011", "e021", "195662009", "Acute viral pharyngitis"],
        ["2020-12-15T08:00:00Z", "p011", "e022", "22298006", "Myocardial infarction"],
        ["2021-07-01T07:00:00Z", "p011", "e023", "38341003", "Hypertension"],
        ["2020-03-10T09:00:00Z", "p012", "e024", "38341003", "Hypertension"],
        ["2021-06-15T08:00:00Z", "p012", "e025", "44054006", "Diabetes mellitus type 2"],
        ["2021-02-01T10:00:00Z", "p013", "e026", "195662009", "Acute viral pharyngitis"],
        ["2022-05-20T08:00:00Z", "p013", "e027", "38341003", "Hypertension"],
        ["2020-10-01T09:00:00Z", "p014", "e028", "44054006", "Diabetes mellitus type 2"],
        ["2021-12-10T08:00:00Z", "p014", "e029", "22298006", "Myocardial infarction"],
        ["2019-07-15T10:00:00Z", "p015", "e030", "38341003", "Hypertension"],
        ["2020-11-01T08:00:00Z", "p015", "e031", "195662009", "Acute viral pharyngitis"],
        ["2021-04-15T09:00:00Z", "p016", "e032", "44054006", "Diabetes mellitus type 2"],
        ["2022-01-05T08:00:00Z", "p016", "e033", "38341003", "Hypertension"],
        ["2020-08-01T10:00:00Z", "p017", "e034", "195662009", "Acute viral pharyngitis"],
        ["2021-10-20T08:00:00Z", "p017", "e035", "22298006", "Myocardial infarction"],
        ["2019-12-01T09:00:00Z", "p018", "e036", "38341003", "Hypertension"],
        ["2021-03-15T08:00:00Z", "p018", "e037", "44054006", "Diabetes mellitus type 2"],
        ["2020-04-10T10:00:00Z", "p019", "e038", "195662009", "Acute viral pharyngitis"],
        ["2021-07-20T08:00:00Z", "p019", "e039", "38341003", "Hypertension"],
        ["2022-02-01T09:00:00Z", "p019", "e040", "44054006", "Diabetes mellitus type 2"],
        ["2021-05-10T08:00:00Z", "p020", "e041", "22298006", "Myocardial infarction"],
        ["2022-08-15T10:00:00Z", "p020", "e042", "38341003", "Hypertension"],
    ]

    # ---- medications.csv ---------------------------------------------
    # Most encounters get a medication
    medications_rows = [
        ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"],
        ["2020-01-10T08:00:00Z", "p001", "e001", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-03-01T09:00:00Z", "p001", "e002", "860975", "Metformin 500 MG", "44054006"],
        ["2021-04-10T10:00:00Z", "p002", "e003", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2021-10-01T08:00:00Z", "p002", "e004", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2022-01-10T07:00:00Z", "p003", "e005", "309362", "Clopidogrel 75 MG", "22298006"],
        ["2021-01-05T08:00:00Z", "p004", "e007", "583214", "Cisplatin 50 MG", "254637007"],
        ["2020-08-15T08:00:00Z", "p005", "e009", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-11-01T07:00:00Z", "p005", "e010", "860975", "Metformin 500 MG", "44054006"],
        ["2020-02-10T08:00:00Z", "p006", "e011", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-05-20T09:00:00Z", "p006", "e012", "860975", "Metformin 500 MG", "44054006"],
        ["2020-06-15T10:00:00Z", "p007", "e013", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2021-09-01T08:00:00Z", "p007", "e014", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2019-11-10T07:00:00Z", "p008", "e015", "309362", "Clopidogrel 75 MG", "22298006"],
        ["2020-07-01T09:00:00Z", "p008", "e016", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-01-15T08:00:00Z", "p009", "e017", "860975", "Metformin 500 MG", "44054006"],
        ["2022-03-10T10:00:00Z", "p009", "e018", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2020-05-05T08:00:00Z", "p010", "e019", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-08-20T09:00:00Z", "p010", "e020", "860975", "Metformin 500 MG", "44054006"],
        ["2020-12-15T08:00:00Z", "p011", "e022", "309362", "Clopidogrel 75 MG", "22298006"],
        ["2021-07-01T07:00:00Z", "p011", "e023", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2020-03-10T09:00:00Z", "p012", "e024", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-06-15T08:00:00Z", "p012", "e025", "860975", "Metformin 500 MG", "44054006"],
        ["2021-02-01T10:00:00Z", "p013", "e026", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2022-05-20T08:00:00Z", "p013", "e027", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2020-10-01T09:00:00Z", "p014", "e028", "860975", "Metformin 500 MG", "44054006"],
        ["2021-12-10T08:00:00Z", "p014", "e029", "309362", "Clopidogrel 75 MG", "22298006"],
        ["2019-07-15T10:00:00Z", "p015", "e030", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2020-11-01T08:00:00Z", "p015", "e031", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2021-04-15T09:00:00Z", "p016", "e032", "860975", "Metformin 500 MG", "44054006"],
        ["2022-01-05T08:00:00Z", "p016", "e033", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2020-08-01T10:00:00Z", "p017", "e034", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2021-10-20T08:00:00Z", "p017", "e035", "309362", "Clopidogrel 75 MG", "22298006"],
        ["2019-12-01T09:00:00Z", "p018", "e036", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2021-03-15T08:00:00Z", "p018", "e037", "860975", "Metformin 500 MG", "44054006"],
        ["2020-04-10T10:00:00Z", "p019", "e038", "746765", "Amoxicillin 250 MG", "195662009"],
        ["2021-07-20T08:00:00Z", "p019", "e039", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
        ["2022-02-01T09:00:00Z", "p019", "e040", "860975", "Metformin 500 MG", "44054006"],
        ["2021-05-10T08:00:00Z", "p020", "e041", "309362", "Clopidogrel 75 MG", "22298006"],
        ["2022-08-15T10:00:00Z", "p020", "e042", "316049", "Hydrochlorothiazide 25 MG", "38341003"],
    ]

    # ---- procedures.csv ----------------------------------------------
    # A subset of encounters get a procedure
    procedures_rows = [
        ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"],
        ["2020-01-11T10:00:00Z", "p001", "e001", "268425006", "Blood pressure monitoring", "38341003"],
        ["2021-03-02T11:00:00Z", "p001", "e002", "698314001", "Glucose measurement", "44054006"],
        ["2021-04-11T09:00:00Z", "p002", "e003", "430193006", "Medication reconciliation", "195662009"],
        ["2022-01-11T08:00:00Z", "p003", "e005", "232717009", "Coronary angiography", "22298006"],
        ["2022-06-02T10:00:00Z", "p003", "e006", "268425006", "Blood pressure monitoring", "38341003"],
        ["2021-01-06T09:00:00Z", "p004", "e007", "703423002", "Lung biopsy", "254637007"],
        ["2021-11-02T10:00:00Z", "p005", "e010", "698314001", "Glucose measurement", "44054006"],
        ["2020-02-11T10:00:00Z", "p006", "e011", "268425006", "Blood pressure monitoring", "38341003"],
        ["2020-06-16T09:00:00Z", "p007", "e013", "430193006", "Medication reconciliation", "195662009"],
        ["2019-11-11T08:00:00Z", "p008", "e015", "232717009", "Coronary angiography", "22298006"],
        ["2021-01-16T10:00:00Z", "p009", "e017", "698314001", "Glucose measurement", "44054006"],
        ["2020-12-16T09:00:00Z", "p011", "e022", "232717009", "Coronary angiography", "22298006"],
        ["2021-07-02T08:00:00Z", "p011", "e023", "268425006", "Blood pressure monitoring", "38341003"],
        ["2020-10-02T10:00:00Z", "p014", "e028", "698314001", "Glucose measurement", "44054006"],
        ["2021-12-11T09:00:00Z", "p014", "e029", "232717009", "Coronary angiography", "22298006"],
        ["2021-10-21T08:00:00Z", "p017", "e035", "232717009", "Coronary angiography", "22298006"],
        ["2022-02-02T10:00:00Z", "p019", "e040", "698314001", "Glucose measurement", "44054006"],
        ["2021-05-11T09:00:00Z", "p020", "e041", "232717009", "Coronary angiography", "22298006"],
    ]

    # Write CSV files
    for filename, rows in [
        ("patients.csv", patients_rows),
        ("encounters.csv", encounters_rows),
        ("conditions.csv", conditions_rows),
        ("medications.csv", medications_rows),
        ("procedures.csv", procedures_rows),
    ]:
        filepath = os.path.join(root, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


# ======================================================================
# Feature set definitions
# ======================================================================

FEATURE_SETS = {
    "conditions_only": ["conditions"],
    "conditions+medications": ["conditions", "medications"],
    "full": ["conditions", "medications", "procedures"],
}

PREDICTION_WINDOWS = [180, 365, 730]


# ======================================================================
# Main
# ======================================================================

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Ablation study for Synthea mortality prediction"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root",
        type=str,
        help="Path to a real Synthea CSV export directory",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Generate and use synthetic demo data (20 patients)",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.demo:
        tmpdir = tempfile.mkdtemp(prefix="synthea_demo_")
        _generate_demo_data(tmpdir)
        data_root = tmpdir
        print(f"[demo] Generated synthetic data in {data_root}")
    else:
        data_root = args.root

    # Collect results for summary table
    results = []

    for feat_name, tables in FEATURE_SETS.items():
        # Load dataset once per feature set (tables differ)
        print(f"\n{'='*60}")
        print(f"Loading SyntheaDataset  tables={tables}")
        print(f"{'='*60}")
        dataset = SyntheaDataset(root=data_root, tables=tables)

        for window in PREDICTION_WINDOWS:
            task = MortalityPredictionSynthea(
                prediction_window_days=window
            )
            sample_dataset = dataset.set_task(task)
            n_samples = len(sample_dataset)

            # Count positives
            n_pos = 0
            for i in range(n_samples):
                label = sample_dataset[i]["mortality"]
                label_int = (
                    int(label.item()) if hasattr(label, "item") else int(label)
                )
                n_pos += label_int

            pos_rate = n_pos / n_samples * 100 if n_samples > 0 else 0.0

            print(
                f"  window={window:>4}d | "
                f"samples={n_samples:>4} | "
                f"positive={n_pos:>3} ({pos_rate:5.1f}%)"
            )

            # ----------------------------------------------------------
            # Model training and evaluation
            # ----------------------------------------------------------
            auroc_str = "N/A"
            f1_str = "N/A"
            try:
                if n_samples < 3:
                    raise ValueError(
                        f"Too few samples ({n_samples}) to split into "
                        "train/val/test sets"
                    )

                # Reproducible splits
                train_ds, val_ds, test_ds = split_by_patient(
                    sample_dataset, [0.8, 0.1, 0.1], seed=42
                )

                # Require at least 1 sample in each split
                if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
                    raise ValueError(
                        f"Empty split: train={len(train_ds)}, "
                        f"val={len(val_ds)}, test={len(test_ds)}"
                    )

                train_dl = get_dataloader(train_ds, batch_size=32, shuffle=True)
                val_dl = get_dataloader(val_ds, batch_size=32, shuffle=False)
                test_dl = get_dataloader(test_ds, batch_size=32, shuffle=False)

                # Build model
                model = RNN(
                    dataset=sample_dataset,
                    embedding_dim=32,
                    hidden_dim=32,
                    num_layers=1,
                    dropout=0.0,
                )

                trainer = Trainer(
                    model=model,
                    metrics=["roc_auc", "f1"],
                    enable_logging=False,
                )
                trainer.train(
                    train_dataloader=train_dl,
                    val_dataloader=val_dl,
                    epochs=5,
                    optimizer_params={"lr": 1e-3},
                    monitor="roc_auc",
                    monitor_criterion="max",
                    load_best_model_at_last=False,
                )

                scores = trainer.evaluate(test_dl)
                auroc_val = scores.get("roc_auc", float("nan"))
                f1_val = scores.get("f1", float("nan"))
                if not (np.isnan(auroc_val)):
                    auroc_str = f"{auroc_val:.3f}"
                if not (np.isnan(f1_val)):
                    f1_str = f"{f1_val:.3f}"
                print(f"    -> AUROC={auroc_str}  F1={f1_str}")

            except Exception as exc:
                print(f"    -> Training failed: {exc}")

            result = {
                "features": feat_name,
                "window": window,
                "n_samples": n_samples,
                "n_positive": n_pos,
                "pos_rate": pos_rate,
                "auroc": auroc_str,
                "f1": f1_str,
            }
            results.append(result)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)
    header = (
        f"{'Feature Set':<28} {'Window':>6} {'Samples':>8} "
        f"{'Pos':>5} {'Rate':>7} {'AUROC':>7} {'F1':>7}"
    )
    print(header)
    print("-" * 80)
    for r in results:
        print(
            f"{r['features']:<28} {r['window']:>5}d {r['n_samples']:>8} "
            f"{r['n_positive']:>5} {r['pos_rate']:>6.1f}% "
            f"{r['auroc']:>7} {r['f1']:>7}"
        )
    print("-" * 80)

    # ------------------------------------------------------------------
    # Note on demo data
    # ------------------------------------------------------------------
    print(
        "\nNOTE: With demo data (20 patients), train/val/test splits are very\n"
        "small and metrics may be unreliable or N/A.  Use a full Synthea\n"
        "dataset (--root) for meaningful performance comparisons.\n"
    )


if __name__ == "__main__":
    main()
