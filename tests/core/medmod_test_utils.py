"""Test helpers for the MedMod in-hospital mortality dataset and task: tiny
synthetic MIMIC-IV v1.0 + MIMIC-CXR subjects.

Default subject: one admission, one 96 h MICU stay (intime 2150-03-01 22:00),
anchor_age 60, one diagnosis, one AP chest X-ray at intime + 10 h with a
CheXpert row. Event tuples are (hours after intime, itemid, value[, table,
stay_id, hadm_id]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets.medmod.extraction import (
    MedModResources,
    SubjectTables,
    process_subject,
)
from pyhealth.datasets.medmod_mimic4 import MedModMIMIC4Dataset

INTIME = pd.Timestamp("2150-03-01 22:00:00")
SID, HADM, STAY = "10000001", "20000001", "30000001"

# MedMod's item map, only the rows of the item ids these tests use (so the
# tests need no download); copied unchanged from itemid_to_variable_map.csv
TEST_ITEMID_MAP = (
    Path(__file__).resolve().parents[2]
    / "test-resources"
    / "medmod"
    / "itemid_to_variable_map.csv"
)

# itemids from MedMod's itemid_to_variable_map.csv (STATUS == ready)
HR, GCS_EYE, CRR, TEMP_F, FIO2, HEIGHT_IN, SPO2, GLUCOSE_LAB, PH_LAB = (
    "220045",
    "220739",
    "3348",
    "223761",
    "223835",
    "226707",
    "220277",
    "50931",
    "50820",
)


def load_test_resources() -> MedModResources:
    """MedMod's files: the test item map and synthetic subject lists."""
    r = MedModResources.load(itemid_map=TEST_ITEMID_MAP)
    r.test_subjects = {"10000002"}
    r.val_subjects = {"10000003"}
    return r


def stay_row(
    stay=STAY, hadm=HADM, intime=INTIME, hours=96.0, first="MICU", last="MICU", los=None
):
    return {
        "stay_id": stay,
        "hadm_id": hadm,
        "first_careunit": first,
        "last_careunit": last,
        "intime": intime,
        "outtime": intime + pd.Timedelta(hours=hours),
        "los": str(hours / 24.0 if los is None else los),
    }


def make_subject(
    sid=SID,
    events=(),
    stays=None,
    admissions=None,
    anchor_age="60",
    dod=None,
    diagnoses=(HADM,),
    cxr=None,
    chexpert=("50000001",),
) -> SubjectTables:
    stays = [stay_row()] if stays is None else stays
    if admissions is None:
        admissions = [
            {
                "hadm_id": HADM,
                "admittime": INTIME - pd.Timedelta(hours=2),
                "dischtime": str(INTIME + pd.Timedelta(days=8)),
                "deathtime": None,
            }
        ]
    if cxr is None:
        cxr = [cxr_row(10.0)]
    ev_rows = []
    for e in events:
        hours, itemid, value = e[:3]
        table = e[3] if len(e) > 3 else "chartevents"
        stay = e[4] if len(e) > 4 else (STAY if table == "chartevents" else None)
        hadm = e[5] if len(e) > 5 else HADM
        ev_rows.append(
            {
                "charttime": INTIME + pd.Timedelta(hours=hours),
                "hadm_id": hadm,
                "stay_id": stay,
                "itemid": itemid,
                "value": value,
                "valuenum": None,
            }
        )
    return SubjectTables(
        subject_id=sid,
        patients=pd.DataFrame([{"anchor_age": anchor_age, "dod": dod}]),
        admissions=pd.DataFrame(admissions),
        icustays=pd.DataFrame(stays),
        events=pd.DataFrame(
            ev_rows,
            columns=["charttime", "hadm_id", "stay_id", "itemid", "value", "valuenum"],
        ),
        diagnoses=pd.DataFrame({"hadm_id": list(diagnoses)}),
        cxr=pd.DataFrame(
            cxr,
            columns=[
                "studydatetime",
                "dicom_id",
                "study_id",
                "viewposition",
                "image_path",
            ],
        ),
        chexpert_study_ids=set(chexpert),
    )


def cxr_row(hours, dicom="d1", study="50000001", view="AP", intime=INTIME):
    return {
        "studydatetime": intime + pd.Timedelta(hours=hours),
        "dicom_id": dicom,
        "study_id": study,
        "viewposition": view,
        "image_path": f"/cxr/files/p10/p{SID}/s{study}/{dicom}.jpg",
    }


def run(st, res, hours_zero="keep", image_dir=None, timestep=1.0):
    """Rule tests use 1-hour bins (the default), so bin i is hour i."""
    return {
        r.stay_id: r for r in process_subject(st, res, hours_zero, image_dir, timestep)
    }


def raw(ehr: np.ndarray) -> np.ndarray:
    """``ehr`` as values. The extraction returns it unnormalized (the processor
    standardizes later), so this is just a float copy."""
    return np.asarray(ehr, dtype=float).copy()


def col(res, name: str) -> int:
    return res.discretizer.output_header().index(name)


# --------------------------------------------- whole-dataset fixtures


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def write_fake_mimic(tmp_path: Path):
    """Three subjects as MIMIC-IV v1.0 + MIMIC-CXR CSVs; returns (ehr, cxr, cache)."""
    ehr, cxr = tmp_path / "mimiciv" / "1.0", tmp_path / "cxr"
    # subject 1: survives; subject 2: dies (deathtime); subject 3: ICU transfer
    _w(
        ehr / "core/patients.csv.gz",
        [
            {
                "subject_id": s,
                "gender": "F",
                "anchor_age": 60,
                "anchor_year": 2150,
                "anchor_year_group": "2008 - 2010",
                "dod": d,
            }
            for s, d in [(10000001, ""), (10000002, "2150-03-04"), (10000003, "")]
        ],
    )
    _w(
        ehr / "core/admissions.csv.gz",
        [
            {
                "subject_id": s,
                "hadm_id": h,
                "admittime": "2150-03-01 20:00:00",
                "dischtime": "2150-03-09 12:00:00",
                "deathtime": dt,
                "admission_type": "EW EMER.",
                "ethnicity": "WHITE",
                "hospital_expire_flag": int(bool(dt)),
            }
            for s, h, dt in [
                (10000001, 20000001, ""),
                (10000002, 20000002, "2150-03-04 10:00:00"),
                (10000003, 20000003, ""),
            ]
        ],
    )
    _w(
        ehr / "icu/icustays.csv.gz",
        [
            {
                "subject_id": s,
                "hadm_id": h,
                "stay_id": st,
                "first_careunit": "MICU",
                "last_careunit": last,
                "intime": "2150-03-01 22:00:00",
                "outtime": "2150-03-05 22:00:00",
                "los": 4.0,
            }
            for s, h, st, last in [
                (10000001, 20000001, 30000001, "MICU"),
                (10000002, 20000002, 30000002, "MICU"),
                (10000003, 20000003, 30000003, "SICU"),
            ]
        ],
    )
    ce = []
    for s, h, st in [
        (10000001, 20000001, 30000001),
        (10000002, 20000002, 30000002),
        (10000003, 20000003, 30000003),
    ]:
        ce += [
            {
                "subject_id": s,
                "hadm_id": h,
                "stay_id": st,
                "charttime": "2150-03-01 22:00:00",
                "storetime": "",
                "itemid": 220045,
                "value": "120",
                "valuenum": 120.0,
                "valueuom": "bpm",
                "warning": 0,
            },
            {
                "subject_id": s,
                "hadm_id": h,
                "stay_id": st,
                "charttime": "2150-03-02 01:30:00",
                "storetime": "",
                "itemid": 220045,
                "value": "88",
                "valuenum": 88.0,
                "valueuom": "bpm",
                "warning": 0,
            },
            {
                "subject_id": s,
                "hadm_id": h,
                "stay_id": st,
                "charttime": "2150-03-02 01:30:00",
                "storetime": "",
                "itemid": 220739,
                "value": "To Speech",
                "valuenum": 3.0,
                "valueuom": "",
                "warning": 0,
            },
            {
                "subject_id": s,
                "hadm_id": h,
                "stay_id": st,
                "charttime": "2150-03-02 01:30:00",
                "storetime": "",
                "itemid": 999999,
                "value": "1",
                "valuenum": 1.0,
                "valueuom": "",
                "warning": 0,
            },
        ]
    _w(ehr / "icu/chartevents.csv.gz", ce)
    _w(
        ehr / "hosp/labevents.csv.gz",
        [
            {
                "labevent_id": 1,
                "subject_id": 10000001,
                "hadm_id": 20000001,
                "specimen_id": 1,
                "itemid": 50931,
                "charttime": "2150-03-02 03:10:00",
                "storetime": "",
                "value": "140",
                "valuenum": 140.0,
                "valueuom": "mg/dL",
                "ref_range_lower": "",
                "ref_range_upper": "",
                "flag": "",
                "priority": "",
                "comments": "",
            }
        ],
    )
    _w(
        ehr / "hosp/diagnoses_icd.csv.gz",
        [
            {
                "subject_id": s,
                "hadm_id": h,
                "seq_num": 1,
                "icd_code": "4019",
                "icd_version": 9,
            }
            for s, h in [
                (10000001, 20000001),
                (10000002, 20000002),
                (10000003, 20000003),
            ]
        ],
    )
    _w(
        ehr / "hosp/d_icd_diagnoses.csv.gz",
        [{"icd_code": "4019", "icd_version": 9, "long_title": "HTN"}],
    )

    meta = [
        {
            "dicom_id": f"d{i}",
            "subject_id": s,
            "study_id": 50000000 + i,
            "PerformedProcedureStepDescription": "CHEST (PORTABLE AP)",
            "ViewPosition": "AP",
            "Rows": 3056,
            "Columns": 2544,
            "StudyDate": 21500302,
            "StudyTime": 81400.0,
            "ProcedureCodeSequence_CodeMeaning": "CHEST (PORTABLE AP)",
            "ViewCodeSequence_CodeMeaning": "antero-posterior",
            "PatientOrientationCodeSequence_CodeMeaning": "Erect",
        }
        for i, s in [(1, 10000001), (2, 10000002), (3, 10000003)]
    ]
    _w(cxr / "mimic-cxr-2.0.0-metadata.csv.gz", meta)
    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]
    _w(
        cxr / "mimic-cxr-2.0.0-chexpert.csv.gz",
        [
            {
                "subject_id": m["subject_id"],
                "study_id": m["study_id"],
                **{c: "" for c in labels},
            }
            for m in meta
        ],
    )
    return ehr, cxr, tmp_path / "cache"


def make_dataset(roots):
    ehr, cxr, cache = roots
    return MedModMIMIC4Dataset(
        ehr_root=ehr,
        cxr_root=cxr,
        cache_dir=cache,
        num_workers=1,
        itemid_map=TEST_ITEMID_MAP,
    )
