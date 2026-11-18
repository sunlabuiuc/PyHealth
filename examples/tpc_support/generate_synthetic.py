# Contributor: Xiao Luo, UIUC NetID xiaoluo6.
# Paper: Temporal Pointwise Convolutional Networks for Length of Stay
# Prediction in the Intensive Care Unit. https://arxiv.org/abs/2007.09483
"""Write five invented patients in the shipped eICU CSV schema."""

import csv
from pathlib import Path
import yaml

TABLES = [
    "lab",
    "vitalperiodic",
    "vitalaperiodic",
    "intakeoutput",
    "respiratorycharting",
    "nursecharting",
]


def generate(root: Path) -> None:
    """Create schema-complete CSVs; all values are invented."""
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "pyhealth/datasets/configs/eicu.yaml").read_text()
    )
    rows = {table: [] for table in ["patient", *TABLES]}
    for sid, age, hours in [
        (101, "18", 12.5),
        (102, "60", 4),
        (103, "> 89", 400.5),
        (104, "17", 12),
        (105, "45", 8),
    ]:
        rows["patient"].append(
            dict(
                patientunitstayid=sid,
                uniquepid=f"synthetic-{sid}",
                patienthealthsystemstayid=sid,
                unitvisitnumber=1,
                hospitalid=1,
                age=age,
                gender="Male" if sid == 103 else "Female",
                admissionheight="" if sid == 105 else 165,
                admissionweight=65,
                unitdischargeoffset=int(hours * 60),
                unitdischargestatus="Alive",
                hospitaladmitoffset=0,
                hospitaldischargeoffset=int(hours * 60),
                hospitaldischargeyear=2020,
                hospitaldischargestatus="Alive",
            )
        )
        # Identical timing, contrasting values: detect cross-patient leakage.
        shift = 10 if sid == 103 else 0
        for minute, hr in [
            (-60, 250),
            (10, 80 + shift),
            (40, 100 + shift),
            (120, 110 + shift),
            (359, 120 + shift),
            (900, 200),
        ]:
            rows["vitalperiodic"].append(
                dict(
                    patientunitstayid=sid,
                    observationoffset=minute,
                    heartrate=hr,
                    sao2=98,
                    respiration=16,
                    systemicsystolic=120,
                    systemicdiastolic=70,
                    systemicmean=85,
                    temperature=37,
                )
            )
        for minute, value in [(60, 4), (90, 6), (240, 3), (300, 99)]:
            rows["lab"].append(
                dict(
                    patientunitstayid=sid,
                    labresultoffset=minute,
                    labname="potassium",
                    labresult=value,
                    labmeasurenameinterface="mmol/L",
                )
            )
        rows["lab"].append(
            dict(
                patientunitstayid=sid,
                labresultoffset=150,
                labname="creatinine",
                labresult=1.2,
                labmeasurenameinterface="mg/dL",
            )
        )
        rows["vitalaperiodic"].append(
            dict(
                patientunitstayid=sid,
                observationoffset=180,
                noninvasivesystolic=130,
                noninvasivediastolic=75,
                noninvasivemean=90,
            )
        )
        for minute, amount in [(65, 100), (95, 50)]:
            rows["intakeoutput"].append(
                dict(
                    patientunitstayid=sid,
                    intakeoutputoffset=minute,
                    celllabel="Urine",
                    cellvaluenumeric=amount,
                )
            )
        rows["respiratorycharting"].append(
            dict(
                patientunitstayid=sid,
                respchartoffset=180,
                respchartvaluelabel="FiO2",
                respchartvalue="40%",
            )
        )
        rows["nursecharting"].append(
            dict(
                patientunitstayid=sid,
                nursingchartoffset=240,
                nursingchartcelltypevalname="Temperature (F)",
                nursingchartvalue="98.6",
            )
        )
    for table, values in rows.items():
        cfg = config["tables"][table]
        fields = list(
            dict.fromkeys(
                [*cfg["attributes"], *(["uniquepid"] if table == "patient" else [])]
            )
        )
        with (root / cfg["file_path"]).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
