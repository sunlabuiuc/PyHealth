"""Improve the benchmark's 48-hour in-hospital mortality model (eICU) with AutoImprover.

Runs task T1 of the PyHealth clinical prediction benchmark plan,
``ihm48_eicu@v1``: in-hospital mortality, predicted 48 h after hospital
admission, for adults still in hospital at 48 h. AutoImprover searches one of
two input views:

- ``tabular``: the benchmark's tabular summary. Last, minimum, maximum and
  count of 15 labs and 6 vital signs in the first 48 h, plus age, sex, the
  number of distinct medications and of prior diagnoses. LogisticRegression,
  MLP and XGBoost, starting from XGBoost (the benchmark's reference model).
- ``sequence``: the same labs and vitals as an hourly 48-step series, plus
  medication and prior-diagnosis codes. LogisticRegression, MLP, RNN,
  Transformer, CNN and RETAIN, starting from the benchmark's GRU config.
  Until #1315 is merged, LogisticRegression and MLP cannot read the
  time series, so the model check removes them and this view searches RNN,
  Transformer, CNN and RETAIN only.

The task, ``InHospitalMortality48hEICU``, is defined in this example and is
not part of ``pyhealth.tasks``. (``MortalityPredictionEICU`` there predicts
mortality at the next visit, a different task.)

Lab and vital values are expressed as (value - normal) / scale with fixed
clinical reference constants (``VARIABLES``), not statistics fitted on the
data, so nothing leaks between splits; a value that was never measured counts
as normal, and its count of 0 tells the model it is missing. GCS is not used:
the demo holds it only in nurse charting.

Splits follow the benchmark (``PatientSplit((0.7, 0.1, 0.2), seed=2026)``).
Each trial is scored by 5-fold patient-grouped cross-validation on the
training patients. The test patients are used once at the end, with APACHE
IVa's predicted hospital mortality on the same stays as a clinical reference.

Data: the eICU-CRD demo (2,520 ICU stays, 186 hospitals) is open access on
PhysioNet; no credentials are needed. Put patient, lab, medication, diagnosis,
vitalPeriodic, vitalAperiodic and apachePatientResult (``.csv.gz``) from
https://physionet.org/content/eicu-crd-demo/2.0.1/ in one folder.

With ``--llm claude`` the proposals come from Claude
(``pip install "anthropic>=1.15"`` and set ANTHROPIC_API_KEY). Prompts hold
the search space, dataset sizes, aggregate label counts and scores, never
patient records.

Usage:
    python examples/autoimprove_eicu_ihm48.py --root ~/pyhealth-demo/eicu-crd-demo-2.0.1
    python examples/autoimprove_eicu_ihm48.py --root ... --view sequence --llm claude
    python examples/autoimprove_eicu_ihm48.py --root ... --check-space
"""

import argparse
import inspect
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pyhealth.autoimprove import (
    DEFAULT_MODELS,
    TRAIN_PARAMS,
    AutoImprover,
    ClaudeLLM,
    LLMProposer,
    Param,
    PipelineConfig,
    RandomProposer,
    SearchSpace,
    check_models,
    with_xgboost,
)
from pyhealth.datasets import PatientSplit, eICUDataset
from pyhealth.metrics import bootstrap_ci, paired_bootstrap_diff
from pyhealth.tasks import BaseTask, FeatureInfo, TaskCard

PREDICTION_MINUTES = 48 * 60
"""Prediction time, in minutes after hospital admission."""

# name: (sources, normal, scale, plausible low, plausible high). Sources are
# (table, column) for vitals and ("lab", labname) for labs. Values outside the
# plausible range are dropped as entry errors.
VARIABLES = {
    "sodium": ([("lab", "sodium")], 140, 5, 100, 180),
    "potassium": ([("lab", "potassium")], 4.2, 0.6, 1.5, 9),
    "chloride": ([("lab", "chloride")], 102, 5, 70, 140),
    "bicarbonate": ([("lab", "bicarbonate")], 24, 4, 5, 50),
    "glucose": (
        [("lab", "glucose"), ("lab", "bedside glucose")], 110, 50, 20, 1500,
    ),
    "calcium": ([("lab", "calcium")], 9.2, 0.8, 4, 16),
    "magnesium": ([("lab", "magnesium")], 2.0, 0.4, 0.5, 6),
    "anion_gap": ([("lab", "anion gap")], 12, 4, 0, 50),
    "phosphate": ([("lab", "phosphate")], 3.5, 1.0, 0.5, 15),
    "creatinine": ([("lab", "creatinine")], 1.0, 1.0, 0.1, 20),
    "bun": ([("lab", "BUN")], 15, 12, 1, 250),
    "hemoglobin": ([("lab", "Hgb")], 12.5, 2, 3, 22),
    "platelets": ([("lab", "platelets x 1000")], 250, 100, 1, 1500),
    "wbc": ([("lab", "WBC x 1000")], 8, 4, 0.1, 200),
    "lactate": ([("lab", "lactate")], 1.2, 1.5, 0.2, 30),
    "heart_rate": ([("vitalperiodic", "heartrate")], 80, 20, 20, 250),
    "resp_rate": ([("vitalperiodic", "respiration")], 16, 6, 3, 70),
    "spo2": ([("vitalperiodic", "sao2")], 97, 3, 50, 100),
    "temperature": ([("vitalperiodic", "temperature")], 37, 0.8, 30, 43),
    "sbp": (
        [("vitalaperiodic", "noninvasivesystolic"),
         ("vitalperiodic", "systemicsystolic")], 120, 20, 40, 260,
    ),
    "dbp": (
        [("vitalaperiodic", "noninvasivediastolic"),
         ("vitalperiodic", "systemicdiastolic")], 70, 12, 20, 180,
    ),
}
LAB_NAMES = {
    source: name
    for name, (sources, *_) in VARIABLES.items()
    for table, source in sources
    if table == "lab"
}
TABULAR_NAMES = (
    ["age", "female"]
    + [f"{v}_{s}" for v in VARIABLES for s in ("last", "min", "max", "n")]
    + ["n_meds", "n_prior_dx"]
)

# name: (words, unit, group, expected effect of a higher value on mortality):
# "+" higher is worse, "-" lower is worse, "u" both extremes are worse.
MEANINGS = {
    "sodium": ("sodium", "mmol/L", "labs: electrolytes", "u"),
    "potassium": ("potassium", "mmol/L", "labs: electrolytes", "u"),
    "chloride": ("chloride", "mmol/L", "labs: electrolytes", "u"),
    "bicarbonate": ("bicarbonate", "mmol/L", "labs: acid-base", "u"),
    "glucose": ("glucose", "mg/dL", "labs: metabolic", "u"),
    "calcium": ("calcium", "mg/dL", "labs: electrolytes", "u"),
    "magnesium": ("magnesium", "mg/dL", "labs: electrolytes", "u"),
    "anion_gap": ("anion gap", "mmol/L", "labs: acid-base", "+"),
    "phosphate": ("phosphate", "mg/dL", "labs: electrolytes", "u"),
    "creatinine": ("creatinine", "mg/dL", "labs: kidney", "+"),
    "bun": ("urea nitrogen", "mg/dL", "labs: kidney", "+"),
    "hemoglobin": ("hemoglobin", "g/dL", "labs: blood count", "-"),
    "platelets": ("platelets", "10^3/uL", "labs: blood count", "-"),
    "wbc": ("white cell count", "10^3/uL", "labs: blood count", "u"),
    "lactate": ("lactate", "mmol/L", "labs: perfusion", "+"),
    "heart_rate": ("heart rate", "/min", "vital signs", "u"),
    "resp_rate": ("respiratory rate", "/min", "vital signs", "u"),
    "spo2": ("oxygen saturation", "%", "vital signs", "-"),
    "temperature": ("temperature", "C", "vital signs", "u"),
    "sbp": ("systolic blood pressure", "mmHg", "vital signs", "-"),
    "dbp": ("diastolic blood pressure", "mmHg", "vital signs", "-"),
}
# A two-sided ("u") variable's lowest value matters when low, its highest when high.
STAT_DIRECTION = {"u": {"last": "u", "min": "-", "max": "+"}}


def task_card(view: str) -> TaskCard:
    """The task card of ``ihm48_eicu@v1``: design rationale and feature dictionary."""
    features = {
        "age": FeatureInfo("age in years, as (age - 65) / 15", "demographics", "+",
                           rationale="mortality rises with age"),
        "female": FeatureInfo("1 if the patient is female", "demographics", ""),
    }
    for name, (words, unit, group, kind) in MEANINGS.items():
        _, normal, scale, _, _ = VARIABLES[name]
        units = f"in steps of {scale:g} {unit} from {normal:g} {unit}"
        if view == "tabular":
            for stat, label in (("last", "last"), ("min", "lowest"), ("max", "highest")):
                direction = STAT_DIRECTION.get(kind, {}).get(stat, kind)
                features[f"{name}_{stat}"] = FeatureInfo(
                    f"{label} {words} in the first 48 h, {units} (0 if unmeasured)",
                    group, direction, unit)
            features[f"{name}_n"] = FeatureInfo(
                f"number of {words} measurements in the first 48 h (log scale)",
                "monitoring intensity", "+",
                rationale="sicker patients are measured more often")
        else:
            features[name] = FeatureInfo(
                f"hourly {words} over the first 48 h, {units}, carried forward",
                group, kind, unit)
    features["n_meds"] = FeatureInfo(
        "number of distinct medications started before 48 h (log scale)",
        "treatment", "+", rationale="more treatment marks a sicker patient")
    features["n_prior_dx"] = FeatureInfo(
        "number of diagnoses from earlier hospital stays (log scale)", "history", "+")
    columns = ({"tabular": TABULAR_NAMES} if view == "tabular" else
               {"vitals_labs": list(VARIABLES), "demographics": ["age", "female"]})
    return TaskCard(
        population=(("Adults (18+) still in hospital 48 h after admission, with "
                     "at least one lab before 48 h"),
                    ("the benchmark's T1 cohort: a prediction at 48 h is only "
                     "useful for patients who are still in hospital")),
        prediction_time=("48 h after hospital admission",
                         "enough data to judge severity, early enough to act"),
        outcome=("death before hospital discharge",
                 "in-hospital mortality, as recorded at discharge"),
        horizon=("until hospital discharge", "the benchmark's T1 definition"),
        exclusions=[
            ("hospital stays shorter than 48 h",
             "the prediction time must fall inside the stay"),
            ("unknown discharge status", "the outcome cannot be labelled"),
        ],
        features=features, columns=columns,
        notes=("Unmeasured labs and vitals count as normal (0), with a count of 0 "
               "telling the model they are missing; GCS is not used because the "
               "demo holds it only in nurse charting; temperature is charted in "
               "few stays. Open eICU-CRD demo data: 186 hospitals, 2014-2015."),
    )

VITAL_TABLES = {
    "vitalperiodic": (
        "vitalPeriodic.csv",
        ["observationoffset", "heartrate", "respiration", "sao2", "temperature",
         "systemicsystolic", "systemicdiastolic"],
    ),
    "vitalaperiodic": (
        "vitalAperiodic.csv",
        ["observationoffset", "noninvasivesystolic", "noninvasivediastolic"],
    ),
}
TABLES = ["lab", "medication", "diagnosis", *VITAL_TABLES]


def write_config(path: Path) -> Path:
    """PyHealth's eICU config plus the two vital-sign tables T1 needs."""
    default = Path(inspect.getfile(eICUDataset)).parent / "configs" / "eicu.yaml"
    with open(default) as f:
        config = yaml.safe_load(f)
    for name, (file_path, columns) in VITAL_TABLES.items():
        config["tables"][name] = {
            "file_path": file_path,
            "patient_id": "uniquepid",
            "timestamp": None,
            "join": [{"file_path": "patient.csv", "on": "patientunitstayid",
                      "how": "inner", "columns": ["uniquepid"]}],
            "attributes": ["patientunitstayid", *columns],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


class InHospitalMortality48hEICU(BaseTask):
    """Benchmark task T1 on eICU (``ihm48_eicu@v1``).

    Defined here for this example only; it is not part of ``pyhealth.tasks``.

    One sample per hospital stay (``patienthealthsystemstayid``), anchored at
    its first ICU stay. Prediction time is 48 h after hospital admission.
    Inclusion: age 18 or over, hospital stay of at least 48 h (so alive at
    48 h), a known discharge status and at least one lab before 48 h. Label:
    ``hospitaldischargestatus == "Expired"``. Features use only events in
    [admission, admission + 48 h); medications are those started before
    48 h; prior diagnoses come from the patient's hospital stays discharged
    in an earlier year (eICU gives no finer order between stays).

    Args:
        view: ``"tabular"`` or ``"sequence"`` (see the module docstring).

    Examples:
        >>> task = InHospitalMortality48hEICU("tabular")
        >>> task.input_schema
        {'tabular': 'tensor'}
    """

    output_schema = {"mortality": "binary"}
    SCHEMAS = {
        "tabular": {"tabular": "tensor"},
        "sequence": {
            "vitals_labs": "timeseries",
            "meds": "sequence",
            "prior_dx": "sequence",
            "demographics": "tensor",
        },
    }

    def __init__(self, view: str = "tabular"):
        super().__init__()
        self.view = view
        self.task_name = f"ihm48_eicu_v1_{view}"
        self.input_schema = dict(self.SCHEMAS[view])

    def task_card(self) -> TaskCard:
        return task_card(self.view)

    def __call__(self, patient):
        units = patient.get_events("patient", return_df=True)
        stays = defaultdict(list)
        for unit in units.iter_rows(named=True):
            stays[unit["patient/patienthealthsystemstayid"]].append(unit)
        events = _events_by_stay(patient, units)
        years = {
            hosp: _num(rows[0]["patient/hospitaldischargeyear"])
            for hosp, rows in stays.items()
        }
        samples = []
        for hosp, rows in stays.items():
            first = min(rows, key=lambda r: _num(r["patient/unitvisitnumber"]))
            age = _age(first["patient/age"])
            admit = _num(first["patient/hospitaladmitoffset"])
            length = _num(first["patient/hospitaldischargeoffset"]) - admit
            status = first["patient/hospitaldischargestatus"]
            if age is None or age < 18 or not length >= PREDICTION_MINUTES:
                continue
            if status not in ("Alive", "Expired"):
                continue
            obs = _observations(events, hosp)
            if not any(VARIABLES[v][0][0][0] == "lab" for _, v, _ in obs):
                continue
            meds = _medications(events["medication"][hosp])
            prior = _prior_diagnoses(events["diagnosis"], years, hosp)
            female = float(first["patient/gender"] == "Female")
            sample = {
                "patient_id": patient.patient_id,
                "visit_id": str(first["patient/patientunitstayid"]),
                "mortality": int(status == "Expired"),
            }
            if self.view == "tabular":
                sample["tabular"] = _summary(obs, age, female, meds, prior)
            else:
                sample["vitals_labs"] = _hourly(obs)
                sample["meds"] = meds or ["<none>"]
                sample["prior_dx"] = prior or ["<none>"]
                sample["demographics"] = [(age - 65) / 15, female]
            samples.append(sample)
        return samples


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _age(value) -> float | None:
    if value in (None, ""):
        return None
    number = 90.0 if str(value).startswith(">") else _num(value)
    return None if math.isnan(number) else number


def _events_by_stay(patient, units):
    """Each table's rows by hospital stay, with minutes since hospital admission.

    eICU offsets count minutes from the admission to the row's own ICU stay;
    subtracting that stay's ``hospitaladmitoffset`` puts every row of a
    hospital stay on one clock.
    """
    unit_info = {
        str(u["patient/patientunitstayid"]): (
            u["patient/patienthealthsystemstayid"],
            _num(u["patient/hospitaladmitoffset"]),
        )
        for u in units.iter_rows(named=True)
    }
    offsets = {"lab": "labresultoffset", "medication": "drugstartoffset",
               "diagnosis": "diagnosisoffset", "vitalperiodic": "observationoffset",
               "vitalaperiodic": "observationoffset"}
    out = {}
    for table in TABLES:
        out[table] = defaultdict(list)
        df = patient.get_events(table, return_df=True)
        for row in df.iter_rows(named=True):
            unit = unit_info.get(str(row[f"{table}/patientunitstayid"]))
            if unit is not None:
                minute = _num(row[f"{table}/{offsets[table]}"]) - unit[1]
                out[table][unit[0]].append((minute, row))
    return out


def _observations(events, hosp):
    """(minute after admission, variable, standardised value) before 48 h."""
    out = []
    for minute, row in events["lab"][hosp]:
        name = LAB_NAMES.get(row["lab/labname"])
        if name is not None:
            out.append((minute, name, _num(row["lab/labresult"])))
    for table in VITAL_TABLES:
        sources = [
            (name, column)
            for name, (srcs, *_) in VARIABLES.items()
            for t, column in srcs
            if t == table
        ]
        for minute, row in events[table][hosp]:
            for name, column in sources:
                out.append((minute, name, _num(row[f"{table}/{column}"])))
    kept = []
    for minute, name, value in out:
        _, normal, scale, low, high = VARIABLES[name]
        if 0 <= minute < PREDICTION_MINUTES and low <= value <= high:
            kept.append((minute, name, (value - normal) / scale))
    return sorted(kept)


def _medications(rows):
    codes = set()
    for start, row in rows:
        cancelled = row["medication/drugordercancelled"] == "Yes"
        if not start < PREDICTION_MINUTES or cancelled:
            continue
        name = (row["medication/drugname"] or "").strip().upper()
        hicl = row["medication/drughiclseqno"]
        if name:
            codes.add(name)
        elif hicl not in (None, ""):
            codes.add(f"HICL:{hicl}")
    return sorted(codes)


def _prior_diagnoses(by_stay, years, hosp):
    earlier = {h for h, year in years.items() if year < years[hosp]}
    codes = set()
    for h in earlier:
        for _, row in by_stay[h]:
            for code in (row["diagnosis/icd9code"] or "").split(","):
                if code.strip():
                    codes.add(code.strip())
    return sorted(codes)


def _summary(obs, age, female, meds, prior):
    by_var = defaultdict(list)
    for _, name, value in obs:
        by_var[name].append(value)
    row = [(age - 65) / 15, female]
    for name in VARIABLES:
        values = by_var.get(name, [])
        if values:
            row += [values[-1], min(values), max(values), math.log1p(len(values))]
        else:
            row += [0.0, 0.0, 0.0, 0.0]
    return row + [math.log1p(len(meds)), math.log1p(len(prior))]


_T0 = datetime(2000, 1, 1)


def _hourly(obs):
    """48 hourly rows; the last value in each hour, NaN where unmeasured."""
    names = list(VARIABLES)
    grid = np.full((PREDICTION_MINUTES // 60, len(names)), np.nan)
    for minute, name, value in obs:
        grid[int(minute // 60), names.index(name)] = value
    times = [_T0 + timedelta(hours=h) for h in range(len(grid))]
    return times, grid


# ---------------------------------------------------------------- search
TRAIN_SPACE = {
    **TRAIN_PARAMS,
    "epochs": Param("int", 5, 60, default=30,
                    meaning=TRAIN_PARAMS["epochs"].meaning),
    "patience": Param("int", 1, 10, default=5,
                      meaning=TRAIN_PARAMS["patience"].meaning),
}
SPACES = {
    "tabular": with_xgboost(
        SearchSpace(
            {k: DEFAULT_MODELS[k] for k in ("LogisticRegression", "MLP")},
            TRAIN_SPACE,
        )
    ),
    "sequence": SearchSpace(
        {
            k: DEFAULT_MODELS[k]
            for k in ("LogisticRegression", "MLP", "RNN", "Transformer", "CNN",
                      "RETAIN")
        },
        TRAIN_SPACE,
    ),
}
BENCHMARK_TRAIN = {"epochs": 30, "batch_size": 256, "lr": 1e-3,
                   "weight_decay": 1e-5, "patience": 5, "pos_weight": "balanced"}
BASELINES = {
    # The benchmark plan's reference model, class-weighted (the plan is not
    # part of this repository).
    "tabular": PipelineConfig(
        "XGBoost",
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": "balanced"},
    ),
    # The benchmark plan's GRU settings.
    "sequence": PipelineConfig(
        "RNN",
        {"rnn_type": "GRU", "embedding_dim": 128, "hidden_dim": 128,
         "dropout": 0.2},
        dict(BENCHMARK_TRAIN),
    ),
}


def test_subgroups(test, view):
    """Age band and sex of every test sample, for subgroup results."""
    key = "tabular" if view == "tabular" else "demographics"
    bands, sex = [], []
    for i in range(len(test)):
        age_z, female = test[i][key][0].item(), test[i][key][1].item()
        age = age_z * 15 + 65
        bands.append("18-64" if age < 65 else "65-79" if age < 80 else "80+")
        sex.append("female" if female > 0.5 else "male")
    return {"age band": bands, "sex": sex}


def apache_reference(root, test, y_true, p_best, groups, n_boot):
    """APACHE IVa predicted hospital mortality on the same test stays."""
    apache = pd.read_csv(Path(root) / "apachePatientResult.csv.gz")
    apache = apache[(apache.apacheversion == "IVa")
                    & (apache.predictedhospitalmortality >= 0)]
    risk = dict(zip(apache.patientunitstayid.astype(str),
                    apache.predictedhospitalmortality))
    stay_of = {i: stay for stay, idx in test.record_to_index.items() for i in idx}
    p_apache = np.array([risk.get(stay_of.get(i), np.nan) for i in range(len(test))])
    has = ~np.isnan(p_apache)
    out = {"n_with_apache": int(has.sum()), "n_test": int(len(test))}
    for metric in ("pr_auc", "roc_auc"):
        out[f"apache_{metric}"] = bootstrap_ci(
            y_true[has], p_apache[has], metric, groups=groups[has],
            n_boot=n_boot, seed=0)
        out[f"best_{metric}_same_stays"] = bootstrap_ci(
            y_true[has], p_best[has], metric, groups=groups[has],
            n_boot=n_boot, seed=0)
        out[f"best_minus_apache_{metric}"] = paired_bootstrap_diff(
            y_true[has], p_best[has], p_apache[has], metric, groups=groups[has],
            n_boot=n_boot, seed=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="eICU demo folder")
    parser.add_argument("--view", choices=list(SPACES), default="tabular")
    parser.add_argument("--llm", choices=["none", "claude"], default="none")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--time-budget", type=float, default=5400)
    parser.add_argument("--run-dir", default="runs/ihm48_eicu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--check-space", action="store_true",
                        help="check which PyHealth models run on the view, then stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("pyhealth.autoimprove").setLevel(logging.INFO)

    run_dir = Path(args.run_dir) / f"{args.llm}_{args.view}"
    config_path = write_config(Path(args.run_dir) / "eicu_t1.yaml")
    dataset = eICUDataset(root=args.root, tables=TABLES, config_path=str(config_path),
                          cache_dir=args.cache_dir)
    train, val, test = dataset.set_task(
        InHospitalMortality48hEICU(args.view),
        split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=2026),
    )
    for name, part in (("train", train), ("val", val), ("test", test)):
        y = np.array([float(part[i]["mortality"]) for i in range(len(part))])
        print(f"{name}: {len(part)} stays, {len(part.patient_to_index)} patients, "
              f"{int(y.sum())} deaths ({y.mean():.1%})")
    space = SPACES[args.view]
    if args.check_space:
        # AutoImprover runs this check itself and drops failing models; here
        # it covers every default model, in the space or not.
        everything = with_xgboost(SearchSpace(DEFAULT_MODELS, TRAIN_SPACE))
        for name, error in check_models(everything, train, device="cpu").items():
            print(f"  {name}: {error or 'ok'}")
        return

    proposer = LLMProposer(ClaudeLLM()) if args.llm == "claude" else RandomProposer(0)
    improver = AutoImprover(
        train, val, proposer, baseline=BASELINES[args.view], space=space,
        metric="pr_auc", report_metrics=["roc_auc"], n_agents=args.agents,
        n_cycles=args.cycles, patience=args.patience, time_budget=args.time_budget,
        seed=0, device="cpu", run_dir=str(run_dir),
        lessons_path=str(run_dir / "lessons_shared.json"), cv_folds=args.cv_folds,
        task_card=task_card(args.view),
    )
    start = time.time()
    result = improver.run()
    print(f"\n{len(result.trials)} trials in {time.time() - start:.0f}s")
    print(result.summary(k=12))
    print("lessons:", *result.lessons, sep="\n- ")

    improver.attribute()   # what each changed setting contributed
    improver.explain()     # how the selected model decides
    scores = improver.evaluate_on_test(test, n_boot=1000,
                                       subgroups=test_subgroups(test, args.view),
                                       return_predictions=True)
    predictions = scores.pop("predictions")
    y_true, p_best = np.ravel(predictions["y_true"]), np.ravel(predictions["best"])
    groups = np.asarray(predictions["patient_ids"])
    scores["best_roc_auc"] = bootstrap_ci(y_true, p_best, "roc_auc",
                                          groups=groups, n_boot=1000, seed=0)
    scores["apache"] = apache_reference(args.root, test, y_true, p_best, groups, 1000)
    out = {
        "task": "ihm48_eicu@v1", "view": args.view, "proposer": args.llm,
        "n_trials": len(result.trials), "selected_trial": result.best.trial_id,
        "top_mean_trial": result.leaderboard(1)[0].trial_id,
        "best_cv_pr_auc": result.best.score,
        "baseline_cv_pr_auc": result.baseline.score if result.baseline else None,
        "lessons": result.lessons, "test": scores,
    }
    os.makedirs(run_dir, exist_ok=True)
    (run_dir / "test_eval.json").write_text(json.dumps(out, indent=1, default=str))
    improver.report()
    print(f"report: {run_dir / 'report.html'}")
    for flag in improver.flags():
        print(f"flag for review: {flag['id']}: {flag['detail']}")


if __name__ == "__main__":
    main()
