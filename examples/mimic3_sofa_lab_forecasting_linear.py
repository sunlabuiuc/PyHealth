"""Example/Ablation for SofaLabForecastingMIMIC3 on synthetic MIMIC-III data. 

Runs a 12h vs 24h lookback ablation over the forecasting task with a
scikit-learn linear regressor, mirroring the paper's `Linear` baseline:
NOTE: Very very very important that the task was for pyhealth model, but issues with vectorization of 
the unique input and nature of the task being a forecasting task scikit-learn was the most
reasonable option to demonstrate the task.

    Staniek et al. (2024), "Early Prediction of Causes (not Effects) in
    Healthcare by Long-Term Clinical Time Series Forecasting."
    https://arxiv.org/abs/2408.03816

This example follows the pattern of Section 5.2 and Appendix B of the paper showing 24h lookback.
The use of MIMIC-III demo data is not allowed, s ynthetic patients are used so the script runs without MIMIC-III access.
This is the same way the test cases for the task was implemented.
Since example is using sythetic data, might as well just use it for ablation too.

This example demonstrates:
1. Building synthetic patients with lab trajectories that mirror the paper's deterioration patterns
2. Applying the SofaLabForecastingMIMIC3 task to collect samples
3. Converting samples into arrays for modeling
4. Running a 12h vs 24h lookback ablation with a linear regression baseline
5. Computing both the paper's masked MSE and a SOFA proxy MSE for evaluation    

"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from pyhealth.data import Patient
from pyhealth.tasks import SofaLabForecastingMIMIC3

# important constants for synthetic data generation and task processing
BILI = "50885"
CREAT = "50912"
PLT = "51265"
T0 = datetime(2023, 1, 1, 0, 0, 0)

NUM_PATIENTS = 2000
LOOKBACK_SHORT = 12
LOOKBACK_LONG = 24
PREDICTION_HOURS = 24
STAY_HOURS = LOOKBACK_LONG + PREDICTION_HOURS + 1

# Baseline values from paper's Appendix B, just using to make synthetic data more real.
BASELINE_INTERCEPTS = {BILI: 0.8, CREAT: 0.9, PLT: 190.0}
BASELINE_SLOPES = {BILI: 0.05, CREAT: 0.03, PLT: -2.5}
DETERIORATION_PATTERNS = [
    {},
    {BILI: 2.5},
    {BILI: 6.5},
    {CREAT: 3.8, PLT: 45.0},
]

def make_patient(
    patient_id: str,
    icu_intime: datetime,
    icu_outtime: datetime,
    lab_events: Sequence[Tuple[datetime, str, float]],
    icustay_id: str,
) -> Patient:
    """Build a synthetic Patient with one ICU stay and lab events."""
    rows: List[dict] = [{
        "event_type": "icustays",
        "timestamp": icu_intime,
        "icustays/icustay_id": icustay_id,
        "icustays/outtime": icu_outtime.strftime("%Y-%m-%d %H:%M:%S"),
    }]
    for ts, itemid, valuenum in lab_events:
        rows.append({
            "event_type": "labevents",
            "timestamp": ts,
            "labevents/itemid": str(itemid),
            "labevents/valuenum": float(valuenum),
        })
    df = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime))
    return Patient(patient_id=patient_id, data_source=df)


def build_synthetic_patients() -> List[Patient]:
    """Creates synthetic patients with two observation events split across 12h halves

    Splitting observations across the short/long halves is what makes the
    12h vs 24h lookback ablation meaningful: the 12h window only sees the
    first event per lab. 
    This is so we can use for example + ablation
    """
    rng = np.random.default_rng(42)
    patients: List[Patient] = []
    pred_lo, pred_hi = LOOKBACK_LONG, LOOKBACK_LONG + PREDICTION_HOURS

    for idx in range(NUM_PATIENTS):
        stay_start = T0 + timedelta(days=idx)
        stay_end = stay_start + timedelta(hours=STAY_HOURS)
        pattern = DETERIORATION_PATTERNS[idx % len(DETERIORATION_PATTERNS)]

        lab_events: List[Tuple[datetime, str, float]] = []
        for lab in (BILI, CREAT, PLT):
            base = BASELINE_INTERCEPTS[lab] + BASELINE_SLOPES[lab] * idx
            future = pattern.get(lab, base)
            hours = (
                int(rng.integers(0, LOOKBACK_SHORT)),
                int(rng.integers(LOOKBACK_SHORT, LOOKBACK_LONG)),
                int(rng.integers(pred_lo, pred_hi)),
                int(rng.integers(pred_lo, pred_hi)),
            )
            values = (base, base + 0.1, future, future)
            for hour, value in zip(hours, values):
                lab_events.append((stay_start + timedelta(hours=hour), lab, value))

        patients.append(make_patient(
            patient_id=f"patient-{idx}",
            icu_intime=stay_start,
            icu_outtime=stay_end,
            lab_events=lab_events,
            icustay_id=f"{100000 + idx}",
        ))
    return patients


def samples_to_arrays(
    samples: Sequence[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert task samples into mask matrix for scikit-learn model"""
    x_train = np.asarray(
        [np.concatenate([s["observation_values"], s["observation_masks"]])
         for s in samples], dtype=np.float32,
    )
    y_predict = np.asarray([s["target_values"] for s in samples], dtype=np.float32)
    m_evaluation = np.asarray([s["target_masks"] for s in samples], dtype=np.float32)
    return x_train, y_predict, m_evaluation


def paper_masked_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: np.ndarray,
    prediction_hours: int = PREDICTION_HOURS,
    num_labs: int = SofaLabForecastingMIMIC3.NUM_LABS,
) -> float:
    """Equation 5 from paper for computing masked MSE, average over patients and timesteps."""
    shape = (-1, prediction_hours, num_labs)
    sq_error = ((y_true.reshape(shape) - y_pred.reshape(shape))
                * masks.reshape(shape)) ** 2
    return float(sq_error.sum() / (y_true.shape[0] * prediction_hours))


def future_lab_sofa(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Undo standardization to apply SOFA direct to each patient from the predicted lab values.

    Applies the SOFA task's Appendix Bthresholds to max-bili, max-creat, and min-platelets per patient.
    Convert back, find worst value in prediction window and apply and sum.
    """
    task = SofaLabForecastingMIMIC3()
    num_labs = SofaLabForecastingMIMIC3.NUM_LABS
    stats = np.asarray(
        [task.LAB_NORMALIZATION_STATS[lab] for lab in (BILI, CREAT, PLT)],
        dtype=np.float32,
    )
    mean, std = stats[:, 0], stats[:, 1]

    v = values.reshape(values.shape[0], -1, num_labs) * std + mean
    m = masks.reshape(masks.shape[0], -1, num_labs) > 0

    scorers = (task._sofa_bilirubin, task._sofa_creatinine, task._sofa_platelets)
    aggregators = (np.max, np.max, np.min)
    scores = np.zeros(values.shape[0], dtype=np.float32)
    for patient_idx in range(values.shape[0]):
        for lab_idx, (aggregate, scorer) in enumerate(zip(aggregators, scorers)):
            observed = v[patient_idx, :, lab_idx][m[patient_idx, :, lab_idx]]
            if observed.size:
                scores[patient_idx] += scorer(float(aggregate(observed)))
    return scores

def run_ablation(lookback_hours: int, patients: Sequence[Patient]) -> dict:
    """Run one arm of the lookback ablation with a linear regression baseline. 24HOUR is default"""
    task = SofaLabForecastingMIMIC3(
        lookback_hours=lookback_hours,
        prediction_hours=PREDICTION_HOURS,
    )
    samples = [sample for patient in patients for sample in task(patient)]
    x, y, masks = samples_to_arrays(samples)
    x_tr, x_te, y_tr, y_te, _, m_te = train_test_split(
        x, y, masks, test_size=0.25, random_state=42,
    )

    model = LinearRegression().fit(x_tr, y_tr)
    y_pred = model.predict(x_te)

    mse_sofa = float(np.mean(
        (future_lab_sofa(y_te, m_te) - future_lab_sofa(y_pred, m_te)) ** 2
    ))
    return {
        "lookback_hours": lookback_hours,
        "num_samples": len(samples),
        "masked_mse": paper_masked_mse(y_te, y_pred, m_te),
        "mse_sofa": mse_sofa,
    }


def print_results(results: Sequence[dict]) -> None:
    """This printing style is from the other task examples like: mp_stagenet_mimic4_interpret.py"""
    print("\nSOFA Lab Forecasting Ablation")
    print("=" * 60)
    print(f"{'lookback':>10} | {'samples':>7} | "
          f"{'masked_mse':>12} | {'mse_sofa':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['lookback_hours']:>10} | {r['num_samples']:>7} | "
              f"{r['masked_mse']:>12.4f} | {r['mse_sofa']:>10.4f}")
    print("=" * 60)

patients = build_synthetic_patients()
results = [
    run_ablation(LOOKBACK_SHORT, patients),
    run_ablation(LOOKBACK_LONG, patients),
]
print_results(results)
