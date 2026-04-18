"""
Dynamic Survival Task for PyHealth.

This module implements a dynamic survival prediction task using:
- Anchor-based sampling
- Observation windows
- Discrete-time survival labels

The task converts longitudinal EHR data into sequence samples
for survival modeling.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple, Type


import numpy as np
from pyhealth.medcode import CrossMap
from pyhealth.tasks.base_task import BaseTask


# ======================
# Global Code Mappers
# ======================

GLOBAL_DIAG_MAPPER = CrossMap("ICD9CM", "CCSCM")
GLOBAL_PROC_MAPPER = CrossMap("ICD9PROC", "CCSPROC")
GLOBAL_DRUG_MAPPER = CrossMap("NDC", "ATC")


# ======================
# Utility Functions
# ======================


def build_daily_time_series_from_df(patient):
    """
    Build daily time series from a dataframe-based patient.

    This function handles the PyHealth main dataset structure where
    all events are stored in a single dataframe (patient.data_source).
    It extracts relevant medical events and aggregates them into
    daily time steps.

    Args:
        patient (Any): A PyHealth patient object with a dataframe
            stored in `patient.data_source`.

    Returns:
        List[Dict[str, Any]]: A list of daily aggregated visits where
        each entry contains:
            - time (int): day index
            - diagnosis (List[str])
            - procedure (List[str])
            - drug (List[str])
    """
    df = patient.data_source

    events = []

    for row in df.iter_rows(named=True):
        timestamp = row.get("timestamp")
        event_type = str(row.get("event_type")).lower()

        # Extract code based on event type
        if event_type == "diagnoses_icd":
            code = row.get("diagnoses_icd/icd9_code")

        elif event_type == "procedures_icd":
            code = row.get("procedures_icd/icd9_code")

        elif event_type == "prescriptions":
            code = row.get("prescriptions/ndc")

        else:
            # Ignore non-medical tables (patients, admissions, icustays)
            continue

        # Skip invalid rows
        if timestamp is None or code is None:
            continue

        events.append((timestamp, code, event_type))

    if not events:
        return []

    # Sort events by time
    events.sort(key=lambda x: x[0])
    first_time = events[0][0]

    # Map time -> codes
    time_to_codes = defaultdict(
        lambda: {"diagnosis": set(), "procedure": set(), "drug": set()}
    )

    for timestamp, code, event_type in events:
        delta_day = (timestamp - first_time).days

        if event_type == "diagnoses_icd":
            time_to_codes[delta_day]["diagnosis"].add(code)

        elif event_type == "procedures_icd":
            time_to_codes[delta_day]["procedure"].add(code)

        elif event_type == "prescriptions":
            time_to_codes[delta_day]["drug"].add(code)

    max_day = max(time_to_codes.keys())

    visits = []
    current_diag, current_proc, current_drug = set(), set(), set()

    # Build cumulative daily visits
    for day in range(max_day + 1):
        if day in time_to_codes:
            current_diag.update(time_to_codes[day]["diagnosis"])
            current_proc.update(time_to_codes[day]["procedure"])
            current_drug.update(time_to_codes[day]["drug"])

        visits.append(
            {
                "time": day,
                "diagnosis": list(current_diag),
                "procedure": list(current_proc),
                "drug": list(current_drug),
            }
        )

    return visits


def build_daily_time_series(patient) -> List[Dict[str, Any]]:
    """
    Convert patient events into a daily time series.

    Args:
        patient: Patient object with visits and event lists.

    Returns:
        List of daily aggregated visits.
    """
    events = []
    
    if hasattr(patient, "data_source"):
        return build_daily_time_series_from_df(patient)


    if hasattr(patient, "get_visits"):
        visits = patient.get_visits()
    else:
        visits = patient.visits.values()

    for visit in visits:
        for table in [
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
        ]:
            for event in visit.event_list_dict.get(table, []):
                timestamp = (
                    event.timestamp
                    if event.timestamp is not None
                    else visit.encounter_time
                )
                if timestamp is None:
                    continue
                events.append((timestamp, event.code, event.vocabulary))

    if not events:
        return []

    events.sort(key=lambda x: x[0])
    first_time = events[0][0]

    time_to_codes = defaultdict(
        lambda: {"diagnosis": set(), "procedure": set(), "drug": set()}
    )

    for timestamp, code, vocab in events:
        delta_day = (timestamp - first_time).days

        if vocab == "ICD9CM":
            time_to_codes[delta_day]["diagnosis"].add(code)
        elif vocab == "ICD9PROC":
            time_to_codes[delta_day]["procedure"].add(code)
        elif vocab == "NDC":
            time_to_codes[delta_day]["drug"].add(code)

    max_day = max(time_to_codes.keys())

    visits = []
    current_diag, current_proc, current_drug = set(), set(), set()

    for day in range(max_day + 1):
        if day in time_to_codes:
            current_diag.update(time_to_codes[day]["diagnosis"])
            current_proc.update(time_to_codes[day]["procedure"])
            current_drug.update(time_to_codes[day]["drug"])

        visits.append(
            {
                "time": day,
                "diagnosis": list(current_diag),
                "procedure": list(current_proc),
                "drug": list(current_drug),
            }
        )

    return visits


# ======================
# Engine
# ======================


class DynamicSurvivalEngine:
    """Core engine for dynamic survival sample generation."""

    def __init__(
        self,
        horizon: int = 24,
        observation_window: int = 24,
        anchor_interval: int = 12,
        anchor_strategy: str = "fixed",
    ):
        self.horizon = horizon
        self.observation_window = observation_window
        self.anchor_interval = anchor_interval
        self.anchor_strategy = anchor_strategy

    def generate_anchors(
        self,
        event_times: List[int],
        outcome_time: Optional[int],
        censor_time: Optional[int] = None,
    ) -> List[int]:
        """Generate anchor points."""
        if not event_times:
            return []

        max_time = (
            outcome_time
            if outcome_time is not None
            else (censor_time if censor_time is not None else max(event_times))
        )

        start_time = min(event_times) + self.observation_window

        if start_time >= max_time:
            return [max_time] if self.anchor_strategy == "single" else []

        if self.anchor_strategy == "fixed":
            anchors = list(
                range(int(start_time), int(max_time), self.anchor_interval)
            )
            return anchors if anchors else [max_time]

        return [max_time]

    def generate_survival_label(
        self,
        anchor_time: int,
        event_time: Optional[int],
        censor_time: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate survival label and mask."""
        y = np.zeros(self.horizon, dtype=float)
        mask = np.ones(self.horizon, dtype=float)

        if event_time is not None:
            delta = int(event_time - anchor_time)

            if delta < 0:
                mask[:] = 0
            elif delta < self.horizon:
                y[delta] = 1
                mask[delta + 1 :] = 0

        elif censor_time is not None:
            delta = int(censor_time - anchor_time)
            if delta < self.horizon:
                # Mask everything after delta
                mask[max(0, delta + 1) :] = 0

        return y, mask

    def process_patient(
        self, patient: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a patient into survival samples."""
        samples = []

        pid = patient.get("patient_id", "unknown")
        visits = patient.get("visits", [])
        event_time = patient.get("outcome_time")
        censor_time = patient.get("censor_time")

        event_times = [v["time"] for v in visits if "time" in v]
        anchors = self.generate_anchors(event_times, event_time, censor_time)

        for anchor in anchors:
            obs_start = anchor - self.observation_window
            seq = []

            for visit in visits:
                if obs_start <= visit["time"] < anchor:
                    if "feature" not in visit:
                        continue

                    feat = np.concatenate(
                        [
                            [
                                (visit["time"] - obs_start)
                                / self.observation_window
                            ],
                            visit["feature"],
                        ]
                    )
                    seq.append(feat)

            if not seq:
                continue

            x = np.array(seq, dtype=float)
            y, mask = self.generate_survival_label(
                anchor, event_time, censor_time
            )

            samples.append(
                {
                    "patient_id": pid,
                    "visit_id": f"{pid}_{anchor}",
                    "x": x.astype(np.float32),
                    "y": y.astype(np.float32),
                    "mask": mask.astype(np.float32),
                }
            )

        return samples

    def __call__(self, patient):
        return self.process_patient(patient)


# ======================
# Task
# ======================


class DynamicSurvivalTask(BaseTask):
    """PyHealth-compatible dynamic survival task."""

    task_name: str = "dynamic_survival"

    input_schema = {
        "x": "tensor",
    }

    output_schema = {
        "y": "tensor",
        "mask": "tensor",
    }
   

    def __init__(
        self,
        dataset,
        horizon: int = 24,
        observation_window: int = 24,
        anchor_interval: int = 12,
        anchor_strategy: str = "fixed",
        use_diag: bool = True,
        use_proc: bool = True,
        use_drug: bool = True,
    ):
        super().__init__()

        self.use_diag = use_diag
        self.use_proc = use_proc
        self.use_drug = use_drug

        self.engine = DynamicSurvivalEngine(
            horizon, observation_window, anchor_interval, anchor_strategy
        )

        self.diag_mapper = GLOBAL_DIAG_MAPPER
        self.proc_mapper = GLOBAL_PROC_MAPPER
        self.drug_mapper = GLOBAL_DRUG_MAPPER

        self.diag_vocab, self.proc_vocab, self.drug_vocab = (
            self.build_vocab(dataset)
        )

    def build_vocab(self, dataset):
        """
        Build vocabularies from a dataset.

        This function supports both PyHealth datasets (which provide
        iter_patients()) and mock datasets (which provide patients dict).

        Args:
            dataset (Any): Dataset object containing patient data.

        Returns:
            None
        """
        diag_set, proc_set, drug_set = set(), set(), set()

        if hasattr(dataset, "iter_patients"):
            patient_iter = dataset.iter_patients()
        else:
            patient_iter = dataset.patients.values()

        for i, patient in enumerate(patient_iter):
            if i > 5:
                break

            visits = build_daily_time_series(patient)

            for visit in visits:
                if self.use_diag:
                    diag_set.update(visit["diagnosis"])
                if self.use_proc:
                    proc_set.update(visit["procedure"])
                if self.use_drug:
                    drug_set.update(visit["drug"])

        self.diag_vocab = {c: i for i, c in enumerate(diag_set)}
        self.proc_vocab = {c: i for i, c in enumerate(proc_set)}
        self.drug_vocab = {c: i for i, c in enumerate(drug_set)}

        return self.diag_vocab, self.proc_vocab, self.drug_vocab

    def encode_multi_hot(self, codes, vocab):
        """Convert codes into multi-hot vector."""
        vec = np.zeros(len(vocab))
        for code in codes:
            if code in vocab:
                vec[vocab[code]] = 1
        return vec

    def __call__(self, patient) -> List[Dict[str, Any]]:
        """
        Convert a patient into dynamic survival samples.

        This function supports three types of patient inputs:
        1. Mock patients with visit dictionaries
        2. Dict-style patients (used in tests)
        3. PyHealth dataframe-based patients

        For dataframe-based patients, mortality is extracted using
        the 'expire_flag' field from patient-level events.

        Args:
            patient (Any): Patient object.

        Returns:
            List[Dict[str, Any]]: Survival samples.
        """

        # -------------------------
        # Mock patient (visits dict)
        # -------------------------
        if hasattr(patient, "visits") and isinstance(patient.visits, dict):
            visits_list = list(patient.visits.values())

            if len(visits_list) == 0:
                return []

            processed_visits = []
            start_time = visits_list[0].encounter_time

            for visit in visits_list:
                features = []

                if self.use_diag:
                    codes = [
                        e.code
                        for e in visit.event_list_dict.get("DIAGNOSES_ICD", [])
                    ]
                    features.append(
                        self.encode_multi_hot(codes, self.diag_vocab)
                    )

                if self.use_proc:
                    codes = [
                        e.code
                        for e in visit.event_list_dict.get("PROCEDURES_ICD", [])
                    ]
                    features.append(
                        self.encode_multi_hot(codes, self.proc_vocab)
                    )

                if self.use_drug:
                    codes = [
                        e.code
                        for e in visit.event_list_dict.get("PRESCRIPTIONS", [])
                    ]
                    features.append(
                        self.encode_multi_hot(codes, self.drug_vocab)
                    )

                x = (
                    np.concatenate(features).astype(np.float32)
                    if features
                    else np.zeros(1, dtype=np.float32)
                )

                time_idx = (visit.encounter_time - start_time).days

                processed_visits.append(
                    {
                        "time": time_idx,
                        "feature": x,
                    }
                )

            death_time = getattr(patient, "death_datetime", None)

            if death_time:
                outcome_time = (death_time - start_time).days
                censor_time = None
            else:
                outcome_time = None
                censor_time = processed_visits[-1]["time"]

            patient_dict = {
                "patient_id": patient.patient_id,
                "visits": processed_visits,
                "outcome_time": outcome_time,
                "censor_time": censor_time,
            }

            return self.engine.process_patient(patient_dict)

        # -------------------------
        # Dict-style patient (tests)
        # -------------------------
        if isinstance(patient, dict):
            return self.engine.process_patient(patient)

        # -------------------------
        # PyHealth dataframe patient
        # -------------------------
        visits_raw = build_daily_time_series(patient)
        if not visits_raw:
            return []

        processed_visits = []

        for visit in visits_raw:
            features = []

            if self.use_diag:
                mapped = [
                    m
                    for c in visit["diagnosis"]
                    for m in self.diag_mapper.map(c)
                    if m
                ]
                features.append(
                    self.encode_multi_hot(mapped, self.diag_vocab)
                )

            if self.use_proc:
                mapped = [
                    m
                    for c in visit["procedure"]
                    for m in self.proc_mapper.map(c)
                    if m
                ]
                features.append(
                    self.encode_multi_hot(mapped, self.proc_vocab)
                )

            if self.use_drug:
                mapped = [
                    m
                    for c in visit["drug"]
                    for m in self.drug_mapper.map(c)
                    if m
                ]
                features.append(
                    self.encode_multi_hot(mapped, self.drug_vocab)
                )

            x = (
                np.concatenate(features).astype(np.float32)
                if features
                else np.zeros(1, dtype=np.float32)
            )

            processed_visits.append(
                {
                    "time": visit["time"],
                    "feature": x,
                }
            )

        # Extract mortality signal
        death_flag = False

        if hasattr(patient, "get_events"):
            for event in patient.get_events():
                if event.event_type == "patients":
                    if event.attr_dict.get("expire_flag") == "1":
                        death_flag = True
                        break

        if death_flag:
            outcome_time = processed_visits[-1]["time"]
            censor_time = None
        else:
            outcome_time = None
            censor_time = processed_visits[-1]["time"]

        patient_dict = {
            "patient_id": getattr(patient, "patient_id", "unknown"),
            "visits": processed_visits,
            "outcome_time": outcome_time,
            "censor_time": censor_time,
        }

        return self.engine.process_patient(patient_dict)