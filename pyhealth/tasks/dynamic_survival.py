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
from typing import Any, Dict, List, Optional, Tuple

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


def build_daily_time_series(patient) -> List[Dict[str, Any]]:
    """
    Convert patient events into a daily time series.

    Args:
        patient: Patient object with visits and event lists.

    Returns:
        List of daily aggregated visits.
    """
    events = []

    for visit in patient.visits.values():
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
                mask[max(0, delta) :] = 0

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
        """Build vocabularies from dataset."""
        diag_set, proc_set, drug_set = set(), set(), set()

        for i, patient in enumerate(dataset.patients.values()):
            if i > 5:
                break

            visits = build_daily_time_series(patient)

            for visit in visits:
                if self.use_diag:
                    diag_set.update(
                        m
                        for c in visit["diagnosis"]
                        for m in self.diag_mapper.map(c)
                        if m
                    )

                if self.use_proc:
                    proc_set.update(
                        m
                        for c in visit["procedure"]
                        for m in self.proc_mapper.map(c)
                        if m
                    )

                if self.use_drug:
                    drug_set.update(
                        m
                        for c in visit["drug"]
                        for m in self.drug_mapper.map(c)
                        if m
                    )

        return (
            {c: i for i, c in enumerate(sorted(diag_set))},
            {c: i for i, c in enumerate(sorted(proc_set))},
            {c: i for i, c in enumerate(sorted(drug_set))},
        )

    def encode_multi_hot(self, codes, vocab):
        """Convert codes into multi-hot vector."""
        vec = np.zeros(len(vocab))
        for code in codes:
            if code in vocab:
                vec[vocab[code]] = 1
        return vec

    def __call__(self, patient) -> List[Dict[str, Any]]:
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

            x = np.concatenate(features) if features else np.zeros(1)

            processed_visits.append({"time": visit["time"], "feature": x})

        first_time = list(patient.visits.values())[0].encounter_time
        death_time = getattr(patient, "death_datetime", None)

        if death_time:
            outcome_time = (death_time - first_time).days
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