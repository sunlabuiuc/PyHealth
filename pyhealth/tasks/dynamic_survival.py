# pyhealth/tasks/dynamic_survival.py

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from .base_task import BaseTask


class DynamicSurvivalTask(BaseTask):
    """
    Dynamic Survival Analysis Task for Early Event Prediction.

    This task converts each patient trajectory into multiple
    anchor-based samples and generates discrete-time survival labels.

    Based on the anchor-based discrete-time survival formulation from:
    Yèche et al. (2024), *Dynamic Survival Analysis for Early Event Prediction*.

    Args:
        horizon (int): Prediction horizon length.
        observation_window (int): Look-back window before each anchor.
        anchor_interval (int): Time interval between anchors.
        anchor_strategy (str): "fixed" or "single".

    Expected Input Format:
        patient (Dict[str, Any]):
            {
                "patient_id": str,
                "visits": List[Dict[str, Any]] with each visit containing:
                    - "time": int (timestamp of event),
                "outcome_time": Optional[int],  # event time if occurred
                "censor_time": Optional[int],   # censoring time if no event
            }

    Output Format:
        List[Dict[str, Any]] where each sample contains:
            {
                "patient_id": str,
                "visit_id": str,
                "x": np.ndarray of shape (T, d),    # input sequence
                "y": np.ndarray of shape (horizon,), # hazard labels
                "mask": np.ndarray of shape (horizon,), # valid steps
            }

    Returns:
        List[Dict[str, Any]]: Processed anchor-based samples.

    Note:
        This task is compatible with PyHealth EHR datasets such as MIMIC-III/IV.
        In real PyHealth datasets, x would be derived from structured EHR features
        (e.g., codes, labs). The current implementation uses minimal synthetic
        features for testing.

        set_task() integration is not yet implemented. Use this task by calling
        it directly on patient dicts.
    """

    task_name: str = "DynamicSurvivalTask"
    input_schema: Dict[str, str] = {"x": "timeseries"}
    output_schema: Dict[str, str] = {"y": "multilabel", "mask": "tensor"}

    def __init__(
        self,
        horizon: int = 24,
        observation_window: int = 24,
        anchor_interval: int = 12,
        anchor_strategy: str = "fixed",
    ):
        super().__init__()

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if observation_window <= 0:
            raise ValueError("observation_window must be positive")
        if anchor_interval <= 0:
            raise ValueError("anchor_interval must be positive")

        self.horizon = horizon
        self.observation_window = observation_window
        self.anchor_interval = anchor_interval
        self.anchor_strategy = anchor_strategy

    def __call__(self, patient: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.process_patient(patient)

    def generate_anchors(
        self,
        event_times: List[int],
        outcome_time: Optional[int],
        censor_time: Optional[int] = None,
    ) -> List[int]:
        """
        Generates anchor times for a patient trajectory.

        Anchors define time points at which predictions are made.
        They must occur after enough observation history is available.

        Args:
            event_times (List[int]): List of visit timestamps.
            outcome_time (Optional[int]): Time of event if it occurred.
            censor_time (Optional[int]): Time of censoring if no event occurred.

        Returns:
            List[int]: Valid anchor timestamps.
        """
        # Determine maximum timeline boundary
        if outcome_time is not None:
            max_time = outcome_time
        elif censor_time is not None:
            max_time = censor_time
        elif len(event_times) > 0:
            max_time = max(event_times)
        else:
            return []

        # If no visit history, cannot construct observation windows
        if len(event_times) == 0:
            return []

        # Ensure anchors start only after sufficient observation window
        min_time = min(event_times)
        start_time = min_time + self.observation_window

        if start_time >= max_time:
            # Not enough room for multiple anchors → fallback to single anchor
            if self.anchor_strategy == "single":
                return [max_time]
            return []

        if self.anchor_strategy == "fixed":
            anchors = list(range(start_time, max_time, self.anchor_interval))

            # Safety: ensure at least one anchor exists
            if len(anchors) == 0:
                anchors = [max_time]

            return anchors

        elif self.anchor_strategy == "single":
            return [max_time]

        else:
            raise ValueError(f"Unknown anchor strategy: {self.anchor_strategy}")

    def generate_survival_label(
        self,
        anchor_time: int,
        event_time: Optional[int],
        censor_time: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates discrete-time hazard label and risk-set mask for a single anchor.

        For events:
            - y[delta] = 1 where delta = event_time - anchor_time
            - mask[delta+1:] = 0 (patient no longer at risk after event)

        For censoring:
            - mask[delta:] = 0 where delta = censor_time - anchor_time

        Args:
            anchor_time (int): The anchor timestamp for this sample.
            event_time (Optional[int]): Time of the event, or None if censored.
            censor_time (Optional[int]): Time of censoring, or None if event occurred.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y, mask) each of shape (horizon,).
        """
        y = np.zeros(self.horizon, dtype=float)
        mask = np.ones(self.horizon, dtype=float)

        if event_time is not None:
            delta = event_time - anchor_time

            if delta < 0:
                mask[:] = 0  # event already happened before anchor
                return y, mask

            if delta < self.horizon:
                y[delta] = 1
                mask[delta + 1:] = 0  # not at risk after event

        elif censor_time is not None:
            delta = censor_time - anchor_time

            if delta < self.horizon:
                mask[delta:] = 0  # censored beyond this point

        return y, mask

    def process_patient(self, patient: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a single patient dict into anchor-based survival samples.

        Args:
            patient (Dict[str, Any]): Patient record with "visits", optionally
                "outcome_time" and "censor_time".

        Returns:
            List[Dict[str, Any]]: One sample per valid anchor point.
        """
        if not isinstance(patient, dict):
            raise TypeError("patient must be a dictionary")

        if "visits" not in patient:
            raise ValueError("Patient must contain 'visits' field")

        if not isinstance(patient["visits"], list):
            raise TypeError("'visits' must be a list of visit records")

        samples = []

        pid = patient.get("patient_id", "unknown")
        visits = patient.get("visits", [])

        event_times = [v["time"] for v in visits]

        event_time = patient.get("outcome_time", None)
        censor_time = patient.get("censor_time", None)

        anchors = self.generate_anchors(event_times, event_time, censor_time)

        for anchor in anchors:
            observation_start = anchor - self.observation_window

            # Build temporal sequence from visits within the observation window
            x = np.array([
                [
                    v["time"] - observation_start,  # relative time
                    1.0  # dummy feature to simulate event presence
                ]
                for v in visits
                if observation_start <= v["time"] < anchor
            ], dtype=float)

            if len(x) == 0:
                continue

            y, mask = self.generate_survival_label(anchor, event_time, censor_time)

            if mask.sum() == 0:
                continue

            samples.append({
                "patient_id": pid,
                "visit_id": f"{pid}_{anchor}",
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "mask": mask.astype(np.float32),
            })

        return samples
