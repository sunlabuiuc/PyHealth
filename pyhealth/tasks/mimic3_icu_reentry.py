"""
PyHealth task for ICU Re-entry classification using the MIMIC-III dataset.

Reproducing and extending: Feature Robustness in Non-Stationary Health Records
(Nestor et al. 2019). This task predicts whether a patient will have an
unplanned return to the ICU within 7 days of their current ICU episode end,
using the first 24 hours of hourly vitals and labs as input features.

A direct transfer (ICU stay beginning within 24 hours of a prior stay) is not
considered a re-entry. Re-entry is defined at the episode level: a patient must
return to the ICU after a gap of > 24 hours and <= 168 hours (7 days) from the
end of their current episode.

Reference:
    Nestor et al. (2019). Feature Robustness in Non-Stationary Health Records:
    Caveats to Deployable Model Performance in Common Clinical Machine Learning
    Tasks. Proceedings of the 4th Machine Learning for Healthcare Conference,
    PMLR 106, 381-405.

Author:
    Jenna Reno (jlreno2@illinois.edu)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from pyhealth.tasks import BaseTask

from pyhealth.tasks.mimic3_clinical_aggregation import (
    CLINICAL_AGGREGATION_MAP,
    CLINICAL_CATEGORIES,
    apply_clinical_aggregation,
)

logger = logging.getLogger(__name__)


class ICUReEntryClassification(BaseTask):
    """
    Task for predicting 7-day unplanned ICU re-entry using MIMIC-III data.

    Given the first 24 hours of hourly vitals and lab values from a patient's
    ICU stay, this task predicts whether the patient will have an unplanned
    return to the ICU within 7 days of their current episode's end.

    Input features are expected to be pre-aggregated into either the clinical
    aggregation representation (65 expert-defined categories, see
    ``CLINICAL_AGGREGATION_MAP``) or the raw LEVEL2 representation from
    MIMIC_Extract, and forward-fill imputed prior to use.

    Direct transfers (ICU stays beginning within 24 hours of a prior stay)
    are excluded from both the cohort and the set of qualifying re-entries.
    Re-entry is evaluated at the episode level: a chain of transfer-linked
    stays is treated as a single episode, and re-entry requires a new episode
    to begin more than 24 hours and no more than 168 hours after the
    episode's end.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): Maps ``vitals_labs`` to ``"tensor"``.
        output_schema (Dict[str, str]): Maps ``reentry_7day`` to ``"binary"``.
        feature_set (str): Feature representation in use. Either
            ``"clinical"`` (65 aggregated categories) or ``"raw"`` (all
            LEVEL2 columns from MIMIC_Extract). Documentation only.
        reentry_window_hours (int): Maximum gap qualifying as re-entry.
            Default: 168 (7 days).
        transfer_threshold_hours (int): Maximum gap treated as direct
            transfer. Default: 24.

    Examples:
        Using ``from_arrays()`` with preprocessed tensors:

        >>> import numpy as np
        >>> from pyhealth.datasets import SampleDataset
        >>> task = ICUReEntryClassification(feature_set="clinical")
        >>> samples = task.from_arrays(
        ...     features=np.zeros((5, 24, 65), dtype="float32"),
        ...     labels=np.array([0, 1, 0, 0, 1]),
        ...     stay_ids=[200001, 200002, 200003, 200004, 200005],
        ... )
        >>> dataset = SampleDataset(samples, dataset_name="mimic3_reentry")
        >>> len(dataset)
        5
    """

    task_name: str = "ICUReEntryClassification"
    input_schema: Dict[str, str] = {"vitals_labs": "tensor"}
    output_schema: Dict[str, str] = {"reentry_7day": "binary"}

    def __init__(
        self,
        feature_set: str = "clinical",
        reentry_window_hours: int = 168,
        transfer_threshold_hours: int = 24,
    ):
        """
        Initializes the ICUReEntryClassification task.

        Args:
            feature_set: Feature representation to document. Either
                ``"clinical"`` or ``"raw"``. Does not affect processing logic.
                Default: ``"clinical"``.
            reentry_window_hours: Hours after episode end within which a new
                ICU episode qualifies as re-entry. Default: 168 (7 days).
            transfer_threshold_hours: Hours between stays below which the
                transition is a direct transfer (not re-entry). Default: 24.

        Raises:
            ValueError: If ``feature_set`` is not ``"clinical"`` or ``"raw"``.
            ValueError: If ``reentry_window_hours`` is not positive.
            ValueError: If ``transfer_threshold_hours`` is not positive or
                is >= ``reentry_window_hours``.
        """
        if feature_set not in ("clinical", "raw"):
            raise ValueError(
                f"feature_set must be 'clinical' or 'raw', got '{feature_set}'"
            )
        if reentry_window_hours <= 0:
            raise ValueError(
                f"reentry_window_hours must be positive, "
                f"got {reentry_window_hours}"
            )
        if transfer_threshold_hours <= 0:
            raise ValueError(
                f"transfer_threshold_hours must be positive, "
                f"got {transfer_threshold_hours}"
            )
        if transfer_threshold_hours >= reentry_window_hours:
            raise ValueError(
                f"transfer_threshold_hours ({transfer_threshold_hours}) must "
                f"be less than reentry_window_hours ({reentry_window_hours})"
            )

        self.feature_set = feature_set
        self.reentry_window_hours = reentry_window_hours
        self.transfer_threshold_hours = transfer_threshold_hours

        logger.info(
            f"ICUReEntryClassification initialized: "
            f"feature_set={feature_set}, "
            f"reentry_window={reentry_window_hours}h, "
            f"transfer_threshold={transfer_threshold_hours}h"
        )

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """
        Processes a single patient for the ICU re-entry classification task.

        Iterates over the patient's ICU stays, groups them into episodes
        (chains of stays separated by less than ``transfer_threshold_hours``),
        and labels each episode based on whether a subsequent episode begins
        within ``reentry_window_hours`` of the current episode's end.

        Args:
            patient: A PyHealth Patient object with ICU stay events accessible
                via ``patient.get_events(event_type="icustays")``. Each event
                must have ``icustay_id``, ``intime``, ``outtime``, and
                ``vitals_labs`` attributes, where ``vitals_labs`` is a
                (24, n_features) float32 numpy array of pre-aggregated,
                forward-fill imputed hourly features.

        Returns:
            List of sample dicts, one per qualifying ICU episode:

            - ``patient_id`` (str): Subject ID.
            - ``visit_id`` (str): ICUSTAY_ID of the index stay.
            - ``vitals_labs`` (np.ndarray): Shape (24, n_features).
            - ``reentry_7day`` (int): 1 if qualifying re-entry, 0 otherwise.

            Returns an empty list if the patient has no ICU stays or if no
            stays have ``vitals_labs`` data.
        """
        samples = []

        icu_stays = patient.get_events(event_type="icustays")
        if not icu_stays:
            return samples

        icu_stays = sorted(icu_stays, key=lambda s: s.intime)

        # ── Group stays into episodes ─────────────────────────────────────
        episodes = []
        current_episode = [icu_stays[0]]
        for stay in icu_stays[1:]:
            gap_hours = (
                (stay.intime - current_episode[-1].outtime).total_seconds()
                / 3600
            )
            if gap_hours < self.transfer_threshold_hours:
                current_episode.append(stay)
            else:
                episodes.append(current_episode)
                current_episode = [stay]
        episodes.append(current_episode)

        # ── Label each episode ────────────────────────────────────────────
        for ep_idx, episode in enumerate(episodes):
            index_stay  = episode[0]
            episode_end = max(s.outtime for s in episode)

            reentry_label = 0
            if ep_idx + 1 < len(episodes):
                next_start = episodes[ep_idx + 1][0].intime
                gap_hours  = (
                    (next_start - episode_end).total_seconds() / 3600
                )
                if (
                    gap_hours > self.transfer_threshold_hours
                    and gap_hours <= self.reentry_window_hours
                ):
                    reentry_label = 1

            if not hasattr(index_stay, "vitals_labs") or \
                    index_stay.vitals_labs is None:
                logger.debug(
                    f"No vitals_labs for stay {index_stay.icustay_id} "
                    f"— skipping."
                )
                continue

            samples.append({
                "patient_id":   str(patient.patient_id),
                "visit_id":     str(index_stay.icustay_id),
                "vitals_labs":  index_stay.vitals_labs,
                "reentry_7day": reentry_label,
            })

        return samples

    def from_arrays(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stay_ids: List[int],
        patient_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Constructs sample dicts from preprocessed numpy arrays.

        Use this method when working with tensors produced by
        ``apply_clinical_aggregation()`` rather than a PyHealth dataset object.
        The returned list is compatible with PyHealth's ``SampleDataset``.

        Args:
            features: Float32 array of shape (n_stays, 24, n_features)
                containing pre-aggregated, forward-fill imputed hourly
                vitals and labs. For the clinical representation, n_features
                should be 65 (matching ``CLINICAL_CATEGORIES``).
            labels: Integer array of shape (n_stays,) containing binary
                re-entry labels (0 or 1).
            stay_ids: List of ICUSTAY_ID integers, length n_stays.
            patient_ids: Optional list of SUBJECT_ID integers, length
                n_stays. If None, stay_ids are used as patient_ids.

        Returns:
            List of sample dicts, one per stay:

            - ``patient_id`` (str): Subject or stay ID.
            - ``visit_id`` (str): ICUSTAY_ID.
            - ``vitals_labs`` (np.ndarray): Shape (24, n_features).
            - ``reentry_7day`` (int): Binary re-entry label.

        Raises:
            ValueError: If array lengths are mismatched.
            ValueError: If ``features`` is not 3-dimensional or does not
                have 24 time steps.

        Examples:
            >>> import numpy as np
            >>> task = ICUReEntryClassification(feature_set="clinical")
            >>> features = np.zeros((3, 24, 65), dtype="float32")
            >>> labels   = np.array([0, 1, 0])
            >>> stay_ids = [200001, 200002, 200003]
            >>> samples  = task.from_arrays(features, labels, stay_ids)
            >>> len(samples)
            3
            >>> samples[1]["reentry_7day"]
            1
        """
        n_stays = len(stay_ids)

        if features.shape[0] != n_stays:
            raise ValueError(
                f"features has {features.shape[0]} rows but "
                f"stay_ids has {n_stays} entries."
            )
        if len(labels) != n_stays:
            raise ValueError(
                f"labels has {len(labels)} entries but "
                f"stay_ids has {n_stays} entries."
            )
        if features.ndim != 3:
            raise ValueError(
                f"features must be 3-dimensional (n_stays, 24, n_features), "
                f"got shape {features.shape}."
            )
        if features.shape[1] != 24:
            raise ValueError(
                f"features must have 24 time steps, got {features.shape[1]}."
            )
        if patient_ids is not None and len(patient_ids) != n_stays:
            raise ValueError(
                f"patient_ids has {len(patient_ids)} entries but "
                f"stay_ids has {n_stays} entries."
            )

        pid_list = patient_ids if patient_ids is not None else stay_ids
        n_pos    = int(sum(labels))

        samples = [
            {
                "patient_id":   str(pid),
                "visit_id":     str(stay_id),
                "vitals_labs":  features[i],
                "reentry_7day": int(labels[i]),
            }
            for i, (stay_id, pid) in enumerate(zip(stay_ids, pid_list))
        ]

        logger.info(
            f"from_arrays: {len(samples)} samples "
            f"(pos={n_pos}, neg={n_stays - n_pos}), "
            f"feature_set={self.feature_set}, "
            f"feature_dim={features.shape[2]}"
        )
        return samples