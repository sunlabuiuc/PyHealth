# Author: Anish Gupta
# NetID: anishg8
# Paper Title: N/A (original task contribution, not a paper reproduction)
# Paper Link: N/A
# Description: Sepsis prediction task for MIMIC-IV. Labels each hospital
#     admission for sepsis onset using a qSOFA-based approximation of the
#     Sepsis-3 clinical criteria (Seymour et al., "Assessment of Clinical
#     Criteria for Sepsis," JAMA 2016, https://doi.org/10.1001/jama.2016.0288),
#     built on the labs/vitals/prescriptions tables PyHealth's MIMIC-IV
#     loader already supports (plus a new `chartevents` table added to
#     mimic4_ehr.yaml for vitals).

from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask


class SepsisPredictionMIMIC4(BaseTask):
    """Task for predicting sepsis onset during a MIMIC-IV hospital admission.

    Each admission is labeled using a Sepsis-3-style two-signal definition:
    a *suspected infection* signal (a new antibiotic order) co-occurring
    with an *organ-dysfunction* signal (qSOFA >= 2).

    qSOFA ("quick SOFA") is the bedside, vitals-only simplification of the
    full Sepsis-3 SOFA score, from the same consensus definitions (Seymour
    et al., "Assessment of Clinical Criteria for Sepsis," JAMA 2016). It
    flags a patient when at least 2 of the following 3 criteria are met:
    respiratory rate >= 22/min, systolic blood pressure <= 100 mmHg, and
    altered mental status (Glasgow Coma Scale < 15). Full SOFA additionally
    requires vasopressor dosing and urine output, which are not currently
    loaded by PyHealth's MIMIC-IV configuration, so qSOFA is used here as a
    documented, citable simplification rather than full Sepsis-3.

    Limitations:
        - qSOFA is a simplified organ-dysfunction proxy, not the full SOFA
          score; it will disagree with a full Sepsis-3 label in some cases.
        - "Suspected infection" is approximated by a curated antibiotic
          drug-name whitelist (``ANTIBIOTIC_DRUG_NAMES``) matched against
          the free-text ``prescriptions.drug`` field. This is a heuristic:
          it will miss antibiotics not on the list and may match ambiguous
          names.
        - Onset is the earliest timestamp where both signals co-occur
          within ``ANTIBIOTIC_WINDOW_HOURS`` of each other. Features are
          censored strictly before that timestamp so the triggering
          observation itself is never part of the model's input.
        - The ``chartevents``/``labevents`` item IDs below are the standard
          MIMIC-IV IDs for these measurements but should be re-verified
          against a real ``icu/d_items.csv.gz`` / ``hosp/d_labitems.csv.gz``
          before use on production data.

    Attributes:
        task_name: The name of the task.
        input_schema: ``{"observations": "timeseries"}`` -- a combined
            labs + vitals timeseries, censored before any sepsis onset.
        output_schema: ``{"sepsis": "binary"}``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import SepsisPredictionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["admissions", "prescriptions", "labevents", "chartevents"],
        ... )
        >>> task = SepsisPredictionMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "SepsisPredictionMIMIC4"
    input_schema: ClassVar[dict[str, str]] = {"observations": "timeseries"}
    output_schema: ClassVar[dict[str, str]] = {"sepsis": "binary"}

    RESP_RATE_ITEMID: ClassVar[str] = "220210"
    SBP_ITEMIDS: ClassVar[list[str]] = ["220179", "220050"]
    GCS_ITEMIDS: ClassVar[list[str]] = ["220739", "223900", "223901"]
    VITAL_ITEMIDS: ClassVar[list[str]] = [RESP_RATE_ITEMID] + SBP_ITEMIDS + GCS_ITEMIDS

    LAB_ITEMIDS: ClassVar[list[str]] = [
        "50813",  # Lactate
        "50912",  # Creatinine
        "51265",  # Platelet Count
        "50885",  # Bilirubin, Total
        "51301",  # White Blood Cells
        "50882",  # Bicarbonate
    ]

    OBSERVATION_ITEMIDS: ClassVar[list[str]] = LAB_ITEMIDS + VITAL_ITEMIDS

    ANTIBIOTIC_DRUG_NAMES: ClassVar[list[str]] = [
        "vancomycin",
        "cefepime",
        "piperacillin",
        "zosyn",
        "meropenem",
        "ceftriaxone",
        "levofloxacin",
        "ciprofloxacin",
        "metronidazole",
        "azithromycin",
        "ampicillin",
        "gentamicin",
        "clindamycin",
        "daptomycin",
        "linezolid",
        "imipenem",
    ]

    ANTIBIOTIC_WINDOW_HOURS: ClassVar[float] = 24.0
    QSOFA_THRESHOLD: ClassVar[int] = 2
    GCS_NORMAL: ClassVar[int] = 15

    def _pivot(
        self, df: pl.DataFrame, table: str, item_ids: list[str]
    ) -> pl.DataFrame:
        """Pivot a filtered events frame to a wide timeseries matrix.

        Matches the pivot pattern used by ``InHospitalMortalityMIMIC4``:
        one row per timestamp, one column per item ID, with missing item
        IDs added as all-null columns so every returned frame has the same
        shape regardless of which items were actually observed.

        Args:
            df: A ``return_df=True`` events frame for a single table,
                already time/``hadm_id``-filtered.
            table: The source event type (e.g. ``"labevents"``), used to
                resolve the ``{table}/itemid`` and ``{table}/valuenum``
                column names.
            item_ids: The item IDs to keep as columns, in output order.

        Returns:
            A DataFrame with a ``timestamp`` column plus one float column
            per entry in ``item_ids``.

        Example:
            Called from ``__call__`` to turn a raw ``chartevents`` or
            ``labevents`` slice into the wide matrix ``_qsofa_onset`` and
            the final ``observations`` tensor both expect, e.g.
            ``self._pivot(vitals_df, "chartevents", self.VITAL_ITEMIDS)``.
        """
        empty = pl.DataFrame({"timestamp": []}).cast({"timestamp": pl.Datetime})
        if df.height == 0:
            return empty.with_columns([pl.lit(None).alias(i) for i in item_ids])

        df = df.filter(pl.col(f"{table}/itemid").is_in(item_ids))
        if df.height == 0:
            return empty.with_columns([pl.lit(None).alias(i) for i in item_ids])

        df = df.select(
            pl.col("timestamp"),
            pl.col(f"{table}/itemid"),
            pl.col(f"{table}/valuenum").cast(pl.Float64),
        )
        df = df.pivot(
            index="timestamp",
            on=f"{table}/itemid",
            values=f"{table}/valuenum",
            aggregate_function="first",
        )
        missing = [i for i in item_ids if i not in df.columns]
        for col in missing:
            df = df.with_columns(pl.lit(None).alias(col))
        return df.select("timestamp", *item_ids)

    def _qsofa_onset(self, vitals: pl.DataFrame) -> datetime | None:
        """Find the earliest timestamp where qSOFA crosses the threshold.

        At each vitals timestamp, scores 1 point each for respiratory
        rate >= 22, systolic BP <= 100 (the minimum across whichever BP
        item IDs are present at that timestamp), and altered mental status
        (summed GCS eye/verbal/motor < ``GCS_NORMAL``, scored only when
        all three components are present).

        Args:
            vitals: The pivoted vitals frame returned by ``_pivot``, with
                one row per timestamp and one column per vital item ID.

        Returns:
            The first timestamp where the qSOFA point total reaches
            ``QSOFA_THRESHOLD``, or ``None`` if it never does.

        Example:
            Called from ``__call__`` on the pivoted full-admission vitals
            to find the onset timestamp used both as the sepsis-label
            trigger and as the feature-censoring cutoff, e.g.
            ``self._qsofa_onset(full_vitals)``.
        """
        vitals = vitals.sort("timestamp")
        for row in vitals.iter_rows(named=True):
            rr = row[self.RESP_RATE_ITEMID]
            sbp_candidates = [
                row[i] for i in self.SBP_ITEMIDS if row[i] is not None
            ]
            sbp = min(sbp_candidates) if sbp_candidates else None
            gcs_components = [
                row[i] for i in self.GCS_ITEMIDS if row[i] is not None
            ]
            gcs_total = sum(gcs_components) if len(gcs_components) == 3 else None

            points = 0
            if rr is not None and rr >= 22:
                points += 1
            if sbp is not None and sbp <= 100:
                points += 1
            if gcs_total is not None and gcs_total < self.GCS_NORMAL:
                points += 1

            if points >= self.QSOFA_THRESHOLD:
                return row["timestamp"]
        return None

    def _antibiotic_times(
        self, patient: Any, hadm_id: Any, start: datetime, end: datetime
    ) -> list[datetime]:
        """Return timestamps of antibiotic orders during an admission.

        Filters the admission's prescriptions to rows whose free-text
        ``drug`` name contains any entry in ``ANTIBIOTIC_DRUG_NAMES``
        (case-insensitive substring match) -- see the class docstring's
        Limitations section for why this is a heuristic, not a guarantee.

        Args:
            patient: The patient to query.
            hadm_id: The admission ID to restrict prescriptions to.
            start: Window start (inclusive).
            end: Window end (inclusive).

        Returns:
            Timestamps of matching antibiotic orders, in no particular
            order.

        Example:
            Called from ``__call__`` once a qSOFA onset time is found, to
            check whether an antibiotic order exists nearby, e.g.
            ``self._antibiotic_times(patient, admission.hadm_id,
            admission.timestamp, dischtime)``.
        """
        prescriptions = patient.get_events(
            event_type="prescriptions",
            start=start,
            end=end,
            filters=[("hadm_id", "==", hadm_id)],
            return_df=True,
        )
        if prescriptions.height == 0:
            return []
        pattern = "|".join(self.ANTIBIOTIC_DRUG_NAMES)
        prescriptions = prescriptions.filter(
            pl.col("prescriptions/drug").str.to_lowercase().str.contains(pattern)
        )
        return prescriptions["timestamp"].to_list()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sepsis-prediction sample per completed admission.

        For each admission: determines the earliest qSOFA-positive
        timestamp (if any) across the full admission window, then checks
        for an antibiotic order within ``ANTIBIOTIC_WINDOW_HOURS`` of it.
        If both signals are found, the admission is labeled ``sepsis=1``
        with features censored strictly before that onset time. Otherwise
        it is labeled ``sepsis=0`` using the full admission window as
        features. Admissions with a missing/unparseable discharge time, or
        with no observations remaining after censoring, are skipped.

        Args:
            patient: The patient whose admissions to process.

        Returns:
            A list of sample dicts, each with ``patient_id``,
            ``admission_id``, ``observations`` (a ``(timestamps, values)``
            tuple matching the ``"timeseries"`` processor's input
            contract), and ``sepsis``.

        Example:
            Never called directly by users -- invoked once per patient by
            ``BaseDataset.set_task()``, e.g.
            ``dataset.set_task(SepsisPredictionMIMIC4())``. See
            ``tests/core/test_sepsis_prediction_mimic4.py`` for direct
            usage against synthetic ``Patient`` objects, and the class
            docstring above for the ``set_task`` example.
        """
        samples: list[dict[str, Any]] = []
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            try:
                # MIMIC-IV timestamps are naive by design (de-identified,
                # date-shifted, no timezone).
                dischtime = datetime.strptime(  # noqa: DTZ007
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (TypeError, ValueError):
                continue
            if dischtime <= admission.timestamp:
                continue

            full_vitals_df = patient.get_events(
                event_type="chartevents",
                start=admission.timestamp,
                end=dischtime,
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            full_vitals = self._pivot(full_vitals_df, "chartevents", self.VITAL_ITEMIDS)
            onset_time = (
                self._qsofa_onset(full_vitals) if full_vitals.height > 0 else None
            )

            sepsis_label = 0
            cutoff = dischtime
            if onset_time is not None:
                antibiotic_times = self._antibiotic_times(
                    patient, admission.hadm_id, admission.timestamp, dischtime
                )
                window = timedelta(hours=self.ANTIBIOTIC_WINDOW_HOURS)
                if any(abs(onset_time - t) <= window for t in antibiotic_times):
                    sepsis_label = 1
                    cutoff = onset_time

            if cutoff <= admission.timestamp:
                continue

            # get_events(end=...) is inclusive, so the observation exactly at
            # `cutoff` (the qSOFA-triggering row, when sepsis_label == 1)
            # must be dropped explicitly -- otherwise the label-defining
            # observation itself would leak into the model's input.
            labs_df = patient.get_events(
                event_type="labevents",
                start=admission.timestamp,
                end=cutoff,
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            if labs_df.height > 0:
                labs_df = labs_df.filter(pl.col("timestamp") < cutoff)
            vitals_df = patient.get_events(
                event_type="chartevents",
                start=admission.timestamp,
                end=cutoff,
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            if vitals_df.height > 0:
                vitals_df = vitals_df.filter(pl.col("timestamp") < cutoff)
            labs = self._pivot(labs_df, "labevents", self.LAB_ITEMIDS)
            vitals = self._pivot(vitals_df, "chartevents", self.VITAL_ITEMIDS)

            observations = labs.join(vitals, on="timestamp", how="full", coalesce=True)
            if observations.height == 0:
                continue
            observations = observations.sort("timestamp")
            observations = observations.select("timestamp", *self.OBSERVATION_ITEMIDS)

            timestamps = observations["timestamp"].to_list()
            values = observations.drop("timestamp").to_numpy()

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "admission_id": admission.hadm_id,
                    "observations": (timestamps, values),
                    "sepsis": sepsis_label,
                }
            )

        return samples
