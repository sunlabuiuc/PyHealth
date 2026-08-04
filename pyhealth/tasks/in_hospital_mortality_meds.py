"""In-hospital mortality prediction for datasets in the Medical Event Data
Standard (MEDS).

This module provides :class:`InHospitalMortalityMEDS`, the MEDS-native
counterpart of
:class:`~pyhealth.tasks.in_hospital_mortality_mimic4.InHospitalMortalityMIMIC4`.
The MIMIC-IV task anchors on a visit object and reads
``admission.hospital_expire_flag``; MEDS represents a hospitalization as two
separate events (``HOSPITAL_ADMISSION//*`` and ``HOSPITAL_DISCHARGE//*``)
that share a ``hadm_id``, so a stay is derived by grouping those events
on that identifier and the label is derived from the discharge code.

Task definition
---------------
Let a *stay* be the set of events sharing one ``hadm_id`` for a subject,
with admission time ``t_a`` (earliest admission event) and discharge time
``t_d`` (latest discharge event). For each completed stay
(``t_d > t_a``) the task produces one sample:

* **Prediction time** ``t_p``.
  ``observation_window="full_stay"`` (default) sets ``t_p = t_d``.
  ``observation_window="first_hours"`` sets ``t_p = t_a + window_hours`` and
  keeps only stays with length of stay strictly greater than
  ``window_hours``, so the window is fully observed and the outcome is
  strictly future.
* **Features.** The ordered sequence of MEDS ``code`` values in the
  half-open interval ``[t_a, t_p)``, excluding every ``HOSPITAL_DISCHARGE//*``
  event and every ``MEDS_DEATH`` event. Both exclusions matter: the half-open
  bound already removes the discharge event when ``t_p = t_d``, and dropping
  ``MEDS_DEATH`` removes the canonical death sentinel (which the demo places a
  few hours after discharge) so the outcome can never leak into the input.
* **Label.** ``mortality = 1`` iff the stay's discharge code is
  ``HOSPITAL_DISCHARGE//DIED``. This is the in-hospital, same-stay
  definition, consistent with ``hospital_expire_flag`` upstream. On the
  public MIMIC-IV demo in MEDS it is a strict superset of ``MEDS_DEATH``
  occurring within a stay: ``MEDS_DEATH`` carries a null ``hadm_id`` there,
  so it cannot be attached to a stay and is deliberately not used as the
  label. Deaths outside the index stay are a subject-level problem and are
  out of scope for this task.

Configuration
-------------
The task reads ``hadm_id``, which is **not** part of the core MEDS schema
(``subject_id``/``time``/``code``/``numeric_value``/``text_value``) but is
present in MIMIC-derived MEDS datasets. It is therefore kept out of the
default ``configs/meds.yaml`` (selecting an absent attribute would raise for
generic MEDS data). A bundled ``configs/meds_with_hadm.yaml`` exposes it;
pass that config (or your own that lists ``hadm_id``) when using this task.

Scope note
----------
The MIMIC-IV task additionally drops pediatric admissions via
``anchor_age``. A MEDS-native age filter is derivable from ``MEDS_BIRTH`` but
is intentionally omitted here: its on-disk representation is not fixed across
MEDS datasets, and silently assuming one would be unsound. Age restriction is
therefore left to a preprocessing step or a future, explicitly parameterized
extension.

References:
    MEDS Working Group. Medical Event Data Standard (MEDS): Facilitating
    Machine Learning for Health. ICLR 2024 Workshop on Learning from Time
    Series For Health. https://openreview.net/forum?id=IsHy2ebjIG
"""

from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask

ADMISSION_PREFIX = "HOSPITAL_ADMISSION"
DISCHARGE_PREFIX = "HOSPITAL_DISCHARGE"
DIED_CODE = "HOSPITAL_DISCHARGE//DIED"
DEATH_CODE = "MEDS_DEATH"

_FULL_STAY = "full_stay"
_FIRST_HOURS = "first_hours"
_VALID_WINDOWS = (_FULL_STAY, _FIRST_HOURS)


class InHospitalMortalityMEDS(BaseTask):
    """In-hospital mortality prediction for MEDS datasets.

    One sample per completed hospital stay. The observation window is the
    half-open interval ``[admission, prediction_time)`` and the binary label
    is whether the stay ended in death (discharge code
    ``HOSPITAL_DISCHARGE//DIED``). MEDS codes observed during the window,
    excluding the terminating discharge event and any ``MEDS_DEATH``, form
    the input sequence. See the module docstring for the full definition.

    Args:
        observation_window (str): ``"full_stay"`` (default) observes the
            entire stay, i.e. ``[admission, discharge)``. ``"first_hours"``
            observes only ``[admission, admission + window_hours)`` and keeps
            stays whose length exceeds ``window_hours`` (an early-warning
            setup with a strictly future outcome).
        window_hours (float): Observation length used when
            ``observation_window="first_hours"``. Ignored for ``"full_stay"``.
            Defaults to ``48.0``, matching ``InHospitalMortalityMIMIC4``.
        code_mapping (Optional[Dict[str, Tuple[str, str]]]): Optional vocab
            mapping forwarded to :class:`BaseTask` (e.g.
            ``{"codes": ("ICD10CM", "CCSCM")}``).

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``codes`` — the sequence of MEDS
            codes observed during the window.
        output_schema (Dict[str, str]): ``mortality`` — binary in-hospital
            mortality.

    Raises:
        ValueError: If ``observation_window`` is not one of
            ``"full_stay"``/``"first_hours"``, or if ``window_hours`` is not
            positive.

    Examples:
        >>> from pathlib import Path
        >>> import pyhealth.datasets.configs as meds_configs
        >>> from pyhealth.datasets import MEDSDataset
        >>> from pyhealth.tasks import InHospitalMortalityMEDS
        >>> # A bundled stay-aware config exposes hadm_id (not a core MEDS
        >>> # field, so it is kept out of the default configs/meds.yaml):
        >>> cfg = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
        >>> dataset = MEDSDataset(
        ...     root="/path/to/mimic-iv-demo-meds/0.0.1",
        ...     config_path=str(cfg),
        ... )
        >>> samples = dataset.set_task(InHospitalMortalityMEDS())
        >>> # Early-warning variant: first 48h, stays longer than 48h only
        >>> early = InHospitalMortalityMEDS(observation_window="first_hours")
    """

    task_name: str = "InHospitalMortalityMEDS"
    input_schema: ClassVar[dict[str, str]] = {"codes": "sequence"}
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    def __init__(
        self,
        observation_window: str = _FULL_STAY,
        window_hours: float = 48.0,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if observation_window not in _VALID_WINDOWS:
            raise ValueError(
                f"observation_window must be one of {_VALID_WINDOWS}, "
                f"got {observation_window!r}."
            )
        if window_hours <= 0:
            raise ValueError(f"window_hours must be positive, got {window_hours}.")
        super().__init__(code_mapping=code_mapping)
        self.observation_window = observation_window
        self.window_hours = float(window_hours)

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the global scan to MEDS events before per-patient calls.

        All MEDS data lives in a single ``meds`` event type, so this narrows
        the frame once rather than per patient.
        """
        return df.filter(pl.col("event_type") == "meds")

    def _group_stays(self, events: pl.DataFrame) -> pl.DataFrame:
        """Builds one row per stay from admission/discharge events.

        Args:
            events (pl.DataFrame): This patient's MEDS events, with an
                integer ``_hadm`` column already attached.

        Returns:
            pl.DataFrame: Columns ``_hadm``, ``admit``, ``discharge``,
            ``discharge_code``, one row per ``hadm_id`` that has both an
            admission and a discharge. Malformed duplicates collapse via
            earliest-admission / latest-discharge aggregation.
        """
        code = pl.col("meds/code")
        admissions = (
            events.filter(code.str.starts_with(ADMISSION_PREFIX))
            .filter(pl.col("_hadm").is_not_null())
            .group_by("_hadm")
            .agg(pl.col("timestamp").min().alias("admit"))
        )
        discharges = (
            events.filter(code.str.starts_with(DISCHARGE_PREFIX))
            .filter(pl.col("_hadm").is_not_null())
            .group_by("_hadm")
            .agg(
                pl.col("timestamp").max().alias("discharge"),
                code.sort_by("timestamp").last().alias("discharge_code"),
            )
        )
        return admissions.join(discharges, on="_hadm", how="inner")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        events = patient.get_events(event_type="meds", return_df=True)
        if events.height == 0:
            return []

        # A nullable integer id is promoted to float through the Dask/pandas
        # pipeline whenever the column carries nulls (e.g. lab events and the
        # MEDS_DEATH sentinel). Cast back to a nullable integer so stays join
        # cleanly and emitted ids stay integral rather than "555.0".
        events = events.with_columns(
            pl.col("meds/hadm_id").cast(pl.Int64, strict=False).alias("_hadm")
        )
        code = pl.col("meds/code")

        stays = self._group_stays(events)
        if stays.height == 0:
            return []

        samples: list[dict[str, Any]] = []
        for stay in stays.sort("admit").iter_rows(named=True):
            admit, discharge = stay["admit"], stay["discharge"]
            if discharge <= admit:
                continue  # degenerate/zero-length stay

            if self.observation_window == _FIRST_HOURS:
                duration_hours = (discharge - admit).total_seconds() / 3600.0
                if duration_hours <= self.window_hours:
                    continue  # window not fully observed within this stay
                predict_time = admit + _timedelta_hours(self.window_hours)
            else:
                predict_time = discharge

            window = events.filter(
                (pl.col("timestamp") >= admit)
                & (pl.col("timestamp") < predict_time)  # half-open: excludes t_p
                & (~code.str.starts_with(DISCHARGE_PREFIX))
                & (code != DEATH_CODE)
            ).sort("timestamp")

            codes = window["meds/code"].to_list()
            if not codes:
                continue  # no observable signal before the prediction time

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "hadm_id": stay["_hadm"],
                    "codes": codes,
                    "mortality": int(stay["discharge_code"] == DIED_CODE),
                }
            )

        return samples


def _timedelta_hours(hours: float):
    """Returns a ``datetime.timedelta`` of ``hours`` (kept import-local)."""
    from datetime import timedelta

    return timedelta(hours=hours)
