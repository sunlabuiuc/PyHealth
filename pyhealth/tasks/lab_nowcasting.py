"""Intra-visit laboratory event nowcasting (the TRACE task) for MIMIC-III/IV.

Paper:
    Yuyang Liang, Yankai Chen, Yixiang Fang, Laks V. S. Lakshmanan and
    Chenhao Ma. "TRACE: Intra-visit Clinical Event Nowcasting via Effective
    Patient Trajectory Encoding." Companion Proceedings of the ACM Web
    Conference 2025 (WWW '25 Companion), short paper.
    https://doi.org/10.1145/3701716.3715545, https://arxiv.org/abs/2503.23072

The task follows Section 2 of the paper: a hospital admission is a
time-ordered trajectory of events (laboratory results with their flag,
medications, procedures); the target is the set of laboratory events drawn
together at the next laboratory time, given everything earlier in the
admission. The authors' code (https://github.com/Amehi/TRACE, no license)
was used only to confirm details such as the 100-event window.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from datetime import datetime
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask

logger = logging.getLogger(__name__)


def lab_flag(
    valuenum: Any = None,
    ref_range_lower: Any = None,
    ref_range_upper: Any = None,
    flag: Any = None,
) -> str:
    """Flags one laboratory result as high, low, normal or abnormal.

    With a numeric value and a reference range (MIMIC-IV), the flag is
    ``"H"`` above the upper bound, ``"L"`` below the lower bound and ``"N"``
    inside. Without them (MIMIC-III, or non-numeric results) it falls back
    to the table's ``flag`` column: ``"A"`` when it is ``"abnormal"`` and
    ``"N"`` otherwise. This is the same rule the task applies to whole
    tables.

    Args:
        valuenum: The numeric result, or None.
        ref_range_lower: Lower bound of the reference range, or None.
        ref_range_upper: Upper bound of the reference range, or None.
        flag: The source ``flag`` column (e.g. ``"abnormal"``), or None.

    Returns:
        One of ``"H"``, ``"L"``, ``"N"`` and ``"A"``.

    Examples:
        >>> lab_flag(6.2, 3.5, 5.1)
        'H'
        >>> lab_flag(4.0, 3.5, 5.1)
        'N'
        >>> lab_flag(None, None, None, "abnormal")
        'A'
    """

    def number(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(out) else out

    value, lower, upper = map(number, (valuenum, ref_range_lower, ref_range_upper))
    if value is not None and (lower is not None or upper is not None):
        if lower is not None and value < lower:
            return "L"
        if upper is not None and value > upper:
            return "H"
        return "N"
    return "A" if flag == "abnormal" else "N"


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _number(column: str) -> pl.Expr:
    # NaN counts as missing, as in lab_flag
    return pl.col(column).cast(pl.Float64, strict=False).fill_nan(None)


class LabNowcasting(BaseTask):
    """Nowcast the laboratory events of the next blood draw in an admission.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` and
    :class:`~pyhealth.datasets.MIMIC3Dataset`.

    For each hospital admission, events from ``event_tables`` that fall
    between admission and discharge form a trajectory of ``(code, time)``
    pairs. Laboratory events are coded with their result flag,
    ``"LAB:<itemid>:<flag>"`` (see :func:`lab_flag`), so a code says both
    which test was run and whether the result was high, low, normal or
    abnormal; other tables are coded ``"<TABLE>:<code>"`` (e.g.
    ``"RX:<drug>"``). A *draw* is a laboratory time: all laboratory events
    with the same timestamp. For a draw at time ``t`` the sample is:

    - ``events``: the codes of the events before ``t`` in the admission,
      oldest first, at most the last ``max_events`` of them;
    - ``event_times``: for each of those events, the hours from the event to
      ``t`` (so the most recent events have the smallest times);
    - ``labs``: the set of laboratory codes of the draw at ``t``, the
      multi-label target.

    With ``target="last"`` (the paper's setup) each admission gives one
    sample, its last draw; ``target="all"`` gives one sample per draw.
    A draw needs at least ``min_history`` earlier events.

    The task subclasses only fix defaults: :class:`LabNowcastingMIMIC4` and
    :class:`LabNowcastingMIMIC3`.

    Args:
        event_tables: Tables whose events make up the trajectory; they must
            be keys of ``CODE_COLUMNS`` and should be loaded in the dataset
            (``set_task`` logs a warning for any that is not, since it would
            add no events). ``"labevents"`` is required (it supplies the
            targets).
        max_events: Number of most recent events kept (99 in the paper's
            code, which appends a mask token to make 100). None keeps all.
        min_history: Minimum number of earlier events for a draw to be a
            sample.
        target: ``"last"`` or ``"all"`` draws of each admission.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import LabNowcastingMIMIC4
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "prescriptions"],
        ... )
        >>> samples = dataset.set_task(LabNowcastingMIMIC4())  # doctest: +SKIP
    """

    task_name: str = "LabNowcasting"
    input_schema: ClassVar[dict[str, str]] = {
        "events": "sequence",
        "event_times": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {"labs": "multilabel"}

    #: Table -> (code prefix, attribute that holds the code).
    CODE_COLUMNS: ClassVar[dict[str, tuple[str, str]]] = {
        "labevents": ("LAB", "itemid"),
        "prescriptions": ("RX", "drug"),
        "procedureevents": ("PROC", "itemid"),
        "inputevents": ("INPUT", "itemid"),
        "outputevents": ("OUTPUT", "itemid"),
        "chartevents": ("CHART", "itemid"),
        "microbiologyevents": ("MICRO", "test_name"),
    }
    DEFAULT_TABLES: ClassVar[tuple[str, ...]] = ("labevents",)

    def __init__(
        self,
        event_tables: tuple[str, ...] | list[str] | None = None,
        max_events: int | None = 99,
        min_history: int = 1,
        target: str = "last",
    ) -> None:
        tables = tuple(event_tables or self.DEFAULT_TABLES)
        if "labevents" not in tables:
            raise ValueError("event_tables must include 'labevents'.")
        unknown = [t for t in tables if t not in self.CODE_COLUMNS]
        if unknown:
            raise ValueError(
                f"No code column known for {unknown}; add them to CODE_COLUMNS."
            )
        if target not in ("last", "all"):
            raise ValueError("target must be 'last' or 'all'.")
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be positive or None.")
        self.event_tables = tables
        self.max_events = max_events
        self.min_history = max(int(min_history), 0)
        self.target = target

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Warns once if an ``event_tables`` entry is not loaded in the dataset.

        A table that is not loaded has no columns in the dataset's event
        frame, and each patient would silently get no events from it.
        ``BaseDataset.set_task`` calls this once before running the task.

        Args:
            df: The dataset's global event frame.

        Returns:
            ``df`` unchanged.

        Examples:
            >>> import polars as pl
            >>> frame = pl.LazyFrame({"labevents/itemid": ["1"]})
            >>> task = LabNowcastingMIMIC4()  # labevents and prescriptions
            >>> task.pre_filter(frame).collect().height  # warns: prescriptions
            1
        """
        columns = df.collect_schema().names()
        missing = [
            table
            for table in self.event_tables
            if not any(c.startswith(f"{table}/") for c in columns)
        ]
        if missing:
            logger.warning(
                "%s: event table(s) %s are not loaded in the dataset, so they "
                "contribute no events; add them to the dataset's tables.",
                self.task_name,
                missing,
            )
        return df

    def _codes(self, df: pl.DataFrame, table: str) -> pl.DataFrame:
        prefix, column = self.CODE_COLUMNS[table]
        code = pl.col(f"{table}/{column}").cast(pl.String)
        if table == "labevents":
            flag = self._flag_expr(df)
            token = pl.concat_str([pl.lit("LAB:"), code, pl.lit(":"), flag])
        else:
            token = pl.concat_str([pl.lit(f"{prefix}:"), code])
        return df.filter(code.is_not_null()).select(
            pl.col("timestamp"),
            token.alias("code"),
            pl.lit(table == "labevents").alias("is_lab"),
        )

    @staticmethod
    def _flag_expr(df: pl.DataFrame) -> pl.Expr:
        """The vectorized :func:`lab_flag` for a ``labevents`` frame."""

        def maybe(name: str) -> pl.Expr:
            column = f"labevents/{name}"
            return _number(column) if column in df.columns else pl.lit(None)

        value = maybe("valuenum")
        lower, upper = maybe("ref_range_lower"), maybe("ref_range_upper")
        abnormal = (
            pl.col("labevents/flag") == "abnormal"
            if "labevents/flag" in df.columns
            else pl.lit(False)
        )
        ranged = value.is_not_null() & (lower.is_not_null() | upper.is_not_null())
        return (
            pl.when(ranged & lower.is_not_null() & (value < lower))
            .then(pl.lit("L"))
            .when(ranged & upper.is_not_null() & (value > upper))
            .then(pl.lit("H"))
            .when(ranged)
            .then(pl.lit("N"))
            .when(abnormal.fill_null(False))
            .then(pl.lit("A"))
            .otherwise(pl.lit("N"))
        )

    def _trajectory(
        self, patient: Any, start: datetime, end: datetime
    ) -> pl.DataFrame:
        frames = []
        for table in self.event_tables:
            df = patient.get_events(
                event_type=table, start=start, end=end, return_df=True
            )
            if df.height:
                frames.append(self._codes(df, table))
        if not frames:
            return pl.DataFrame(
                schema={"timestamp": pl.Datetime, "code": pl.String,
                        "is_lab": pl.Boolean}
            )
        # Patient events are not stably ordered within a timestamp; sort by
        # code too so the same data always gives the same sequence.
        return pl.concat(frames, how="vertical_relaxed").sort(["timestamp", "code"])

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = []
        for admission in patient.get_events(event_type="admissions"):
            start, end = admission.timestamp, _parse_time(admission.dischtime)
            if start is None or end is None:
                continue
            events = self._trajectory(patient, start, end)
            times = events["timestamp"].to_list()
            codes = events["code"].to_list()
            draw_labs: dict[datetime, set[str]] = {}
            for t, code, is_lab in zip(times, codes, events["is_lab"].to_list()):
                if is_lab:
                    draw_labs.setdefault(t, set()).add(code)
            draws = sorted(draw_labs)
            if self.target == "last":
                draws = draws[-1:]
            for draw in draws:
                n_before = bisect_left(times, draw)
                if n_before < max(self.min_history, 1):
                    continue
                first = 0
                if self.max_events is not None:
                    first = max(0, n_before - self.max_events)
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": str(admission.hadm_id),
                        "events": codes[first:n_before],
                        "event_times": [
                            (draw - t).total_seconds() / 3600.0
                            for t in times[first:n_before]
                        ],
                        "labs": sorted(draw_labs[draw]),
                    }
                )
        return samples


class LabNowcastingMIMIC4(LabNowcasting):
    """:class:`LabNowcasting` on MIMIC-IV: laboratory results and medications.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset`.

    Laboratory codes carry a high/low/normal flag from the numeric result
    and its reference range (``"A"`` when only the ``flag`` column marks a
    result abnormal), as in the paper's MIMIC-IV setup. Medications come
    from ``prescriptions``; add ``"procedureevents"`` (ICU procedures) to
    ``event_tables`` for timed procedures (``procedures_icd`` has only a
    discharge time in PyHealth's config, so it cannot be ordered within an
    admission).

    Args:
        event_tables: Defaults to ``("labevents", "prescriptions")``.
        max_events: See :class:`LabNowcasting`.
        min_history: See :class:`LabNowcasting`.
        target: See :class:`LabNowcasting`.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import LabNowcastingMIMIC4
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "prescriptions"],
        ... )
        >>> samples = dataset.set_task(LabNowcastingMIMIC4())  # doctest: +SKIP
        >>> samples[0]["labs"]  # doctest: +SKIP
    """

    task_name: str = "LabNowcastingMIMIC4"
    DEFAULT_TABLES: ClassVar[tuple[str, ...]] = ("labevents", "prescriptions")


class LabNowcastingMIMIC3(LabNowcasting):
    """:class:`LabNowcasting` on MIMIC-III: laboratory results only.

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset`.

    MIMIC-III's ``LABEVENTS`` has no reference range, so laboratory codes
    carry a binary flag, ``"A"`` (abnormal) or ``"N"``, as in the paper's
    MIMIC-III setup. Add ``"prescriptions"`` to ``event_tables`` to include
    medications (MIMIC-III prescriptions have a start date, not a time).

    Args:
        event_tables: Defaults to ``("labevents",)``.
        max_events: See :class:`LabNowcasting`.
        min_history: See :class:`LabNowcasting`.
        target: See :class:`LabNowcasting`.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import LabNowcastingMIMIC3
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iii/1.4", tables=["labevents"]
        ... )
        >>> samples = dataset.set_task(LabNowcastingMIMIC3())  # doctest: +SKIP
    """

    task_name: str = "LabNowcastingMIMIC3"
    DEFAULT_TABLES: ClassVar[tuple[str, ...]] = ("labevents",)
