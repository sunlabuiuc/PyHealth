# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
"""Rotterdam breast-cancer cohort with separate recurrence and death outcomes."""

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_dataset import BaseDataset

if TYPE_CHECKING:
    from pyhealth.tasks import MultiEventSurvivalRotterdam

ROTTERDAM_COLUMNS: tuple[str, ...] = (
    "pid",
    "year",
    "age",
    "meno",
    "size",
    "grade",
    "nodes",
    "pgr",
    "er",
    "hormon",
    "chemo",
    "rtime",
    "recur",
    "dtime",
    "death",
)
ROTTERDAM_SIZE_CATEGORIES: tuple[str, ...] = ("<=20", "20-50", ">50")

_BINARY = frozenset({"meno", "hormon", "chemo", "recur", "death"})
_POSITIVE = frozenset({"age", "rtime", "dtime"})
_NON_NEGATIVE = frozenset({"nodes", "pgr", "er"})
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def parse_rotterdam_record(
    record: Mapping[str, Any], pid: str
) -> dict[str, float | str]:
    """Validate one Rotterdam source row without imputation or label repair.

    Every value is checked and returned as read: numeric columns become
    floats and ``size`` stays one of its three source categories. Times are
    never shifted, rounded or replaced, so a recurrence censored before a
    later death keeps its own ``rtime``.

    Args:
        record: Mapping from the 14 non-ID column names to raw values,
            typically strings read from the CSV.
        pid: Patient identifier, used only in error messages.

    Returns:
        The 14 validated values keyed by column name.

    Raises:
        ValueError: If a value is missing, non-numeric, non-finite or outside
            its allowed range, or if ``rtime`` exceeds ``dtime``. The message
            names the column and the patient.

    Examples:
        >>> values = parse_rotterdam_record(
        ...     {"year": "1992", "age": "74", "meno": "1", "size": "<=20",
        ...      "grade": "3", "nodes": "0", "pgr": "35", "er": "291",
        ...      "hormon": "0", "chemo": "0", "rtime": "1799", "recur": "0",
        ...      "dtime": "1799", "death": "0"},
        ...     pid="1",
        ... )
        >>> values["size"], values["rtime"], values["recur"]
        ('<=20', 1799.0, 0.0)
    """
    values: dict[str, float | str] = {}
    for column in ROTTERDAM_COLUMNS[1:]:
        raw = record.get(column)
        text = "" if raw is None else str(raw).strip()
        if not text:
            raise ValueError(f"Rotterdam pid {pid!r}: {column} is missing")
        if column == "size":
            if text not in ROTTERDAM_SIZE_CATEGORIES:
                raise ValueError(
                    f"Rotterdam pid {pid!r}: size must be one of "
                    f"{list(ROTTERDAM_SIZE_CATEGORIES)}, got {text!r}"
                )
            values[column] = text
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"Rotterdam pid {pid!r}: {column} must be numeric, got {text!r}"
            ) from None
        if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
            raise ValueError(
                f"Rotterdam pid {pid!r}: {column} must be finite float32, got {text!r}"
            )
        if column in _BINARY and value not in (0.0, 1.0):
            raise ValueError(
                f"Rotterdam pid {pid!r}: {column} must be 0 or 1, got {text!r}"
            )
        if column in _POSITIVE and (value <= 0 or np.float32(value) <= 0):
            raise ValueError(
                f"Rotterdam pid {pid!r}: {column} must be strictly positive, "
                f"got {text!r}"
            )
        if column in _NON_NEGATIVE and value < 0:
            raise ValueError(
                f"Rotterdam pid {pid!r}: {column} must be non-negative, got {text!r}"
            )
        values[column] = value
    if values["rtime"] > values["dtime"]:
        raise ValueError(
            f"Rotterdam pid {pid!r}: rtime ({values['rtime']:g}) exceeds dtime "
            f"({values['dtime']:g}); recurrence follow-up cannot outlast "
            "death follow-up"
        )
    return values


def _validate_rotterdam_csv(path: Path) -> int:
    """Check the whole source file before any caching and count its patients.

    Args:
        path: Location of rotterdam.csv.

    Returns:
        Number of patient rows.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a column is missing, a pid is empty or repeated, a
            value is invalid, or the file has no patient rows.
    """
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(ROTTERDAM_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Rotterdam CSV missing columns: {sorted(missing)}")
        for row in reader:
            pid = row["pid"] or ""
            if not pid.strip() or pid in seen:
                raise ValueError(
                    f"Rotterdam pid must be nonempty and unique, got {pid!r}"
                )
            parse_rotterdam_record(row, pid)
            seen.add(pid)
    if not seen:
        raise ValueError("Rotterdam CSV must contain at least one patient row")
    return len(seen)


class RotterdamDataset(BaseDataset):
    """Rotterdam breast-cancer cohort with recurrence and death outcomes.

    One row per patient from the Rotterdam tumor bank: 2,982 primary breast
    cancer patients with ten baseline covariates and two outcomes, each with
    its own follow-up time in days. ``rtime``/``recur`` give time to
    recurrence or recurrence censoring, and ``dtime``/``death`` give time to
    death or censoring. A patient can have both events, one or neither.

    All patient times are kept exactly as recorded. In the source, 43
    patients died without a recorded recurrence after recurrence follow-up
    had ended (``recur=0``, ``death=1``, ``rtime < dtime``). Their recurrence
    stays censored at ``rtime`` and is not extended to the death time. Tied
    recurrence and death times are also kept. No value is imputed, rounded or
    remapped: tumor ``size`` stays one of ``"<=20"``, ``"20-50"`` or ``">50"``
    (mm), and numeric coding belongs to the task. The default task is
    :class:`~pyhealth.tasks.MultiEventSurvivalRotterdam`.

    This differs from :class:`~pyhealth.datasets.GBSGDataset`, which loads
    DeepSurv's pooled Rotterdam/GBSG benchmark: 2,232 records, seven
    features and a single recurrence-free survival endpoint in months. This
    loader uses the full cohort released with the R ``survival`` package
    (``survival::rotterdam``), as also distributed in the MENSA authors'
    repository, and keeps both events separate for multi-event survival
    analysis.

    Source: R ``survival`` package, ``survival::rotterdam``
    (https://stat.ethz.ch/R-manual/R-devel/library/survival/html/rotterdam.html),
    from Royston and Altman (2013), https://doi.org/10.1186/1471-2288-13-33.

    The whole CSV is validated when the dataset is created, before anything
    is cached. Validation requires all 15 columns; nonempty, unique ``pid``
    values; no missing values; finite numeric covariates; ``meno``,
    ``hormon``, ``chemo``, ``recur`` and ``death`` coded 0/1; positive
    ``age``, ``rtime`` and ``dtime``; non-negative ``nodes``, ``pgr`` and
    ``er``; one of the three ``size`` categories; and ``rtime <= dtime``.
    No network request is made by this loader.

    Args:
        root: Directory containing rotterdam.csv.
        tables: Optional table selection; only ``["rotterdam"]`` is supported.
        dataset_name: Optional cache/display name, default ``"rotterdam"``.
        config_path: Optional YAML configuration using the same column
            contract. Defaults to ``configs/rotterdam.yaml``.
        **kwargs: BaseDataset options, including cache_dir and num_workers.

    Raises:
        ValueError: If the table selection or configuration is unsupported,
            or the CSV fails validation. Messages name the column and pid.
        FileNotFoundError: If rotterdam.csv does not exist under ``root``.

    Examples:
        >>> from pyhealth.datasets import RotterdamDataset
        >>> dataset = RotterdamDataset(  # doctest: +SKIP
        ...     root="../data", cache_dir="../cache/rotterdam"
        ... )
        >>> len(dataset.unique_patient_ids)  # doctest: +SKIP
        2982
        >>> samples = dataset.set_task()  # doctest: +SKIP
        >>> patient = dataset.get_patient("1")  # doctest: +SKIP
        >>> event = patient.get_events("rotterdam")[0]  # doctest: +SKIP
        >>> event.attr_dict["rtime"], event.attr_dict["recur"]  # doctest: +SKIP
        ('1799', '0')
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        tables = ["rotterdam"] if tables is None else tables
        if tables != ["rotterdam"]:
            raise ValueError('RotterdamDataset supports only tables=["rotterdam"]')
        super().__init__(
            root=str(Path(root).expanduser().resolve()),
            tables=tables,
            dataset_name=dataset_name or "rotterdam",
            config_path=config_path
            or str(Path(__file__).parent / "configs" / "rotterdam.yaml"),
            **kwargs,
        )
        table = self.config.tables.get("rotterdam")
        if (
            table is None
            or table.patient_id != "pid"
            or table.timestamp is not None
            or not set(ROTTERDAM_COLUMNS[1:]).issubset(table.attributes)
        ):
            raise ValueError(
                "Rotterdam config must use pid, no timestamp, and all 14 source "
                "attributes"
            )
        self.num_source_patients = _validate_rotterdam_csv(
            Path(self.root) / table.file_path
        )

    @property
    def default_task(self) -> "MultiEventSurvivalRotterdam":
        """Recurrence/death task with ten features and source times in days.

        Examples:
            >>> dataset = RotterdamDataset(root="../data")  # doctest: +SKIP
            >>> type(dataset.default_task).__name__  # doctest: +SKIP
            'MultiEventSurvivalRotterdam'
        """
        from pyhealth.tasks import MultiEventSurvivalRotterdam

        return MultiEventSurvivalRotterdam()
