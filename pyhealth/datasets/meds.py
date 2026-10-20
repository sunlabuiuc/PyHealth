"""MEDS (Medical Event Data Standard) dataset for PyHealth.

MEDS distributes event data as *typed*, sharded Parquet, already flattened to
one row per measurement -- ``(subject_id, time, code, numeric_value, ...)`` --
plus a canonical subject-to-split mapping at
``metadata/subject_splits.parquet``. See the MEDS schema documentation for
the canonical subject-to-split mapping:
https://medical-event-data-standard.github.io/
This maps almost one-to-one onto
PyHealth's canonical event schema
(``patient_id | event_type | timestamp | <table>/<attribute>``).

Parquet scanning and the typed-timestamp fast-path live in
:class:`BaseDataset`. ``MEDSDataset`` adds three MEDS-specific pieces:

1. **Schema contract at construction.** :meth:`_validate_event_schema` reads
   Parquet footers only and raises ``TypeError`` when the configured
   timestamp column is missing, not a timestamp type, or timezone-aware
   (MEDS reference ``DataSchema`` is ``timestamp[us]``, tz-naive).
2. **Split-aware loading.** ``subset=`` keeps only the patients of one
   canonical split, via ``split_source`` (``"metadata"`` or ``"directory"``).
3. **Cache disambiguation.** Subset instances nest a dedicated cache
   directory so different splits never share a processing cache.

MEDS spec: https://github.com/Medical-Event-Data-Standard/meds
"""

import logging
from pathlib import Path
from typing import Literal

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds

from .base_dataset import BaseDataset, clean_path

logger = logging.getLogger(__name__)

#: Canonical MEDS split names, in canonical order (MEDS spec).
MEDS_SPLITS: tuple[str, ...] = ("train", "tuning", "held_out")

#: MEDS-normative locations, relative to the dataset root.
DATA_RELPATH = "data"
SUBJECT_SPLITS_RELPATH = "metadata/subject_splits.parquet"

SplitSource = Literal["metadata", "directory"]


class MEDSDataset(BaseDataset):
    """Dataset for MEDS (Medical Event Data Standard) sources.

    MEDS data is distributed as sharded, typed Parquet under per-split
    directories (``data/train/*.parquet``, ``data/tuning/*.parquet``,
    ``data/held_out/*.parquet``) plus a canonical subject-to-split map at
    ``metadata/subject_splits.parquet``. See the MEDS schema documentation:
    https://medical-event-data-standard.github.io/

    ``time`` must be a timezone-naive timestamp (MEDS reference schema);
    violations raise ``TypeError`` at construction.

    Split handling:
        The canonical split is available two ways, both optional:

        * as **events**: load the ``subject_splits`` table
          (``tables=["meds", "subject_splits"]``) and each subject carries one
          ``subject_splits`` event with attribute ``subject_splits/split`` --
          the exact pattern of EHRShot's ``splits`` table, usable from
          ``Task.pre_filter`` or per-patient logic;
        * as a **loader filter**: ``subset="train"`` (or ``"tuning"`` /
          ``"held_out"``) keeps only that split's patients in every loaded
          table, via the same patient-``isin`` mechanic as dev mode.

        ``split_source`` controls where ``subset`` gets its patient list:
        ``"metadata"`` (default) reads the canonical mapping file --
        authoritative per the MEDS spec and independent of directory layout;
        ``"directory"`` derives it from which ``data/<split>/`` directory
        subjects appear in -- useful when an export omits the metadata file.
        The two sources *should* agree; whether PyHealth must verify that
        equivalence is an open question for the upstream maintainer, so
        this class does not silently pick one when they could diverge: it
        uses exactly the source you asked for, and caches them separately.

    Note:
        ``event_type`` is the table name (``"meds"``) for every row; the
        clinically meaningful event kind lives in the ``meds/code``
        attribute. This mirrors EHRShot, whose single ``ehrshot`` table also
        carries an event vocabulary in a ``code`` attribute. Whether upstream
        prefers mapping MEDS ``code`` onto ``event_type`` instead is a design
        question for the maintainer.

    Args:
        root: Root directory of the MEDS dataset (the directory that
            contains ``data/`` and ``metadata/``).
        tables: Tables to load, as named in ``configs/meds.yaml``. Defaults
            to ``["meds"]``; add ``"subject_splits"`` to expose the canonical
            split as events.
        subset: ``"train"``, ``"tuning"``, ``"held_out"``, or ``"all"``
            (default). Anything but ``"all"`` filters every loaded table to
            that split's patients.
        split_source: Where ``subset`` gets its patient list from; see
            above. Ignored when ``subset="all"``.
        dataset_name: Dataset name. Defaults to ``"meds"``.
        config_path: Path to the YAML config. Defaults to
            ``configs/meds.yaml``.
        **kwargs: Forwarded to :class:`BaseDataset` (``cache_dir``,
            ``num_workers``, ``dev``). Note dev mode's 1000-patient cap is
            applied downstream of ``load_table`` (in
            ``BaseDataset._event_transform``), so it composes with
            ``subset`` with no extra handling here.

    Examples:
        >>> from pyhealth.datasets import MEDSDataset
        >>> dataset = MEDSDataset(
        ...     root="/path/to/mimic-iv-demo-meds/0.0.1",
        ... )  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> # Canonical training split only, split map exposed as events:
        >>> train = MEDSDataset(
        ...     root="/path/to/mimic-iv-demo-meds/0.0.1",
        ...     tables=["meds", "subject_splits"],
        ...     subset="train",
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        subset: str = "all",
        split_source: SplitSource = "metadata",
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if subset not in (*MEDS_SPLITS, "all"):
            raise ValueError(
                f"subset must be one of {(*MEDS_SPLITS, 'all')}, got {subset!r}"
            )
        if split_source not in ("metadata", "directory"):
            raise ValueError(
                f"split_source must be 'metadata' or 'directory', got {split_source!r}"
            )

        # Set before super().__init__: _init_cache_dir (called by the base
        # constructor) reads them.
        self.subset = subset
        self.split_source = split_source
        self._subset_patient_ids_cache: list[str] | None = None

        if config_path is None:
            logger.info("No config path provided, using default MEDS config")
            config_path = Path(__file__).parent / "configs" / "meds.yaml"

        if tables is None:
            tables = ["meds"]

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "meds",
            config_path=config_path,
            **kwargs,
        )

        # Fail fast on schema-contract violations (footer read only).
        self._validate_event_schema()

    # ------------------------------------------------------------------
    # Cache keying
    # ------------------------------------------------------------------

    def _init_cache_dir(self, cache_dir) -> Path:
        """Nest a subset-specific directory under the standard cache key.

        The base cache key hashes only ``{root, tables, dataset_name, dev}``
        (``BaseDataset._init_cache_dir``); ``subset`` changes the *content*
        of the cached ``global_event_df`` because rows are filtered in
        ``load_data``, so instances with different subsets (or different
        split sources) must not share a cache. ``subset="all"`` (default)
        keeps the exact upstream cache layout.
        """
        base = super()._init_cache_dir(cache_dir)
        if self.subset == "all":
            return base
        sub = base / f"subset-{self.split_source}-{self.subset}"
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    # ------------------------------------------------------------------
    # Split handling (canonical split as events + optional subset filter)
    # ------------------------------------------------------------------

    def _subset_patient_ids(self) -> list[str] | None:
        """Patient IDs belonging to ``self.subset``; ``None`` for ``"all"``.

        * ``split_source="metadata"``: read the canonical mapping file. One
          row per subject, so plain pandas is enough. Column names
          ``subject_id`` / ``split``.
        * ``split_source="directory"``: subjects found under
          ``data/<subset>/``. Column projection keeps the read cheap.

        Computed once per instance and reused across tables, so multi-table
        loads pay the read a single time.
        """
        if self.subset == "all":
            return None
        if self._subset_patient_ids_cache is None:
            if self.split_source == "metadata":
                path = Path(clean_path(f"{self.root}/{SUBJECT_SPLITS_RELPATH}"))
                if not path.exists():
                    raise FileNotFoundError(
                        f"subset={self.subset!r} with split_source='metadata' "
                        f"requires {SUBJECT_SPLITS_RELPATH} under "
                        f"{self.root!r}. Pass split_source='directory' to "
                        "derive the split from the data/<split>/ layout, or "
                        "use subset='all'."
                    )
                splits = pd.read_parquet(path).rename(columns=str.lower)
                ids = splits.loc[splits["split"] == self.subset, "subject_id"]
            else:  # "directory"
                split_dir = Path(
                    clean_path(f"{self.root}/{DATA_RELPATH}/{self.subset}")
                )
                if not split_dir.is_dir():
                    raise FileNotFoundError(
                        f"subset={self.subset!r} with split_source="
                        f"'directory' requires the directory "
                        f"{DATA_RELPATH}/{self.subset} under {self.root!r}."
                    )
                ids = (
                    self._scan_parquet(str(split_dir))["subject_id"].unique().compute()
                )
            self._subset_patient_ids_cache = ids.astype("string").dropna().tolist()
            logger.info(
                f"MEDS subset={self.subset!r} via split_source="
                f"{self.split_source!r}: "
                f"{len(self._subset_patient_ids_cache)} patients"
            )
        return self._subset_patient_ids_cache

    def load_data(self) -> dd.DataFrame:
        """Load all configured tables, restricted to the subset if any.

        Returns:
            dd.DataFrame: The concatenated event frame, filtered to the
            subjects of ``self.subset`` when a split was requested.
        """
        df = super().load_data()
        subset_ids = self._subset_patient_ids()
        if subset_ids is not None:
            df = df[df["patient_id"].isin(subset_ids)]
        return df

    def _validate_event_schema(self) -> None:
        """Fails fast when a Parquet event table violates the MEDS contract.

        Only Parquet footers are read (no data, no Dask). For every selected
        table whose source is Parquet and whose timestamp is a single column,
        that column must exist and be a timezone-naive timestamp type: the
        MEDS reference ``DataSchema`` defines ``time`` as ``timestamp[us]``
        without a timezone (verified against the ``meds`` 0.4.1 package).

        This closes, at construction time, the silent-parse hazard of
        date-like integers: an ``int64`` column holding ``20240101`` is
        rejected here by dtype instead of being parsed as a date deep
        inside the Dask graph.

        Raises:
            TypeError: If the timestamp column is missing from the Parquet
                schema, is not a timestamp type, or is timezone-aware.
        """
        for name in self.tables:
            table_cfg = self.config.tables.get(name.lower())
            if table_cfg is None:
                continue  # unknown table: load_table raises the proper error
            ts_col = table_cfg.timestamp
            if not ts_col or isinstance(ts_col, list):
                continue
            source = Path(clean_path(f"{self.root}/{table_cfg.file_path}"))
            if source.suffix not in (".parquet", ".pq") and not source.is_dir():
                continue  # non-Parquet source: string-parse contract applies
            schema = pa_ds.dataset(str(source), format="parquet").schema
            fields = {field.name.lower(): field for field in schema}
            field = fields.get(ts_col.lower())
            if field is None:
                raise TypeError(
                    f"MEDS table '{name}': timestamp column '{ts_col}' is "
                    f"missing from the Parquet schema {schema.names}."
                )
            if not pa.types.is_timestamp(field.type):
                raise TypeError(
                    f"MEDS table '{name}': column '{ts_col}' must be a "
                    f"timestamp in the Parquet schema, got '{field.type}'. "
                    "Date-like integers or strings parse unreliably; convert "
                    "the column upstream (e.g. with MEDS-Transform)."
                )
            if field.type.tz is not None:
                raise TypeError(
                    f"MEDS table '{name}': column '{ts_col}' is timezone-"
                    f"aware ('{field.type}'), but the MEDS reference schema "
                    "is timezone-naive (timestamp[us]). Normalize upstream, "
                    "e.g. tz_convert('UTC').tz_localize(None)."
                )
