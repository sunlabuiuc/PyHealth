"""Generic FHIR ingestion using flattened resource tables.

Architecture
------------
1. Stream NDJSON/NDJSON.GZ FHIR resources from disk.
2. Normalize each resource type into a 2D table via a declarative
   :class:`~pyhealth.datasets.fhir.utils.ResourceSpec` registry
   (``self.resource_specs``) — see :mod:`~pyhealth.datasets.fhir.utils`.
3. Feed those tables through the standard YAML-driven
   :class:`~pyhealth.datasets.BaseDataset` pipeline so downstream task
   processing operates on :class:`~pyhealth.data.Patient` and
   ``global_event_df`` rows.

``FHIRDataset`` is generic: it owns the streaming/cache/validation machinery but
no specific resource specs or config. Use it directly by passing
``resource_specs=`` + ``config_path=``, or subclass it for a concrete source
(e.g. :class:`~pyhealth.datasets.fhir.mimic4.MIMIC4FHIR`) that bakes those in as
class attributes.

Authors:
    John Wu and Evan Febrianto
"""

from __future__ import annotations

import functools
import hashlib
import logging
import operator
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import dask.dataframe as dd
import narwhals as nw
import orjson
import pandas as pd
import platformdirs
from yaml import safe_load

from ..base_dataset import BaseDataset
from .utils import (
    FHIR_SCHEMA_VERSION,
    SUPPORTED_OUTPUT_FORMATS,
    ResourceSpec,
    filter_flat_tables_by_patient_ids,
    load_resource_specs_from_yaml,
    sorted_patient_ids_from_flat_tables,
    stream_fhir_ndjson_to_flat_tables,
    table_file_name,
    tables_from_specs,
)

logger = logging.getLogger(__name__)


def read_fhir_settings_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        data = safe_load(stream)
    return data if isinstance(data, dict) else {}


def _strip_tz_to_naive_ms(part: pd.Series) -> pd.Series:
    if getattr(part.dtype, "tz", None) is not None:
        part = part.dt.tz_localize(None)
    return part.astype("datetime64[ms]")


class FHIRDataset(BaseDataset):
    """FHIR resources flattened into per-type tables, then the standard pipeline.

    Streams raw FHIR NDJSON/NDJSON.GZ exports into flattened tables (one per
    configured resource type) and pipelines them through
    :class:`~pyhealth.datasets.BaseDataset` for downstream task processing
    (global event dataframe, patient iteration, task sampling).

    The entire ingest is driven by a single YAML config with three top-level
    sections — ``glob_patterns:`` (which NDJSON files to open),
    ``resource_specs:`` (how to project each FHIR resource type into a flat
    row), and ``tables:`` (how those rows are exposed as events downstream).
    See ``pyhealth/datasets/fhir/configs/mimic4fhir.yaml`` for a complete
    worked example and the FHIRDataset rst page for a section-by-section guide.

    Pass ``config_path=...`` directly, or subclass and set
    ``DEFAULT_CONFIG_PATH`` to bundle a default (see
    :class:`~pyhealth.datasets.fhir.mimic4.MIMIC4FHIR`).

    Args:
        root: Path to the NDJSON/NDJSON.GZ export directory.
        config_path: Path to the FHIR ingest YAML. Defaults to the class
            attribute ``DEFAULT_CONFIG_PATH``. The YAML must contain a
            ``resource_specs:`` block; any ``glob_patterns:`` and ``tables:``
            blocks are also read from here.
        glob_pattern: Single glob for NDJSON files; overrides the YAML's
            ``glob_patterns``. Mutually exclusive with *glob_patterns*.
        glob_patterns: Multiple glob patterns; overrides the YAML's
            ``glob_patterns``. Mutually exclusive with *glob_pattern*.
        output_format: Flat-table format, one of ``parquet`` (default),
            ``csv``, ``tsv``. Defaults to the class attribute
            ``DEFAULT_OUTPUT_FORMAT``.
        max_patients: Limit ingest to the first *N* unique patient IDs.
        ingest_num_shards: Ignored; retained for API compatibility.
        cache_dir: Cache directory root (UUID subdir appended per config).
        num_workers: Worker processes for task sampling.
        dev: Development mode; limits to 1000 patients if *max_patients* is
            ``None``.

    Examples:
        >>> # ad-hoc, no subclass
        >>> ds = FHIRDataset(
        ...     root="/data/fhir",
        ...     config_path="my_fhir.yaml",
        ... )
        >>> # or a preconfigured source subclass
        >>> from pyhealth.datasets import MIMIC4FHIR
        >>> ds = MIMIC4FHIR(root="/data/mimic-iv-fhir", max_patients=500)
    """

    #: Default ingest YAML path; set by source subclasses to bundle a config.
    DEFAULT_CONFIG_PATH: Optional[str] = None
    #: Default flat-table output format.
    DEFAULT_OUTPUT_FORMAT: str = "parquet"
    #: Dataset name used for cache identity / logging.
    DATASET_NAME: str = "fhir"

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        glob_patterns: Optional[Sequence[str]] = None,
        output_format: Optional[str] = None,
        max_patients: Optional[int] = None,
        ingest_num_shards: Optional[int] = None,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        del ingest_num_shards

        resolved_config = config_path or type(self).DEFAULT_CONFIG_PATH
        if resolved_config is None:
            raise ValueError(
                "FHIRDataset requires config_path: pass config_path=... or use a "
                "subclass that defines DEFAULT_CONFIG_PATH."
            )
        self._fhir_config_path = str(Path(resolved_config).resolve())
        self._fhir_settings = read_fhir_settings_yaml(self._fhir_config_path)

        # Section 2 of the YAML: how each FHIR resource type projects into a row.
        self.resource_specs: Mapping[str, ResourceSpec] = (
            load_resource_specs_from_yaml(self._fhir_settings)
        )

        # Cross-validate: every table the specs declare must have a downstream
        # `tables:` block (Section 3). Catches typos at startup.
        spec_tables = set(tables_from_specs(self.resource_specs))
        declared_tables = set((self._fhir_settings.get("tables") or {}).keys())
        missing = spec_tables - declared_tables
        if missing:
            raise ValueError(
                f"config {self._fhir_config_path}: resource_specs references "
                f"table(s) {sorted(missing)} not declared in the 'tables:' "
                f"block. Add a matching tables.<name> entry (patient_id, "
                f"timestamp, attributes) for each."
            )

        self.output_format = output_format or type(self).DEFAULT_OUTPUT_FORMAT
        if self.output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output_format {self.output_format!r}; "
                f"expected one of {SUPPORTED_OUTPUT_FORMATS}."
            )

        if glob_pattern is not None and glob_patterns is not None:
            raise ValueError("Pass at most one of glob_pattern and glob_patterns.")
        if glob_patterns is not None:
            self.glob_patterns: List[str] = list(glob_patterns)
        elif glob_pattern is not None:
            self.glob_patterns = [glob_pattern]
        else:
            raw_list = self._fhir_settings.get("glob_patterns")
            if raw_list:
                if not isinstance(raw_list, list):
                    raise TypeError("config glob_patterns must be a list of strings.")
                self.glob_patterns = [str(x) for x in raw_list]
            elif self._fhir_settings.get("glob_pattern") is not None:
                self.glob_patterns = [str(self._fhir_settings["glob_pattern"])]
            else:
                self.glob_patterns = ["**/*.ndjson.gz"]

        self.glob_pattern = (
            self.glob_patterns[0]
            if len(self.glob_patterns) == 1
            else "; ".join(self.glob_patterns)
        )
        self.max_patients = 1000 if dev and max_patients is None else max_patients

        self._fhir_tables = tables_from_specs(self.resource_specs)

        resolved_root = str(Path(root).expanduser().resolve())
        super().__init__(
            root=resolved_root,
            tables=list(self._fhir_tables),
            dataset_name=type(self).DATASET_NAME,
            config_path=self._fhir_config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    # ------------------------------------------------------------------
    # Cache identity
    # ------------------------------------------------------------------

    def _init_cache_dir(self, cache_dir: str | Path | None) -> Path:
        try:
            yaml_digest = hashlib.sha256(
                Path(self._fhir_config_path).read_bytes()
            ).hexdigest()[:16]
        except OSError:
            yaml_digest = "missing"
        identity = orjson.dumps(
            {
                "root": self.root,
                "tables": sorted(self.tables),
                "dataset_name": self.dataset_name,
                "dev": self.dev,
                "glob_patterns": self.glob_patterns,
                "max_patients": self.max_patients,
                "output_format": self.output_format,
                "fhir_schema_version": FHIR_SCHEMA_VERSION,
                "fhir_yaml_digest16": yaml_digest,
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8")
        cache_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, identity))
        out = (
            Path(platformdirs.user_cache_dir(appname="pyhealth")) / cache_id
            if cache_dir is None
            else Path(cache_dir) / cache_id
        )
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache dir: {out}")
        return out

    # ------------------------------------------------------------------
    # NDJSON -> flat tables ingest
    # ------------------------------------------------------------------

    @property
    def prepared_tables_dir(self) -> Path:
        return self.cache_dir / "flattened_tables"

    def _ensure_prepared_tables(self) -> None:
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(f"FHIR root not found: {root}")

        expected = [
            self.prepared_tables_dir / table_file_name(t, self.output_format)
            for t in self._fhir_tables
        ]
        if all(p.is_file() for p in expected):
            return
        if self.prepared_tables_dir.exists():
            shutil.rmtree(self.prepared_tables_dir)

        try:
            staging_root = self.create_tmpdir()
            staging = staging_root / "flattened_fhir_tables"
            staging.mkdir(parents=True, exist_ok=True)
            stream_fhir_ndjson_to_flat_tables(
                root,
                self.glob_patterns,
                staging,
                self.resource_specs,
                self.output_format,
            )
            if self.max_patients is None:
                shutil.move(str(staging), str(self.prepared_tables_dir))
                return

            filtered_root = self.create_tmpdir()
            filtered = filtered_root / "filtered"
            pids = sorted_patient_ids_from_flat_tables(
                staging, self._fhir_tables, self.output_format
            )
            filter_flat_tables_by_patient_ids(
                staging,
                filtered,
                pids[: self.max_patients],
                self._fhir_tables,
                self.output_format,
            )
            shutil.move(str(filtered), str(self.prepared_tables_dir))
        finally:
            self.clean_tmpdir()

    def _event_transform(self, output_dir: Path) -> None:
        self._ensure_prepared_tables()
        super()._event_transform(output_dir)

    # ------------------------------------------------------------------
    # Table loading (flat tables instead of source CSVs)
    # ------------------------------------------------------------------

    def _read_flat_table(self, path: Path) -> dd.DataFrame:
        if self.output_format == "parquet":
            return dd.read_parquet(
                str(path), split_row_groups=True, blocksize="64MB"
            ).replace("", pd.NA)
        sep = "\t" if self.output_format == "tsv" else ","
        return dd.read_csv(
            str(path), sep=sep, dtype=str, blocksize="64MB"
        ).replace("", pd.NA)

    def load_table(self, table_name: str) -> dd.DataFrame:
        """Load one flattened table into the standard event schema.

        Deviations from ``BaseDataset.load_table`` (CSV via ``_scan_csv_tsv_gz``):

        * Reads pre-built flat tables (parquet/csv/tsv) under
          ``prepared_tables_dir``.
        * Timestamp parsing uses ``errors="coerce"`` + ``utc=True`` (FHIR ISO
          strings include timezone suffix or partial dates).
        * Strips tz-aware timestamps to naive UTC for Dask compat.
        * Drops rows with null ``patient_id`` before returning.
        """
        assert self.config is not None
        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        table_cfg = self.config.tables[table_name]
        path = self.prepared_tables_dir / table_file_name(
            table_name, self.output_format
        )
        if not path.exists():
            raise FileNotFoundError(f"Flattened table not found: {path}")

        logger.info(f"Scanning FHIR flattened table: {table_name} from {path}")
        df: dd.DataFrame = self._read_flat_table(path)
        df = df.rename(columns=str.lower)

        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info(
                f"Preprocessing FHIR table: {table_name} "
                f"with {preprocess_func.__name__}"
            )
            df = preprocess_func(nw.from_native(df)).to_native()  # type: ignore[union-attr]

        for join_cfg in table_cfg.join:
            join_path = self.prepared_tables_dir / Path(join_cfg.file_path).name
            if not join_path.exists():
                raise FileNotFoundError(f"FHIR join table not found: {join_path}")
            logger.info(f"Joining FHIR table {table_name} with {join_path}")
            join_df: dd.DataFrame = self._read_flat_table(join_path)
            join_df = join_df.rename(columns=str.lower)
            join_key = join_cfg.on.lower()
            cols = [c.lower() for c in join_cfg.columns]
            df = df.merge(join_df[[join_key] + cols], on=join_key, how=join_cfg.how)

        ts_col = table_cfg.timestamp
        if ts_col:
            ts = (
                functools.reduce(
                    operator.add,
                    (df[c].astype("string") for c in ts_col),
                )
                if isinstance(ts_col, list)
                else df[ts_col].astype("string")
            )
            ts = dd.to_datetime(
                ts, format=table_cfg.timestamp_format, errors="coerce", utc=True
            )
            df = df.assign(timestamp=ts.map_partitions(_strip_tz_to_naive_ms))
        else:
            df = df.assign(timestamp=pd.NaT)

        if table_cfg.patient_id:
            df = df.assign(patient_id=df[table_cfg.patient_id].astype("string"))
        else:
            df = df.reset_index(drop=True)
            df = df.assign(patient_id=df.index.astype("string"))

        df = df.dropna(subset=["patient_id"])
        df = df.assign(event_type=table_name)
        rename_attr = {
            attr.lower(): f"{table_name}/{attr}" for attr in table_cfg.attributes
        }
        df = df.rename(columns=rename_attr)
        return df[
            ["patient_id", "event_type", "timestamp"]
            + [rename_attr[a.lower()] for a in table_cfg.attributes]
        ]

    # ------------------------------------------------------------------
    # Patient IDs (deterministic sorted order)
    # ------------------------------------------------------------------

    @property
    def unique_patient_ids(self) -> List[str]:
        if self._unique_patient_ids is None:
            self._unique_patient_ids = (
                self.global_event_df.select("patient_id")
                .unique()
                .sort("patient_id")
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
            logger.info(f"Found {len(self._unique_patient_ids)} unique patient IDs")
        return self._unique_patient_ids
