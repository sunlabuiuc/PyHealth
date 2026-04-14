"""MIMIC-IV FHIR ingestion using flattened resource tables.

Architecture
------------
1. Stream NDJSON/NDJSON.GZ FHIR resources from disk.
2. Normalize each resource type into a 2D table (Patient, Encounter, Condition,
   Observation, MedicationRequest, Procedure) via :mod:`~pyhealth.datasets.fhir_ingest`.
3. Feed those tables through the standard YAML-driven
   :class:`~pyhealth.datasets.BaseDataset` pipeline so downstream task processing
   operates on :class:`~pyhealth.data.Patient` and ``global_event_df`` rows.

Module layout
-------------
- :mod:`~pyhealth.datasets.fhir_ingest` -- NDJSON → Parquet pipeline.
- :mod:`~pyhealth.datasets.fhir_cehr`   -- CEHR tokenization, ConceptVocab, labels.
- This module                            -- :class:`MIMIC4FHIRDataset` wiring ingest
  into :class:`~pyhealth.datasets.BaseDataset`.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import logging
import operator
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dask.dataframe as dd
import narwhals as nw
import orjson
import pandas as pd
import platformdirs
import polars as pl
from litdata.processing.data_processor import in_notebook
from yaml import safe_load

from ..data import Patient
from .base_dataset import BaseDataset
from .fhir_cehr import ConceptVocab, ensure_special_tokens, warm_mpf_vocab_from_patient
from .fhir_ingest import (
    FHIR_SCHEMA_VERSION,
    FHIR_TABLE_FILE_NAMES,
    FHIR_TABLES,
    _sorted_patient_ids_from_flat_tables,
    filter_flat_tables_by_patient_ids,
    stream_fhir_ndjson_to_flat_tables,
)

logger = logging.getLogger(__name__)


def read_fhir_settings_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_fhir.yaml")
    with open(path, encoding="utf-8") as stream:
        data = safe_load(stream)
    return data if isinstance(data, dict) else {}


def _strip_tz_to_naive_ms(part: pd.Series) -> pd.Series:
    if getattr(part.dtype, "tz", None) is not None:
        part = part.dt.tz_localize(None)
    return part.astype("datetime64[ms]")


class MIMIC4FHIRDataset(BaseDataset):
    """MIMIC-IV FHIR with flattened resource tables and standard PyHealth task flow.

    Streams raw MIMIC-IV FHIR NDJSON/NDJSON.GZ exports into six flattened Parquet
    tables then pipelines them through :class:`~pyhealth.datasets.BaseDataset` for
    standard downstream task processing (global event dataframe, patient iteration,
    task sampling).

    **Ingest flow (out-of-core):**

    1. Scan NDJSON files matching ``glob_patterns`` (defaults to six Mimic* families).
    2. Parse and flatten each FHIR resource into a row in the appropriate table.
    3. Cache normalized tables under ``cache_dir / {uuid} / flattened_tables/``.
    4. Load tables into ``global_event_df`` via YAML config.

    **Cache fingerprinting:** includes ``glob_patterns`` and YAML digest, so changes
    to either create a new independent cache.
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        glob_patterns: Optional[Sequence[str]] = None,
        max_patients: Optional[int] = None,
        ingest_num_shards: Optional[int] = None,
        vocab_path: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        """Initialize a MIMIC-IV FHIR dataset.

        Args:
            root: Path to the NDJSON/NDJSON.GZ export directory.
            config_path: Path to a custom YAML config. Defaults to
                ``pyhealth/datasets/configs/mimic4_fhir.yaml``.
            glob_pattern: Single glob for NDJSON files. Mutually exclusive
                with ``glob_patterns``.
            glob_patterns: Multiple glob patterns. Mutually exclusive with
                ``glob_pattern``.
            max_patients: Limit ingest to the first N unique patient IDs.
            ingest_num_shards: Ignored; retained for API compatibility.
            vocab_path: Path to a pre-built ConceptVocab JSON file.
            cache_dir: Cache directory root (UUID subdir appended per config).
            num_workers: Worker processes for task sampling.
            dev: Development mode; limits to 1000 patients if max_patients is None.
        """
        del ingest_num_shards

        default_cfg = os.path.join(os.path.dirname(__file__), "configs", "mimic4_fhir.yaml")
        self._fhir_config_path = str(Path(config_path or default_cfg).resolve())
        self._fhir_settings = read_fhir_settings_yaml(self._fhir_config_path)

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
                    raise TypeError("mimic4_fhir.yaml glob_patterns must be a list of strings.")
                self.glob_patterns = [str(x) for x in raw_list]
            elif self._fhir_settings.get("glob_pattern") is not None:
                self.glob_patterns = [str(self._fhir_settings["glob_pattern"])]
            else:
                self.glob_patterns = ["**/*.ndjson.gz"]

        self.glob_pattern = (
            self.glob_patterns[0] if len(self.glob_patterns) == 1 else "; ".join(self.glob_patterns)
        )
        self.max_patients = 1000 if dev and max_patients is None else max_patients
        self.source_root = str(Path(root).expanduser().resolve())
        self.vocab = (
            ConceptVocab.load(vocab_path)
            if vocab_path and os.path.isfile(vocab_path)
            else ConceptVocab()
        )
        super().__init__(
            root=self.source_root,
            tables=FHIR_TABLES,
            dataset_name="mimic4_fhir",
            config_path=self._fhir_config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _init_cache_dir(self, cache_dir: str | Path | None) -> Path:
        try:
            yaml_digest = hashlib.sha256(Path(self._fhir_config_path).read_bytes()).hexdigest()[:16]
        except OSError:
            yaml_digest = "missing"
        identity = orjson.dumps(
            {
                "source_root": self.source_root,
                "tables": sorted(self.tables),
                "dataset_name": self.dataset_name,
                "dev": self.dev,
                "glob_patterns": self.glob_patterns,
                "max_patients": self.max_patients,
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
        logger.info("Cache dir: %s", out)
        return out

    @property
    def prepared_tables_dir(self) -> Path:
        return self.cache_dir / "flattened_tables"

    def _ensure_prepared_tables(self) -> None:
        root = Path(self.source_root)
        if not root.is_dir():
            raise FileNotFoundError(f"MIMIC4 FHIR root not found: {root}")

        expected = [self.prepared_tables_dir / FHIR_TABLE_FILE_NAMES[t] for t in FHIR_TABLES]
        if all(p.is_file() for p in expected):
            return
        if self.prepared_tables_dir.exists():
            shutil.rmtree(self.prepared_tables_dir)

        try:
            staging_root = self.create_tmpdir()
            staging = staging_root / "flattened_fhir_tables"
            staging.mkdir(parents=True, exist_ok=True)
            stream_fhir_ndjson_to_flat_tables(root, self.glob_patterns, staging)

            if self.max_patients is None:
                shutil.move(str(staging), str(self.prepared_tables_dir))
                return

            filtered_root = self.create_tmpdir()
            filtered = filtered_root / "flattened_fhir_tables_filtered"
            patient_ids = _sorted_patient_ids_from_flat_tables(staging)
            filter_flat_tables_by_patient_ids(staging, filtered, patient_ids[: self.max_patients])
            shutil.move(str(filtered), str(self.prepared_tables_dir))
        finally:
            self.clean_tmpdir()

    def _event_transform(self, output_dir: Path) -> None:
        self._ensure_prepared_tables()
        super()._event_transform(output_dir)

    def load_table(self, table_name: str) -> dd.DataFrame:
        """Load one flattened Parquet table into the standard event schema.

        Deviations from BaseDataset.load_table (which reads CSV via _scan_csv_tsv_gz):
        - Reads from pre-built Parquet under prepared_tables_dir.
        - Timestamp parsing uses errors="coerce" + utc=True (FHIR ISO strings
          include timezone suffix or are partial dates).
        - Strips tz-aware timestamps to naive UTC for Dask compatibility.
        - Drops rows with null patient_id before returning.
        """
        assert self.config is not None
        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        table_cfg = self.config.tables[table_name]
        path = self.prepared_tables_dir / table_cfg.file_path
        if not path.exists():
            raise FileNotFoundError(f"Flattened table not found: {path}")

        logger.info("Scanning FHIR flattened table: %s from %s", table_name, path)
        df: dd.DataFrame = dd.read_parquet(
            str(path), split_row_groups=True, blocksize="64MB"
        ).replace("", pd.NA)
        df = df.rename(columns=str.lower)

        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info("Preprocessing FHIR table: %s with %s", table_name, preprocess_func.__name__)
            df = preprocess_func(nw.from_native(df)).to_native()  # type: ignore[union-attr]

        for join_cfg in table_cfg.join:
            join_path = self.prepared_tables_dir / Path(join_cfg.file_path).name
            if not join_path.exists():
                raise FileNotFoundError(f"FHIR join table not found: {join_path}")
            logger.info("Joining FHIR table %s with %s", table_name, join_path)
            join_df: dd.DataFrame = dd.read_parquet(
                str(join_path), split_row_groups=True, blocksize="64MB"
            ).replace("", pd.NA)
            join_df = join_df.rename(columns=str.lower)
            join_key = join_cfg.on.lower()
            cols = [c.lower() for c in join_cfg.columns]
            df = df.merge(join_df[[join_key] + cols], on=join_key, how=join_cfg.how)

        ts_col = table_cfg.timestamp
        if ts_col:
            ts = (
                functools.reduce(operator.add, (df[c].astype("string") for c in ts_col))
                if isinstance(ts_col, list)
                else df[ts_col].astype("string")
            )
            ts = dd.to_datetime(ts, format=table_cfg.timestamp_format, errors="coerce", utc=True)
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
        rename_attr = {attr.lower(): f"{table_name}/{attr}" for attr in table_cfg.attributes}
        df = df.rename(columns=rename_attr)
        return df[["patient_id", "event_type", "timestamp"] + [rename_attr[a.lower()] for a in table_cfg.attributes]]

    @property
    def unique_patient_ids(self) -> List[str]:
        if self._unique_patient_ids is None:
            self._unique_patient_ids = (
                self.global_event_df
                .select("patient_id")
                .unique()
                .sort("patient_id")
                .collect(engine="streaming")["patient_id"]
                .to_list()
            )
            logger.info("Found %d unique patient IDs", len(self._unique_patient_ids))
        return self._unique_patient_ids

    def set_task(
        self,
        task: Any = None,
        num_workers: Optional[int] = None,
        input_processors: Optional[Any] = None,
        output_processors: Optional[Any] = None,
    ) -> Any:
        self._main_guard(self.set_task.__name__)
        if task is None:
            raise ValueError("Pass a task instance, e.g. MPFClinicalPredictionTask(max_len=512).")

        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        if isinstance(task, MPFClinicalPredictionTask):
            worker_count = (
                1 if in_notebook() else (num_workers if num_workers is not None else self.num_workers)
            )
            warmup_pids = self._mpf_patient_ids_for_task(task)
            patient_count = len(warmup_pids)
            effective_workers = min(worker_count, patient_count) if patient_count else 1
            ensure_special_tokens(self.vocab)
            self._warm_mpf_vocabulary(task, warmup_pids)
            task.frozen_vocab = effective_workers > 1
            task.vocab = self.vocab
            task._specials = ensure_special_tokens(self.vocab)

        return super().set_task(task, num_workers, input_processors, output_processors)

    def _mpf_patient_ids_for_task(self, task: Any) -> List[str]:
        return (
            task.pre_filter(self.global_event_df)
            .select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .sort()
            .to_list()
        )

    def _warm_mpf_vocabulary(self, task: Any, patient_ids: List[str]) -> None:
        clinical_cap = max(0, task.max_len - 2)
        base = self.global_event_df
        for batch in itertools.batched(patient_ids, 128):
            patients = (
                base.filter(pl.col("patient_id").is_in(batch))
                .collect(engine="streaming")
                .partition_by("patient_id", as_dict=True)
            )
            for patient_key, patient_df in patients.items():
                warm_mpf_vocab_from_patient(
                    self.vocab, Patient(patient_id=patient_key[0], data_source=patient_df), clinical_cap
                )

    def gather_samples(self, task: Any) -> List[Dict[str, Any]]:
        task.vocab = self.vocab
        task._specials = None
        task.frozen_vocab = False
        samples: List[Dict[str, Any]] = []
        for patient in self.iter_patients():
            samples.extend(task(patient))
        return samples
