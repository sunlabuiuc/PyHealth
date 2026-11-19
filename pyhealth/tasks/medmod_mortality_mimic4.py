"""PyHealth task: MedMod in-hospital mortality (EHR + latest AP chest X-ray).

Used like any PyHealth task::

    from pyhealth.datasets import get_dataloader
    from pyhealth.datasets import MedModMIMIC4Dataset, MedModSplit
    from pyhealth.tasks import MedModInHospitalMortality

    dataset = MedModMIMIC4Dataset(
        ehr_root="/data/mimiciv_medmod_slim/1.0",
        cxr_root="/data/mimic-cxr-jpg/2.0.0",
        cache_dir="/data/pyhealth_cache",
    )
    task = MedModInHospitalMortality(
        hours_zero="keep", image_dir="/data/mimic-cxr-jpg/2.0.0/resized"
    )
    train, val, test = dataset.set_task(task, split=MedModSplit(), num_workers=4)
    train_loader = get_dataloader(train, batch_size=32, shuffle=True)

Every rule follows MedMod's released code, which produced the paper's numbers
(where the paper says otherwise, see the docs page, "Paper vs code"). Two EHR
settings can be changed: ``timestep`` (bin width, default 1 h) and
``normalizer`` (default: the file MedMod's code loads; ``None`` fits one on the
train split). The defaults reproduce MedMod's own extraction bit for bit and
match MedFuse's released encoders (the frozen Early Fusion encoders).

``MedModMIMIC4Dataset`` (pyhealth/datasets/medmod_mimic4.py) is a
``MIMIC4Dataset`` with the v1.0
config and MedMod's row filters applied before PyHealth's disk-heavy sort.

PyHealth references are ``file:line`` in github.com/sunlabuiuc/PyHealth
(pyhealth/...), MedMod references as in ``pyhealth/datasets/medmod/extraction.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import polars as pl

from ..datasets.medmod.discretizer import PH_NORMALIZER
from ..datasets.medmod.extraction import (
    DEFAULT_TIMESTEP,
    N_HOURS,
    HoursZero,
    MedModResources,
    StayResult,
    SubjectTables,
    process_subject,
)
from ..processors.medmod_ehr_processor import MedModEHRProcessor
from .base_task import BaseTask


def _table(patient, event_type: str, ts_name: str) -> pd.DataFrame:
    """One event type as pandas, with the ``<event_type>/`` prefix removed
    (attribute naming from base_dataset.py:1029-1033) and ``timestamp``
    renamed to ``ts_name``."""
    df: pl.DataFrame = patient.get_events(event_type=event_type, return_df=True)
    prefix = f"{event_type}/"
    keep = ["timestamp"] + [c for c in df.columns if c.startswith(prefix)]
    df = df.select(keep).rename(
        {c: c[len(prefix) :] for c in keep if c.startswith(prefix)}
    )
    out = df.to_pandas()
    return out.rename(columns={"timestamp": ts_name})


class MedModInHospitalMortality(BaseTask):
    """One sample per ICU stay that survives MedMod's IHM pipeline.

    Sample: ``patient_id`` (subject_id), ``visit_id`` (stay_id), ``ehr``
    (float32, (48 / timestep) x 76: 48 x 76 by default), ``image_path``
    (latest AP CXR within 48 h of intime), ``label`` (1 = died in hospital),
    ``split`` (train / val / test from MedMod's subject lists).

    Set it with ``split=MedModSplit()``: MedMod's fixed lists, and with
    ``normalizer=None`` the normalizer is fitted on the train split only.

    Args:
        hours_zero: required. ``"keep"`` rows at Hours == 0 (MedFuse loader,
            Early Fusion) or ``"drop"`` them, i.e. 0 < Hours <= 48 (MedMod loader,
            ConVIRT). The cohort
            is the same either way; only ``ehr`` changes. (The paper is silent.)
        timestep: bin width in hours; 48 must be a multiple. Default 1.0, the
            released code's (--timestep default); e.g. 2.0 gives 24 x 76.
        normalizer: a MedMod normalizer file, used as is. Default
            ``PH_NORMALIZER``, the one the released code loads (and MedFuse's
            checkpoints were trained with; its statistics are over 1-hour
            bins). ``None``: means and stds fitted on the train split by
            ``MedModEHRProcessor``.
        image_dir: folder of resized images (``resized/<dicom_id>.jpg``, as
            written by ``pyhealth.datasets.medmod.resize_cxr``). If set,
            ``image_path`` points
            there and stays whose file is missing are dropped (funnel stage
            ``resized_image_found``). If None, ``image_path`` is PyHealth's
            original ``files/...`` path, not checked on disk. set_task caches by
            the task's arguments, not by the folder's contents: after adding
            images, use a fresh cache_dir or delete the task cache.
        resources: MedMod's item map, split lists and discretizer config.
            Defaults to MedMod's files bundled in ``pyhealth/datasets/medmod/files``
            (``MedModResources.load()``); pass your own ``MedModResources``
            only to use different files.

    Examples:
        >>> from pyhealth.tasks import MedModInHospitalMortality
        >>> task = MedModInHospitalMortality(hours_zero="keep")
        >>> task.task_name, task.output_schema
        ('MedModInHospitalMortality', {'label': 'binary'})
        >>> # train, val, test = dataset.set_task(task, split=MedModSplit())
    """

    task_name: str = "MedModInHospitalMortality"
    input_schema: ClassVar[dict[str, Any]] = {
        "ehr": MedModEHRProcessor,
        "image_path": "raw",
    }
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        *,
        hours_zero: HoursZero,
        timestep: float = DEFAULT_TIMESTEP,
        normalizer: str | Path | None = PH_NORMALIZER,
        image_dir: str | Path | None = None,
        resources: MedModResources | None = None,
    ):
        super().__init__()
        if hours_zero not in ("keep", "drop"):
            raise ValueError(
                f'hours_zero must be "keep" (Early Fusion) or "drop" (ConVIRT), '
                f"got {hours_zero!r}"
            )
        bins = N_HOURS / timestep if timestep > 0 else 0
        if bins < 1 or abs(bins - round(bins)) > 1e-9:
            raise ValueError(f"timestep must divide {N_HOURS} h, got {timestep}")
        if normalizer is not None and not Path(normalizer).is_file():
            raise FileNotFoundError(f"normalizer file not found: {normalizer}")
        if image_dir is not None and not Path(image_dir).is_dir():
            raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
        # set_task builds its cache key from vars(task) with json.dumps(default=str)
        # (base_dataset.py:1340-1354): plain values, and a resources object whose
        # str() is its source paths.
        self.hours_zero = hours_zero
        self.timestep = float(timestep)
        self.image_dir = None if image_dir is None else str(Path(image_dir).resolve())
        self.resources = resources if resources is not None else MedModResources.load()
        # The processor spec carries the normalizer choice, so PyHealth's cache
        # key (which includes input_schema) changes with it.
        norm = None if normalizer is None else str(Path(normalizer).resolve())
        self.input_schema = {
            "ehr": (MedModEHRProcessor, {"normalizer": norm}),
            "image_path": "raw",
        }

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drop chartevents/labevents rows outside the 114 ready itemids before
        patients are materialised (hook called at base_dataset.py:1126). This
        matches MedMod, which keeps only mapped itemids (extract_subj.py:91-96,
        bench_prep.py:159-164)."""
        names = df.collect_schema().names()
        ids = [str(i) for i in sorted(self.resources.itemids)]
        expr = pl.lit(True)
        for et in ("chartevents", "labevents"):
            col = f"{et}/itemid"
            if col in names:
                expr = expr & ((pl.col("event_type") != et) | pl.col(col).is_in(ids))
        return df.filter(expr)

    def subject_tables(self, patient) -> SubjectTables:
        ce = _table(patient, "chartevents", "charttime")
        le = _table(patient, "labevents", "charttime")
        events = pd.concat([ce, le], ignore_index=True, sort=False)
        for col in ("hadm_id", "stay_id", "itemid", "value", "valuenum"):
            if col not in events.columns:
                events[col] = pd.NA

        icu = _table(patient, "icustays", "intime")
        if not icu.empty:
            # MIMIC times are ISO 8601 ("2150-03-05 14:10:00"); naming the format
            # avoids pandas' per-element dateutil fallback (and its warning) when
            # it cannot guess, e.g. if a value is missing. Same parsed values.
            icu["outtime"] = pd.to_datetime(icu["outtime"], format="ISO8601")

        cxr = _table(patient, "metadata", "studydatetime")
        chex = _table(patient, "chexpert", "studydatetime")
        return SubjectTables(
            subject_id=str(patient.patient_id),
            patients=_table(patient, "patients", "timestamp"),
            admissions=_table(patient, "admissions", "admittime"),
            icustays=icu,
            events=events,
            diagnoses=_table(patient, "diagnoses_icd", "dischtime"),
            cxr=cxr,
            chexpert_study_ids=(
                set(chex["study_id"].astype(str)) if "study_id" in chex else set()
            ),
        )

    def process_patient(self, patient) -> list[StayResult]:
        """Every ICU stay of the patient with the last stage it survived."""
        return process_subject(
            self.subject_tables(patient),
            self.resources,
            self.hours_zero,
            self.image_dir,
            self.timestep,
        )

    def __call__(self, patient) -> list[dict]:
        return [r.sample() for r in self.process_patient(patient) if r.kept]
