"""MedMod in-hospital mortality (Elsharief et al., 2025) on MIMIC-IV v1.0 and
MIMIC-CXR-JPG 2.0.0: the dataset and MedMod's fixed train / val / test split.

Usage::

    from pyhealth.datasets import MedModMIMIC4Dataset, MedModSplit
    from pyhealth.tasks import MedModInHospitalMortality

    dataset = MedModMIMIC4Dataset(ehr_root=..., cxr_root=..., cache_dir=...)
    task = MedModInHospitalMortality(hours_zero="keep", image_dir=...)
    train, val, test = dataset.set_task(task, split=MedModSplit())

``MedModMIMIC4Dataset`` is ``MIMIC4Dataset`` with a MIMIC-IV v1.0 config, the
tables the task reads, and MedMod's row filters applied before PyHealth caches
the events. PyHealth sorts the whole concatenated event table by patient
(base_dataset.py), which with all of chartevents + labevents needs tens of GB
of scratch space; the ``preprocess_<table>`` hooks drop rows first. Every row
dropped is one MedMod drops anyway, so the cohort and the samples are
unchanged:

* itemid not among the 114 ready itemids: extract_subjects_iv.py:91-96 and the
  inner merge in preprocessing.py:159-164 (MedMod mimic4extract)
* hadm_id empty: validate_events.py:56-57
* value empty: subject.py:30 (the task still applies MedMod's full pandas-1.1.5
  NA rule afterwards; this only removes true empties)

MedMod's split comes from fixed subject lists (testset_iv.csv, then
valset_iv.csv among the rest; split_train_and_test.py, split_train_val.py):
the paper's random 70/10/20 patient split (§3.2.4) as MedMod drew it, giving
Table 2's 4,485 / 488 / 1,242.

* ``MedModSplit``: a PyHealth ``Split`` for ``set_task``; processors are
  fitted on its training part only.
* ``split_by_medmod(samples)``: splits a built ``SampleDataset`` by its
  ``split`` field, like ``split_by_patient``; for synthetic samples, whose
  patients are not on MedMod's lists.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np

from .medmod.downloads import SHA256, medmod_file_path
from .medmod.extraction import (
    read_itemid_to_variable_map,
    read_split_file,
    split_of,
)
from .mimic4 import MIMIC4Dataset, MIMIC4EHRDataset
from .sample_dataset import SampleDataset
from .splitter import Split

if TYPE_CHECKING:
    from ..tasks.base_task import BaseTask

EHR_V10_CONFIG = Path(__file__).resolve().parent / "configs" / "mimic4_ehr_v1.0.yaml"
EHR_TABLES = ("chartevents", "labevents", "diagnoses_icd")
CXR_TABLES = ("metadata", "chexpert")


def _ready_itemids(itemid_map: str | Path | None = None) -> list[str]:
    return sorted(str(i) for i in read_itemid_to_variable_map(itemid_map)["itemid"])


def _keep_medmod_rows(frame: nw.LazyFrame, itemids: list[str]) -> nw.LazyFrame:
    return frame.filter(
        nw.col("itemid").is_in(itemids)
        & ~nw.col("hadm_id").is_null()
        & ~nw.col("value").is_null()
    )


class MedModEHRDataset(MIMIC4EHRDataset):
    """MIMIC-IV EHR sub-dataset with MedMod's row filters as PyHealth hooks.

    Examples:
        >>> from pyhealth.datasets.medmod_mimic4 import MedModEHRDataset
        >>> MedModEHRDataset.itemid_map is None  # MedMod's map, downloaded
        True
        >>> # Built for you by MedModMIMIC4Dataset (sub_datasets["ehr"]).
    """

    #: MedMod's item map; None downloads it on first use.
    itemid_map: str | None = None

    def preprocess_chartevents(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Keeps only chartevents rows of MedMod's mapped item ids.

        Args:
            frame: The raw chartevents table.

        Returns:
            The rows whose ``itemid`` maps to one of the 17 variables.
        """
        return _keep_medmod_rows(frame, _ready_itemids(self.itemid_map))

    def preprocess_labevents(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Keeps only labevents rows of MedMod's mapped item ids.

        Args:
            frame: The raw labevents table.

        Returns:
            The rows whose ``itemid`` maps to one of the 17 variables.
        """
        return _keep_medmod_rows(frame, _ready_itemids(self.itemid_map))


class MedModMIMIC4Dataset(MIMIC4Dataset):
    """``MIMIC4Dataset`` with the v1.0 config, the tables the task reads, and
    the row filters above.

    ``dataset_name`` is part of PyHealth's cache key
    (``BaseDataset._init_cache_dir``), so this cache never mixes with an
    unfiltered ``MIMIC4Dataset`` cache. The key does not cover the item map,
    so a custom ``itemid_map`` adds a digest of its file to ``dataset_name``:
    each map gets its own cache.

    Args:
        ehr_root: folder holding ``core/``, ``hosp/``, ``icu/`` (MIMIC-IV v1.0).
        cxr_root: folder holding the MIMIC-CXR-JPG CSVs.
        cache_dir: where PyHealth writes its processed copy of the data. Put it
            next to the data; without it PyHealth uses its default cache folder.
        itemid_map: MedMod's ``itemid_to_variable_map.csv``. Without it, the
            file is downloaded from MedMod's GitHub on first use. A custom map
            gets its own cache.

    Examples:
        >>> from pyhealth.datasets import MedModMIMIC4Dataset
        >>> dataset = MedModMIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="/data/mimiciv/1.0",
        ...     cxr_root="/data/mimic-cxr-jpg/2.0.0",
        ...     cache_dir="/data/pyhealth_cache",
        ... )
    """

    def __init__(
        self,
        ehr_root: str | Path,
        cxr_root: str | Path,
        cache_dir: str | Path | None = None,
        dev: bool = False,
        num_workers: int = 1,
        ehr_config_path: str | Path = EHR_V10_CONFIG,
        itemid_map: str | Path | None = None,
    ):
        ehr_root, cxr_root = os.fspath(ehr_root), os.fspath(cxr_root)
        cfg = os.fspath(ehr_config_path)
        # The default map is pinned and SHA-256 checked, so it needs no digest.
        dataset_name = "medmod_mimic4"
        if itemid_map is not None:
            dataset_name += f"_{_digest(itemid_map)}"
        super().__init__(
            ehr_root=ehr_root,
            cxr_root=cxr_root,
            ehr_tables=list(EHR_TABLES),
            cxr_tables=list(CXR_TABLES),
            ehr_config_path=cfg,
            dataset_name=dataset_name,
            dev=dev,
            cache_dir=None if cache_dir is None else os.fspath(cache_dir),
            num_workers=num_workers,
        )
        # Swap in the filtering EHR sub-dataset (built the same way as
        # MIMIC4Dataset.__init__). Nothing is loaded until global_event_df is used.
        self.sub_datasets["ehr"] = MedModEHRDataset(
            root=ehr_root,
            tables=list(EHR_TABLES),
            config_path=cfg,
            cache_dir=str(self.cache_dir),
            dev=dev,
            num_workers=num_workers,
        )
        if itemid_map is not None:
            self.sub_datasets["ehr"].itemid_map = os.fspath(itemid_map)

    def set_task(
        self, task: BaseTask | None = None, *args: Any, **kwargs: Any
    ) -> SampleDataset | tuple[SampleDataset, ...]:
        """Runs PyHealth's ``set_task`` after checking the task's item map.

        The dataset drops events outside its own item map before caching, so a
        MedMod task with item ids outside that map would silently miss their
        events. This raises an error instead.

        Args:
            task: The task, usually ``MedModInHospitalMortality``.
            *args: Passed on to ``BaseDataset.set_task``.
            **kwargs: Passed on to ``BaseDataset.set_task``.

        Returns:
            What ``BaseDataset.set_task`` returns.

        Raises:
            ValueError: If some of the task's item ids are not in the dataset's
                item map.

        Examples:
            >>> train, val, test = dataset.set_task(  # doctest: +SKIP
            ...     task, split=MedModSplit()
            ... )
        """
        resources = getattr(task, "resources", None)
        if resources is not None:
            kept = set(_ready_itemids(self.sub_datasets["ehr"].itemid_map))
            missing = sorted({str(i) for i in resources.itemids} - kept)
            if missing:
                raise ValueError(
                    f"{len(missing)} item ids of the task are not in this "
                    f"dataset's itemid_map, so their events were dropped before "
                    f"caching (first: {missing[:5]}). Use the same item map for "
                    "the dataset and the task."
                )
        return super().set_task(task, *args, **kwargs)


SPLITS = ("train", "val", "test")


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class MedModSplit(Split):
    """MedMod's subject lists as a PyHealth ``Split`` (train, val, test).

    Args:
        testset: path of a test subject list (default: MedMod's
            ``testset_iv.csv``, downloaded from MedMod's GitHub on first use).
        valset: path of a validation subject list (default: MedMod's
            ``valset_iv.csv``, the same way).

    Examples:
        >>> from pyhealth.datasets import MedModSplit
        >>> MedModSplit().to_dict()["kind"]
        'medmod'
        >>> # train, val, test = dataset.set_task(task, split=MedModSplit())
    """

    testset: str | None = None
    valset: str | None = None

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Assigns samples to MedMod's fixed train, validation and test split.

        Args:
            patient_to_index: Sample positions of each patient.

        Returns:
            The sorted sample positions of train, val and test.
        """
        test = read_split_file(self.testset or medmod_file_path("testset_iv.csv"))
        val = read_split_file(self.valset or medmod_file_path("valset_iv.csv"))
        parts: dict[str, list[int]] = {name: [] for name in SPLITS}
        for pid, indices in patient_to_index.items():
            parts[split_of(str(pid), test, val)].extend(indices)
        empty = [name for name in SPLITS if not parts[name]]
        if empty:  # normal for PyHealth dev mode or tiny test data
            warnings.warn(f"MedModSplit: no samples in split(s) {empty}", stacklevel=2)
        return [np.array(sorted(parts[name]), dtype=np.int64) for name in SPLITS]

    def to_dict(self) -> dict:
        """Describes the split for PyHealth's cache key.

        Returns:
            The split kind and the digests of the test and val lists, so the
            cache changes whenever a list's content does.
        """
        return {
            "kind": "medmod",
            "testset_sha256": (
                _digest(self.testset) if self.testset else SHA256["testset_iv.csv"][:16]
            ),
            "valset_sha256": (
                _digest(self.valset) if self.valset else SHA256["valset_iv.csv"][:16]
            ),
        }


def split_positions(splits: Iterable[str]) -> dict[str, list[int]]:
    """Positions of each split value, in sample order.

    Examples:
        >>> from pyhealth.datasets.medmod_mimic4 import split_positions
        >>> split_positions(["train", "test", "train"])
        {'train': [0, 2], 'val': [], 'test': [1]}
    """
    out: dict[str, list[int]] = {name: [] for name in SPLITS}
    for i, name in enumerate(splits):
        if name not in out:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        out[name].append(i)
    return out


def split_by_medmod(
    samples: SampleDataset,
) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    """Split a MedMod ``SampleDataset`` by its ``split`` field.

    Args:
        samples: a ``SampleDataset`` from ``set_task`` without ``split``.

    Returns:
        train, val, test: ``SampleDataset`` views (``samples.subset``), as
        returned by PyHealth's ``split_by_patient``.

    Examples:
        >>> from pyhealth.datasets import split_by_medmod
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> train, val, test = split_by_medmod(samples)  # doctest: +SKIP
    """
    positions = split_positions(samples[i]["split"] for i in range(len(samples)))
    empty = [name for name, idx in positions.items() if not idx]
    if empty:  # normal for PyHealth dev mode or tiny test data; worth a heads-up
        warnings.warn(f"split_by_medmod: no samples in split(s) {empty}", stacklevel=2)
    return tuple(samples.subset(positions[name]) for name in SPLITS)
