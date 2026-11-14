"""MIMIC-IV events and the Chest ImaGenome tables in one dataset.

Chest ImaGenome annotates the images of MIMIC-CXR, whose ``subject_id`` is the
MIMIC-IV ``subject_id``, so its rows join MIMIC-IV's by patient. This dataset
composes :class:`~pyhealth.datasets.MIMIC4Dataset` and
:class:`~pyhealth.datasets.ChestImaGenomeDataset` without changing either, the
way ``MIMIC4Dataset`` composes its EHR, note and CXR parts.
"""

from __future__ import annotations

import logging
from typing import Any

import dask.dataframe as dd

from .base_dataset import BaseDataset
from .chest_imagenome import ChestImaGenomeDataset
from .mimic4 import MIMIC4Dataset

logger = logging.getLogger(__name__)

# MIMIC4Dataset arguments that go into the cache key.
_MIMIC4_ROOTS = ("ehr_root", "note_root", "cxr_root", "ed_root")
_MIMIC4_TABLES = ("ehr_tables", "note_tables", "cxr_tables", "ed_tables")
# The tables ChestImaGenomeDataset loads by default.
_IMAGENOME_DEFAULT_TABLES = (
    "silver_images",
    "silver_objects",
    "silver_comparisons",
    "gold_comparisons",
)


class MIMIC4ImaGenomeDataset(BaseDataset):
    """MIMIC-IV (EHR, notes, CXR) together with Chest ImaGenome, by patient.

    Each patient holds the events of a :class:`~pyhealth.datasets.MIMIC4Dataset`
    built from ``mimic4_kwargs`` and the requested tables of a
    :class:`~pyhealth.datasets.ChestImaGenomeDataset`, for tasks such as
    :class:`~pyhealth.tasks.DiseaseProgressionMIMIC4` that need both.

    The data come from PhysioNet and need credentialed access:

    - MIMIC-IV: https://physionet.org/content/mimiciv/
    - MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/
    - Chest ImaGenome: https://physionet.org/content/chest-imagenome/1.0.0/
      (Wu et al., "Chest ImaGenome Dataset for Clinical Reasoning",
      NeurIPS Datasets and Benchmarks 2021)

    The composite has no config of its own, so its cache key covers the
    roots, tables, name and dev flag but not the source files. If files
    under the same roots change, delete this dataset's cache directory
    (:attr:`cache_dir`) to rebuild; the sub-datasets fingerprint their own
    sources.

    Args:
        imagenome_root: Root directory of Chest ImaGenome 1.0.0.
        imagenome_tables: Chest ImaGenome tables to include, for example
            ``["silver_images", "silver_comparisons", "gold_comparisons"]``.
            Defaults to the tables ChestImaGenomeDataset loads by default
            (silver images, objects and comparisons, and gold comparisons).
        dataset_name: Name of the dataset.
        dev: Whether to enable dev mode (limit to 1000 patients).
        cache_dir: Directory for cached data, as in
            :class:`~pyhealth.datasets.BaseDataset`.
        num_workers: Number of worker processes.
        **mimic4_kwargs: Arguments of
            :class:`~pyhealth.datasets.MIMIC4Dataset`, such as ``ehr_root``,
            ``cxr_root``, ``ehr_tables`` and ``cxr_tables``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4ImaGenomeDataset
        >>> dataset = MIMIC4ImaGenomeDataset(  # doctest: +SKIP
        ...     imagenome_root="/path/to/chest-imagenome/1.0.0",
        ...     imagenome_tables=["silver_images", "silver_comparisons"],
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     cxr_root="/path/to/mimic-cxr-jpg/2.0.0",
        ...     ehr_tables=["chartevents", "labevents"],
        ...     cxr_tables=["metadata"],
        ... )
        >>> sorted(dataset.sub_datasets)  # doctest: +SKIP
        ['imagenome', 'mimic4']
    """

    def __init__(
        self,
        imagenome_root: str,
        imagenome_tables: list[str] | None = None,
        dataset_name: str = "mimic4_imagenome",
        dev: bool = False,
        cache_dir: str | None = None,
        num_workers: int = 1,
        **mimic4_kwargs: Any,
    ) -> None:
        imagenome_tables = list(imagenome_tables or _IMAGENOME_DEFAULT_TABLES)
        roots = [str(mimic4_kwargs.get(name)) for name in _MIMIC4_ROOTS]
        tables = [
            table
            for name in _MIMIC4_TABLES
            for table in mimic4_kwargs.get(name) or []
        ]
        super().__init__(
            root="|".join([*roots, str(imagenome_root)]),
            tables=tables + imagenome_tables,
            dataset_name=dataset_name,
            config_path=None,
            dev=dev,
            cache_dir=cache_dir,
            num_workers=num_workers,
        )
        logger.info(
            "Initializing MIMIC4ImaGenomeDataset with Chest ImaGenome tables: "
            f"{imagenome_tables} (dev mode: {dev})"
        )
        self.sub_datasets: dict[str, BaseDataset] = {
            "mimic4": MIMIC4Dataset(
                dev=dev,
                cache_dir=str(self.cache_dir),
                num_workers=num_workers,
                **mimic4_kwargs,
            ),
            "imagenome": ChestImaGenomeDataset(
                root=str(imagenome_root),
                tables=imagenome_tables,
                cache_dir=str(self.cache_dir),
                dev=dev,
                num_workers=num_workers,
            ),
        }

    def load_data(self) -> dd.DataFrame:
        """Combines the events of both datasets.

        Returns:
            The events of the MIMIC-IV parts and of Chest ImaGenome.
        """
        frames = [dataset.load_data() for dataset in self.sub_datasets.values()]
        return dd.concat(frames, axis=0, join="outer")
