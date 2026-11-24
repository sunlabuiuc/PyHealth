"""MedMod's ConVIRT pretraining set: phenotyping stays, whole ICU stay + CXR.

MedMod pretrains ConVIRT with ``--task`` left at its default, phenotyping
(src/parser.py:13; scripts/pretrain/simclr.sh), on the paired train split, and
uses val for its early stopping. This module builds those samples from
MedMod's own extraction output (``extracted_root``: ``root/all_stays.csv`` and
``phenotyping/{train,val}_listfile.csv`` with the episode files, all in
``phenotyping/train/``) and the MIMIC-CXR-JPG metadata and CheXpert CSVs.

Image pairing, as MedMod (src/data/utils.py): every image of the patient joined
to every stay (``get_all_metadata``), the latest AP image with
intime <= StudyDateTime <= outtime (``get_recent_cxr``, 200-211, 280-286), only
images whose study has a CheXpert row (``get_final_meta``, 342-365), stays on
the split's listfile. Stays whose resized image is missing are dropped.

EHR, as MedMod's loader (src/data/datasets.py):

* rows with 0 < Hours < full_stay_time, strict on both sides
  (``_read_timeseries``, 137-160), where full_stay_time =
  round((outtime - intime) in hours, 3) (utils.py:194-195); MedMod's two dummy
  rows if none is left (155-160);
* discretized with end = the listfile's period_length (196, 201): 1-hour bins,
  previous-value imputation, masks; ``bins = int(period_length + 1 - 1e-6)``,
  so the length varies per stay;
* standardized by ``MedModEHRSequenceProcessor`` (the phenotyping normalizer
  by default), whose collate zero-pads a batch and returns the true lengths.

Real data: 7,755 train / 882 val stays (the paper's Table 2 has 7,756 / 882;
one train image is not served by PhysioNet).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ...processors.medmod_ehr_processor import MedModEHRSequenceProcessor
from ..sample_dataset import SampleDataset, create_sample_dataset
from .discretizer import PH_NORMALIZER, Discretizer
from .extraction import MEDMOD_DUMMY_ROWS

PRETRAINING_SPLITS = ("train", "val")


def _read_table(path: Path) -> pd.DataFrame:
    for p in (path, Path(f"{path}.gz")):
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{path}(.gz) not found")


def pretraining_cohort(
    extracted_root: str | Path,
    cxr_root: str | Path,
    splits: Sequence[str] = PRETRAINING_SPLITS,
) -> pd.DataFrame:
    """One row per paired stay: ``split, subject_id, stay_id, study_id, dicom_id``.

    Args:
        extracted_root: MedMod's extraction output (``root/``, ``phenotyping/``).
        cxr_root: folder with ``mimic-cxr-2.0.0-metadata.csv(.gz)`` and
            ``mimic-cxr-2.0.0-chexpert.csv(.gz)``.
        splits: listfiles to pair (MedMod pretrains on train, validates on val).

    Returns:
        The paired stays, in MedMod's order (split, then the merges' order).

    Examples:
        >>> from pyhealth.datasets.medmod.pretraining import pretraining_cohort
        >>> cohort = pretraining_cohort("/data/medmod_extracted",
        ...                             "/data/mimic-cxr-jpg/2.0.0")  # doctest: +SKIP
        >>> cohort["split"].value_counts().to_dict()  # doctest: +SKIP
        {'train': 7756, 'val': 882}
    """
    extracted_root, cxr_root = Path(extracted_root), Path(cxr_root)
    meta = _read_table(cxr_root / "mimic-cxr-2.0.0-metadata.csv")
    stays = pd.read_csv(extracted_root / "root" / "all_stays.csv")
    merged = meta.merge(stays[["subject_id", "stay_id", "intime", "outtime"]],
                        how="inner", on="subject_id")
    merged["StudyTime"] = merged["StudyTime"].apply(lambda x: f"{int(float(x)):06}")
    merged["StudyDateTime"] = pd.to_datetime(
        merged["StudyDate"].astype(str) + " " + merged["StudyTime"],
        format="%Y%m%d %H%M%S",
    )
    merged["intime"] = pd.to_datetime(merged["intime"])
    merged["outtime"] = pd.to_datetime(merged["outtime"])
    during = merged[(merged["StudyDateTime"] >= merged["intime"])
                    & (merged["StudyDateTime"] <= merged["outtime"])]
    ap = during[during["ViewPosition"] == "AP"]
    latest = ap.sort_values("StudyDateTime", kind="stable").groupby("stay_id").tail(1)

    chexpert = _read_table(cxr_root / "mimic-cxr-2.0.0-chexpert.csv")
    labeled = meta.merge(chexpert[["study_id"]], how="inner", on="study_id")
    labeled = labeled.drop_duplicates(subset=["dicom_id"])[["dicom_id"]]
    parts = []
    for split in splits:
        listfile = pd.read_csv(extracted_root / "phenotyping" / f"{split}_listfile.csv")
        part = latest.merge(listfile, how="inner", on="stay_id")
        part = part.merge(labeled, how="inner", on="dicom_id")
        part.insert(0, "split", split)
        parts.append(part)
    cohort = pd.concat(parts, ignore_index=True)
    return cohort[["split", "subject_id", "stay_id", "study_id", "dicom_id"]]


def read_timeseries(
    path: str | Path, lower: float, upper: float
) -> tuple[np.ndarray, list[str], bool]:
    """MedMod's ``_read_timeseries`` (datasets.py:137-160), line for line.

    Returns:
        The rows ``lower < Hours < upper`` (strings), the header, and whether
        MedMod's dummy rows replaced an empty window.

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.datasets.medmod.pretraining import read_timeseries
        >>> path = os.path.join(tempfile.mkdtemp(), "episode.csv")
        >>> _ = open(path, "w").write("Hours,HR\\n0.0,70\\n1.5,80\\n9.0,90\\n")
        >>> rows, header, dummy = read_timeseries(path, 0, 5.0)
        >>> rows.tolist(), header, dummy
        ([['1.5', '80']], ['Hours', 'HR'], False)
    """
    ret = []
    with open(path) as tsfile:
        header = tsfile.readline().strip().split(",")
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(",")
            t = float(mas[0])
            if t < lower:
                continue
            elif (t > lower) & (t < upper):
                ret.append(np.array(mas))
            elif t > upper:
                break
    if ret:
        return np.stack(ret), header, False
    return np.stack([np.array(r) for r in MEDMOD_DUMMY_ROWS]), header, True


def read_listfile(path: str | Path) -> dict[str, tuple[str, float]]:
    """``stay_id -> (episode file, period_length)``, parsed as MedMod does
    (``line.split(',')``, ``float(mas[1])``; datasets.py:96-121).

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.datasets.medmod.pretraining import read_listfile
        >>> path = os.path.join(tempfile.mkdtemp(), "train_listfile.csv")
        >>> _ = open(path, "w").write("stay,period_length,stay_id\\n"
        ...                           "1_episode1_timeseries.csv,30.5,201\\n")
        >>> read_listfile(path)
        {'201': ('1_episode1_timeseries.csv', 30.5)}
    """
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            mas = line.strip().split(",")
            out[str(int(float(mas[2])))] = (mas[0], float(mas[1]))
    return out


def full_stay_times(all_stays: str | Path) -> dict[str, float]:
    """``stay_id -> round((outtime - intime) in hours, 3)`` (utils.py:194-195).

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.datasets.medmod.pretraining import full_stay_times
        >>> path = os.path.join(tempfile.mkdtemp(), "all_stays.csv")
        >>> _ = open(path, "w").write("stay_id,intime,outtime\\n"
        ...                           "201,2150-01-01 00:00,2150-01-02 06:00\\n")
        >>> full_stay_times(path)
        {'201': 30.0}
    """
    stays = pd.read_csv(all_stays, usecols=["stay_id", "intime", "outtime"])
    hours = (pd.to_datetime(stays["outtime"]) - pd.to_datetime(stays["intime"])).apply(
        lambda x: np.round(x.total_seconds() / 60 / 60, 3)
    )
    return dict(zip(stays["stay_id"].astype(str), hours))


def load_pretraining_samples(
    extracted_root: str | Path,
    cxr_root: str | Path,
    image_dir: str | Path,
    cohort: pd.DataFrame | None = None,
) -> tuple[list[dict], dict]:
    """Raw pretraining samples and a funnel of counts.

    Args:
        extracted_root: MedMod's extraction output.
        cxr_root: folder with the MIMIC-CXR-JPG metadata and CheXpert CSVs.
        image_dir: resized images, ``<image_dir>/<dicom_id>.jpg``.
        cohort: ``pretraining_cohort(...)``; computed if None.

    Returns:
        ``(samples, funnel)``. Each sample has ``patient_id``, ``visit_id``,
        ``ehr`` (unnormalized float64, ``(bins, 76)``), ``image_path`` and
        ``split``. ``funnel`` holds counts per split.

    Examples:
        >>> from pyhealth.datasets.medmod.pretraining import (
        ...     load_pretraining_samples,
        ... )
        >>> samples, funnel = load_pretraining_samples(  # doctest: +SKIP
        ...     "/data/medmod_extracted", "/data/mimic-cxr-jpg/2.0.0",
        ...     "/data/mimic-cxr-jpg/2.0.0/resized")
    """
    extracted_root, image_dir = Path(extracted_root), Path(image_dir)
    if cohort is None:
        cohort = pretraining_cohort(extracted_root, cxr_root)
    pheno = extracted_root / "phenotyping"
    upper = full_stay_times(extracted_root / "root" / "all_stays.csv")
    disc = Discretizer()
    samples, funnel = [], {}
    for split in dict.fromkeys(cohort["split"]):
        listfile = read_listfile(pheno / f"{split}_listfile.csv")
        part = cohort[cohort["split"] == split]
        f = funnel[split] = {"listfile_stays": len(listfile), "paired": len(part),
                             "image_missing": 0, "dummy_rows": 0, "kept": 0}
        for row in part.itertuples(index=False):
            stay_id = str(row.stay_id)
            image = image_dir / f"{row.dicom_id}.jpg"
            if not image.is_file():
                f["image_missing"] += 1
                continue
            name, period_length = listfile[stay_id]
            X, header, dummy = read_timeseries(
                pheno / "train" / name, 0, upper[stay_id]
            )
            f["dummy_rows"] += int(dummy)
            samples.append({
                "patient_id": str(row.subject_id),
                "visit_id": stay_id,
                "ehr": disc.transform(X, header=header, end=period_length),
                "image_path": str(image),
                "split": split,
            })
            f["kept"] += 1
    return samples, funnel


def create_pretraining_dataset(
    samples: list[dict], normalizer: str | Path | None = PH_NORMALIZER
) -> SampleDataset:
    """A PyHealth ``SampleDataset``: ``ehr`` (varying length, standardized by
    ``MedModEHRSequenceProcessor``), ``image_path`` and ``split``.

    Args:
        samples: from ``load_pretraining_samples`` (or
            ``make_synthetic_pretraining_samples``).
        normalizer: a MedMod normalizer file (default: the phenotyping one
            MedMod's code loads), or None to fit on the samples.

    Examples:
        >>> import tempfile
        >>> from pyhealth.datasets.medmod.pretraining import (
        ...     create_pretraining_dataset,
        ... )
        >>> from pyhealth.datasets.medmod.synthetic import (
        ...     make_synthetic_pretraining_samples,
        ... )
        >>> raw = make_synthetic_pretraining_samples(tempfile.mkdtemp(), 4)
        >>> len(create_pretraining_dataset(raw))  # doctest: +SKIP
        4
    """
    return create_sample_dataset(
        samples,
        input_schema={
            "ehr": (MedModEHRSequenceProcessor,
                    {"normalizer": None if normalizer is None else str(normalizer)}),
            "image_path": "raw",
        },
        output_schema={},
        dataset_name="medmod_pretraining",
        task_name="MedModConVIRTPretraining",
        in_memory=False,
    )
