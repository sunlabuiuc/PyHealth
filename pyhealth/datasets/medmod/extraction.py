"""MedMod in-hospital-mortality extraction, one subject at a time.

Pure pandas, no PyHealth import, so every rule can be unit-tested on tiny
synthetic tables. ``pyhealth.tasks.medmod_mortality_mimic4`` adapts a PyHealth
``Patient`` to these inputs.

References are ``file:line`` in the MedMod repo (github.com/nyuad-cai/MedMod,
commit in ``files/MEDMOD_COMMIT``):

* ``mimic3csv.py``      = mimic4extract/mimic3benchmark/mimic3csv.py
* ``subject.py``        = mimic4extract/mimic3benchmark/subject.py
* ``bench_prep.py``     = mimic4extract/mimic3benchmark/preprocessing.py
* ``extract_subj.py``   = mimic4extract/mimic3benchmark/scripts/extract_subjects_iv.py
* ``validate.py``       = mimic4extract/mimic3benchmark/scripts/validate_events.py
* ``extract_ep.py``     = mimic4extract/mimic3benchmark/scripts/
                          extract_episodes_from_subjects.py
* ``create_ihm.py``     = mimic4extract/mimic3benchmark/scripts/
                          create_in_hospital_mortality.py
* ``split_test.py``     = mimic4extract/mimic3benchmark/scripts/split_train_and_test.py
* ``split_val.py``      = mimic4extract/mimic3models/split_train_val.py
* ``datasets.py`` / ``utils.py`` = src/data/datasets.py, src/data/utils.py
* MedFuse ``ehr_dataset.py`` = github.com/nyuad-cai/MedFuse datasets/ehr_dataset.py
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .discretizer import Discretizer
from .downloads import MEDMOD_COMMIT, medmod_file_path

EPS = 1e-6  # create_ihm.py:12
N_HOURS = 48  # create_ihm.py:12
# Default bin width in hours: MedMod's released code, --timestep default 1.0
# (src/parser.py:30), never overridden. (The papers say "every two hours",
# MedMod §3.2.2, MedFuse §4.1.1; the task's ``timestep`` setting changes it.)
DEFAULT_TIMESTEP = 1.0

# Strings pandas 1.1.5 (MedMod pins pandas==1.1.5, environment.yml:248) reads
# as NaN: pandas/_libs/parsers.pyx:1354-1373 at tag v1.1.5. Note "None" is NOT
# in this set (it was added in pandas 2.0, parsers.pyx:1382 at v2.0.0), so the
# GCS eye-opening value "None" survives MedMod's read_csv round trips.
PANDAS_115_NA = frozenset(
    {
        "-1.#IND",
        "1.#QNAN",
        "1.#IND",
        "-1.#QNAN",
        "#N/A N/A",
        "#N/A",
        "N/A",
        "n/a",
        "NA",
        "<NA>",
        "#NA",
        "NULL",
        "null",
        "NaN",
        "-NaN",
        "nan",
        "-nan",
        "",
    }
)

HoursZero = Literal["keep", "drop"]

# Funnel stages, in the order of the filters. A stay "passes" a
# stage if it survives that filter and every earlier one.
STAGES = (
    "icustays",  # all rows of icu/icustays for the subject
    "same_careunit",  # mimic3csv.py:77-81
    "has_admission_and_patient",  # inner merges, extract_subj.py:48-49
    "one_stay_per_admission",  # mimic3csv.py:120-125 (extract_subj.py:50)
    "age_ge_18",  # mimic3csv.py:93-98, 128-131 -> all_stays.csv
    "los_ge_48h",  # create_ihm.py:34-40
    "event_in_48h",  # extract_ep.py:73-76 + create_ihm.py:45-52
    "has_diagnosis",  # bench_prep.py:59 + create_ihm.py:28-30
    "ap_cxr_in_48h",  # utils.py:269-277
    "cxr_has_chexpert",  # utils.py:357-363
    # image_dir/<dicom_id>.jpg exists (only checked when image_dir is set)
    "resized_image_found",
)


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------


def read_itemid_to_variable_map(path: str | Path | None = None) -> pd.DataFrame:
    """bench_prep.py:141-156: keep rows with LEVEL2 != '', COUNT > 0,
    STATUS == 'ready'. Returns columns ``variable, itemid (int), mimic_label``.

    Without ``path``, MedMod's file is downloaded on first use
    (``downloads.medmod_file_path``).

    Examples:
        >>> from pyhealth.datasets.medmod.extraction import read_itemid_to_variable_map
        >>> var_map = read_itemid_to_variable_map()  # doctest: +SKIP
        >>> list(var_map.columns), var_map["variable"].nunique()  # doctest: +SKIP
        (['variable', 'itemid', 'mimic_label'], 17)
    """
    if path is None:
        path = medmod_file_path("itemid_to_variable_map.csv")
    var_map = pd.read_csv(path).fillna("").astype(str)
    var_map["COUNT"] = var_map["COUNT"].astype(int)
    var_map = var_map[(var_map["LEVEL2"] != "") & (var_map["COUNT"] > 0)]
    var_map = var_map[var_map["STATUS"] == "ready"]
    var_map["ITEMID"] = var_map["ITEMID"].astype(int)
    var_map = var_map[["LEVEL2", "ITEMID", "MIMIC LABEL"]]
    var_map = var_map.rename(
        columns={"LEVEL2": "variable", "ITEMID": "itemid", "MIMIC LABEL": "mimic_label"}
    )
    return var_map.reset_index(drop=True)


def split_of(subject_id: str, test_subjects: set[str], val_subjects: set[str]) -> str:
    """MedMod's split of a subject: test list first, then the val list among
    the rest, else train (split_test.py:33-34, split_val.py:27-28).

    Examples:
        >>> from pyhealth.datasets.medmod.extraction import split_of
        >>> split_of("101", test_subjects={"101"}, val_subjects={"101"})
        'test'
        >>> split_of("102", test_subjects=set(), val_subjects=set())
        'train'
    """
    if subject_id in test_subjects:
        return "test"
    if subject_id in val_subjects:
        return "val"
    return "train"


def read_split_file(path: str | Path) -> set[str]:
    """split_test.py:25-29 / split_val.py:15-19: lines ``subject_id,flag``;
    subjects with ``int(flag) == 1`` are in the set (compared as strings).

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.datasets.medmod.extraction import read_split_file
        >>> path = os.path.join(tempfile.mkdtemp(), "testset.csv")
        >>> _ = open(path, "w").write("101,1\\n102,0\\n")
        >>> read_split_file(path)
        {'101'}
    """
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            x, y = line.split(",")
            if int(y) == 1:
                out.add(x)
    return out


def _source_name(path: str | Path) -> str:
    """A file given by the user, by its absolute path."""
    return str(Path(path).resolve())


@dataclass
class MedModResources:
    """MedMod's files: item map, split lists, discretizer config.

    The discretizer is MedMod's (previous-value imputation, masks, start at
    intime; preprocessing.py:247-252); ``process_subject`` sets the bin width.
    Standardizing is not done here but by ``MedModEHRProcessor``
    (medmod_ehr_processor.py): a MedMod normalizer file, or fitted on the
    train split.

    The item map and the split lists are read on first use (``var_map``,
    ``test_subjects``, ``val_subjects``); by default they are MedMod's,
    downloaded from MedMod's GitHub on first use
    (``downloads.medmod_file_path``), so building a task, its schema or
    synthetic samples needs no network.

    ``repr`` shows only the source paths (``sources``), so it is short and the
    same in every run. PyHealth puts ``str()`` of each task attribute into the
    task's cache key (``BaseDataset._task_cache_dir``), so this matters.

    Examples:
        >>> from pyhealth.datasets.medmod.extraction import MedModResources
        >>> res = MedModResources.load()  # no download until a file is used
        >>> res.sources["itemid_map"], len(res.discretizer.channels)
        ('medmod@7a47c0b:itemid_to_variable_map.csv', 17)
    """

    sources: dict[str, str]
    discretizer: Discretizer = field(repr=False)
    itemid_map: Path | None = field(default=None, repr=False)
    testset: Path | None = field(default=None, repr=False)
    valset: Path | None = field(default=None, repr=False)
    _test_subjects: set[str] | None = field(default=None, repr=False)
    _val_subjects: set[str] | None = field(default=None, repr=False)
    _var_map: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def load(
        cls,
        itemid_map: str | Path | None = None,
        testset: str | Path | None = None,
        valset: str | Path | None = None,
    ) -> MedModResources:
        """Defaults: MedMod's item map and split lists (downloaded on first
        use). Pass ``itemid_map`` / ``testset`` / ``valset`` to use other
        files."""
        sources = {
            "itemid_map": (
                f"medmod@{MEDMOD_COMMIT[:7]}:itemid_to_variable_map.csv"
                if itemid_map is None
                else _source_name(itemid_map)
            ),
            "testset": (
                f"medmod@{MEDMOD_COMMIT[:7]}:testset_iv.csv"
                if testset is None
                else _source_name(testset)
            ),
            "valset": (
                f"medmod@{MEDMOD_COMMIT[:7]}:valset_iv.csv"
                if valset is None
                else _source_name(valset)
            ),
        }
        return cls(
            sources=sources,
            discretizer=Discretizer(),
            itemid_map=None if itemid_map is None else Path(itemid_map),
            testset=None if testset is None else Path(testset),
            valset=None if valset is None else Path(valset),
        )

    @property
    def test_subjects(self) -> set[str]:
        """Subjects on MedMod's test list (read, or downloaded, on first use)."""
        if self._test_subjects is None:
            path = self.testset or medmod_file_path("testset_iv.csv")
            self._test_subjects = read_split_file(path)
        return self._test_subjects

    @test_subjects.setter
    def test_subjects(self, subjects: set[str]) -> None:
        self._test_subjects = set(subjects)

    @property
    def val_subjects(self) -> set[str]:
        """Subjects on MedMod's validation list (read, or downloaded, on first use)."""
        if self._val_subjects is None:
            path = self.valset or medmod_file_path("valset_iv.csv")
            self._val_subjects = read_split_file(path)
        return self._val_subjects

    @val_subjects.setter
    def val_subjects(self, subjects: set[str]) -> None:
        self._val_subjects = set(subjects)

    @property
    def var_map(self) -> pd.DataFrame:
        """MedMod's item map, ready rows only (read, or downloaded, on first use)."""
        if self._var_map is None:
            self._var_map = read_itemid_to_variable_map(self.itemid_map)
        return self._var_map

    @property
    def itemids(self) -> set[int]:
        """The MIMIC-IV item ids mapped to MedMod's 17 variables."""
        return {int(i) for i in self.var_map["itemid"]}


# --------------------------------------------------------------------------
# Value cleaning, bench_prep.py:193-335 (applied per variable, in clean_fns order)
# --------------------------------------------------------------------------

_NUM_RE = r"^(\d+(\.\d*)?|\.\d+)$"


def clean_sbp(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:194-198
    """Systolic blood pressure: the first number of ``120/80``.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_sbp
        >>> clean_sbp(pd.DataFrame({"value": ["120/80", "110"]})).tolist()
        [120.0, 110.0]
    """
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match(r"^(\d+)/(\d+)$", s).group(1))
    return v.astype(float)


def clean_dbp(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:201-205
    """Diastolic blood pressure: the second number of ``120/80``.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_dbp
        >>> clean_dbp(pd.DataFrame({"value": ["120/80", "70"]})).tolist()
        [80.0, 70.0]
    """
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match(r"^(\d+)/(\d+)$", s).group(2))
    return v.astype(float)


def clean_crr(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:209-219
    """Capillary refill rate: normal/brisk -> 0, abnormal/delayed -> 1.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_crr
        >>> clean_crr(pd.DataFrame({"value": ["Brisk", "Delayed", "other"]})).tolist()
        [0.0, 1.0, nan]
    """
    v = pd.Series(np.nan, index=df.index, dtype=float)
    s = df.value.astype(str)
    v.loc[(s == "Normal <3 secs") | (s == "Brisk")] = 0
    v.loc[(s == "Abnormal >3 secs") | (s == "Delayed")] = 1
    return v


def clean_fio2(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:223-244
    """Fraction inspired oxygen: divided by 100 (see the comment).

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_fio2
        >>> clean_fio2(pd.DataFrame({"value": ["40"], "valuenum": ["40"]})).tolist()
        [0.4]
    """
    v = df.value.astype(float).copy()
    # Line 240 builds `is_str` with np.array(map(...), dtype=np.bool). Under
    # Python 3 a map object is not a sequence, so numpy makes a 0-d array whose
    # value is bool(map_object) == True; `is_str | (...)` is then True for every
    # row. `valuenum` holds the numeric valuenum as a string for chartevents and
    # labevents (mimic3csv.py:220, subject.py:34), so 'torr' is never in it.
    # Net effect: every FiO2 value is divided by 100. (The ph normalizer's FiO2
    # mean, 0.294, is consistent with this.)
    is_str = True
    idx = (
        df.valuenum.fillna("").astype(str).apply(lambda s: "torr" not in s.lower())
        & is_str
    )
    v.loc[idx] = v[idx] / 100.0
    return v


def clean_lab(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:248-252
    """Glucose and pH: non-numeric strings become NaN.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_lab
        >>> clean_lab(pd.DataFrame({"value": ["7.4", "ERROR"]})).tolist()
        [7.4, nan]
    """
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match(_NUM_RE, s))
    v.loc[idx] = np.nan
    return v.astype(float)


def clean_o2sat(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:256-265
    """Oxygen saturation: fractions (<= 1) become percentages.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_o2sat
        >>> clean_o2sat(pd.DataFrame({"value": ["0.95", "98"]})).tolist()
        [95.0, 98.0]
    """
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match(_NUM_RE, s))
    v.loc[idx] = np.nan
    v = v.astype(float)
    idx = v <= 1
    v.loc[idx] = v[idx] * 100.0
    return v


def clean_temperature(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:269-273
    """Temperature: values >= 79 are Fahrenheit, converted to Celsius.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_temperature
        >>> df = pd.DataFrame(
        ...     {"value": ["212", "37.5"], "valuenum": ["", ""],
        ...      "mimic_label": ["", ""]}
        ... )
        >>> clean_temperature(df).tolist()
        [100.0, 37.5]
    """
    v = df.value.astype(float).copy()
    # Faithful to MedMod: `'F' in s.lower()` can never be True, so only the
    # `v >= 79` rule converts Fahrenheit to Celsius.
    idx = (
        df.valuenum.fillna("").astype(str).apply(lambda s: "F" in s.lower())
        | df.mimic_label.apply(lambda s: "F" in s.lower())
        | (v >= 79)
    )
    v.loc[idx] = (v[idx] - 32) * 5.0 / 9
    return v


def clean_weight(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:278-286
    """Weight: ounces and pounds converted to kg.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_weight
        >>> df = pd.DataFrame(
        ...     {"value": ["70", "10"], "valuenum": ["", ""],
        ...      "mimic_label": ["Daily Weight", "Weight (lb)"]}
        ... )
        >>> [round(v, 2) for v in clean_weight(df)]
        [70.0, 4.54]
    """
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna("").astype(str).apply(
        lambda s: "oz" in s.lower()
    ) | df.mimic_label.apply(lambda s: "oz" in s.lower())
    v.loc[idx] = v[idx] / 16.0
    idx = (
        idx
        | df.valuenum.fillna("").astype(str).apply(lambda s: "lb" in s.lower())
        | df.mimic_label.apply(lambda s: "lb" in s.lower())
    )
    v.loc[idx] = v[idx] * 0.453592
    return v


def clean_height(df: pd.DataFrame) -> pd.Series:  # bench_prep.py:291-295
    """Height: inches converted to cm, rounded.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_height
        >>> df = pd.DataFrame(
        ...     {"value": ["70", "180"], "valuenum": ["", ""],
        ...      "mimic_label": ["Height (in)", "Height (cm)"]}
        ... )
        >>> clean_height(df).tolist()
        [178.0, 180.0]
    """
    v = df.value.astype(float).copy()
    # Faithful: itemid 226707 ("Height", unit Inch) is NOT converted because
    # neither its label nor valuenum contains "in".
    idx = df.valuenum.fillna("").astype(str).apply(
        lambda s: "in" in s.lower()
    ) | df.mimic_label.apply(lambda s: "in" in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


CLEAN_FNS = {  # bench_prep.py:308-319 (same order)
    "Capillary refill rate": clean_crr,
    "Diastolic blood pressure": clean_dbp,
    "Systolic blood pressure": clean_sbp,
    "Fraction inspired oxygen": clean_fio2,
    "Oxygen saturation": clean_o2sat,
    "Glucose": clean_lab,
    "pH": clean_lab,
    "Temperature": clean_temperature,
    "Weight": clean_weight,
    "Height": clean_height,
}


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """bench_prep.py:322-335. MedMod calls exit() if a cleaner raises; here
    the exception propagates. Rows whose cleaned value is NaN are dropped.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import clean_events
        >>> ev = pd.DataFrame({
        ...     "variable": ["Systolic blood pressure", "Glucose"],
        ...     "value": ["120/80", "ERROR"],
        ...     "valuenum": ["", ""],
        ...     "mimic_label": ["", ""],
        ... })
        >>> clean_events(ev)["value"].tolist()
        [120.0]
    """
    events = events.copy()
    events["value"] = events["value"].astype(object)
    for var_name, fn in CLEAN_FNS.items():
        idx = events["variable"] == var_name
        if idx.any():
            events.loc[idx, "value"] = fn(events[idx]).astype(object)
    return events.loc[events["value"].notnull()]


# --------------------------------------------------------------------------
# Inputs / outputs
# --------------------------------------------------------------------------


@dataclass
class SubjectTables:
    """One subject's rows. ID and value columns are raw strings (as PyHealth
    stores them); timestamp columns are pandas datetimes.

    * patients:   anchor_age, dod            (0 or 1 row)
    * admissions: hadm_id, admittime, dischtime, deathtime
    * icustays:   stay_id, hadm_id, first_careunit, last_careunit, intime, outtime, los
    * events:     charttime, hadm_id, stay_id (NA for labevents), itemid, value,
                  valuenum
    * diagnoses:  hadm_id  (diagnoses_icd rows that matched d_icd_diagnoses)
    * cxr:        studydatetime, dicom_id, study_id, viewposition, image_path
    * chexpert_study_ids: study_ids with a row in mimic-cxr-2.0.0-chexpert

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import (
        ...     MedModResources, SubjectTables, process_subject,
        ... )
        >>> e = pd.DataFrame()
        >>> st = SubjectTables("101", e, e, e, e, e, e)  # a subject with no rows
        >>> process_subject(st, MedModResources.load())
        []
    """

    subject_id: str
    patients: pd.DataFrame
    admissions: pd.DataFrame
    icustays: pd.DataFrame
    events: pd.DataFrame
    diagnoses: pd.DataFrame
    cxr: pd.DataFrame
    chexpert_study_ids: set[str] = field(default_factory=set)


@dataclass
class StayResult:
    """One ICU stay in ``process_subject``: the last stage it passed
    and, for kept stays, the sample's fields (``sample()``).

    Examples:
        >>> from pyhealth.datasets.medmod.extraction import StayResult
        >>> r = StayResult(subject_id="101", stay_id="201", hadm_id="301", passed=0)
        >>> r.kept
        False
    """
    subject_id: str
    stay_id: str
    hadm_id: str | None
    passed: int  # index into STAGES of the last stage this stay survived
    label: int | None = None
    split: str | None = None
    image_path: str | None = None
    dicom_id: str | None = None
    original_image_path: str | None = (
        None  # files/p../p../s../<dicom>.jpg, set even if image_dir lacks it
    )
    ehr: np.ndarray | None = None
    used_dummy_rows: bool = False  # MedMod empty-window fallback fired

    @property
    def kept(self) -> bool:
        """Whether the stay passed every cohort stage (is in the cohort)."""
        return self.passed == len(STAGES) - 1

    def sample(self) -> dict:
        """Returns the stay as a PyHealth sample.

        Returns:
            A dict with ``patient_id``, ``visit_id``, ``ehr``, ``image_path``,
            ``label`` and ``split``. Use it only for a stay in the cohort.
        """
        return {
            "patient_id": self.subject_id,
            "visit_id": self.stay_id,
            "ehr": self.ehr,
            "image_path": self.image_path,
            "label": int(self.label),
            "split": self.split,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.where(~s.astype(str).isin(PANDAS_115_NA)), errors="coerce")


def _id_str(x: Any) -> str:
    return str(int(x))


def _pandas_like_values(values: pd.Series) -> pd.Series:
    """Mimic pandas' per-file dtype inference in subject.py:28 (read_events):
    if every non-null value parses as a number the column is float, otherwise
    every value stays a string. Only matters for the same-timestamp tie-break
    in convert_events_to_timeseries and for clean_lab's string check.
    Approximation: MedMod infers over the subject's whole events.csv (all
    itemids in the full item map); this uses the 17-variable events only."""
    try:
        return values.astype(float).astype(object)
    except (TypeError, ValueError):
        return values.astype(str).astype(object)


def in_hospital_mortality(
    admittime: pd.Timestamp,
    dischtime: pd.Timestamp,
    deathtime: pd.Timestamp | None,
    dod: pd.Timestamp | None,
) -> int:
    """mimic3csv.py:101-108. dod or deathtime within [admittime, dischtime].

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import in_hospital_mortality
        >>> t = pd.Timestamp
        >>> admit, disch, death = t("2150-01-01"), t("2150-01-10"), t("2150-01-05")
        >>> in_hospital_mortality(admit, disch, death, pd.NaT)
        1
    """

    def within(t: pd.Timestamp | None) -> bool:
        return pd.notnull(t) and admittime <= t <= dischtime

    return int(within(dod) or within(deathtime))


# --------------------------------------------------------------------------
# Time series (subject.py:56-68, extract_ep.py:53-89)
# --------------------------------------------------------------------------


def cleaned_subject_events(events: pd.DataFrame, var_map: pd.DataFrame) -> pd.DataFrame:
    """Validated raw events -> mapped to variables, typed like pandas, cleaned.
    One row per raw event (before the per-charttime pick).

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import (
        ...     cleaned_subject_events, read_itemid_to_variable_map,
        ... )
        >>> ev = pd.DataFrame({"charttime": [pd.Timestamp("2150-01-01 01:00")],
        ...                    "stay_id": [201], "itemid": [220045],  # Heart Rate
        ...                    "value": ["80"], "valuenum": ["80"]})
        >>> out = cleaned_subject_events(ev, read_itemid_to_variable_map())
        >>> out[["variable", "value"]].values.tolist()
        [['Heart Rate', 80.0]]
    """
    ev = events.merge(var_map, on="itemid")  # bench_prep.py:159-164 (inner)
    if ev.empty:
        return ev
    ev = ev.copy()
    ev["value"] = _pandas_like_values(ev["value"])
    return clean_events(ev)  # extract_ep.py:55


def subject_timeseries(events: pd.DataFrame, var_map: pd.DataFrame) -> pd.DataFrame:
    """Validated raw events -> one row per charttime.

    Returns columns ``charttime``, ``stay_ids`` (set of stay_ids of the events
    at that time) and one column per variable (object; NaN = missing).

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import (
        ...     read_itemid_to_variable_map, subject_timeseries,
        ... )
        >>> ev = pd.DataFrame({"charttime": [pd.Timestamp("2150-01-01 01:00")],
        ...                    "stay_id": [201], "itemid": [220045],  # Heart Rate
        ...                    "value": ["80"], "valuenum": ["80"]})
        >>> ts = subject_timeseries(ev, read_itemid_to_variable_map())
        >>> float(ts.loc[0, "Heart Rate"]), ts.loc[0, "stay_ids"]
        (80.0, {201})
    """
    ev = cleaned_subject_events(events, var_map)
    if ev.empty:  # extract_ep.py:57-59
        return pd.DataFrame(columns=["charttime", "stay_ids"])

    # subject.py:59-61: sort by (charttime, variable, value) and keep the LAST
    # row per (charttime, variable), i.e. the largest value. Each variable's
    # values share one type (float if cleaned / numeric file, else str).
    picked = ev.groupby(["charttime", "variable"], sort=True)["value"].agg(
        lambda s: sorted(s.tolist())[-1]  # noqa: FURB192 (MedMod's sort, keep last)
    )
    wide = picked.unstack("variable")
    stays_at = ev.groupby("charttime")["stay_id"].agg(lambda s: {int(x) for x in s})
    return wide.join(stays_at.rename("stay_ids")).sort_index().reset_index()


def _fmt(v: Any) -> str:
    """Value as it would appear in MedMod's episode CSV ('' = missing)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)  # floats: str(float) == repr in Python 3, as pandas.to_csv writes


def stay_rows(
    ts: pd.DataFrame,
    stay_id: int,
    intime: pd.Timestamp,
    outtime: pd.Timestamp,
    channels: Sequence[str],
) -> list[list]:
    """Rows of one stay, ``[hours(float), v_1(str), ..., v_17(str)]``, sorted.

    subject.py:39-45: a row belongs to the stay if one of its events has that
    stay_id OR its charttime is within [intime, outtime].
    subject.py:48-53: Hours = (charttime - intime) in hours.
    Columns follow extract_ep.py:85-87 (alphabetical), which equals the
    discretizer config order.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.discretizer import Discretizer
        >>> from pyhealth.datasets.medmod.extraction import stay_rows
        >>> ts = pd.DataFrame({"charttime": [pd.Timestamp("2150-01-01 01:30")],
        ...                    "stay_ids": [{201}], "Heart Rate": [80.0]})
        >>> channels = Discretizer().channels
        >>> intime, outtime = pd.Timestamp("2150-01-01"), pd.Timestamp("2150-01-03")
        >>> rows = stay_rows(ts, 201, intime, outtime, channels)
        >>> rows[0][0], rows[0][1 + channels.index("Heart Rate")]
        (1.5, '80.0')
    """
    if ts.empty:
        return []
    sel = ts["stay_ids"].apply(lambda s: stay_id in s) | (
        (ts["charttime"] >= intime) & (ts["charttime"] <= outtime)
    )
    sub = ts[sel]
    out = []
    for _, r in sub.iterrows():
        hours = (r["charttime"] - intime).total_seconds() / 60.0 / 60
        out.append([hours] + [_fmt(r[c]) if c in sub.columns else "" for c in channels])
    out.sort(key=lambda x: x[0])
    return out


def ihm_window(rows: list[list]) -> list[list]:
    """create_ihm.py:47-48: keep -eps < t < 48 + eps.

    Examples:
        >>> from pyhealth.datasets.medmod.extraction import ihm_window
        >>> ihm_window([[-1.0, "a"], [0.0, "b"], [47.5, "c"], [60.0, "d"]])
        [[0.0, 'b'], [47.5, 'c']]
    """
    return [r for r in rows if -EPS < r[0] < N_HOURS + EPS]


# MedMod's fallback when the loader window is empty, datasets.py:156-163
# (columns: Hours + 17 channels; '109' = Heart Rate, '30' = Respiratory rate...).
MEDMOD_DUMMY_ROWS = [
    [
        "0.11666666666666667",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "109",
        "",
        "",
        "",
        "30",
        "",
        "",
        "",
        "",
    ],
    [
        "0.16666666666666666",
        "",
        "61.0",
        "",
        "",
        "",
        "",
        "",
        "",
        "109",
        "",
        "64",
        "97.0",
        "29",
        "74.0",
        "",
        "",
        "",
    ],
]


def loader_rows(
    rows: list[list], hours_zero: HoursZero, full_stay_hours: float
) -> tuple[list[list], bool]:
    """Rows the training loader feeds to the discretizer.

    * ``keep`` (MedFuse, datasets/ehr_dataset.py:47-60): every row of the IHM
      file, i.e. -eps < t < 48 + eps, Hours == 0 included.
    * ``drop`` (MedMod, datasets.py:137-163 with lower=0 and
      upper=full_stay_time from get_recent_cxr, utils.py:195-196, 208-209):
      keep 0 < t < upper. Because the IHM file already stops at 48 + eps and
      upper >= 48 for every LOS >= 48 h stay, this is 0 < t <= 48 in practice;
      a row at exactly t == upper is dropped. If nothing is left, MedMod
      substitutes hard-coded dummy rows; the second return value flags that.

    Examples:
        >>> from pyhealth.datasets.medmod.extraction import loader_rows
        >>> rows = [[0.0, "80"], [1.5, "82"]]
        >>> loader_rows(rows, "keep", full_stay_hours=72.0)
        ([['0.0', '80'], ['1.5', '82']], False)
        >>> loader_rows(rows, "drop", full_stay_hours=72.0)
        ([['1.5', '82']], False)
    """
    if hours_zero == "keep":
        return [[repr(float(r[0]))] + list(r[1:]) for r in rows], False
    if hours_zero != "drop":
        raise ValueError(f"hours_zero must be 'keep' or 'drop', got {hours_zero!r}")
    lower, upper = 0.0, full_stay_hours
    kept = []
    for r in rows:
        t = float(r[0])
        if t < lower:
            continue
        elif (t > lower) and (t < upper):
            kept.append([repr(t)] + list(r[1:]))
        elif t > upper:
            break
    if not kept:
        return [list(r) for r in MEDMOD_DUMMY_ROWS], True
    return kept, False


def build_ehr(rows: list[list], discretizer: Discretizer) -> np.ndarray:
    """datasets.py:196-201: discretize with end = period_length = 48 (the IHM
    listfile writes period_length 0, create_ihm.py:72). Returns the
    unnormalized float64 array, (48 / timestep) x 76. Standardizing (line
    202-203) is ``MedModEHRProcessor``'s job, so its statistics can come from
    the train split.

    Examples:
        >>> from pyhealth.datasets.medmod.discretizer import Discretizer
        >>> from pyhealth.datasets.medmod.extraction import build_ehr
        >>> disc = Discretizer()
        >>> row = ["1.5"] + [""] * len(disc.channels)
        >>> row[1 + disc.channels.index("Heart Rate")] = "80"
        >>> build_ehr([row], disc).shape
        (48, 76)
    """
    return discretizer.transform(rows, end=float(N_HOURS))


# --------------------------------------------------------------------------
# Main per-subject routine
# --------------------------------------------------------------------------


def resolve_image_path(
    pyhealth_path: str, dicom_id: str, image_dir: str | Path | None
) -> str | None:
    """Without ``image_dir``: PyHealth's original path (files/p../p../s../<dicom>.jpg,
    not checked on disk). With ``image_dir``: ``image_dir/<dicom_id>.jpg``, the flat
    layout of MedFuse resize.py:29, 37; None if that file does not exist.

    Examples:
        >>> import tempfile
        >>> from pyhealth.datasets.medmod.extraction import resolve_image_path
        >>> resolve_image_path("files/p12/p123/s456/abc.jpg", "abc", None)
        'files/p12/p123/s456/abc.jpg'
        >>> resolve_image_path("files/p12/p123/s456/abc.jpg", "abc", tempfile.mkdtemp())
    """
    if image_dir is None:
        return pyhealth_path
    p = Path(image_dir) / f"{dicom_id}.jpg"
    return str(p) if p.is_file() else None


def validated_events(
    st: SubjectTables, all_stays: pd.DataFrame, res: MedModResources
) -> pd.DataFrame:
    """validate_events.py logic (validate.py:54-80) on the subject's raw events.
    ``all_stays`` needs columns ``stay_int`` and ``hadm_int``.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import (
        ...     MedModResources, SubjectTables, validated_events,
        ... )
        >>> events = pd.DataFrame({"charttime": [pd.Timestamp("2150-01-01 01:00")],
        ...                        "hadm_id": ["301"], "stay_id": ["201"],
        ...                        "itemid": ["220045"], "value": ["80"],
        ...                        "valuenum": ["80"]})
        >>> e = pd.DataFrame()
        >>> st = SubjectTables("101", e, e, e, events, e, e)
        >>> all_stays = pd.DataFrame({"stay_int": [201], "hadm_int": [301]})
        >>> out = validated_events(st, all_stays, MedModResources.load())
        >>> out[["stay_id", "itemid", "value"]].values.tolist()
        [[201, 220045, '80']]
    """
    ev = st.events.copy()
    for col in ("stay_id", "valuenum"):
        if col not in ev.columns:
            ev[col] = pd.NA
    if ev.empty:
        return ev
    ev["itemid"] = _to_num(ev["itemid"])
    ev = ev[ev["itemid"].isin(res.itemids)]  # extract_subj.py:91-96 (superset filter)
    ev["itemid"] = ev["itemid"].astype(int)
    # subject.py:30: drop null values (pandas 1.1.5 NA strings)
    ev = ev[ev["value"].notna() & ~ev["value"].astype(str).isin(PANDAS_115_NA)]
    ev["hadm_num"] = _to_num(ev["hadm_id"])
    ev = ev[ev["hadm_num"].notna()]  # validate.py:56-57
    hadm_to_stay = dict(zip(all_stays["hadm_int"], all_stays["stay_int"]))
    ev = ev[ev["hadm_num"].isin(hadm_to_stay.keys())]  # validate.py:59-65
    stay_r = ev["hadm_num"].map(hadm_to_stay)
    stay_ev = _to_num(ev["stay_id"]).fillna(
        stay_r
    )  # validate.py:71 (labevents get the admission's stay)
    same = stay_ev == stay_r  # validate.py:79-80
    ev = ev[same].assign(stay_id=stay_r[same].astype(int))
    return ev[["charttime", "stay_id", "itemid", "value", "valuenum"]]


def process_subject(
    st: SubjectTables,
    res: MedModResources,
    hours_zero: HoursZero = "keep",
    image_dir: str | Path | None = None,
    timestep: float = DEFAULT_TIMESTEP,
) -> list[StayResult]:
    """Every ICU stay of one subject with the last stage it survived; kept
    stays carry the unnormalized ``ehr`` at ``timestep``-hour bins.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.medmod.extraction import (
        ...     MedModResources, SubjectTables, process_subject,
        ... )
        >>> e = pd.DataFrame()
        >>> st = SubjectTables("101", e, e, e, e, e, e)  # a subject with no rows
        >>> process_subject(st, MedModResources.load())
        []
        >>> # In the task: process_subject(task.subject_tables(patient), task.resources)
    """
    disc = res.discretizer.with_timestep(timestep)
    sid = st.subject_id
    stays = st.icustays.copy()
    if stays.empty:
        return []
    results: dict[int, StayResult] = {}
    for _, s in stays.iterrows():
        results[int(s["stay_id"])] = StayResult(
            sid, _id_str(s["stay_id"]), _id_str(s["hadm_id"]), passed=0
        )

    def bump(stay_ids: Iterable[Any], stage: str) -> None:
        k = STAGES.index(stage)
        for x in stay_ids:
            results[int(x)].passed = k

    # 1. first_careunit == last_careunit (mimic3csv.py:79)
    stays = stays[stays["first_careunit"] == stays["last_careunit"]]
    bump(stays["stay_id"], "same_careunit")

    # Label for every stay that has an admission row (used for funnel death
    # rates before the admissions merge; mimic3csv.py:101-108).
    adm = st.admissions.copy()
    adm["hadm_int"] = _to_num(adm["hadm_id"])
    adm = adm.drop_duplicates("hadm_int").set_index("hadm_int")
    dod = pd.NaT
    if (
        not st.patients.empty
        and "dod" in st.patients
        and pd.notnull(st.patients["dod"].iloc[0])
    ):
        dod = pd.to_datetime(st.patients["dod"].iloc[0])
    for r in results.values():
        h = int(r.hadm_id)
        if h in adm.index:
            a = adm.loc[h]
            r.label = in_hospital_mortality(
                pd.to_datetime(a["admittime"]),
                pd.to_datetime(a["dischtime"]),
                pd.to_datetime(a["deathtime"]),
                dod,
            )

    # 2. inner merges with admissions (subject, hadm) and patients (subject)
    #    (extract_subj.py:48-49)
    if st.patients.empty:
        stays = stays.iloc[0:0]
    stays = stays[_to_num(stays["hadm_id"]).isin(adm.index)]
    bump(stays["stay_id"], "has_admission_and_patient")

    # 3. exactly one remaining ICU stay per admission (mimic3csv.py:120-125).
    #    Counted AFTER the transfer filter, so an admission whose other stay
    #    was a transfer still qualifies.
    counts = stays.groupby("hadm_id")["stay_id"].transform("count")
    stays = stays[counts == 1]
    bump(stays["stay_id"], "one_stay_per_admission")

    # 4. age = anchor_age (negative -> 90), keep >= 18 (mimic3csv.py:93-98, 128-131)
    if not stays.empty:
        age = float(_to_num(st.patients["anchor_age"]).iloc[0])
        age = 90.0 if age < 0 else age
        if not age >= 18:
            stays = stays.iloc[0:0]
    bump(stays["stay_id"], "age_ge_18")
    all_stays = stays.copy()  # == this subject's rows of root/all_stays.csv
    if all_stays.empty:
        return list(results.values())
    all_stays["stay_int"] = _to_num(all_stays["stay_id"]).astype(int)
    all_stays["hadm_int"] = _to_num(all_stays["hadm_id"]).astype(int)

    ev = validated_events(st, all_stays, res)
    ts = (
        subject_timeseries(ev, res.var_map)
        if not ev.empty
        else pd.DataFrame(columns=["charttime", "stay_ids"])
    )

    channels = res.discretizer.channels
    diag_hadms = (
        set(_to_num(st.diagnoses["hadm_id"]).dropna().astype(int))
        if not st.diagnoses.empty
        else set()
    )

    cxr = st.cxr.copy()
    if not cxr.empty:
        cxr = cxr[cxr["viewposition"] == "AP"]  # utils.py:274

    for _, s in all_stays.iterrows():
        r = results[int(s["stay_int"])]
        intime, outtime = s["intime"], s["outtime"]

        # 5. LOS >= 48 h from icustays.los (days): create_ihm.py:34-40
        los = float(_to_num(pd.Series([s["los"]])).iloc[0])
        if pd.isnull(los) or 24.0 * los < N_HOURS - EPS:
            continue
        r.passed = STAGES.index("los_ge_48h")

        # 6. at least one time-series row with -eps < t < 48 + eps
        rows = ihm_window(stay_rows(ts, int(s["stay_int"]), intime, outtime, channels))
        if not rows:
            continue
        r.passed = STAGES.index("event_in_48h")

        # 7. at least one diagnosis for the admission (inner merge in
        #    assemble_episodic_data, bench_prep.py:59, leaves an empty episode
        #    file that create_ihm.py:28-30 skips).
        if int(s["hadm_int"]) not in diag_hadms:
            continue
        r.passed = STAGES.index("has_diagnosis")

        r.split = split_of(sid, res.test_subjects, res.val_subjects)

        # 8. latest AP CXR with intime <= StudyDateTime <= intime + 48 h
        #    (utils.py:269-277, get_recent_cxr utils.py:200-211).
        #    Ties at the same StudyDateTime: MedMod sorts with pandas' default
        #    quicksort and takes tail(1). Under its pinned numpy 1.19 that sort
        #    is stable for small groups, so the LAST tied row in metadata-file
        #    order wins (the merge in utils.py:184 keeps the metadata order).
        #    mimic-cxr-2.0.0-metadata.csv is sorted by (subject_id, study_id,
        #    dicom_id) (checked on the real file, 2026-11-09), so for one
        #    subject that is the largest (StudyDateTime, study_id, dicom_id).
        #    Computed from the row itself, never from PyHealth's event order:
        #    PyHealth 2.1.0 sorts a patient's events by timestamp only, so its
        #    order of tied images is arbitrary (pyhealth/data/data.py).
        if cxr.empty:
            continue
        end = intime + pd.DateOffset(hours=N_HOURS)
        win = cxr[(cxr["studydatetime"] >= intime) & (cxr["studydatetime"] <= end)]
        if win.empty:
            continue
        r.passed = STAGES.index("ap_cxr_in_48h")
        win = win.assign(
            _study=_to_num(win["study_id"].astype(str)),
            _dicom=win["dicom_id"].astype(str),
        )
        latest = win.sort_values(
            ["studydatetime", "_study", "_dicom"], kind="mergesort"
        ).iloc[-1]

        # 9. its study must have a CheXpert row (utils.py:357-363)
        if str(latest["study_id"]) not in st.chexpert_study_ids:
            continue
        r.passed = STAGES.index("cxr_has_chexpert")
        r.dicom_id = str(latest["dicom_id"])
        r.original_image_path = latest["image_path"]

        # 10. optional: the resized copy must exist (no fallback to another image)
        r.image_path = resolve_image_path(latest["image_path"], r.dicom_id, image_dir)
        if r.image_path is None:
            continue
        r.passed = STAGES.index("resized_image_found")

        full_stay_hours = float(
            np.round((outtime - intime).total_seconds() / 60 / 60, 3)
        )  # utils.py:195-196
        lrows, r.used_dummy_rows = loader_rows(rows, hours_zero, full_stay_hours)
        r.ehr = build_ehr(lrows, disc)

    return list(results.values())
