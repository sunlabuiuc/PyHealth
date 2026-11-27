"""MedMod in-hospital mortality end to end in PyHealth: synthetic MIMIC-IV v1.0 +
MIMIC-CXR folders -> MedModMIMIC4Dataset -> set_task(split=MedModSplit())."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from pyhealth.datasets import MedModMIMIC4Dataset, MedModSplit, PatientSplit
from pyhealth.datasets.medmod.discretizer import PH_NORMALIZER, medmod_ehr_funcs
from pyhealth.datasets.medmod.extraction import STAGES, MedModResources
from pyhealth.tasks import MedModInHospitalMortality
from tests.core.medmod_test_utils import (
    TEST_ITEMID_MAP,
    load_test_resources,
    make_dataset,
    write_fake_mimic,
)


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """One fake MIMIC folder and PyHealth cache for the whole module.

    Building the event table starts PyHealth's Dask worker processes. That
    takes about 3 s, whatever the data size, so the table is built once here.
    Each test then makes its own dataset object and task on this cache, which
    takes milliseconds.
    """
    roots = write_fake_mimic(tmp_path_factory.mktemp("fake_mimic"))
    make_dataset(roots).global_event_df.collect()
    return roots


def test_rows_filtered_before_cache(roots):
    df = make_dataset(roots).global_event_df.collect()
    items = (
        df.filter(pl.col("event_type") == "chartevents")["chartevents/itemid"]
        .unique()
        .to_list()
    )
    assert "999999" not in items and set(items) == {
        "220045",
        "220739",
    }  # non-MedMod itemid dropped early
    assert "chartevents/valueuom" not in df.columns
    assert df.filter(pl.col("event_type") == "labevents").height == 1


def _map_without_220739(tmp_path):
    """The test item map without item 220739."""
    lines = TEST_ITEMID_MAP.read_text().splitlines(keepends=True)
    smaller = tmp_path / "itemid_map_without_220739.csv"
    smaller.write_text("".join(line for line in lines if ",220739," not in line))
    return smaller


def test_custom_item_map_gets_its_own_cache(roots, tmp_path):
    # PyHealth's cache key has no item map, so a dataset with another map must
    # not reuse the table the first map filtered.
    ehr, cxr, cache = roots
    smaller = _map_without_220739(tmp_path)
    other = MedModMIMIC4Dataset(
        ehr_root=ehr, cxr_root=cxr, cache_dir=cache, num_workers=1, itemid_map=smaller
    )
    assert other.cache_dir != make_dataset(roots).cache_dir
    df = other.global_event_df.collect()
    items = df.filter(pl.col("event_type") == "chartevents")["chartevents/itemid"]
    assert set(items.unique().to_list()) == {"220045"}
    default = MedModMIMIC4Dataset(ehr_root=ehr, cxr_root=cxr, cache_dir=cache)
    assert default.dataset_name == "medmod_mimic4"  # pinned map, no digest


def test_task_item_ids_must_be_in_the_dataset_map(roots, tmp_path):
    # The dataset dropped item 220739 before caching, so a task that reads it
    # would silently miss its events.
    ehr, cxr, cache = roots
    dataset = MedModMIMIC4Dataset(
        ehr_root=ehr,
        cxr_root=cxr,
        cache_dir=cache,
        num_workers=1,
        itemid_map=_map_without_220739(tmp_path),
    )
    task = MedModInHospitalMortality(hours_zero="keep", resources=load_test_resources())
    with pytest.raises(ValueError, match=r"\['220739'\]"):
        dataset.set_task(task)


def _lists(tmp_path, test=(), val=()):
    """MedMod-style subject lists: "subject_id,1" marks a member."""
    paths = []
    for name, ids in (("test", test), ("val", val)):
        f = tmp_path / f"{name}set.csv"
        f.write_text("".join(f"{i},1\n" for i in ids) + "99999999,0\n")
        paths.append(str(f))
    return paths


@pytest.mark.parametrize("hours_zero", ["keep", "drop"])
def test_set_task_end_to_end_2h_fitted_normalizer(roots, tmp_path, hours_zero):
    # Both stays in train: PyHealth fits the binary label processor on the
    # training part too, which needs both labels. Train-only fitting with a
    # held-out part is tested in test_medmod_mimic4_processor.py.
    testset, valset = _lists(tmp_path)
    res = MedModResources.load(
        itemid_map=TEST_ITEMID_MAP, testset=testset, valset=valset
    )
    task = MedModInHospitalMortality(
        hours_zero=hours_zero, timestep=2.0, normalizer=None, resources=res
    )
    with pytest.warns(UserWarning, match="no samples in split"):  # no val, test
        train, val, test = make_dataset(roots).set_task(
            task, split=MedModSplit(testset=testset, valset=valset), num_workers=1
        )
    assert (len(train), len(val), len(test)) == (2, 0, 0)
    got = {train[i]["visit_id"]: train[i] for i in range(2)}
    assert set(got) == {"30000001", "30000002"}  # 30000003 is a transfer
    for stay, label in [("30000001", 0), ("30000002", 1)]:
        s = got[stay]
        assert tuple(s["ehr"].shape) == (24, 76) and s["ehr"].dtype == torch.float32
        assert float(s["label"].reshape(-1)[0]) == label and s["split"] == "train"
        assert s["image_path"].endswith(".jpg") and "/files/p10/" in s["image_path"]

    # The normalizer was learned from these (train) stays: over their bins,
    # every standardized column has mean 0.
    proc = train.input_processors["ehr"]
    assert proc.learns_statistics and proc.normalizer_file is None
    cont = task.resources.discretizer.continuous_columns()
    x = np.concatenate([np.asarray(s["ehr"], dtype=float) for s in got.values()])
    np.testing.assert_allclose(x[:, cont].mean(axis=0), 0, atol=1e-5)
    assert train.fit_split["kind"] == "medmod"

    # Hours == 0 row (HR 120 at intime): mask set only when kept
    mask_hr = task.resources.discretizer.output_header().index("mask->Heart Rate")
    assert float(got["30000001"]["ehr"][0, mask_hr]) == (
        1.0 if hours_zero == "keep" else 0.0
    )


def test_set_task_default_is_released_code_settings(roots, tmp_path):
    testset, valset = _lists(tmp_path)
    task = MedModInHospitalMortality(hours_zero="keep", resources=load_test_resources())
    ds = make_dataset(roots)
    with pytest.warns(UserWarning, match="no samples in split"):
        train, val, test = ds.set_task(
            task, split=MedModSplit(testset=testset, valset=valset), num_workers=1
        )
    samples = [p[i] for p in (train, val, test) for i in range(len(p))]
    assert {s["visit_id"] for s in samples} == {"30000001", "30000002"}
    proc = train.input_processors["ehr"]
    assert not proc.learns_statistics and proc.normalizer_file == str(
        PH_NORMALIZER.resolve()
    )
    # same numbers as MedMod's own pipeline: the task's raw 1-hour tensor, ph
    # normalizer in float64, then float32
    _, ph = medmod_ehr_funcs()
    for s in samples:
        assert tuple(s["ehr"].shape) == (48, 76)
        raws = {r["visit_id"]: r["ehr"] for r in task(ds.get_patient(s["patient_id"]))}
        np.testing.assert_array_equal(
            np.asarray(s["ehr"]), ph.transform(raws[s["visit_id"]]).astype(np.float32)
        )


def test_image_dir_end_to_end(roots, tmp_path):
    img_dir = tmp_path / "resized"
    img_dir.mkdir()
    for d in ("d1", "d2"):
        (img_dir / f"{d}.jpg").write_bytes(b"x")
    parts = make_dataset(roots).set_task(
        MedModInHospitalMortality(
            hours_zero="keep", image_dir=img_dir, resources=load_test_resources()
        ),
        split=PatientSplit(ratios=(1.0, 0.0)),  # every stay in train
        num_workers=1,
    )
    paths = sorted(p[i]["image_path"] for p in parts for i in range(len(p)))
    assert paths == [
        str(img_dir.resolve() / "d1.jpg"),
        str(img_dir.resolve() / "d2.jpg"),
    ]

    (img_dir / "d2.jpg").unlink()
    ds = make_dataset(roots)
    task = MedModInHospitalMortality(
        hours_zero="keep", image_dir=img_dir, resources=load_test_resources()
    )
    results = [
        r
        for pid in ds.unique_patient_ids
        for r in task.process_patient(ds.get_patient(pid))
    ]
    chexpert = STAGES.index("cxr_has_chexpert")
    assert sum(r.passed >= chexpert for r in results) == 2
    assert sum(r.kept for r in results) == 1  # the stay whose image is gone drops


def test_task_validates_its_arguments(tmp_path):
    with pytest.raises(ValueError, match="keep"):
        MedModInHospitalMortality(hours_zero="maybe")
    with pytest.raises(FileNotFoundError, match="image_dir"):
        MedModInHospitalMortality(hours_zero="keep", image_dir=tmp_path / "missing")
    with pytest.raises(TypeError):
        MedModInHospitalMortality("keep")  # keyword-only, so the choice is explicit
    with pytest.raises(ValueError, match="divide"):
        MedModInHospitalMortality(hours_zero="keep", timestep=5.0)
    with pytest.raises(FileNotFoundError, match="normalizer"):
        MedModInHospitalMortality(hours_zero="keep", normalizer=tmp_path / "none")
    t = MedModInHospitalMortality(hours_zero="keep")
    assert t.timestep == 1.0
    assert t.input_schema["ehr"][1]["normalizer"].endswith(".normalizer")
    t2 = MedModInHospitalMortality(hours_zero="drop", timestep=2.0, normalizer=None)
    assert t2.timestep == 2.0 and t2.input_schema["ehr"][1] == {"normalizer": None}
    assert t2.hours_zero == "drop"
