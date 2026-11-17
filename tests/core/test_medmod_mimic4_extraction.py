"""MedMod in-hospital mortality: cohort, labels, cleaning, time series, CXR
pairing and split, on tiny synthetic subjects (no PyHealth dataset build)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.medmod.extraction import (
    MEDMOD_DUMMY_ROWS,
    STAGES,
    loader_rows,
    process_subject,
)
from tests.core.medmod_test_utils import (
    load_test_resources,
    HR,
    GCS_EYE,
    CRR,
    TEMP_F,
    FIO2,
    HEIGHT_IN,
    SPO2,
    GLUCOSE_LAB,
    PH_LAB,
    INTIME,
    SID,
    HADM,
    STAY,
    stay_row,
    make_subject,
    cxr_row,
    run,
    raw,
    col,
)


@pytest.fixture(scope="module")
def res():
    return load_test_resources()


KEPT = len(STAGES) - 1


NORMAL_HR = 86.0  # discretizer_config.json normal_values["Heart Rate"]


def kept(st, res, hours_zero="keep"):
    r = run(st, res, hours_zero)[STAY]
    assert r.kept, f"stay dropped after stage {STAGES[r.passed]}"
    return r


def hr_raw(r, res):
    return raw(r.ehr, res)[:, col(res, "Heart Rate")]


def hr_mask(r, res):
    return r.ehr[:, col(res, "mask->Heart Rate")]


def test_sample_format(res):
    r = kept(make_subject(events=[(5.0, HR, "90")]), res)
    s = r.sample()
    assert set(s) == {"patient_id", "visit_id", "ehr", "image_path", "label", "split"}
    # unnormalized float64; MedModEHRProcessor standardizes and casts later
    assert s["ehr"].dtype == np.float64 and s["ehr"].shape == (48, 76)
    default = process_subject(make_subject(events=[(5.0, HR, "90")]), res)[0]
    assert default.ehr.shape == (48, 76)  # the released code's 1-hour bins
    # 2-hour bins: hour 5.0 lands in bin int(5 / 2 - eps) = 2
    two_hour = run(make_subject(events=[(5.0, HR, "90")]), res, timestep=2.0)[STAY]
    assert two_hour.ehr.shape == (24, 76)
    assert hr_mask(two_hour, res).nonzero()[0].tolist() == [2]
    assert hr_raw(two_hour, res)[2] == 90.0
    assert s["patient_id"] == "10000001" and s["visit_id"] == STAY
    assert s["label"] == 0 and s["split"] == "train"
    assert s["image_path"].endswith("/s50000001/d1.jpg")


def test_event_window_edges(res):
    # create_ihm.py:47-48 keeps -eps < t < 48 + eps
    r = kept(
        make_subject(
            events=[(-1 / 60, HR, "50"), (48.0, HR, "120"), (48 + 1 / 60, HR, "130")]
        ),
        res,
    )
    hr, m = hr_raw(r, res), hr_mask(r, res)
    assert m[47] == 1 and hr[47] == pytest.approx(120, abs=1e-3)  # t == 48 -> last bin
    assert m[:47].sum() == 0  # t < 0 dropped
    assert hr[0] == pytest.approx(
        NORMAL_HR, abs=1e-3
    )  # imputed normal, not the t<0 value


def test_stay_with_events_only_outside_window_is_dropped(res):
    r = run(make_subject(events=[(-0.5, HR, "80"), (49.0, HR, "80")]), res)[STAY]
    assert STAGES[r.passed] == "los_ge_48h"


def test_los_edges(res):
    ev = [(5.0, HR, "90")]
    assert run(make_subject(events=ev, stays=[stay_row(los=2.0)]), res)[STAY].kept
    assert run(make_subject(events=ev, stays=[stay_row(los=(48 - 1e-7) / 24)]), res)[
        STAY
    ].kept  # eps
    r = run(make_subject(events=ev, stays=[stay_row(los=47.9 / 24)]), res)[STAY]
    assert STAGES[r.passed] == "age_ge_18"


def test_cxr_window_edges(res):
    ev = [(5.0, HR, "90")]
    for hours, ok in [
        (0.0, True),
        (48.0, True),
        (48 + 1 / 3600, False),
        (-1 / 3600, False),
    ]:
        r = run(make_subject(events=ev, cxr=[cxr_row(hours)]), res)[STAY]
        assert r.kept is ok, hours
        if not ok:
            assert STAGES[r.passed] == "has_diagnosis"


def test_cxr_latest_ap_and_chexpert(res):
    ev = [(5.0, HR, "90")]
    cxr = [
        cxr_row(2, "early", "50000001"),
        cxr_row(30, "late", "50000002"),
        cxr_row(40, "pa", "50000003", view="PA"),
    ]
    r = kept(make_subject(events=ev, cxr=cxr, chexpert=("50000001", "50000002")), res)
    assert r.dicom_id == "late"  # latest AP; the later PA is ignored
    # latest AP has no CheXpert row -> stay dropped (no fallback to an earlier image)
    r = run(make_subject(events=ev, cxr=cxr, chexpert=("50000001",)), res)[STAY]
    assert STAGES[r.passed] == "ap_cxr_in_48h"


def test_cxr_tie_takes_last_metadata_row(res):
    # Same study, same time: the metadata file lists a study's images in
    # ascending dicom_id, so MedMod's tail(1) takes the largest dicom_id,
    # whatever order the rows arrive in (PyHealth 2.1.0 does not keep file order).
    ev = [(5.0, HR, "90")]
    for rows in (
        [cxr_row(10, "aaa"), cxr_row(10, "bbb")],
        [cxr_row(10, "bbb"), cxr_row(10, "aaa")],
    ):
        assert kept(make_subject(events=ev, cxr=rows), res).dicom_id == "bbb"
    # Two studies at the same time: the file is sorted by study_id (numeric),
    # so the larger one wins.
    rows = [cxr_row(10, "zzz", "50000009"), cxr_row(10, "aaa", "50000010")]
    st = make_subject(events=ev, cxr=rows, chexpert=("50000009", "50000010"))
    assert kept(st, res).dicom_id == "aaa"
    # A later image still beats a larger dicom_id.
    rows = [cxr_row(11, "aaa"), cxr_row(10, "zzz")]
    assert kept(make_subject(events=ev, cxr=rows), res).dicom_id == "aaa"


def test_hours_zero_keep_vs_drop(res):
    st = make_subject(events=[(0.0, HR, "140"), (1.5, HR, "100")])
    k = kept(st, res, "keep")
    d = kept(st, res, "drop")
    assert hr_mask(k, res)[0] == 1 and hr_raw(k, res)[0] == pytest.approx(140, abs=1e-3)
    assert hr_mask(d, res)[0] == 0 and hr_raw(d, res)[0] == pytest.approx(
        NORMAL_HR, abs=1e-3
    )
    assert hr_raw(d, res)[1] == pytest.approx(100, abs=1e-3)
    assert not k.used_dummy_rows and not d.used_dummy_rows


def test_hours_zero_drop_only_row_at_zero_uses_medmod_dummy(res):
    st = make_subject(events=[(0.0, HR, "140")])
    d = kept(st, res, "drop")  # cohort unchanged: stay still kept
    assert d.used_dummy_rows
    expected = res.discretizer.transform(MEDMOD_DUMMY_ROWS, end=48.0)
    np.testing.assert_array_equal(d.ehr, expected)
    assert hr_raw(d, res)[0] == pytest.approx(109, abs=1e-3)  # dummy HR


def test_drop_mode_upper_bound_is_full_stay_time():
    rows = [[0.0] + [""] * 17, [10.0] + ["1"] * 17, [48.0] + ["2"] * 17]
    out, dummy = loader_rows(rows, "drop", full_stay_hours=96.0)
    assert [float(r[0]) for r in out] == [10.0, 48.0] and not dummy
    out, _ = loader_rows(rows, "drop", full_stay_hours=48.0)  # t == upper is dropped
    assert [float(r[0]) for r in out] == [10.0]
    out, _ = loader_rows(rows, "keep", full_stay_hours=48.0)
    assert [float(r[0]) for r in out] == [0.0, 10.0, 48.0]


def test_binning_edges_and_last_value_wins(res):
    ev = [
        (0.5, HR, "61"),
        (1.0, HR, "62"),  # both bin 0 (int(1 - 1e-6) = 0); later wins
        (1.0 + 1 / 60, HR, "63"),  # bin 1
        (2.1, HR, "70"),
        (2.9, HR, "75"),  # bin 2, later wins
        (47.5, HR, "99"),
    ]  # bin 47
    r = kept(make_subject(events=ev), res)
    hr, m = hr_raw(r, res), hr_mask(r, res)
    assert hr[0] == pytest.approx(62, abs=1e-3)
    assert hr[1] == pytest.approx(63, abs=1e-3)
    assert hr[2] == pytest.approx(75, abs=1e-3)
    assert hr[47] == pytest.approx(99, abs=1e-3)
    assert m[[0, 1, 2, 47]].tolist() == [1, 1, 1, 1] and m.sum() == 4


def test_same_timestamp_duplicates_keep_medmod_sort_max(res):
    # subject.py:59-61 keeps the last value after sorting; values are strings
    # here (the GCS text makes pandas read the column as str), so "99" > "100".
    ev = [(3.0, HR, "100"), (3.0, HR, "99"), (4.0, GCS_EYE, "To Speech")]
    assert hr_raw(kept(make_subject(events=ev), res), res)[3] == pytest.approx(
        99, abs=1e-3
    )
    # all-numeric subject: numeric sort, so 100 wins
    ev = [(3.0, HR, "100"), (3.0, HR, "99")]
    assert hr_raw(kept(make_subject(events=ev), res), res)[3] == pytest.approx(
        100, abs=1e-3
    )


def test_imputation_normal_then_carry_forward(res):
    r = kept(make_subject(events=[(10.5, HR, "120"), (20.5, HR, "80")]), res)
    hr, m = hr_raw(r, res), hr_mask(r, res)
    np.testing.assert_allclose(hr[:10], NORMAL_HR, atol=1e-3)
    np.testing.assert_allclose(hr[10:20], 120, atol=1e-3)
    np.testing.assert_allclose(hr[20:], 80, atol=1e-3)
    assert np.flatnonzero(m).tolist() == [10, 20]


def test_categorical_imputation_carries_one_hot(res):
    r = kept(make_subject(events=[(5.5, GCS_EYE, "To Speech")]), res)
    eye_cols = [
        col(res, f"Glascow coma scale eye opening->{v}")
        for v in res.discretizer._possible_values["Glascow coma scale eye opening"]
    ]
    block = r.ehr[:, eye_cols]
    normal = eye_cols.index(col(res, "Glascow coma scale eye opening->4 Spontaneously"))
    speech = eye_cols.index(col(res, "Glascow coma scale eye opening->To Speech"))
    assert (block[:5, normal] == 1).all() and block[:5].sum() == 5
    assert (block[5:, speech] == 1).all() and block[5:].sum() == 43
    assert (
        r.ehr[:, col(res, "mask->Glascow coma scale eye opening")].tolist()
        == [0] * 5 + [1] + [0] * 42
    )


def test_one_hot_layout_and_masks(res):
    h = res.discretizer.output_header()
    assert len(h) == 76
    assert len(res.discretizer.continuous_columns()) == 12
    assert sum("->" in x and not x.startswith("mask->") for x in h) == 47
    assert h[59:] == [f"mask->{c}" for c in res.discretizer.channels]
    r = kept(
        make_subject(
            events=[(2.5, GCS_EYE, "None"), (2.5, CRR, "Delayed"), (2.5, HR, "90")]
        ),
        res,
    )
    assert (
        r.ehr[2, col(res, "Glascow coma scale eye opening->None")] == 1
    )  # "None" kept (pandas 1.1.5)
    assert r.ehr[2, col(res, "Capillary refill rate->1.0")] == 1
    assert r.ehr[2, col(res, "Capillary refill rate->0.0")] == 0
    onehot = [i for i, x in enumerate(h) if "->" in x and not x.startswith("mask->")]
    assert set(np.unique(r.ehr[:, onehot])) <= {0.0, 1.0}
    masks = r.ehr[:, 59:]
    assert set(np.unique(masks)) <= {0.0, 1.0}
    observed = {res.discretizer.channels[j] for j in np.flatnonzero(masks[2])}
    assert observed == {
        "Glascow coma scale eye opening",
        "Capillary refill rate",
        "Heart Rate",
    }
    assert masks[[0, 1, 3]].sum() == 0


def test_unknown_categorical_value_raises_like_medmod(res):
    with pytest.raises(ValueError):
        run(make_subject(events=[(2.5, GCS_EYE, "Eyes closed")]), res)


def test_ehr_is_unnormalized_the_processor_zscores(res):
    pytest.importorskip("pyhealth")
    from pyhealth.datasets.medmod.discretizer import PH_NORMALIZER
    from pyhealth.processors.medmod_ehr_processor import MedModEHRProcessor

    r = kept(make_subject(events=[(2.5, HR, "90")]), res)
    c = col(res, "Heart Rate")
    assert r.ehr[2, c] == 90.0  # raw value in the task output
    proc = MedModEHRProcessor(PH_NORMALIZER)
    z = proc.process(r.ehr).numpy()
    expected = (90 - proc.normalizer.means[c]) / proc.normalizer.stds[c]
    assert z.dtype == np.float32 and z[2, c] == pytest.approx(expected, rel=1e-5)
    assert z[2, col(res, "mask->Heart Rate")] == 1  # masks not normalized


def test_value_cleaning_quirks(res):
    ev = [
        (2.5, TEMP_F, "98.6"),
        (3.5, TEMP_F, "78"),  # >= 79 -> Celsius; 78 left alone
        (2.5, FIO2, "50"),  # every FiO2 / 100
        (2.5, HEIGHT_IN, "70"),  # itemid 226707 not converted
        (2.5, SPO2, "0.95"),  # <= 1 -> * 100
        (2.5, GLUCOSE_LAB, "ERROR", "labevents"),  # non-numeric lab -> dropped
        (3.5, GLUCOSE_LAB, "130", "labevents"),
        (2.5, PH_LAB, "7.31", "labevents"),
    ]
    r = kept(make_subject(events=ev), res)
    x = raw(r.ehr, res)
    assert x[2, col(res, "Temperature")] == pytest.approx(37.0, abs=1e-3)
    assert x[3, col(res, "Temperature")] == pytest.approx(78.0, abs=1e-3)
    assert x[2, col(res, "Fraction inspired oxygen")] == pytest.approx(0.5, abs=1e-4)
    assert x[2, col(res, "Height")] == pytest.approx(70.0, abs=1e-3)
    assert x[2, col(res, "Oxygen saturation")] == pytest.approx(95.0, abs=1e-2)
    assert r.ehr[2, col(res, "mask->Glucose")] == 0
    assert x[3, col(res, "Glucose")] == pytest.approx(130, abs=1e-2)
    assert x[2, col(res, "pH")] == pytest.approx(7.31, abs=1e-4)


def test_na_strings_dropped_but_none_kept(res):
    r = kept(make_subject(events=[(2.5, HR, "NA"), (3.5, HR, "88")]), res)
    assert hr_mask(r, res)[2] == 0 and hr_mask(r, res)[3] == 1


def test_event_assignment(res):
    ev = [
        (2.5, HR, "70", "chartevents", STAY),
        (3.5, GLUCOSE_LAB, "150", "labevents", None),  # no stay_id: recovered from hadm
        (4.5, HR, "71", "chartevents", "39999999"),  # other stay_id -> dropped
        (5.5, GLUCOSE_LAB, "160", "labevents", None, None),
    ]  # no hadm_id -> dropped
    r = kept(make_subject(events=ev), res)
    assert r.ehr[3, col(res, "mask->Glucose")] == 1
    assert r.ehr[4, col(res, "mask->Heart Rate")] == 0
    assert r.ehr[5, col(res, "mask->Glucose")] == 0


def test_cohort_filters(res):
    ev = [(5.0, HR, "90")]
    r = run(make_subject(events=ev, stays=[stay_row(first="MICU", last="SICU")]), res)[
        STAY
    ]
    assert STAGES[r.passed] == "icustays"
    r = run(make_subject(events=ev, anchor_age="17"), res)[STAY]
    assert STAGES[r.passed] == "one_stay_per_admission"
    r = run(make_subject(events=ev, diagnoses=()), res)[STAY]
    assert STAGES[r.passed] == "event_in_48h"


def test_one_stay_per_admission_counted_after_transfer_filter(res):
    ev = [(5.0, HR, "90")]
    second = stay_row(
        stay="30000002", intime=INTIME + pd.Timedelta(days=5), first="CCU", last="SICU"
    )
    out = run(make_subject(events=ev, stays=[stay_row(), second]), res)
    assert out[STAY].kept  # the other stay was a transfer, removed first
    third = stay_row(stay="30000003", intime=INTIME + pd.Timedelta(days=5))
    out = run(make_subject(events=ev, stays=[stay_row(), third]), res)
    assert STAGES[out[STAY].passed] == "has_admission_and_patient"
    assert STAGES[out["30000003"].passed] == "has_admission_and_patient"


def test_labels(res):
    ev = [(5.0, HR, "90")]
    adm = lambda death: [
        {
            "hadm_id": HADM,
            "admittime": INTIME - pd.Timedelta(hours=2),
            "dischtime": str(INTIME + pd.Timedelta(days=8)),
            "deathtime": death,
        }
    ]
    assert (
        kept(
            make_subject(events=ev, admissions=adm(str(INTIME + pd.Timedelta(days=8)))),
            res,
        ).label
        == 1
    )
    assert (
        kept(make_subject(events=ev, dod=str(INTIME + pd.Timedelta(days=3))), res).label
        == 1
    )
    assert (
        kept(
            make_subject(events=ev, dod=str(INTIME + pd.Timedelta(days=30))), res
        ).label
        == 0
    )


def test_split_assignment(res):
    ev = [(5.0, HR, "90")]
    assert kept(make_subject(sid="10000002", events=ev), res).split == "test"
    assert kept(make_subject(sid="10000003", events=ev), res).split == "val"
    assert kept(make_subject(sid="10000004", events=ev), res).split == "train"


def test_split_file_parsing(tmp_path):
    from pyhealth.datasets.medmod.extraction import read_split_file

    p = tmp_path / "t.csv"
    p.write_text("111,0\n222,1\n333,1\n")
    assert read_split_file(p) == {"222", "333"}


def test_resources_read_split_lists_lazily(tmp_path, monkeypatch):
    from pyhealth.datasets.medmod import split_lists
    from pyhealth.datasets.medmod.extraction import MedModResources

    def no_network(*args, **kwargs):
        raise AssertionError("downloaded")

    monkeypatch.setattr(split_lists.urllib.request, "urlopen", no_network)
    r = MedModResources.load()  # no split list needed yet: no download
    assert r.sources["testset"].startswith("medmod@")
    (tmp_path / "t.csv").write_text("111,1\n222,0\n")
    (tmp_path / "v.csv").write_text("333,1\n")
    r = MedModResources.load(testset=tmp_path / "t.csv", valset=tmp_path / "v.csv")
    assert r.test_subjects == {"111"} and r.val_subjects == {"333"}


def test_normalizer_fit_matches_medmod_formula():
    from pyhealth.datasets.medmod.discretizer import Normalizer

    rng = np.random.default_rng(0)
    arrays = [rng.normal(3, 2, size=(int(rng.integers(1, 9)), 5)) for _ in range(30)]
    arrays[0][:, 4] = 7.0
    norm = Normalizer(fields=[0, 1, 4]).fit(iter(arrays))  # streamed, like PyHealth
    stacked = np.concatenate(arrays)
    np.testing.assert_allclose(norm.means, stacked.mean(axis=0))
    np.testing.assert_allclose(norm.stds[:4], stacked.std(axis=0, ddof=1)[:4])
    x = arrays[1]
    assert np.array_equal(norm.transform(x)[:, 2:4], x[:, 2:4])  # not in fields
    assert " at 0x" not in repr(norm) and "sha256:" in repr(norm)


def test_normalizer_fit_constant_column_gets_the_floor_not_nan():
    from pyhealth.datasets.medmod.discretizer import Normalizer

    x = np.full((7, 2), 0.1)  # 0.1 is inexact in binary: var rounds below 0
    x[:, 1] = 97.3
    norm = Normalizer().fit([x[:3], x[3:]])
    assert np.all(np.isfinite(norm.stds)) and norm.stds[0] == 1e-7


# --------------------------------------------------------- image_dir


def test_image_path_without_image_dir_is_pyhealth_path(res):
    r = run(make_subject(events=[(5.0, HR, "90")]), res)[STAY]
    assert r.kept
    assert r.image_path == f"/cxr/files/p10/p{SID}/s50000001/d1.jpg"


def test_image_path_with_image_dir(res, tmp_path):
    (tmp_path / "d1.jpg").write_bytes(b"x")
    r = run(make_subject(events=[(5.0, HR, "90")]), res, image_dir=tmp_path)[STAY]
    assert r.kept
    assert r.image_path == str(tmp_path / "d1.jpg")
    assert r.sample()["image_path"] == str(tmp_path / "d1.jpg")


def test_missing_resized_file_drops_stay_and_is_counted(res, tmp_path):
    ev = [(5.0, HR, "90")]
    (tmp_path / "d1.jpg").write_bytes(b"x")  # only subject 1's image was resized
    results = []
    for sid, dicom in [("10000001", "d1"), ("10000004", "d4")]:
        st = make_subject(sid=sid, events=ev, cxr=[cxr_row(10.0, dicom)])
        results += list(run(st, res, image_dir=tmp_path).values())
    by_sid = {r.subject_id: r for r in results}
    assert by_sid["10000001"].kept
    missing = by_sid["10000004"]
    assert not missing.kept and missing.image_path is None and missing.ehr is None
    assert (
        STAGES[missing.passed] == "cxr_has_chexpert"
    )  # dropped at resized_image_found
    chexpert = STAGES.index("cxr_has_chexpert")
    assert sum(r.passed >= chexpert for r in results) == 2
    assert sum(r.kept for r in results) == 1  # passed resized_image_found


def test_no_fallback_to_older_resized_image(res, tmp_path):
    (tmp_path / "early.jpg").write_bytes(b"x")  # only the older image exists
    cxr = [cxr_row(2.0, "early", "50000001"), cxr_row(30.0, "late", "50000002")]
    st = make_subject(
        events=[(5.0, HR, "90")], cxr=cxr, chexpert=("50000001", "50000002")
    )
    r = run(st, res, image_dir=tmp_path)[STAY]
    assert not r.kept and r.dicom_id == "late"


def test_without_image_dir_new_stage_is_passed_automatically(res):
    results = list(run(make_subject(events=[(5.0, HR, "90")]), res).values())
    assert len(results) == 1 and results[0].kept
    assert results[0].passed == STAGES.index("resized_image_found")


# ------------------------------------------------------------- split


def test_split_positions_keep_sample_order_and_reject_unknown_splits():
    from pyhealth.datasets.medmod_mimic4 import split_positions

    out = split_positions(["train", "test", "train", "val", "test"])
    assert out == {"train": [0, 2], "val": [3], "test": [1, 4]}
    assert split_positions([]) == {"train": [], "val": [], "test": []}
    with pytest.raises(ValueError, match="unknown split"):
        split_positions(["train", "holdout"])


def test_split_by_medmod_returns_subset_views_like_pyhealth():
    from pyhealth.datasets.medmod_mimic4 import split_by_medmod

    class FakeSamples:  # the two methods split_by_medmod uses
        def __init__(self, splits):
            self.splits = splits

        def __len__(self):
            return len(self.splits)

        def __getitem__(self, i):
            return {"split": self.splits[i]}

        def subset(self, indices):
            return [self.splits[i] for i in indices]

    train, val, test = split_by_medmod(FakeSamples(["test", "train", "val", "train"]))
    assert (train, val, test) == (["train", "train"], ["val"], ["test"])
    with pytest.warns(UserWarning, match="no samples"):
        train, val, test = split_by_medmod(FakeSamples(["train", "train"]))
    assert (len(train), len(val), len(test)) == (2, 0, 0)


def test_resources_repr_is_short_and_stable():
    from pyhealth.datasets.medmod.extraction import MedModResources

    a, b = repr(MedModResources.load()), repr(MedModResources.load())
    assert a == b and "bundled:itemid_to_variable_map.csv" in a
    assert "DataFrame" not in a and " at 0x" not in a and len(a) < 600
