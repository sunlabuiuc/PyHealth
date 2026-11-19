"""MedMod in-hospital mortality: synthetic samples with the task's contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyhealth.datasets.medmod.discretizer import medmod_ehr_funcs
from pyhealth.datasets.medmod.extraction import StayResult
from pyhealth.datasets.medmod.synthetic import (
    make_synthetic_samples,
    make_synthetic_stays,
)


DISC, NORM = medmod_ehr_funcs()
HEADER = DISC.output_header()


@pytest.fixture(scope="module")
def keep(tmp_path_factory):
    return make_synthetic_samples(
        "keep", tmp_path_factory.mktemp("syn"), n_stays=120, seed=3
    )


def test_keys_and_types_match_the_real_task(keep):
    real_keys = set(
        StayResult("p", "s", "h", passed=0, label=0, split="train").sample()
    )
    for s in keep:
        assert set(s) == real_keys
        # what the task returns: unnormalized float64, 1-hour bins by default
        assert s["ehr"].dtype == np.float64 and s["ehr"].shape == (48, 76)
        assert isinstance(s["patient_id"], str) and s["patient_id"].startswith("syn-")
        assert isinstance(s["visit_id"], str) and s["visit_id"].startswith("syn-stay-")
        assert s["label"] in (0, 1) and s["split"] in ("train", "val", "test")


def test_ehr_layout_is_consistent(keep):
    masks = [i for i, n in enumerate(HEADER) if n.startswith("mask->")]
    assert len(masks) == 17 and len(DISC.continuous_columns()) == 12
    for s in keep:
        x = s["ehr"]
        assert np.isfinite(x).all()
        assert set(np.unique(x[:, masks])) <= {0.0, 1.0}
        for ch in DISC.channels:
            cols = [i for i, n in enumerate(HEADER) if n.startswith(ch + "->")]
            if cols:  # categorical: exactly one hot per hour (observed or imputed)
                assert (x[:, cols].sum(axis=1) == 1).all(), ch
    observed = np.stack([s["ehr"][:, masks] for s in keep]).mean()
    assert 0.05 < observed < 0.9  # some gaps, some observations


def test_splits_per_patient_both_labels_and_rates(keep):
    by_patient = {}
    for s in keep:
        by_patient.setdefault(s["patient_id"], set()).add(s["split"])
    assert all(len(v) == 1 for v in by_patient.values())
    for split in ("train", "val", "test"):
        labels = {s["label"] for s in keep if s["split"] == split}
        assert labels == {0, 1}, split
    assert sum(s["split"] == "train" for s in keep) > sum(
        s["split"] == "test" for s in keep
    )


def test_dummy_images(keep):
    for s in keep[:5]:
        with Image.open(s["image_path"]) as img:
            assert img.format == "JPEG" and img.mode == "L" and img.size[0] == 512


def test_deterministic_and_drop_keep_differ_only_with_a_row_at_hour_zero(tmp_path):
    keep_a = make_synthetic_samples("keep", tmp_path, n_stays=60, seed=7)
    keep_b = make_synthetic_samples("keep", tmp_path, n_stays=60, seed=7)
    drop = make_synthetic_samples("drop", tmp_path, n_stays=60, seed=7)
    stays = make_synthetic_stays(60, seed=7)
    assert all(np.array_equal(a["ehr"], b["ehr"]) for a, b in zip(keep_a, keep_b))
    for k, d, st in zip(keep_a, drop, stays):
        assert (k["visit_id"], k["label"], k["split"]) == (
            d["visit_id"],
            d["label"],
            d["split"],
        )
        has_zero = any(float(r[0]) == 0.0 for r in st.rows)
        if not has_zero:  # only a row at Hours == 0 can make drop and keep differ
            assert np.array_equal(k["ehr"], d["ehr"])
    # ...and some stays with such a row do differ (others don't: a later value in
    # the first hour overwrites it, "later value wins", as in real data)
    differ = [not np.array_equal(k["ehr"], d["ehr"]) for k, d in zip(keep_a, drop)]
    assert any(differ) and sum(differ) <= sum(
        any(float(r[0]) == 0.0 for r in st.rows) for st in stays
    )


def test_hours_zero_is_required(tmp_path):
    with pytest.raises(ValueError):
        make_synthetic_samples("maybe", tmp_path, n_stays=5)


def test_create_synthetic_samples_works_like_the_real_pipeline(tmp_path):
    pytest.importorskip("pyhealth")
    import torch
    from pyhealth.datasets import get_dataloader

    from pyhealth.datasets import split_by_medmod
    from pyhealth.datasets.medmod.synthetic import create_synthetic_samples

    samples = create_synthetic_samples("keep", n_stays=60, seed=1, out_dir=tmp_path)
    train, val, test = split_by_medmod(samples)
    assert len(train) + len(val) + len(test) == len(samples) == 60
    s = train[0]
    assert s["split"] == "train" and tuple(s["ehr"].shape) == (48, 76)
    assert s["ehr"].dtype == torch.float32
    proc = samples.input_processors["ehr"]  # the released code's file
    assert not proc.learns_statistics and proc.normalizer_file.endswith(".normalizer")
    assert Path(s["image_path"]).exists()
    assert set(s) >= {"patient_id", "visit_id", "ehr", "image_path", "label", "split"}
    batch = next(iter(get_dataloader(train, batch_size=8, shuffle=True, seed=0)))
    assert tuple(batch["ehr"].shape) == (8, 48, 76)

    # 2-hour bins, normalizer fitted on the train split only
    other = create_synthetic_samples(
        "keep", n_stays=60, seed=1, out_dir=tmp_path, timestep=2.0, normalizer=None
    )
    ptrain, _, ptest = split_by_medmod(other)
    assert tuple(ptrain[0]["ehr"].shape) == (24, 76)
    assert other.input_processors["ehr"].learns_statistics
    cont = DISC.continuous_columns()
    x = np.concatenate([np.asarray(ptrain[i]["ehr"]) for i in range(len(ptrain))])
    np.testing.assert_allclose(x[:, cont].mean(axis=0), 0, atol=1e-4)
    xt = np.concatenate([np.asarray(ptest[i]["ehr"]) for i in range(len(ptest))])
    assert np.abs(xt[:, cont].mean(axis=0)).max() > 1e-3  # not fitted on test


def test_rows_are_never_empty_like_real_pivot_rows():
    for st in make_synthetic_stays(80, seed=11):
        assert all(any(v != "" for v in r[1:]) for r in st.rows)


def test_synthetic_pretraining_samples_vary_in_length(tmp_path):
    from pyhealth.datasets.medmod.synthetic import make_synthetic_pretraining_samples

    raw = make_synthetic_pretraining_samples(tmp_path, n_stays=40, seed=2)
    assert len(raw) == 40
    assert {s["split"] for s in raw} <= {"train", "val"}
    bins = [s["ehr"].shape[0] for s in raw]
    assert len(set(bins)) > 10 and min(bins) >= 6 and max(bins) <= 481
    assert all(s["ehr"].shape[1] == 76 for s in raw)
    assert all(Path(s["image_path"]).exists() for s in raw)
    again = make_synthetic_pretraining_samples(tmp_path, n_stays=40, seed=2)
    assert all(np.array_equal(a["ehr"], b["ehr"]) for a, b in zip(raw, again))


def test_synthetic_pretraining_batches_carry_true_lengths(tmp_path):
    from pyhealth.datasets import get_dataloader, split_by_medmod
    from pyhealth.datasets.medmod.synthetic import (
        create_synthetic_pretraining_samples,
    )

    samples = create_synthetic_pretraining_samples(30, seed=1, out_dir=tmp_path)
    train, val, test = split_by_medmod(samples)
    assert len(train) + len(val) == 30 and len(test) == 0
    ehr, lengths = next(iter(get_dataloader(train, batch_size=8)))["ehr"]
    assert ehr.shape[0] == len(lengths) == 8 and ehr.shape[2] == 76
    assert ehr.shape[1] == int(lengths.max())
    for i, n in enumerate(lengths.tolist()):
        assert not ehr[i, n:].any()
