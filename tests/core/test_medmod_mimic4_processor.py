"""MedMod in-hospital mortality: MedModSplit (a PyHealth Split) and
MedModEHRProcessor (MedMod's normalizer file, or one fitted on the train split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhealth.datasets.medmod.discretizer import (
    PH_NORMALIZER,
    Normalizer,
    medmod_ehr_funcs,
)
from pyhealth.datasets.medmod_mimic4 import MedModSplit
from pyhealth.processors.medmod_ehr_processor import MedModEHRProcessor


DISC, PH = medmod_ehr_funcs()
CONT = DISC.continuous_columns()


def lists(tmp_path, test=(), val=()):
    out = []
    for name, ids in (("test", test), ("val", val)):
        f = tmp_path / f"{name}set.csv"
        f.write_text("".join(f"{i},1\n" for i in ids) + "99999999,0\n")
        out.append(str(f))
    return out


def raw_ehr(rng, bins=24, shift=0.0):
    x = np.zeros((bins, 76))
    x[:, CONT] = rng.normal(50 + shift, 10, size=(bins, len(CONT)))
    return x


def test_medmod_split_follows_the_lists(tmp_path):
    testset, valset = lists(tmp_path, test=["2"], val=["3", "2"])
    split = MedModSplit(testset=testset, valset=valset)
    parts = split.split_indices({"1": [0, 4], "2": [1], "3": [3], "5": [2]})
    # test wins over val, like MedMod (test list first, then val among the rest)
    assert [p.tolist() for p in parts] == [[0, 2, 4], [3], [1]]
    assert all(p.dtype == np.int64 for p in parts)

    before = split.to_dict()
    Path(valset).write_text("3,1\n")
    assert split.to_dict() != before  # cache key follows the list contents
    assert before["kind"] == "medmod"


class _Response:
    def __init__(self, data: bytes):
        self.data = data

    def read(self) -> bytes:
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_split_list_download_checksum_cache_and_fallback(tmp_path, monkeypatch):
    import hashlib

    from pyhealth.datasets.medmod import split_lists as sl

    good = b"111,1\n222,0\n"
    monkeypatch.setitem(sl.SHA256, "testset_iv.csv", hashlib.sha256(good).hexdigest())
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        if "/MedMod/" in url:
            raise OSError("offline")  # first source fails: falls back to MedFuse
        return _Response(good)

    monkeypatch.setattr(sl.urllib.request, "urlopen", fake_urlopen)
    path = sl.split_list_path("testset_iv.csv", cache_dir=tmp_path)
    assert path.read_bytes() == good and len(calls) == 2
    assert all(sl.MEDMOD_COMMIT in u or sl.MEDFUSE_COMMIT in u for u in calls)
    sl.split_list_path("testset_iv.csv", cache_dir=tmp_path)  # cached
    assert len(calls) == 2

    path.write_bytes(b"tampered")  # a bad cache is fetched again, and checked
    monkeypatch.setattr(
        sl.urllib.request, "urlopen", lambda url, timeout: _Response(b"wrong")
    )
    with pytest.raises(OSError, match="SHA-256"):
        sl.split_list_path("testset_iv.csv", cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.partial"))


def test_default_medmod_split_cache_key_needs_no_download(monkeypatch):
    from pyhealth.datasets.medmod import split_lists as sl

    def no_network(*args, **kwargs):
        raise AssertionError("downloaded")

    monkeypatch.setattr(sl.urllib.request, "urlopen", no_network)
    key = MedModSplit().to_dict()
    assert key["testset_sha256"] == sl.SHA256["testset_iv.csv"][:16]
    assert key["valset_sha256"] == sl.SHA256["valset_iv.csv"][:16]


def test_processor_is_fitted_on_the_train_split_only(tmp_path):
    """PyHealth's SampleBuilder with MedModSplit, as set_task does it."""
    from pyhealth.datasets.sample_dataset import SampleBuilder

    rng = np.random.default_rng(0)
    testset, valset = lists(tmp_path, test=["t1", "t2"], val=["v1"])
    samples, train_arrays = [], []
    for pid, split, shift, label in [
        ("a", "train", 0, 0), ("b", "train", 0, 1), ("c", "train", 0, 0),
        ("v1", "val", 30, 1), ("t1", "test", 60, 0), ("t2", "test", 60, 1),
    ]:  # fmt: skip
        x = raw_ehr(rng, shift=shift)
        samples.append({"patient_id": pid, "ehr": x, "label": label, "split": split})
        if split == "train":
            train_arrays.append(x)

    builder = SampleBuilder(
        input_schema={"ehr": (MedModEHRProcessor, {"normalizer": None})},
        output_schema={"label": "binary"},
    )
    builder.fit(samples, split=MedModSplit(testset=testset, valset=valset))
    proc = builder.input_processors["ehr"]
    want = Normalizer(fields=CONT).fit(train_arrays)
    np.testing.assert_array_equal(proc.normalizer.means, want.means)
    np.testing.assert_array_equal(proc.normalizer.stds, want.stds)
    everyone = Normalizer(fields=CONT).fit(s["ehr"] for s in samples)
    assert not np.allclose(proc.normalizer.means[CONT], everyone.means[CONT])
    assert [p.tolist() for p in builder.split_indices] == [[0, 1, 2], [3], [4, 5]]


def test_processor_output_matches_medmods_arithmetic():
    """float64 z-score, then float32 (MedMod's .float()): what the released-code
    samples always were, so compare_medmod's bit-identity still holds."""
    rng = np.random.default_rng(1)
    x = raw_ehr(rng, bins=48)
    proc = MedModEHRProcessor(PH_NORMALIZER)
    got = proc.process(x)
    assert got.dtype.is_floating_point and got.dtype.itemsize == 4
    np.testing.assert_array_equal(got.numpy(), PH.transform(x).astype(np.float32))
    assert not proc.learns_statistics  # a fixed file learns nothing
    proc.fit([{"ehr": raw_ehr(rng)}], "ehr")  # no-op
    np.testing.assert_array_equal(proc.process(x).numpy(), got.numpy())


def test_processor_cache_key_is_stable():
    """With pre-fitted processors PyHealth keys its cache on
    json.dumps(vars(processor), default=str) (base_dataset.py:1207-1230)."""
    import json

    def key(p):
        return json.dumps(vars(p), default=str, sort_keys=True)

    a, b = MedModEHRProcessor(PH_NORMALIZER), MedModEHRProcessor(PH_NORMALIZER)
    assert key(a) == key(b) and " at 0x" not in key(a)
    fitted = MedModEHRProcessor()
    fitted.fit([{"ehr": raw_ehr(np.random.default_rng(2))}], "ehr")
    assert "sha256:" in key(fitted) and " at 0x" not in key(fitted)
    assert fitted.schema() == ("value",) and fitted.dim() == (2,)
    assert fitted.is_token() is False


def test_sequence_processor_pads_at_the_end_and_returns_lengths():
    import torch

    from pyhealth.processors.medmod_ehr_processor import MedModEHRSequenceProcessor

    rng = np.random.default_rng(3)
    seq = MedModEHRSequenceProcessor(PH_NORMALIZER)
    raw = [rng.normal(size=(n, 76)) for n in (4, 9, 1)]
    values = [seq.process(x) for x in raw]
    # each stay standardized exactly as the fixed-length processor does
    fixed = MedModEHRProcessor(PH_NORMALIZER)
    for v, x in zip(values, raw):
        assert torch.equal(v, fixed.process(x))
    ehr, lengths = seq.collate(values)
    assert lengths.tolist() == [4, 9, 1] and lengths.dtype == torch.long
    assert tuple(ehr.shape) == (3, 9, 76) and ehr.dtype == torch.float32
    for i, n in enumerate((4, 9, 1)):  # MedMod pad_zeros: zeros after the stay
        assert torch.equal(ehr[i, :n], values[i])
        assert not ehr[i, n:].any()
    assert seq.schema() == ("value", "lengths")
