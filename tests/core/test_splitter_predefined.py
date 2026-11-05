import numpy as np
import torch
from pyhealth.datasets.splitter import split_by_predefined


class FakeDataset:
    def __init__(self, samples):
        self.samples = samples
        self.patient_to_index = {}
        for i, s in enumerate(samples):
            self.patient_to_index.setdefault(s["patient_id"], []).append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def subset(self, indices):
        return [int(i) for i in indices]


def make_ds():
    s = [{"patient_id": f"t{p}", "split": "train"} for p in range(10) for _ in range(3)]
    s += [{"patient_id": f"e{p}", "split": "eval"} for p in range(4) for _ in range(2)]
    return FakeDataset(s)


def test_test_set_is_eval_partition():
    ds = make_ds()
    *_, test = split_by_predefined(ds, [0.8, 0.2], seed=0)
    assert all(ds[i]["split"] == "eval" for i in test) and len(test) == 8


def test_patient_level_has_no_leakage():
    ds = make_ds()
    train, val, cal, test = split_by_predefined(ds, [0.6, 0.2, 0.2], seed=0, by_patient=True)
    groups = [{ds[i]["patient_id"] for i in part} for part in (train, val, cal, test)]
    for a in range(4):
        for b in range(a + 1, 4):
            assert groups[a].isdisjoint(groups[b])


def test_seed_is_deterministic_and_get_index():
    ds = make_ds()
    a = split_by_predefined(ds, [0.8, 0.2], seed=1, get_index=True)
    b = split_by_predefined(ds, [0.8, 0.2], seed=1, get_index=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert sum(len(p) for p in a) == len(ds)
