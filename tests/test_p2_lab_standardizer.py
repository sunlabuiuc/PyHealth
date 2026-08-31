"""Proof that lab z-scores fit on observed train rows, not a WORLD_SIZE shard.

``SampleDataset`` subclasses ``litdata.StreamingDataset``. Under ``torchrun``,
``WORLD_SIZE`` is set before ``torch.distributed`` is initialised, so
``__len__`` / ``__iter__`` silently yield 1/N of the train split (the same
shard on every rank). Measured on real litdata with 20 samples: ``len()``
reports 5 under ``WORLD_SIZE=4`` while ``region_of_interest`` still sums to 20.

Fitting padded 0.0 as if it were a measurement also moves sodium's mean from
140 to 105. ``patient_to_index`` is unusable after ``subset()``: it still
holds parent indices and raised ``ValueError: index 237 didn't find a match
within the chunk intervals``.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p2_lab_standardizer.py -q
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch


def _lab_samples(n: int = 40):
    torch.manual_seed(0)
    return [
        {
            "labs": torch.stack(
                [140.0 + torch.randn(1) * 4, 1.0 + torch.randn(1) * 0.2]
            ).view(1, 2),
            "labs_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        for _ in range(n)
    ]


class TestP2LabStandardizer(unittest.TestCase):
    def test_fit_ignores_padded_zeros(self):
        from pyhealth.processors import fit_lab_standardizer

        samples = [
            {
                "labs": torch.tensor([[140.0, 1.0], [0.0, 0.0]]),
                "labs_mask": torch.tensor([[True, True], [False, False]]),
            },
            {
                "labs": torch.tensor([[142.0, 1.2], [138.0, 0.8]]),
                "labs_mask": torch.tensor([[True, True], [True, True]]),
            },
        ]
        standardizer = fit_lab_standardizer(samples)
        # Observed sodium 140, 142, 138. Mean 140, not 105 from the padded 0.0.
        self.assertAlmostEqual(standardizer.mean[0].item(), 140.0, places=4)

    def test_unobserved_slot_maps_to_zero(self):
        from pyhealth.processors import fit_lab_standardizer

        standardizer = fit_lab_standardizer(_lab_samples())
        values = torch.tensor([[[140.0, 1.0], [0.0, 0.0]]])
        observed = torch.tensor([[[True, True], [False, False]]])
        out = standardizer(values, observed)
        self.assertEqual(out[0, 1].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(out).all())

    def test_world_size_does_not_shrink_the_fit(self):
        from pyhealth.processors import fit_lab_standardizer

        samples = _lab_samples(40)

        class _ShardedByWorldSize:
            def __init__(self, records):
                self._records = records
                self.region_of_interest = [(0, len(records))]

            def _visible(self):
                world = int(os.environ.get("WORLD_SIZE", "1"))
                return self._records[: len(self._records) // world]

            def __len__(self):
                return len(self._visible())

            def __iter__(self):
                return iter(self._visible())

            def __getitem__(self, index):
                return self._records[index]

        dataset = _ShardedByWorldSize(samples)
        single = fit_lab_standardizer(dataset)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            sharded = fit_lab_standardizer(dataset)
        self.assertTrue(torch.allclose(single.mean, sharded.mean))
        self.assertTrue(torch.allclose(single.std, sharded.std))
        self.assertEqual(
            int(single.observed_count.sum()), int(sharded.observed_count.sum())
        )

    def test_statistics_travel_in_the_state_dict(self):
        from pyhealth.processors import fit_lab_standardizer

        standardizer = fit_lab_standardizer(_lab_samples())
        keys = set(standardizer.state_dict())
        self.assertTrue({"mean", "std"} <= keys)
        self.assertEqual(tuple(standardizer.state_dict()["mean"].shape), (2,))
