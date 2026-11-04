"""Tests for foreground patch sampling."""

import unittest

import numpy as np

from pyhealth.sampler.patch_sampler import PatchSampler


class TestPatchSampler(unittest.TestCase):
    def test_foreground_patch_contains_foreground(self):
        sampler = PatchSampler((4, 4, 4), foreground_prob=1.0)
        mask = np.zeros((10, 10, 10), dtype=np.int64)
        mask[1, 9, 0] = 1
        for _ in range(20):
            location = sampler.sample_location((10, 10, 10), mask)
            patch = mask[
                location.origin[0] : location.origin[0] + 4,
                location.origin[1] : location.origin[1] + 4,
                location.origin[2] : location.origin[2] + 4,
            ]
            self.assertGreater(int(patch.sum()), 0)

    def test_random_fallback_without_foreground(self):
        sampler = PatchSampler((4, 4, 4), foreground_prob=1.0)
        mask = np.zeros((10, 10, 10), dtype=np.int64)
        location = sampler.sample_location((10, 10, 10), mask)
        self.assertEqual(len(location.origin), 3)
        self.assertIsNone(location.slice_index)

    def test_2d_patch_from_3d_volume(self):
        sampler = PatchSampler((4, 4), foreground_prob=1.0)
        mask = np.zeros((5, 10, 10), dtype=np.int64)
        mask[3, 6, 7] = 1
        location = sampler.sample_location((5, 10, 10), mask)
        self.assertIsNotNone(location.slice_index)
        self.assertEqual(location.slice_index, 3)
        self.assertEqual(len(location.origin), 2)

    def test_center_location(self):
        sampler = PatchSampler((4, 4, 4))
        location = sampler.center_location((10, 10, 10))
        self.assertEqual(location.origin, (3, 3, 3))
        self.assertIsNone(location.slice_index)


if __name__ == "__main__":
    unittest.main()
