import math
import unittest

import torch

from pyhealth.models import SinusoidalTimeEmbedding


class TestSinusoidalTimeEmbedding(unittest.TestCase):
    def test_dim_two_is_finite(self):
        embedding = SinusoidalTimeEmbedding(dim=2, max_hours=24.0)

        result = embedding(torch.tensor([0.0, 6.0, 24.0]))

        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(torch.isfinite(result).all())

    def test_standard_dimension_values_are_unchanged(self):
        embedding = SinusoidalTimeEmbedding(dim=6, max_hours=24.0)
        time = torch.tensor([6.0])
        frequencies = torch.tensor([1.0, 0.01, 0.0001])
        arguments = time.unsqueeze(-1) / 24.0 * 2 * math.pi * frequencies
        expected = torch.cat([arguments.sin(), arguments.cos()], dim=-1)

        self.assertTrue(torch.allclose(embedding(time), expected))


if __name__ == "__main__":
    unittest.main()
