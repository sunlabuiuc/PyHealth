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
        # Wavelengths run 1h..24h here; the old t/720 formulation wrapped
        # every frequency every 30 days, aliasing later stays onto earlier ones.
        embedding = SinusoidalTimeEmbedding(dim=6, max_hours=24.0)
        time = torch.tensor([6.0])
        periods = torch.exp(torch.linspace(math.log(1.0), math.log(24.0), 3))
        arguments = time.unsqueeze(-1) * (2 * math.pi / periods)
        expected = torch.cat([arguments.sin(), arguments.cos()], dim=-1)

        self.assertTrue(torch.allclose(embedding(time), expected))

    def test_long_span_stays_monotonic_in_phase(self):
        embedding = SinusoidalTimeEmbedding(dim=6)
        early = embedding(torch.tensor([6.0]))
        late = embedding(torch.tensor([9606.0]))

        self.assertFalse(torch.allclose(early, late))

    def test_invalid_span_raises(self):
        with self.assertRaises(ValueError):
            SinusoidalTimeEmbedding(dim=6, max_hours=1.0, min_hours=24.0)


if __name__ == "__main__":
    unittest.main()
