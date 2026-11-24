import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.models import SelectiveSSM
from pyhealth.models.armr import SelectiveSSM as ARMRSelectiveSSM
from pyhealth.models.selective_ssm import selective_scan


class _ARMRReferenceSSM(nn.Module):
    """ARMR's SelectiveSSM before it moved to selective_ssm.py (#1371)."""

    def __init__(self, dim, state_size=16, expand=2, conv_size=4):
        super().__init__()
        inner = expand * dim
        self.dt_rank = math.ceil(dim / 16)
        self.state_size = state_size
        self.in_proj = nn.Linear(dim, 2 * inner)
        self.conv = nn.Conv1d(
            inner, inner, conv_size, groups=inner, padding=conv_size - 1
        )
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        a = torch.arange(1, state_size + 1, dtype=torch.float).repeat(inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, x):
        steps = x.size(1)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :steps].transpose(1, 2)
        u = F.silu(u)
        dt, b, c = self.x_proj(u).split(
            [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        a = -torch.exp(self.a_log)
        da = torch.exp(dt.unsqueeze(-1) * a)
        dbu = dt.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
        h = torch.zeros_like(da[:, 0])
        ys = []
        for t in range(steps):
            h = da[:, t] * h + dbu[:, t]
            ys.append((h * c[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.d
        return self.out_proj(y * F.silu(gate))


class TestSelectiveSSM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.block = SelectiveSSM(dim=8, state_size=4)
        self.x = torch.randn(3, 12, 8)
        self.mask = torch.arange(12)[None, :] < torch.tensor([[12], [7], [1]])

    def _parallel(self):
        block = SelectiveSSM(dim=8, state_size=4, scan="parallel", chunk_size=5)
        block.load_state_dict(self.block.state_dict())
        return block

    def test_one_class_shared_with_armr(self):
        self.assertIs(ARMRSelectiveSSM, SelectiveSSM)

    def test_default_matches_the_previous_armr_block(self):
        reference = _ARMRReferenceSSM(dim=8, state_size=4)
        reference.load_state_dict(self.block.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(self.block(self.x), reference(self.x)))

    def test_parallel_scan_matches_the_loop(self):
        parallel = self._parallel()
        with torch.no_grad():
            torch.testing.assert_close(parallel(self.x), self.block(self.x),
                                       atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(parallel(self.x, self.mask),
                                       self.block(self.x, self.mask),
                                       atol=1e-5, rtol=1e-4)

    def test_mask_ignores_padding(self):
        for block in (self.block, self._parallel()):
            with torch.no_grad():
                out = block(self.x, self.mask)
                self.assertTrue((out[~self.mask] == 0).all())
                for i, length in enumerate((12, 7, 1)):
                    alone = block(self.x[i : i + 1, :length])
                    torch.testing.assert_close(out[i : i + 1, :length], alone,
                                               atol=1e-5, rtol=1e-4)

    def test_mixer_is_causal(self):
        changed = self.x.clone()
        changed[:, 6:] = torch.randn(3, 6, 8)
        for block in (self.block, self._parallel()):
            with torch.no_grad():
                torch.testing.assert_close(block(self.x)[:, :6],
                                           block(changed)[:, :6])

    def test_invalid_scan(self):
        with self.assertRaises(ValueError):
            SelectiveSSM(dim=8, scan="fast")


class TestSelectiveScan(unittest.TestCase):
    def test_matches_the_recurrence(self):
        torch.manual_seed(0)
        log_decay = -torch.rand(2, 50, 3, 4) * 3
        drive = torch.randn(2, 50, 3, 4)
        state = torch.zeros(2, 3, 4)
        expected = []
        for t in range(50):
            state = torch.exp(log_decay[:, t]) * state + drive[:, t]
            expected.append(state)
        for chunk_size in (1, 7, 32, 64):
            torch.testing.assert_close(
                selective_scan(log_decay, drive, chunk_size),
                torch.stack(expected, 1), atol=1e-4, rtol=1e-4,
            )

    def test_strong_decay_stays_finite(self):
        log_decay = torch.full((1, 400, 2), -20.0, requires_grad=True)
        drive = torch.randn(1, 400, 2, requires_grad=True)
        states = selective_scan(log_decay, drive)
        torch.testing.assert_close(states, drive.detach(), atol=1e-4, rtol=1e-3)
        states.sum().backward()
        self.assertTrue(torch.isfinite(log_decay.grad).all())
        self.assertTrue(torch.isfinite(drive.grad).all())


if __name__ == "__main__":
    unittest.main()
