"""Tests for KEEP pipeline's device resolution logic.

Covers the ``resolve_device`` helper in ``run_pipeline.py`` that lets
callers pass ``device="auto"`` and have it resolved to the best available
torch device (cuda > mps > cpu) at runtime.

Uses mocks instead of depending on whichever GPU happens to be available
in the test environment, so the same tests pass on a laptop, H200, and
CPU-only CI.
"""

from unittest.mock import patch

import pytest

from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
    resolve_device,
)


# ---------------------------------------------------------------------------
# Explicit device values pass through unchanged
# ---------------------------------------------------------------------------

class TestExplicitDevicesPassThrough:
    """When device is not 'auto', resolve_device returns it unchanged."""

    def test_cuda_passes_through(self):
        """'cuda' stays 'cuda' regardless of what's actually available."""
        # Even if cuda is not available, an explicit 'cuda' should not be
        # silently downgraded — let the caller see the error downstream.
        assert resolve_device("cuda") == "cuda"

    def test_cpu_passes_through(self):
        """'cpu' stays 'cpu' (useful for forcing CPU-only reproducibility)."""
        assert resolve_device("cpu") == "cpu"

    def test_mps_passes_through(self):
        """'mps' stays 'mps' regardless of platform."""
        assert resolve_device("mps") == "mps"

    def test_unknown_device_passes_through(self):
        """Unknown values pass through; torch will raise later if invalid.

        This keeps the helper simple and lets torch own device validation.
        """
        assert resolve_device("cuda:0") == "cuda:0"
        assert resolve_device("cuda:1") == "cuda:1"


# ---------------------------------------------------------------------------
# 'auto' resolution — uses mocked torch.cuda / torch.backends.mps
# ---------------------------------------------------------------------------

class TestAutoResolution:
    """When device='auto', picks cuda > mps > cpu based on availability."""

    def test_auto_picks_cuda_when_available(self):
        """cuda is preferred even if mps is also available."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.backends.mps.is_available", return_value=True):
            assert resolve_device("auto") == "cuda"

    def test_auto_picks_mps_when_no_cuda(self):
        """mps is preferred over cpu when cuda is unavailable."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            assert resolve_device("auto") == "mps"

    def test_auto_falls_back_to_cpu(self):
        """cpu is the final fallback when neither cuda nor mps is available."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            assert resolve_device("auto") == "cpu"

    def test_auto_falls_back_to_cpu_when_mps_backend_missing(self):
        """Older torch builds without torch.backends.mps should not crash.

        On pre-1.12 torch (pre-Apple Silicon support), ``torch.backends.mps``
        may not exist at all. The helper uses ``getattr(..., None)`` to
        handle that gracefully.
        """
        import torch
        # Simulate missing torch.backends.mps by patching getattr to return None
        with patch("torch.cuda.is_available", return_value=False), \
             patch.object(torch.backends, "mps", None, create=True):
            # Even with mps backend missing, we should fall through to cpu
            # without raising AttributeError
            assert resolve_device("auto") == "cpu"


# ---------------------------------------------------------------------------
# Integration smoke — real torch, whatever device the test env has
# ---------------------------------------------------------------------------

class TestRealTorchIntegration:
    """Sanity check that 'auto' returns *something* valid on the real env."""

    def test_auto_returns_valid_device_string(self):
        """The resolved string must be one of cuda/mps/cpu."""
        resolved = resolve_device("auto")
        assert resolved in ("cuda", "mps", "cpu")

    def test_auto_returns_device_torch_accepts(self):
        """The resolved value should be a torch-valid device string.

        Smoke test that torch can actually construct a tensor on the
        resolved device.
        """
        import torch
        resolved = resolve_device("auto")
        # Should not raise — creating a tiny tensor on the resolved device
        t = torch.zeros(1, device=resolved)
        assert str(t.device).split(":")[0] == resolved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
