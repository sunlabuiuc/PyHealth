"""Pytest hooks and environment fixes for the PyHealth test suite.

``litdata`` (imported by ``pyhealth.datasets.base_dataset``) references
``torch.uint16``, which exists in PyTorch 2.3+. Some environments (notably
macOS x86_64) only receive PyTorch 2.2.x wheels from PyPI, which triggers
``AttributeError`` on import. CI and Linux graders typically use current
torch wheels; this shim only applies when the attribute is missing.
"""

from __future__ import annotations

import torch

if not hasattr(torch, "uint16"):
    # Same storage width as uint16; satisfies litdata's import-time dtype map.
    # For real uint16 tensor I/O, use PyTorch 2.3+ (see project dependencies).
    torch.uint16 = torch.int16
