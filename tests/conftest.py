"""conftest.py: stub optional heavy dependencies unavailable in this environment."""

import sys
import types

_STUBS = [
    "mamba_ssm",
    "mamba_ssm.modules",
    "mamba_ssm.modules.mamba_simple",
    "linear_attention_transformer",
    "ogb",
    "ogb.graphproppred",
    "ogb.graphproppred.mol_encoder",
]

for _name in _STUBS:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = []  # mark as package so submodule lookups work
        sys.modules[_name] = _mod
