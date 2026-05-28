"""Unified training CLI for PyHealth multimodal experiments.

Reads a YAML config (with optional _inherit chain) then fires the E2E script
with all resolved arguments.  CLI flags override YAML values.

Usage:
    python scripts/train_unified.py --config configs/train/e2e_baseline.yaml \
        --model transformer --seed 0

    # Smoke test
    python scripts/train_unified.py --config configs/train/smoke.yaml

    # Full run with frozen encoder
    python scripts/train_unified.py --config configs/train/e2e_balanced.yaml \
        --model ehrmamba --freeze-encoder --seed 42

The script resolves _model_overrides from the YAML, merges CLI overrides, then
calls unified_embedding_e2e_mimic4.py via subprocess so it can be used from
condor jobs without reimplementing the full arg-build logic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_config(config_path: Path) -> Dict[str, Any]:
    """Load config and recursively merge _inherit chain."""
    cfg = _load_yaml(config_path)
    inherit = cfg.pop("_inherit", None)
    if inherit:
        parent_path = config_path.parent / inherit
        parent = _resolve_config(parent_path)
        # parent values are defaults; child values win
        merged = {**parent, **cfg}
        return merged
    return cfg


def _apply_model_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge _model_overrides[model] into cfg if present."""
    overrides = cfg.pop("_model_overrides", {})
    model = cfg.get("model", "mlp")
    if model in overrides:
        cfg = {**cfg, **overrides[model]}
    return cfg


def _to_args(cfg: Dict[str, Any]) -> list[str]:
    """Convert resolved config dict to CLI args for unified_embedding_e2e_mimic4.py."""
    # Map config keys to CLI flags (underscores → hyphens, bool flags → store_true)
    bool_flags = {
        "icd_codes", "freeze_encoder", "include_vitals",
        "balanced_sampling", "bidirectional",
    }
    skip_keys = {"_inherit", "_model_overrides"}

    args = []
    for key, val in cfg.items():
        if key in skip_keys or val is None:
            continue
        flag = "--" + key.replace("_", "-")
        if key in bool_flags:
            if val:
                args.append(flag)
        else:
            args += [flag, str(val)]
    return args


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified PyHealth training CLI — loads a YAML config then fires the E2E script."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file (e.g. configs/train/e2e_baseline.yaml).",
    )
    # Pass-through overrides — any key from the YAML can be overridden here.
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--dev", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true", default=None)
    parser.add_argument("--balanced-sampling", action="store_true", default=None)
    parser.add_argument("--icd-codes", action="store_true", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ehr-root", default=None)
    parser.add_argument("--note-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    cli = _parse_cli()

    config_path = Path(cli.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = _resolve_config(config_path)
    cfg = _apply_model_overrides(cfg)

    # Apply CLI overrides (non-None values win over YAML)
    cli_overrides = {
        "model": cli.model,
        "seed": cli.seed,
        "task": cli.task,
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "embedding_dim": cli.embedding_dim,
        "dev": cli.dev,
        "freeze_encoder": cli.freeze_encoder if cli.freeze_encoder else None,
        "balanced_sampling": cli.balanced_sampling if cli.balanced_sampling else None,
        "icd_codes": cli.icd_codes if cli.icd_codes else None,
        "output_dir": cli.output_dir,
        "ehr_root": cli.ehr_root,
        "note_root": cli.note_root,
        "cache_dir": cli.cache_dir,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    script = Path(__file__).parent.parent / "examples" / "mortality_prediction" / "unified_embedding_e2e_mimic4.py"
    cmd = [sys.executable, str(script)] + _to_args(cfg)

    print("Resolved command:")
    print("  " + " \\\n  ".join(cmd))

    if cli.dry_run:
        print("\n[dry-run] Not executing.")
        return

    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(script.parent.parent.parent)})
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
