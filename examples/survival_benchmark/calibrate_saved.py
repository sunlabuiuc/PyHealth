"""Authors: Aritra Saha and Ugochukwu Njoku.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html

Score exact saved observed-time probabilities without refitting a model.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pyhealth.metrics import d_calibration


def calibrate(path: str | Path, bins: int = 10) -> dict[str, Any]:
    """Score exact observed-time probabilities from a saved prediction archive.

    Args:
        path: NPZ containing survival_at_observed_time and event arrays.
        bins: Number of equal-width probability bins, at least two.

    Returns:
        The public PyHealth d_calibration result, ready for JSON serialization.

    Raises:
        ValueError: If exact predictions or matching event flags are absent.

    Examples:
        >>> callable(calibrate)
        True
    """
    with np.load(path, allow_pickle=False) as saved:
        if "survival_at_observed_time" not in saved:
            raise ValueError(
                "Exact observed-time probabilities are required; regenerate them "
                "from the checkpoint, not a coarse interpolated grid"
            )
        if "event" not in saved:
            raise ValueError("Prediction archive requires event flags")
        return d_calibration(saved["survival_at_observed_time"], saved["event"], bins)


def main(argv: Sequence[str] | None = None) -> None:
    """Score one or more saved archives from the command line.

    Args:
        argv: CLI arguments, or None to read the process command line.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", type=Path, nargs="+")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    result = [
        {"predictions": str(path), "calibration": calibrate(path, a.bins)}
        for path in a.predictions
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
