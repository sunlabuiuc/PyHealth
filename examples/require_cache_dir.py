"""Keep dataset caches in controlled storage when working with identified data.

A PyHealth dataset caches processed copies of its source data (the event table
and task samples). Without ``cache_dir`` they go to your user cache folder and
PyHealth logs a warning naming it. Setting ``PYHEALTH_REQUIRE_CACHE_DIR=1`` turns
a missing ``cache_dir`` into an error, so no script or notebook in the project
can write data copies outside approved storage by accident.

Uses the small MIMIC-III demo bundled in ``test-resources/``; no download.

Usage (from the repository root):
    PYHEALTH_REQUIRE_CACHE_DIR=1 python examples/require_cache_dir.py
"""

import os
import tempfile
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset

DEMO_ROOT = (
    Path(__file__).resolve().parent.parent / "test-resources" / "core" / "mimic3demo"
)


def main():
    os.environ.setdefault("PYHEALTH_REQUIRE_CACHE_DIR", "1")

    # 1) Forgetting cache_dir is now an error, raised before anything is written.
    try:
        MIMIC3Dataset(root=str(DEMO_ROOT), tables=["diagnoses_icd"])
    except ValueError as err:
        print("without cache_dir:", err)

    # 2) With cache_dir inside storage you control, it works as usual.
    #    (A temporary folder stands in for your project's secure storage here.)
    with tempfile.TemporaryDirectory() as project_cache:
        dataset = MIMIC3Dataset(
            root=str(DEMO_ROOT), tables=["diagnoses_icd"], cache_dir=project_cache
        )
        dataset.stats()
        print("cache written under:", dataset.cache_dir)


if __name__ == "__main__":
    main()
