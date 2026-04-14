#!/usr/bin/env python3
"""Warm the shared Table 2 cache from a real Python file.

This exists because Python 3.12 multiprocessing with spawn cannot safely
re-import a `python - <<'PY'` stdin script as `__main__`.
"""

from __future__ import annotations

import os

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesICDLabsMIMIC4


def main() -> None:
    cache_root = os.environ["TABLE2_SHARED_CACHE_ROOT"]
    dataset = MIMIC4Dataset(
        ehr_root=os.environ["EHR_ROOT"],
        ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        note_root=os.environ["NOTE_ROOT"],
        note_tables=["discharge", "radiology"],
        cache_dir=cache_root,
        dev=os.environ.get("TABLE2_DEV_MODE", "0") == "1",
        num_workers=int(os.environ["TABLE2_CACHE_WARM_NUM_WORKERS"]),
    )
    dataset.set_task(
        ClinicalNotesICDLabsMIMIC4(window_hours=24),
        num_workers=int(os.environ["TABLE2_CACHE_WARM_NUM_WORKERS"]),
    )


if __name__ == "__main__":
    main()
