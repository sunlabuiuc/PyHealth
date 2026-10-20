"""Working with Event timestamps, including events that have no time.

Some tables have no timestamp column (e.g. MIMIC-III ``patients``). Their
events have ``timestamp=None``, the same ``null`` shown by
``get_events(..., return_df=True)``. This script shows how to read them, how to
keep only timed events before comparing times, and that events can be pickled
(e.g. stored in task samples).

Uses the small MIMIC-III demo bundled in ``test-resources/``; no download.

Usage (from the repository root):
    python examples/event_timestamps_mimic3demo.py
"""

import pickle
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset

DEMO_ROOT = (
    Path(__file__).resolve().parent.parent / "test-resources" / "core" / "mimic3demo"
)


def main():
    dataset = MIMIC3Dataset(root=str(DEMO_ROOT), tables=["diagnoses_icd"])
    patient = dataset.get_patient("10006")

    # Demographics have no time: the Event agrees with the DataFrame view.
    demographics = patient.get_events(event_type="patients")[0]
    df = patient.get_events(event_type="patients", return_df=True)
    print("patients event timestamp:", demographics.timestamp)
    print("patients row timestamp:  ", df["timestamp"].to_list()[0])
    print("gender:", demographics.gender)

    # Keep only timed events before comparing or sorting by time.
    admission = patient.get_events(event_type="admissions")[0]
    events = patient.get_events()
    timed = [e for e in events if e.timestamp is not None]
    before_admission = [e for e in timed if e.timestamp <= admission.timestamp]
    print(
        f"{len(events)} events, {len(timed)} with a time, "
        f"{len(before_admission)} at or before the first admission"
    )

    # Events survive pickling, so tasks can put them in samples.
    restored = pickle.loads(pickle.dumps(admission))
    print("pickle round trip equal:", restored == admission)


if __name__ == "__main__":
    main()
