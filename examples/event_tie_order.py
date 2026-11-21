"""Events that share a timestamp keep their source-file order.

Builds a tiny dataset whose diagnoses all share one timestamp (as MIMIC
diagnoses share their admission's discharge time), then shows that a patient's
diagnoses come out in the order of the source file, which is their priority
order, on every cache build. It is a demonstration: with five rows, an
unordered build can also come out in this order by chance, so the regression
tests are in ``tests/core/test_event_tie_order.py``.

Run:
    python examples/event_tie_order.py
"""

import tempfile
from pathlib import Path

from pyhealth.datasets import BaseDataset

CONFIG = """version: "1.0"
tables:
  diagnoses:
    file_path: "diagnoses.csv"
    patient_id: "patient_id"
    timestamp: "dischtime"
    attributes: ["seq_num", "code"]
"""

CODES = ["428.0", "250.00", "401.9", "038.9", "486"]  # priority order


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        root.mkdir()
        (root / "config.yaml").write_text(CONFIG)
        rows = ["patient_id,dischtime,seq_num,code"] + [
            f"p1,2020-01-05 12:00:00,{i + 1},{code}" for i, code in enumerate(CODES)
        ]
        (root / "diagnoses.csv").write_text("\n".join(rows) + "\n")

        for build in range(1, 4):
            dataset = BaseDataset(
                root=str(root),
                tables=["diagnoses"],
                dataset_name="tie_order_demo",
                config_path=str(root / "config.yaml"),
                cache_dir=str(Path(tmp) / f"cache{build}"),  # a fresh build each time
            )
            events = dataset.get_patient("p1").get_events(event_type="diagnoses")
            codes = [e.code for e in events]
            print(f"build {build}:", [(e.seq_num, e.code) for e in events])
            assert codes == CODES, f"build {build} lost the priority order: {codes}"
        print("All builds keep the priority order.")


if __name__ == "__main__":
    main()
