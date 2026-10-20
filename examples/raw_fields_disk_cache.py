"""Variable-length raw fields round-trip through the disk cache.

EHR samples often carry fields whose length varies by patient: notes per
admission, measurement times, codes per visit. Declare them as "raw" and they
are stored as one value per sample, so the disk-backed dataset returns exactly
what went in, next to ordinary tensor fields.

Runs in a few seconds on CPU with synthetic data; no download.

Usage:
    python examples/raw_fields_disk_cache.py
"""

from pyhealth.datasets import create_sample_dataset, get_dataloader

samples = [
    {
        "patient_id": f"p{i}",
        "notes": [f"note {j}" for j in range(i % 4)],  # 0-3 notes
        "lab_times": [0.5 * j for j in range(1 + i % 5)],  # 1-5 times
        "visits": [["I10", "E11"][: 1 + v % 2] for v in range(1 + i % 3)],
        "label": i % 2,
    }
    for i in range(12)
]


def main():
    dataset = create_sample_dataset(
        samples,
        input_schema={"notes": "raw", "lab_times": "raw", "visits": "nested_sequence"},
        output_schema={"label": "binary"},
        in_memory=False,  # write to the disk (litdata) cache, as set_task does
    )
    for i in (0, 3, 5):
        s = dataset[i]
        print(s["patient_id"], "notes:", s["notes"], "| lab_times:", s["lab_times"],
              "| visits tensor:", tuple(s["visits"].shape))
    print("all raw fields intact:",
          all(dataset[i]["notes"] == samples[i]["notes"]
              and dataset[i]["lab_times"] == samples[i]["lab_times"]
              for i in range(len(samples))))
    batch = next(iter(get_dataloader(dataset, batch_size=4)))
    print("batch notes:", batch["notes"])


if __name__ == "__main__":
    main()
