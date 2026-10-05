"""Create reordered and nested SampleDataset subsets."""

from pyhealth.datasets import create_sample_dataset


def main():
    samples = [
        {"patient_id": "p1", "record_id": "r1", "feature": 1, "label": 0},
        {"patient_id": "p2", "record_id": "r2", "feature": 2, "label": 1},
        {"patient_id": "p1", "record_id": "r3", "feature": 3, "label": 0},
        {"patient_id": "p3", "record_id": "r4", "feature": 4, "label": 1},
    ]
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"feature": "raw"},
        output_schema={"label": "raw"},
        in_memory=False,
    )

    subset = dataset.subset([3, 0, 2])
    nested_subset = subset.subset([2, 0])

    print(subset.patient_to_index)
    print(nested_subset.patient_to_index)
    dataset.close()


if __name__ == "__main__":
    main()
