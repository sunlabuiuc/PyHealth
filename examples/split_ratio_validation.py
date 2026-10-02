"""Demonstrate valid and invalid patient split ratios on synthetic data."""

from pyhealth.datasets import create_sample_dataset, split_by_patient


def main() -> None:
    samples = []
    for patient_index in range(6):
        for record_number in range(2):
            record_index = patient_index * 2 + record_number
            samples.append(
                {
                    "patient_id": f"patient-{patient_index:02d}",
                    "record_id": f"record-{record_index:02d}",
                    "feature": record_index,
                    "label": record_index % 2,
                }
            )
    source_record_ids = {sample["record_id"] for sample in samples}
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"feature": "raw"},
        output_schema={"label": "raw"},
        in_memory=True,
    )
    try:
        train, validation, test = split_by_patient(
            dataset, ratios=[0.5, 0.25, 0.25], seed=42
        )
        patient_ids = [
            {part[index]["patient_id"] for index in range(len(part))}
            for part in (train, validation, test)
        ]
        split_record_ids = [
            [part[index]["record_id"] for index in range(len(part))]
            for part in (train, validation, test)
        ]
        if not (
            patient_ids[0].isdisjoint(patient_ids[1])
            and patient_ids[0].isdisjoint(patient_ids[2])
            and patient_ids[1].isdisjoint(patient_ids[2])
        ):
            raise RuntimeError("a patient appears in more than one split")
        flattened_record_ids = [
            record_id for split in split_record_ids for record_id in split
        ]
        if (
            set(flattened_record_ids) != source_record_ids
            or len(flattened_record_ids) != len(source_record_ids)
        ):
            raise RuntimeError("source records were not covered exactly once")
        patient_overlap_count = sum(
            len(patient_ids[left] & patient_ids[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        print(
            "valid split record counts:",
            [len(part) for part in (train, validation, test)],
        )
        print("patient overlap count:", patient_overlap_count)
        print("source records covered exactly once:", len(flattened_record_ids))

        try:
            split_by_patient(dataset, ratios=[0.8, -0.2, 0.4], seed=42)
        except ValueError as error:
            print("invalid ratios rejected:", error)
        else:
            raise RuntimeError("negative split ratios were not rejected")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
