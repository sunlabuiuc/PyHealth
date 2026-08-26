"""Drug recommendation on an OMOP CDM dataset.

Run with a local OMOP CDM v5.3 export, e.g. the CMS SynPUF 1k sample.
"""

from pyhealth.datasets import OMOPDataset, get_dataloader, split_by_patient
from pyhealth.tasks import DrugRecommendationOMOP


def main() -> None:
    dataset = OMOPDataset(
        root="/path/to/omop_cdm",
        tables=[
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
        ],
    )
    dataset.stats()

    samples = dataset.set_task(DrugRecommendationOMOP())
    print(samples[0])

    train, _val, _test = split_by_patient(samples, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train, batch_size=32, shuffle=True)
    print(next(iter(train_loader)).keys())


if __name__ == "__main__":
    main()
