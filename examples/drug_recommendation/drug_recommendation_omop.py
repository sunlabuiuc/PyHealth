"""Drug recommendation samples from an OMOP CDM dataset.

Loads condition, procedure and drug tables from an OMOP CDM v5.3 export,
applies ``DrugRecommendationOMOP``, prints one sample and one batch from a
patient-level split. It stops before modelling; plug the samples into any
multilabel model (see the MIMIC-III drug recommendation examples).

Data: any OMOP CDM v5.3 CSV export, e.g. the CMS SynPUF 1k sample or the
open-access MIMIC-IV demo in OMOP format
(https://physionet.org/content/mimic-iv-demo-omop/). Set ``root`` below.

Run: ``python examples/drug_recommendation/drug_recommendation_omop.py``.
Without downloading anything, set ``root`` to ``test-resources/omop``.
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
