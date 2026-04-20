from typing import Any

from pyhealth.tasks import BaseTask


class MedLingoTask(BaseTask):
    """
    Task for MedLingo-style clinical abbreviation expansion.

    Contributor:
        Tedra Birch (tbirch2@illinois.edu)

    Paper:
        Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?
        https://arxiv.org/abs/2505.15024

    This task converts MedLingo dataset records into model-ready input/target pairs.
    """

    task_name: str = "medlingo_task"
    input_schema = {"input": "str"}
    output_schema = {"target": "str"}

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample: dict[str, Any]) -> dict[str, str]:
        """
        Convert a single MedLingo sample into task-ready format.

        Args:
            sample: A dictionary containing the fields 'context' and 'label'.

        Returns:
            A dictionary with the processed input and target fields.
        """
        return {
            "input": sample["context"],
            "target": sample["label"],
        }

    def process(self, dataset):
        """
        Convert processed MedLingo records into task-ready samples.

        Args:
            dataset: Output of MedLingoDataset.process().

        Returns:
            A list of dictionaries with input and target fields.
        """
        output = []

        for record in dataset:
            for sample in record["medlingo"]:
                output.append(self(sample))

        return output