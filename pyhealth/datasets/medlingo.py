import json
from pathlib import Path
from typing import Dict, List

from pyhealth.datasets import BaseDataset

class MedLingoDataset(BaseDataset):
    """
    MedLingo Dataset for clinical abbreviation interpretation.

    Contributor: 
        Tedra Birch(tbirch2@illinois.edu)

    Paper: 
        Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information? 
        https://arxiv.org/abs/2505.15024

    This dataset is inspired by the MedLingo benchmark and is constructed from 
    cleaned, curated clinical abbreviation samples.

    Each sample contains:
        - abbr: clinical abbreviation string
        - context: short clinical text snippet
        - label: ground truth expanded meaning
        - source: source of the sample (e.g. "mimic_iv", "synthetic_demo")

    Args:
        root: Root directory containing medlingo_samples.json (e.g., "test-resources" for demo usage)
        config_path: Optional path to dataset config yaml.

    Example:
        >>> dataset = MedLingoDataset(root="data")
        >>> records = dataset.process()
    """

    def __init__(
        self,
        root: str = "",
        config_path: str | None = None,
    ) -> None:
        tables = ["medlingo"]  # single table dataset
        super().__init__(
            root=root,
            tables=tables,
            dataset_name="medlingo",
            config_path=config_path,
        )
    
    @classmethod
    def from_json(cls, filepath: str | Path) -> "MedLingoDataset":
        dataset = cls(root=str(Path(filepath).parent))
        return dataset

    
    def process(self) -> List[Dict]:
        """
        Load MedLingo JSON samples and convert them into PyHealth-style records.

        Returns:
            A list of patient/visit records with a medlingo table.
        """
        file_path = Path(self.root) / "medlingo_samples.json"


        # Check if the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")


        with open(file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        data = []

        # Convert each sample into the standardized format
        for i, sample in enumerate(samples):
            data.append({
                "patient_id": f"patient_{i}",
                "visit_id": f"visit_{i}",
                "medlingo": [
                    {
                        "abbr": sample["abbr"],
                        "context": sample["context"],
                        "label": sample["label"],
                    }
                ]
            })

        return data