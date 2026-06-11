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

    This dataset is inspired by the MedLingo benchmark and uses synthetic/curated clinical
    abbreviation samples for demonstration and testing purposes.

    No real patient data or MIMIC data is included in this dataset.

    Each sample contains:
        - abbr: clinical abbreviation string
        - context: short clinical text snippet
        - label: ground truth expanded meaning
        - source: source of the sample (e.g. "synthetic_demo")

    Args:
        root: Root directory (used for demo/example purposes only)
        config_path: Optional path to dataset config yaml.

    Example:
        >>> samples = [{"abbr": "SOB", "context": "Patient has SOB.", "label": "shortness of breath"}]
        >>> dataset = MedLingoDataset(samples=samples)
        >>> records = dataset.process()
    """

    def __init__(
        self,
        samples: List[Dict[str, str]] | None = None,
        root: str = "",
        config_path: str | None = None,
    ) -> None:
        self.samples = samples or []
        tables = ["medlingo"] 
        
        super().__init__(
            root=root,
            tables=tables,
            dataset_name="medlingo",
            config_path=config_path,
        )

    
    def process(self) -> List[Dict]:
        """
        Convert MedLingo samples into PyHealth-style records.

        Returns:
            A list of patient/visit records with a medlingo table.
        """

        data = []

        # Convert each sample into the standardized format
        for i, sample in enumerate(self.samples):
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