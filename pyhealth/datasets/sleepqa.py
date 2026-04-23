import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional
import pandas as pd

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class SleepQADataset(BaseDataset):
    """Dataset class for the SleepQA dataset.

    SleepQA is a health coaching dataset consisting of passages and 
    corresponding question-answer pairs related to sleep hygiene.

    Args:
        root: root directory of the raw data.
        config_path: path to the configuration file. Default is sleepqa.yaml.
        download: whether to download the dataset. Default is False.
        **kwargs: additional arguments for BaseDataset.

    Examples:
        >>> from pyhealth.datasets import SleepQADataset
        >>> dataset = SleepQADataset(root="./data", download=True)
        >>> dataset.stat()
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "sleepqa.yaml"),
        download: bool = False,
        **kwargs,
    ) -> None:
        self._json_path = os.path.join(root, "sleepqa.json")
        if download:
            self._download(root)
        self._verify_data(root)
        self._index_data(root)

        super().__init__(
            root=root,
            tables=["sleepqa"],
            dataset_name="SleepQA",
            config_path=config_path,
            **kwargs,
        )
        

    @property
    def default_task(self):
        """Returns the default SleepQAExtractiveQA task."""
        from pyhealth.tasks.sleepqa_extractive_qa import SleepQAExtractiveQA
        return SleepQAExtractiveQA()

    def _download(self, root: str) -> None:
        """Downloads raw SleepQA JSON from the official source."""
        os.makedirs(root, exist_ok=True)
        link = "link = "https://raw.githubusercontent.com/IvaBojic/SleepQA/main/data/training/sleep-train.json"
        logger.info(f"Downloading SleepQA to {self._json_path}...")
        urllib.request.urlretrieve(link, self._json_path)

    def _verify_data(self, root: str) -> None:
        """Verifies that the raw JSON file exists."""
        if not os.path.isfile(self._json_path):
            raise FileNotFoundError(
                "Dataset path must contain 'sleepqa.json'!")

    def _index_data(self, root: str) -> pd.DataFrame:
        """Parses SleepQA JSON into a relational CSV for PyHealth indexing."""
        with open(self._json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for item in data.get("data", []):
            p_id = str(item.get("passage_id", ""))
            txt = item.get("text", "")
            for qa in item.get("qas", []):
                ans = qa.get("answers", [{}])[0]
                rows.append({
                    "patient_id": p_id,
                    "visit_id": f"v_{p_id}",
                    "question_id": str(qa.get("id", "")),
                    "question": qa.get("question", ""),
                    "passage": txt,
                    "answer_text": ans.get("text", ""),
                    "answer_start": ans.get("answer_start", 0),
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(
            root, "sleepqa-metadata-pyhealth.csv"), index=False)
        return df
