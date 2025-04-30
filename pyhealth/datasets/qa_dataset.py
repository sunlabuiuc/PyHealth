import logging
from pathlib import Path
from typing import Optional
import json
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class QuestionAnsweringDataset(BaseDataset):
    """Question answering dataset.
       Based on the paper: Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records
       Link to the dataset: https://github.com/cyc1am3n/text2program-for-ehr/blob/main/data/natural/train.json

    Args:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.

    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.root = root
        return

    def parse_json(self):
        self.samples = []
        with open(self.root,"r") as f:
            for line in f:
                if line.strip():
                    curr_line = json.loads(line)
                    sample = {
                        "question": curr_line.get("question", ""),
                        "trace": curr_line.get("trace", ""),
                        "answer": curr_line.get("answer", [])
                    }
                    self.samples.append(sample)



if __name__ == "__main__":


    # change the root as needed when testing

    root = "/srv/local/data/train.json"

    dataset = QuestionAnsweringDataset(root=root)
    dataset.parse_json()

    samples = dataset.samples
    
    print(f"Loaded {len(samples)} samples.")
    first_sample = samples[0]

    print("Here is an example entry from the dataset:")

    print(f'Question: {first_sample.get("question", "")}')
    print(f'Execution trace: {first_sample.get("trace", "")}')
    print(f'Answer: {first_sample.get("answer", "")}')