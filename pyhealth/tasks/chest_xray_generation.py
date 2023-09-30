from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

from pyhealth.tasks.task_template import TaskTemplate


@dataclass(frozen=True)
class ChestXrayGeneration(TaskTemplate):
    task_name: str = "ChestXrayGeneration"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"text": "text"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"text": "text"})

    def __call__(self, patient):
        sample = {
            "text": patient["report"],
        }
        return [sample]


if __name__ == "__main__":
    task = ChestXrayGeneration()
    print(task)
    print(type(task))
