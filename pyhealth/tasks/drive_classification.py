from dataclasses import dataclass, field
from typing import Dict

from pyhealth.tasks.task_template import TaskTemplate


@dataclass(frozen=True)
class DriveClassification(TaskTemplate):
    task_name: str = "DriveClassification"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"path": "image"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"label": "label"})

    def __call__(self, patient):
        return [patient]


if __name__ == "__main__":
    task = DriveClassification()
    print(task)
    print(type(task))
