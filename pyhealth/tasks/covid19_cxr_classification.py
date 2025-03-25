from dataclasses import dataclass, field
from typing import Dict

from pyhealth.tasks.task_template import TaskTemplate


@dataclass(frozen=True)
class COVID19CXRClassification(TaskTemplate):
    task_name: str = "COVID19CXRClassification"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"path": "image"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"label": "label"})

    def __call__(self, patient):
        path = patient.attr_dict["path"]
        label = patient.attr_dict["label"]
        return [{"path": path, "label": label}]


if __name__ == "__main__":
    task = COVID19CXRClassification()
    print(task)
    print(type(task))