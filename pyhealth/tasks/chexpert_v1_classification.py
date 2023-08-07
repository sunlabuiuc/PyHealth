from dataclasses import dataclass, field
from typing import Dict

from tasks.task_template import TaskTemplate # from pyhealth.tasks import TaskTemplate


@dataclass(frozen=True)
class CheXpertV1Classification(TaskTemplate):
    task_name: str = "CheXpertV1Classification"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"path": "image"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"label": "label"})

    def __call__(self, patient):
        return [patient]


if __name__ == "__main__":
    task = CheXpertV1Classification()
    print(task)
    print(type(task))
