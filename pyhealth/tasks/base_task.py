from abc import ABC, abstractmethod
from typing import Dict, List


class BaseTask(ABC):
    task_name: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]

    @abstractmethod
    def __call__(self, patient) -> List[Dict]:
        raise NotImplementedError