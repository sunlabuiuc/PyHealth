from abc import ABC, abstractmethod
from typing import Dict, List, Union, Type

import pandas as pd


class BaseTask(ABC):
    task_name: str
    input_schema: Dict[str, Union[str, Type]]
    output_schema: Dict[str, Union[str, Type]]

    def pre_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def __call__(self, patient) -> List[Dict]:
        raise NotImplementedError
