from abc import ABC, abstractmethod
from typing import Dict, List, Union, Type

import polars as pl


class BaseTask(ABC):
    task_name: str
    input_schema: Dict[str, Union[str, Type]]
    output_schema: Dict[str, Union[str, Type]]

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    @abstractmethod
    def __call__(self, patient) -> List[Dict]:
        raise NotImplementedError
