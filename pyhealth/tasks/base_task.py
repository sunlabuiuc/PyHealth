from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import polars as pl


class BaseTask(ABC):
    task_name: str
    input_schema: Dict[str, Union[str, Type]]
    output_schema: Dict[str, Union[str, Type]]

    def __init__(
        self,
        code_mapping: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        """Initialize a task with optional code mapping.

        Args:
            code_mapping: optional dict mapping feature keys to
                ``(source_vocab, target_vocab)`` tuples.  For example::

                    code_mapping={
                        "conditions": ("ICD9CM", "CCSCM"),
                        "procedures": ("ICD9PROC", "CCSPROC"),
                        "drugs": ("NDC", "ATC"),
                    }

                When provided, the corresponding ``input_schema`` entries
                are upgraded from ``"sequence"`` to
                ``("sequence", {"code_mapping": (src, tgt)})`` so that the
                ``SequenceProcessor`` maps raw codes at fit/process time.
        """
        if code_mapping is not None:
            schema = dict(self.input_schema)
            for field, mapping in code_mapping.items():
                if field in schema:
                    base = schema[field]
                    if isinstance(base, tuple):
                        base, kwargs = base
                        kwargs = dict(kwargs)
                    else:
                        kwargs = {}
                    kwargs["code_mapping"] = mapping
                    schema[field] = (base, kwargs)
            self.input_schema = schema

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    def prepare_for_dataset(
        self, dataset: Any, num_workers: int
    ) -> None:
        """Hook called by :meth:`~pyhealth.datasets.BaseDataset.set_task` before
        the LitData caching pipeline.

        Override this to perform task-specific preparation that requires access
        to the dataset (e.g., vocabulary warming that must happen in the main
        process before workers are forked).

        Args:
            dataset: The :class:`~pyhealth.datasets.BaseDataset` instance.
            num_workers: Effective number of workers that will be used.
        """

    @abstractmethod
    def __call__(self, patient) -> List[Dict]:
        raise NotImplementedError
