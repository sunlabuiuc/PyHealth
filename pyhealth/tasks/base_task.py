from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Type
from typing import TYPE_CHECKING

import polars as pl

from .fingerprint import record_init_args

if TYPE_CHECKING:
    from .task_card import TaskCard


class BaseTask(ABC):
    """A prediction task: turns one patient's events into samples.

    Subclasses set ``task_name``, ``input_schema`` and ``output_schema`` and
    implement ``__call__``. They may also describe themselves with
    :meth:`task_card`.

    Init arguments, class-level configuration, and ``version`` are part of
    the ``set_task`` cache key. Bump ``version`` when ``__call__`` or
    ``pre_filter`` changes in a way that alters generated samples.

    Examples:
        >>> class Mortality(BaseTask):
        ...     task_name = "mortality"
        ...     input_schema = {"labs": "tensor"}
        ...     output_schema = {"death": "binary"}
        ...     def __call__(self, patient):
        ...         return []
        >>> Mortality().task_name
        'mortality'
    """

    task_name: str
    input_schema: Dict[str, Union[str, Type]]
    output_schema: Dict[str, Union[str, Type]]

    #: Bump when ``__call__`` or ``pre_filter`` logic changes in a way that
    #: alters the generated samples. Init args alone cannot detect this.
    version: str = "1"

    #: Attribute names that do not affect the generated samples and must not
    #: invalidate the cache (e.g. ``num_workers``, ``verbose``). Denylist, not
    #: allowlist: forgetting an entry here costs a spurious rebuild, whereas
    #: forgetting to allowlist a semantic arg silently reuses a stale cache.
    fingerprint_exclude: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        """Record the effective ``__init__`` arguments of every task instance."""
        super().__init_subclass__(**kwargs)
        record_init_args(cls)

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

    def task_card(self) -> "TaskCard | None":
        """Optional: what the task predicts, for whom and when, and why.

        Override to return a :class:`~pyhealth.tasks.TaskCard`, with the
        rationale for the cohort, prediction time, outcome and exclusions,
        and a dictionary of the task's features. Reports, such as those of
        :mod:`pyhealth.autoimprove`, use it to explain a model in clinical
        terms. The default returns ``None``.

        Examples:
            >>> class Mortality(BaseTask):
            ...     task_name = "mortality"
            ...     input_schema = {"labs": "tensor"}
            ...     output_schema = {"death": "binary"}
            ...     def __call__(self, patient):
            ...         return []
            >>> Mortality().task_card() is None
            True
        """
        return None

    @abstractmethod
    def __call__(self, patient) -> List[Dict]:
        raise NotImplementedError
