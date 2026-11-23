import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

if TYPE_CHECKING:
    from .task_card import TaskCard


class BaseTask(ABC):
    """A prediction task: turns one patient's events into samples.

    Subclasses set ``task_name``, ``input_schema`` and ``output_schema`` and
    implement ``__call__``. They may also describe themselves with
    :meth:`task_card`.

    **Views.** A task may offer several *views*: named alternatives for
    ``input_schema`` over the same samples. ``__call__`` runs once and its
    samples, labels and split are shared by every view; a view only decides
    which processor, with which settings, turns each field into model input.
    Set ``views`` (view name to input schema) and ``default_view`` instead of
    ``input_schema``, and choose a view when creating the task::

        task = MyTask(view="dense")   # by name
        task = MyTask(model=RETAIN)   # the first view RETAIN accepts
        task = MyTask()               # the default view

    A task without ``views`` has the one ``input_schema`` it declares.

    **Your own version of a view.** Samples, labels and split are fixed by
    the task; how a view processes them is yours to change:

    - change settings of a view with ``view_overrides``, field by field::

          MyTask(view="dense", view_overrides={"dynamic": {"normalize": False}})

      The given settings replace those of the field's processor (a nested
      setting is replaced whole); a processor spec (a name, class or
      ``(name, settings)`` pair) replaces the field's processor.
    - pass a whole input schema as ``view``; such a view is named
      ``"custom"`` and any model may be built on it.
    - subclass the task and add to ``views`` to give a view a name::

          class MyTaskPlus(MyTask):
              views = {**MyTask.views, "mine": {...}}

    Each version is cached separately; all of them read the task's one
    cache of samples (a subclass, being another task, has its own).

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
    input_schema: dict[str, str | type]
    output_schema: dict[str, str | type]

    #: View name to input schema. Empty for a task with a single input schema.
    views: ClassVar[Mapping[str, dict[str, Any]]] = {}
    #: The view used when neither ``view`` nor ``model`` is given.
    default_view: ClassVar[str | None] = None
    #: The name of the selected view (``"custom"`` for a schema passed as
    #: ``view``), or None for a task without views.
    view: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls.__dict__
        if "views" in own:
            views = own["views"]
            if not views:
                return
            default = cls.default_view
            if default is None:
                default = next(iter(views))
                cls.default_view = default
            if default not in views:
                raise TypeError(
                    f"{cls.__name__}.default_view is {default!r}, which is not "
                    f"one of its views: {sorted(views)}."
                )
            # Class-level access (``MyTask.input_schema``) gives the default view.
            cls.input_schema = views[default]
            cls.view = default
        elif "input_schema" in own and cls.views:
            # A subclass that declares its own input schema does not inherit
            # the views of its base, which describe other fields.
            cls.views = {}
            cls.default_view = None
            cls.view = None

    def __init__(
        self,
        code_mapping: dict[str, tuple[str, str]] | None = None,
        view: str | dict[str, Any] | None = None,
        model: Any = None,
        view_overrides: Mapping[str, Any] | None = None,
    ):
        """Initialize a task, choosing its view and optional code mapping.

        Args:
            view: the name of one of the task's ``views``, or an input schema
                of your own for the task's fields. Defaults to
                ``default_view``.
            model: a model class (or instance) to choose the view for: the
                first of its ``accepted_views`` that the task offers. Give
                ``view`` or ``model``, not both.
            view_overrides: changes to the selected view, by input key: a
                dict of processor settings to replace, or a processor spec
                to use instead.
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

        Raises:
            ValueError: if ``view`` and ``model`` are both given, the task
                has no views, the view is unknown, or the model accepts none
                of the task's views.
        """
        self._select_view(view, model)
        if view_overrides:
            self._override_view(view_overrides)
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

    def _select_view(self, view: str | dict[str, Any] | None, model: Any) -> None:
        """Sets ``input_schema`` and ``view`` from the ``view``/``model`` arguments."""
        name = type(self).__name__
        if view is not None and model is not None:
            raise ValueError(f"{name}: give view or model, not both.")
        if view is None and model is None:
            return  # the class-level default, set by __init_subclass__
        if not self.views:
            raise ValueError(
                f"{name} has a single input schema and no views, so view= and "
                "model= do not apply."
            )
        offered = ", ".join(repr(v) for v in self.views)
        if model is not None:
            model_name = getattr(model, "__name__", type(model).__name__)
            accepted = tuple(getattr(model, "accepted_views", ()) or ())
            if not accepted:
                raise ValueError(
                    f"{model_name} does not declare accepted_views, so {name} "
                    f"cannot choose a view for it. Pass view= instead; {name} "
                    f"offers {offered}."
                )
            view = next((v for v in accepted if v in self.views), None)
            if view is None:
                raise ValueError(
                    f"{model_name} reads the views "
                    f"{', '.join(repr(v) for v in accepted)}; {name} offers "
                    f"{offered}. No view fits both."
                )
        if isinstance(view, str):
            if view not in self.views:
                raise ValueError(
                    f"{name} has no view {view!r}. It offers {offered}."
                )
            self.input_schema = self.views[view]
            self.view = view
        elif isinstance(view, Mapping) and view:
            self.input_schema = dict(view)
            self.view = "custom"
        else:
            raise ValueError(
                f"{name}: view must be a view name or an input schema, got {view!r}."
            )

    def _override_view(self, overrides: Mapping[str, Any]) -> None:
        """Applies ``view_overrides`` to a copy of the selected input schema."""
        name = type(self).__name__
        schema = copy.deepcopy(dict(self.input_schema))
        unknown = sorted(set(overrides) - set(schema))
        if unknown:
            raise ValueError(
                f"{name}: view_overrides names {unknown}, which are not inputs "
                f"of the view {self.view!r}; its inputs are {sorted(schema)}."
            )
        for key, change in overrides.items():
            if not isinstance(change, Mapping):
                schema[key] = change  # another processor for this field
                continue
            spec = schema[key]
            processor, settings = spec if isinstance(spec, tuple) else (spec, {})
            schema[key] = (processor, {**settings, **copy.deepcopy(dict(change))})
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
    def __call__(self, patient) -> list[dict]:
        raise NotImplementedError
