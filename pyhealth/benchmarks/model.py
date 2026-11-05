"""Trains and tests PyHealth models on benchmark tasks, from config dicts.

:class:`ModelBenchmark` takes an environment dict (``env.yaml``: device and
paths) and one dict per task (``<task>.yaml``: task, dataset, split, training
and models). Each ``(task id, model id, version)`` is one evaluation that
trains and tests the model on every split seed; its saved result is reused
until the model's ``version`` (its code's commit SHA) changes.
"""

import contextlib
import gc
import json
import logging
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, override

from ._serialization import _resolve
from .base import Benchmark, Evaluation, Result

logger = logging.getLogger(__name__)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Merges ``override`` over ``base``, recursing into nested mappings.

    Examples:
        >>> deep_merge({"a": 1, "o": {"lr": 1, "wd": 0}}, {"o": {"lr": 2}})
        {'a': 1, 'o': {'lr': 2, 'wd': 0}}
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            value = deep_merge(merged[key], value)
        merged[key] = value
    return merged


def summarize(runs: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and population std of each metric over its finite values.

    Examples:
        >>> summarize([{"auc": 0.8}, {"auc": 0.6}, {"auc": float("nan")}])
        {'auc': {'mean': 0.7, 'std': 0.1}}
    """
    summary = {}
    for name in sorted({name for run in runs for name in run}):
        values = [v for run in runs if math.isfinite(v := run.get(name, math.nan))]
        if values:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            summary[name] = {"mean": round(mean, 10), "std": round(std, 10)}
    return summary


def _write_json(path: Path, data: Any) -> None:
    """Writes JSON atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp makes it 0600
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _save_processors_and_split(parts: Sequence[Any], seed_dir: Path) -> None:
    """Saves one seed's fitted processors and patient split to ``seed_dir``.

    Writes ``processors.pkl``, a copy of the ``schema.pkl`` that ``set_task``
    caches next to the processed samples (the fitted input and output
    processors and the schemas), and ``split.json``, the sorted patient IDs of
    each part (``train``, ``val``, ``test``).

    TODO: temporary solution until the Trainer saves processors and splits
    with its checkpoints.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(parts[0].path) / "schema.pkl", seed_dir / "processors.pkl")
    _write_json(
        seed_dir / "split.json",
        {
            name: sorted(part.patient_to_index)
            for name, part in zip(("train", "val", "test"), parts)
        },
    )


class ModelBenchmark(Benchmark):
    """One evaluation per ``(task id, model id, version)``, over all seeds.

    Outputs of a pair, in ``<output_dir>[/dev]/<task id>/<model id>/``:
        - ``result.json``: the result of the last run, with its ``version``.
        - ``seed<N>/``: the last run's training log and checkpoints, fitted
          processors (``processors.pkl``) and patient split (``split.json``).

    The folder is emptied before a pair runs, so it only ever holds one run.

    A pair is run again only when its ``result.json`` is missing or holds
    another ``version``; bump ``version`` to rerun. ``cache_dir`` only holds the
    datasets' preprocessing cache, so pairs of the same task share the parsed
    dataset and processed splits.

    A result's ``metrics`` hold each test metric's mean and std over the
    seeds, and ``metadata["seeds"]`` each seed's ``metrics``.

    Args:
        env: ``{"compute": {"device"}, "paths": {"input_dir", "cache_dir",
            "output_dir"}, "dev"}``. A relative dataset ``root`` in a task is
            resolved against ``input_dir``; absolute roots and URLs are used
            as given.
        tasks: Task dicts with ``task`` (``id``, ``target``, ``params``,
            ``dataset``), ``split`` (``target``, ``params`` with ``seeds``),
            ``training`` (``batch_size``, ``epochs``, ``metrics``,
            ``optimizer``, ``trainer``: :meth:`~pyhealth.trainer.Trainer.train`
            kwargs) and ``models`` (``id``, ``version``, ``target``,
            ``params``, optional ``training`` overrides).

    Examples:
        >>> task = {
        ...     "task": {"id": "mortality"},
        ...     "models": [
        ...         {"id": "rnn", "version": "abc123"},
        ...         {"id": "transformer", "version": "abc123"},
        ...     ],
        ... }
        >>> bench = ModelBenchmark({"compute": {"device": "cpu"}}, [task])
        >>> [ev.params["model_id"] for ev in bench.run_configs]
        ['rnn', 'transformer']
        >>> bench.run_output_dir(next(iter(bench.run_configs)), Path("out"))
        PosixPath('out/mortality/rnn')

        ``bench.run()`` then trains and tests each model on every split seed.
    """

    def __init__(
        self,
        env: Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]],
    ) -> None:
        self.env, self.tasks = env, tasks
        self.dev = bool(env.get("dev", False))
        self.device = env.get("compute", {}).get("device", "cpu")
        self.run_configs = self.build_run_configs(tasks)

    @staticmethod
    def build_run_configs(
        tasks: Sequence[Mapping[str, Any]],
    ) -> dict[Evaluation, tuple[Mapping[str, Any], Mapping[str, Any]]]:
        """Maps each (task, model) pair's evaluation to its task and model configs.

        An evaluation's params are only the task id, model id and model
        ``version``; the configs it runs with are the dict's values.
        """
        return {
            Evaluation(
                {"task_id": task["task"]["id"], "model_id": model["id"], "version": str(model["version"])}
            ): (task, model)
            for task in tasks
            for model in task["models"]
        }

    @property
    @override
    def name(self) -> str:
        return "model"

    def run(self) -> dict[Evaluation, Result]:
        """Evaluates every task and model with the env's ``paths``."""
        paths = self.env.get("paths", {})
        cache_dir, output_dir = paths.get("cache_dir"), paths.get("output_dir")
        return self.evaluate(
            list(self.run_configs.keys()),
            Path(cache_dir) if cache_dir else None,
            Path(output_dir) if output_dir else None,
        )

    def run_output_dir(self, ev: Evaluation, output_dir: Path) -> Path:
        """``<output_dir>[/dev]/<task id>/<model id>/``, the outputs of ``ev``."""
        root = output_dir / "dev" if self.dev else output_dir
        return root / ev.params["task_id"] / ev.params["model_id"]

    def saved(self, ev: Evaluation, output_dir: Path | None) -> Result | None:
        """The saved result if it is of ``ev``'s version, else None (must run)."""
        if output_dir is None:
            return None
        path = self.run_output_dir(ev, output_dir) / "result.json"
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != ev.params["version"]:
            logger.info(f"[new version] {path}: {data.get('version')} -> {ev.params['version']}")
            return None
        return Result("success", data["metrics"], data["metadata"])

    @override
    def _evaluate(
        self,
        evaluations: Sequence[Evaluation],
        cache_dir: Path | None,
        output_dir: Path | None,
    ) -> dict[Evaluation, Result]:
        results: dict[Evaluation, Result] = {}
        for ev in evaluations:
            task_config, model_config = self.run_configs[ev]
            label = f"{ev.params['task_id']} / {ev.params['model_id']}"
            if (hit := self.saved(ev, output_dir)) is not None:
                logger.info(f"[done before] {label}")
                results[ev] = hit
                continue
            try:
                dataset = self._load_dataset(task_config, cache_dir)  # cheap: lazy, cached
                run_output_dir = self.run_output_dir(ev, output_dir) if output_dir else None
                if run_output_dir is not None:  # holds one run only, whatever its seeds
                    shutil.rmtree(run_output_dir, ignore_errors=True)
                result = self._evaluate_one(task_config, model_config, dataset, run_output_dir)
            except Exception as exc:  # one failed pair spares the rest
                logger.exception(f"[failed] {label}")
                results[ev] = Result("failed", {}, {"error": str(exc), "error_type": type(exc).__name__})
                continue
            if run_output_dir is not None:
                _write_json(
                    run_output_dir / "result.json",
                    {
                        "version": ev.params["version"],
                        "metrics": result.metrics,
                        "metadata": result.metadata,
                    },
                )
            logger.info(f"[done] {label}: {result.metrics}")
            results[ev] = result
        return results

    def _load_dataset(self, task: Mapping[str, Any], cache_dir: Path | None) -> Any:
        """Builds the task's dataset, with ``cache_dir`` and the env's dev mode."""
        cfg = dict(task["task"]["dataset"])
        input_dir = self.env.get("paths", {}).get("input_dir")
        root = cfg.get("root")
        if input_dir and root and "://" not in root and not Path(root).is_absolute():
            cfg["root"] = str(Path(input_dir) / root)
        return _resolve(cfg.pop("target"))(
            **cfg, dev=self.dev, cache_dir=str(cache_dir) if cache_dir else None
        )

    def _evaluate_one(
        self,
        task: Mapping[str, Any],
        model: Mapping[str, Any],
        dataset: Any,
        output_dir: Path | None,
    ) -> Result:
        """Trains and tests one model on one task, once per split seed."""
        task_cfg, split_cfg = task["task"], task["split"]
        task_obj = _resolve(task_cfg["target"])(**task_cfg.get("params", {}))
        split_params = {k: v for k, v in split_cfg["params"].items() if k != "seeds"}
        runs: dict[int, dict[str, Any]] = {}
        for seed in split_cfg["params"]["seeds"]:
            split = _resolve(split_cfg["target"])(seed=seed, **split_params)
            train, val, test = dataset.set_task(task_obj, split=split)  # cached on disk
            if output_dir is not None:
                _save_processors_and_split((train, val, test), output_dir / f"seed{seed}")
            runs[seed] = self._train_and_test(task, model, seed, (train, val, test), output_dir)
            gc.collect()
        seeds = {
            str(seed): {"metrics": {k: v if math.isfinite(v) else None for k, v in metrics.items()}}
            for seed, metrics in runs.items()
        }
        return Result("success", summarize(list(runs.values())), {"seeds": seeds})

    def _train_and_test(
        self,
        task: Mapping[str, Any],
        model: Mapping[str, Any],
        seed: int,
        parts: tuple[Any, Any, Any],
        output_dir: Path | None,
    ) -> dict[str, float]:
        """Trains one model on one seed's (train, val, test).

        Returns:
            The test metrics.
        """
        from pyhealth.datasets import get_dataloader
        from pyhealth.trainer import Trainer
        from pyhealth.utils import set_seed

        train, val, test = parts
        training = deep_merge(task["training"], model.get("training", {}))
        optimizer = training.get("optimizer", {})
        batch_size = training["batch_size"]

        # Trainer adds a log.txt handler to its module logger and never removes
        # it, so later runs would also log into this seed's log.txt; remove it
        # once this seed is done.
        # TODO: temporary solution until the Trainer cleans up its handler.
        trainer_logger = logging.getLogger(Trainer.__module__)
        handlers = list(trainer_logger.handlers)
        try:
            set_seed(seed)
            trainer = Trainer(
                model=_resolve(model["target"])(dataset=train, **model.get("params", {})),
                metrics=training["metrics"],
                device=self.device,
                enable_logging=output_dir is not None,
                output_path=str(output_dir) if output_dir else None,
                exp_name=f"seed{seed}",
            )
            trainer.train(
                train_dataloader=get_dataloader(train, batch_size, shuffle=True, seed=seed),
                val_dataloader=get_dataloader(val, batch_size),
                epochs=training["epochs"],
                optimizer_class=_resolve(optimizer.get("target", "torch.optim.Adam")),
                optimizer_params=optimizer.get("params"),
                **training.get("trainer", {}),
            )
            scores = trainer.evaluate(get_dataloader(test, batch_size))
        finally:
            for handler in trainer_logger.handlers:
                if handler not in handlers:
                    trainer_logger.removeHandler(handler)
                    handler.close()

        return {k: float(v) for k, v in scores.items()}
