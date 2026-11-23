"""Where a run lives on disk: everything under ``run_dir``.

A :class:`RunStore` is the only code that knows the layout of a run directory:

==========================  =================================================
``trials.jsonl``            one line per trial, appended as they finish
``meta.json``               the settings the trials depend on (checked on resume)
``run.json``, ``card.json`` what a report needs besides the trials
``lessons.json``            the proposer's lessons
``preds/trial_<id>.npz``    each trial's out-of-fold predictions
``code/trial_<id>.py``      generated code of code trials
``evaluations.json``,       proposed evaluations and their code
``evaluations/<name>.py``
``analysis/``,              analyses, their results and the findings
``findings.json``
``logs/``                   output of generated code (never shown to an LLM)
``generated/``              one JSON record per piece of generated code, and
                            ``index.json`` listing them (:meth:`RunStore.log_code`)
``test.json``, ...          results of later steps (test, confirm, explain)
==========================  =================================================

Without a ``run_dir`` nothing is written, and results are only kept in memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RunStore:
    """Reads and writes one run's files.

    Args:
        run_dir: The run's directory, or ``None`` to keep nothing on disk.
        lessons_path: A JSON file of lessons shared across runs, keyed by
            search fingerprint.

    Examples:
        >>> store = RunStore(None)
        >>> store.save_result("test.json", {"auc": 0.8}, default=float)
        >>> store.read_json("test.json")
        {'auc': 0.8}
    """

    def __init__(self, run_dir: str | None, lessons_path: str | None = None):
        self.run_dir = run_dir
        self.lessons_path = lessons_path
        self._results: dict[str, Any] = {}
        self._code: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------- paths
    def path(self, relative: str) -> str | None:
        """The absolute path of a run file, or ``None`` without a run_dir."""
        return os.path.join(self.run_dir, relative) if self.run_dir else None

    def exists(self, relative: str) -> bool:
        path = self.path(relative)
        return path is not None and os.path.exists(path)

    def log_dir(self) -> str | None:
        """Where generated code's output goes."""
        return self.path("logs")

    def log_path(self, name: str) -> str | None:
        return self.path(os.path.join("logs", f"{name}.log"))

    # ------------------------------------------------------ text and JSON
    def write_text(self, relative: str, text: str) -> None:
        path = self.path(relative)
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def read_text(self, relative: str) -> str | None:
        if not self.exists(relative):
            return None
        with open(self.path(relative)) as f:
            return f.read()

    def write_json(self, relative: str, data: Any, default=None) -> None:
        self.write_text(relative, json.dumps(data, indent=1, default=default))

    def read_json(self, relative: str) -> Any:
        """A result kept in memory, else the file, else ``None``."""
        if relative in self._results:
            return self._results[relative]
        text = self.read_text(relative)
        return None if text is None else json.loads(text)

    def remove(self, relative: str) -> None:
        """Forgets a result, in memory and on disk."""
        self._results.pop(relative, None)
        if self.exists(relative):
            os.remove(self.path(relative))

    def save_result(self, name: str, out: dict[str, Any], default) -> None:
        """Keeps a result in memory and, with a run_dir, writes it there.

        The in-memory copy is what flags and reports read, so they work
        without a run_dir; it is the same JSON as the file.
        """
        text = json.dumps(out, indent=1, default=default)
        self._results[name] = json.loads(text)
        self.write_text(name, text)

    # ------------------------------------------------------------ trials
    def append_trial(self, record: dict[str, Any]) -> None:
        path = self.path("trials.jsonl")
        if path is None:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_trials(self) -> list[dict[str, Any]]:
        text = self.read_text("trials.jsonl")
        return [json.loads(line) for line in (text or "").splitlines() if line.strip()]

    # -------------------------------------------------------- predictions
    def save_arrays(self, relative: str, **arrays: Any) -> None:
        path = self.path(relative)
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)

    def load_arrays(self, relative: str) -> dict[str, np.ndarray] | None:
        if not self.exists(relative):
            return None
        with np.load(self.path(relative), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    # ----------------------------------------------------------- settings
    def check_settings(self, settings: dict[str, Any]) -> None:
        """Records the run's settings in ``meta.json``, or checks them on resume.

        A changed search space alone is allowed: it only bounds new proposals.

        Raises:
            ValueError: If the run_dir holds a run with different settings.
        """
        stored = self.read_json("meta.json")
        if stored is not None:
            changed = [k for k, v in settings.items() if k in stored and stored[k] != v]
            if changed == ["space"]:
                logger.warning(
                    "%s was started with a different search space; its trials "
                    "are kept and new proposals use the current space",
                    self.run_dir,
                )
                self.write_json("meta.json", settings)
                return
            if changed:
                raise ValueError(
                    f"{self.run_dir} holds a run with a different "
                    f"{', '.join(changed)}; resuming would mix its trials with "
                    "this run's. Use a new run_dir, or delete this one to start "
                    "over."
                )
            return
        if self.exists("trials.jsonl"):
            logger.warning(
                "%s has no meta.json, so its settings cannot be checked; the "
                "current ones are recorded", self.run_dir,
            )
        self.write_json("meta.json", settings)

    # ------------------------------------------------------------ lessons
    def read_shared_lessons(self) -> dict[str, Any]:
        """Lessons per search fingerprint; an older plain list is set aside."""
        if not self.lessons_path or not os.path.exists(self.lessons_path):
            return {}
        with open(self.lessons_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.warning(
            "%s holds lessons without a search fingerprint; they are not "
            "reused and will be replaced",
            self.lessons_path,
        )
        return {}

    def save_lessons(self, lessons: list[str], fingerprint: str, search: str) -> None:
        """Writes the run's lessons, and the shared lessons file if any."""
        self.write_json("lessons.json", lessons)
        if self.lessons_path:
            stored = self.read_shared_lessons()
            stored[fingerprint] = {"search": search, "lessons": lessons}
            with open(self.lessons_path, "w") as f:
                json.dump(stored, f, indent=1)

    # ---------------------------------------------------- generated code
    def generated(self) -> list[dict[str, Any]]:
        """Every logged piece of generated code, oldest first (resumed runs
        included)."""
        if self._code is None:
            index = self.read_json("generated/index.json") or []
            self._code = [self.read_json(f"generated/{row['file']}") or row
                          for row in index]
        return self._code

    def log_code(self, kind: str, code: str, **fields: Any) -> int:
        """Records one piece of generated code, whatever happened to it.

        Every attempt is kept, also those rejected before they ran (a failed
        static check, a repeat) or that failed while running, so a run's
        generated code can be read in one place: ``generated/NNNN_<kind>.json``
        per attempt, listed in ``generated/index.json``. Without a run_dir
        the records are kept in memory only.

        Args:
            kind: ``code`` (a model), ``evaluation`` or ``analysis``.
            code: The source.
            **fields: What is known so far: ``cycle``, ``agent``,
                ``strategy``, ``name``, ``rationale``, ``status``,
                ``reason``...

        Returns:
            The record's number, for :meth:`update_code`.

        Examples:
            >>> store = RunStore(None)
            >>> n = store.log_code("code", "def build_and_train(): ...",
            ...                    status="proposed")
            >>> store.update_code(n, status="scored", score=0.41)
            >>> store.generated()[0]["status"], store.generated()[0]["score"]
            ('scored', 0.41)
        """
        records = self.generated()
        number = len(records)
        record = {"number": number, "kind": kind,
                  "file": f"{number:04d}_{kind}.json",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "code_sha": hashlib.sha256(code.encode()).hexdigest()[:12],
                  **fields, "code": code}
        records.append(record)
        self._write_code(record)
        return number

    def update_code(self, number: int, **fields: Any) -> None:
        """Adds what happened next to a logged piece of code."""
        record = self.generated()[number]
        code = record.pop("code")
        record.update(fields)
        record["code"] = code  # keep the source last, after the summary
        self._write_code(record)

    def _write_code(self, record: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        self.write_json(f"generated/{record['file']}", record, default=str)
        summary = ("number", "kind", "cycle", "agent", "name", "status", "trial_id",
                   "score", "file")
        self.write_json("generated/index.json",
                        [{k: r.get(k) for k in summary} for r in self.generated()],
                        default=str)
