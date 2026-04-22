import logging
from pathlib import Path
from typing import Any

import narwhals as nw

from ..tasks.medlingo_jargon_expansion import MedLingoJargonExpansionTask
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# Expected public export from Flora-jia-jfr/diagnosing_our_datasets:
# datasets/MedLingo/questions.csv with columns word1, word2, question, answer.
_REQUIRED_QUESTION_COLUMNS = frozenset({"word1", "word2", "question", "answer"})


class MedLingoDataset(BaseDataset):
    """MedLingo jargon QA rows from the *Diagnosing our datasets* line of work.

    Public MedLingo data (e.g. ``questions.csv``) is released with the paper
    *Diagnosing our datasets* (Jia, Sontag & Agrawal, CHIL 2025,
    https://arxiv.org/abs/2505.15024). Place ``questions.csv`` under ``root``
    (same layout as ``datasets/MedLingo/questions.csv`` in the paper's data
    repo). Each CSV row becomes one synthetic patient with a single
    ``questions`` event; attributes are ``word1``, ``word2``, ``question``,
    and ``answer`` (column names are matched case-insensitively after load).

    Args:
        root: Directory containing ``questions.csv``.
        dataset_name: Optional override for the dataset name.
        config_path: YAML config path; defaults to ``configs/medlingo.yaml``.
        cache_dir: Optional cache root (see :class:`BaseDataset`).
        num_workers: Workers for task/sample transforms.
        dev: If True, limits to the first 1000 patients (see ``BaseDataset``).

    Note:
        :meth:`default_task` uses ``MedLingoJargonExpansionTask(shot_mode=
        \"one_shot\")`` so ``set_task()`` matches the released CSV prompts.
        Pass ``MedLingoJargonExpansionTask(shot_mode=\"zero_shot\")`` for the
        ablation that rebuilds the prompt from ``word1``/``word2`` only.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        cache_dir=None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default MedLingo config")
            config_path = Path(__file__).parent / "configs" / "medlingo.yaml"
        default_tables = ["questions"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "medlingo",
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @property
    def default_task(self) -> MedLingoJargonExpansionTask:
        """Default MedLingo task using the released one-shot ``question`` text."""
        return MedLingoJargonExpansionTask(shot_mode="one_shot")

    def preprocess_questions(self, df: Any) -> Any:
        """Ensure required MedLingo columns exist after lowercasing names."""
        lf = nw.from_native(df)
        names = set(lf.columns)
        missing = _REQUIRED_QUESTION_COLUMNS - names
        if missing:
            raise ValueError(
                "questions.csv is missing required column(s): "
                f"{sorted(missing)}. Expected columns: "
                f"{sorted(_REQUIRED_QUESTION_COLUMNS)} (case-insensitive)."
            )
        return lf
