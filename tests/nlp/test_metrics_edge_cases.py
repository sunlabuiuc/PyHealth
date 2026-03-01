from dataclasses import dataclass
from typing import Iterable, Tuple
from unittest.mock import patch

from tests.base import BaseTestCase
from pyhealth.nlp.metrics import FloatScore, ScoreContext, ScoreMethod, ScoreSet, Scorer


@dataclass
class MissingDependencyScoreMethod(ScoreMethod):
    @classmethod
    def _get_external_modules(cls) -> Tuple[str, ...]:
        return ("missing_test_dependency_package>=0.0.1",)

    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        for _ in context.pairs:
            yield FloatScore(1.0)


@dataclass
class ConstantScoreMethod(ScoreMethod):
    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        for _ in context.pairs:
            yield FloatScore(1.0)


class TestMetricsEdgeCases(BaseTestCase):
    def test_get_missing_modules_handles_failed_installs(self):
        scorer = Scorer(methods={"missing": MissingDependencyScoreMethod()})
        with patch.object(scorer, "_install_all", return_value=False):
            missing = scorer._get_missing_modules()

        self.assertEqual(("missing_test_dependency_package>=0.0.1",), missing)
        self.assertNotIn("missing", scorer.methods)

    def test_as_numpy_supports_empty_score_sets(self):
        score_set = ScoreSet(results=tuple())
        columns, values = score_set.as_numpy()

        self.assertEqual([], columns)
        self.assertEqual((0, 0), values.shape)
        self.assertEqual((0, 0), score_set.as_dataframe().shape)

    def test_scorer_with_empty_pairs_returns_empty_scoreset(self):
        scorer = Scorer(methods={"constant": ConstantScoreMethod()})
        score_set = scorer.score(ScoreContext(pairs=tuple(), methods={"constant"}))

        self.assertEqual(0, len(score_set.results))
        columns, values = score_set.as_numpy()
        self.assertEqual([], columns)
        self.assertEqual((0, 0), values.shape)
