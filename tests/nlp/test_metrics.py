import unittest
from pathlib import Path
from typing import List

import pandas as pd

from pyhealth.nlp.metrics import (
    LevenshteinDistanceScoreMethod,
    ScoreContext,
    Scorer,
    ScoreResult,
    ScoreSet,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        s1: str = 'The boy threw the ball. He practiced every day.'
        s2: str = 'The boy threw X the ball. He practiced every day.'
        self.pairs: List[List[str]] = [[s1, s1], [s1, s2]]

    def test_object_graph(self):
        # configure only the edit distance method
        scorer = Scorer(
            methods={'editdistance': LevenshteinDistanceScoreMethod()},
        )
        ss: ScoreSet = scorer.score(ScoreContext(self.pairs))
        self.assertEqual(ScoreSet, type(ss))
        self.assertEqual(2, len(ss.results))
        res1: ScoreResult = ss.results[0]
        self.assertEqual(ScoreResult, type(res1))
        self.assertEqual(1, len(res1.scores))
        self.assertTrue('editdistance' in res1.scores)
        self.assertEqual(1., res1.scores['editdistance'].value)

    def test_pandas(self):
        WRITE: bool = False
        should_file: Path = Path('test-resources/nlp/metrics.csv')
        scorer = Scorer()
        ss: ScoreSet = scorer.score(ScoreContext(self.pairs))
        df: pd.DataFrame = ss.as_dataframe()
        # give tolerance for arch high sig digits that might be off by epsilon
        df = df.round(4)
        if WRITE:
            should_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(should_file, index=False)
        should: pd.DataFrame = pd.read_csv(should_file)
        self.assertEqual(should.to_string(), df.to_string())
