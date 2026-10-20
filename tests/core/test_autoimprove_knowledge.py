"""pyhealth.autoimprove.knowledge: citation checks, curated tables, privacy, mapping."""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from test_autoimprove_search import SPACE, BASELINE, World  # noqa: E402

from pyhealth.autoimprove import (  # noqa: E402
    AutoImprover,
    Document,
    Evidence,
    FeatureSpace,
    KnowledgeBase,
    QueueProposer,
    parse_effect,
    pubmed_query,
)
from pyhealth.tasks import TaskCard  # noqa: E402

ABSTRACT = ("Children with two or more ED visits in the prior year had a higher "
            "risk of exacerbation (OR 2.4, 95% CI 1.8-3.1). Controller use was "
            "associated with fewer exacerbations (RR 0.7, 0.5-0.9).")
CARD = TaskCard(population="children with asthma", outcome="asthma exacerbation")


def _reply(data):
    return f"```json\n{json.dumps(data)}\n```"


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, system, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class TestCitationCheck(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()
        self.kb.add_documents([Document("PMID:1", ABSTRACT)])

    def verdicts(self, record):
        self.kb.evidence = [record]
        return self.kb.verify()[0]

    def test_real_record_is_kept_and_direction_set_from_the_value(self):
        e = self.verdicts(Evidence(
            "prior ED visits", {"type": "OR", "value": 2.4, "ci": [1.8, 3.1],
                                "direction": "-"}, "PMID:1",
            "had a higher risk of exacerbation (OR 2.4, 95% CI 1.8-3.1)"))
        self.assertTrue(e.verified)
        self.assertEqual(e.effect["direction"], "+")  # corrected from the value

    def test_invented_citation_and_numbers_are_rejected(self):
        cases = [
            (Evidence("x", {"type": "OR", "value": 2.4}, "PMID:999",
                      "(OR 2.4, 95% CI 1.8-3.1)"), "not among the gathered"),
            (Evidence("x", {"type": "OR", "value": 3.1}, "PMID:1",
                      "Children with two or more ED visits had triple the risk"),
             "quote is not in the source"),
            (Evidence("x", {"type": "OR", "value": 2.9}, "PMID:1",
                      "risk of exacerbation (OR 2.4, 95% CI 1.8-3.1)"),
             "effect size 2.9 is not in the quote"),
            (Evidence("x", {"type": "OR", "value": 2.4, "ci": [1.5, 3.1]}, "PMID:1",
                      "risk of exacerbation (OR 2.4, 95% CI 1.8-3.1)"),
             "effect size 1.5 is not in the quote"),
        ]
        for record, reason in cases:
            e = self.verdicts(record)
            self.assertFalse(e.verified)
            self.assertIn(reason, e.reason)

    def test_extraction_cannot_choose_its_source(self):
        llm = ScriptedLLM([_reply({"records": [
            {"predictor": "prior ED visits", "effect": {"type": "OR", "value": 2.4,
             "ci": [1.8, 3.1]}, "quote": "risk of exacerbation (OR 2.4, 95% CI "
             "1.8-3.1)", "citation": "PMID:12345"},
            {"predictor": "made up", "effect": {"type": "OR", "value": 5.0},
             "quote": "an invented sentence about OR 5.0"},
        ]})])
        kb = KnowledgeBase(llm=llm)
        kb.add_documents([Document("PMID:1", ABSTRACT)])
        found = kb.extract(CARD)
        self.assertEqual({e.source for e in found}, {"PMID:1"})
        kb.verify()
        self.assertEqual([e.predictor for e in kb.records()], ["prior ED visits"])
        self.assertIn("children with asthma", llm.prompts[0])


class TestTablesAndQueries(unittest.TestCase):
    def test_curated_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "risk_factors.csv")
            with open(path, "w") as f:
                f.write("Risk factor,Effect estimate (95% CI),Population,References\n"
                        "Prior exacerbation,RR 1.94 (1.61–2.32),>=12 yrs,[3]\n"
                        "Controller use,OR 0.62 (0.45-0.85),children,[4]\n"
                        "Prior exacerbation,OR 2.0–4.1,children,[2]\n")
            kb = KnowledgeBase()
            self.assertEqual(kb.add_table_evidence(
                path, "Risk factor", "Effect estimate (95% CI)", "Population",
                "References"), 3)
            records = kb.verify()
            self.assertTrue(all(e.verified for e in records))
            self.assertEqual(records[0].effect, {"type": "RR", "value": 1.94,
                                                 "ci": [1.61, 2.32], "direction": "+"})
            self.assertEqual(records[1].effect["direction"], "-")
            self.assertEqual(kb.direction_of("prior exacerbation"), "+")

    def test_parse_effect(self):
        self.assertEqual(parse_effect("aHR 1.5 (1.1-2.0)")["type"], "aHR")
        self.assertEqual(parse_effect("AUROC 0.84")["type"], "AUC")

    def test_pubmed_queries_hold_task_card_text_only(self):
        urls = []

        def fetch(url):
            urls.append(url)
            if "esearch" in url:
                return json.dumps({"esearchresult": {"idlist": ["111", "222"]}})
            return "Title. Abstract text with OR 1.5."

        kb = KnowledgeBase(fetch=fetch)
        docs = kb.search_pubmed(CARD, max_results=2)
        self.assertEqual([d.id for d in docs], ["PMID:111", "PMID:222"])
        card_words = set(re.findall(r"[a-z]+", "children with asthma asthma "
                                    "exacerbation"))
        allowed = card_words | {"and", "or", "predict", "risk", "factor"}
        for query in kb.queries:
            words = set(re.findall(r"[a-z]+", query.lower()))
            self.assertLessEqual(words, allowed, query)
        self.assertEqual(kb.queries[0], pubmed_query(CARD))
        self.assertEqual(len(urls), 3)  # one search, two abstracts


class TestMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()

    def _kb(self, mappings):
        kb = KnowledgeBase(llm=ScriptedLLM([_reply({"mappings": mappings})]))
        kb.add_documents([Document("PMID:1", "Recent visits raised the risk "
                                             "(OR 3.0). Controller use raised the "
                                             "risk (OR 1.4). Smoke exposure raised "
                                             "the risk (OR 1.6)."),
                          Document("PMID:2", "Controller use lowered the risk "
                                             "(OR 0.8).")])
        kb.add_evidence([
            Evidence("recent visits", {"type": "OR", "value": 3.0}, "PMID:1",
                     "Recent visits raised the risk (OR 3.0)."),
            Evidence("controller use", {"type": "OR", "value": 1.4}, "PMID:1",
                     "Controller use raised the risk (OR 1.4)."),
            Evidence("controller use", {"type": "OR", "value": 0.8}, "PMID:2",
                     "Controller use lowered the risk (OR 0.8)."),
            Evidence("smoke exposure", {"type": "OR", "value": 1.6}, "PMID:1",
                     "Smoke exposure raised the risk (OR 1.6)."),
        ])
        kb.verify()
        return kb

    def test_have_build_and_not_in_data(self):
        fs = FeatureSpace(self.world, time_key="t", min_patients=5,
                          columns={"x": ["noise"]})
        card = TaskCard(population="test patients", outcome="the label",
                        columns={"x": ["noise"]})
        kb = self._kb([
            {"predictor": "recent visits", "status": "build",
             "spec": {"op": "count", "window": 90, "select": {"event_type": "visits"}},
             "population_match": 1.0, "data_support": 1.0},
            {"predictor": "controller use", "status": "build",
             "spec": {"op": "count", "window": 45, "select": {"event_type": "visits"}},
             "population_match": 0.5, "data_support": 0.5},
            {"predictor": "smoke exposure", "status": "not in data",
             "note": "no smoke exposure field"},
        ])
        kb.map(fs, card)
        gap = kb.gap_report()
        self.assertEqual([r["predictor"] for r in gap["build"]], ["recent visits"])
        self.assertEqual(gap["build"][0]["name"], "count_90d__visits")
        refused = next(r for r in gap["not in data"] if r["predictor"] == "controller use")
        self.assertIn("spec refused", refused["note"])
        self.assertIn("smoke exposure", [r["predictor"] for r in gap["not in data"]])
        self.assertEqual(kb.direction_of("controller use"), "")  # sources disagree
        ranked = kb.ranked_specs()
        self.assertEqual(ranked[0]["citations"], ["PMID:1"])
        self.assertEqual(ranked[0]["direction"], "+")
        self.assertIn("visits", kb.llm.prompts[0])  # the menu, not data

        with tempfile.TemporaryDirectory() as tmp:
            kb.save(os.path.join(tmp, "kb.json"))
            again = KnowledgeBase.load(os.path.join(tmp, "kb.json"))
            self.assertEqual(again.gap_report(), gap)

            improver = AutoImprover(
                self.world.train, self.world.val, QueueProposer(ranked),
                baseline=BASELINE, space=SPACE, cv_folds=5, features=fs,
                run_dir=tmp, check=False, knowledge=kb, n_agents=1, n_cycles=2)
            result = improver.run()
            self.assertEqual([s["name"] for s in result.best.features],
                             ["count_90d__visits"])
            text = improver.report().to_markdown()
            self.assertIn("## Literature", text)
            self.assertIn("Gap report", text)
            self.assertIn("PMID:1", text)


if __name__ == "__main__":
    unittest.main()
