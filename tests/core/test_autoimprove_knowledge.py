"""pyhealth.autoimprove.knowledge: citation checks, curated tables, privacy, mapping."""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from test_autoimprove_search import BASELINE, SPACE, World  # noqa: E402

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
from pyhealth.autoimprove.knowledge import _numbers  # noqa: E402
from pyhealth.autoimprove.proposers import build_feature_prompt  # noqa: E402
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

    def test_negative_effects_and_percentages_are_found(self):
        self.kb.add_documents([Document("PMID:2", (
            "Exacerbation rates did not differ (MD -0.37, 95% CI -0.8 to 0.06). "
            "Wheeze intensity discriminated best (AUC 84% [95% CI, 82%-85%])."))])
        for effect, quote in [
            ({"type": "MD", "value": -0.37, "ci": [-0.8, 0.06]},
             "(MD -0.37, 95% CI -0.8 to 0.06)"),
            ({"type": "AUC", "value": 0.84, "ci": [0.82, 0.85]},
             "(AUC 84% [95% CI, 82%-85%])"),
        ]:
            e = self.verdicts(Evidence("x", effect, "PMID:2", quote))
            self.assertTrue(e.verified, e.reason)
        e = self.verdicts(Evidence("x", {"type": "AUC", "value": 0.86}, "PMID:2",
                                   "(AUC 84% [95% CI, 82%-85%])"))
        self.assertFalse(e.verified)

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

    def test_sign_percent_and_position_mismatches_are_rejected(self):
        self.kb.add_documents([Document("PMID:3", (
            "Symptom days fell (MD -0.37, 95% CI -0.8 to 0.06). "
            "Night waking rose (MD 0.37, 95% CI 0.1 to 0.6)."))])
        quote = "risk of exacerbation (OR 2.4, 95% CI 1.8-3.1)"
        cases = [
            # the "95%" confidence level is not an effect of 0.95
            (Evidence("x", {"type": "OR", "value": 0.95}, "PMID:1", quote),
             "effect size 0.95"),
            # a CI bound cannot stand in for the estimate
            (Evidence("x", {"type": "OR", "value": 1.8, "ci": [1.8, 3.1]}, "PMID:1",
                      quote), "effect size 1.8"),
            # signs must match, both ways
            (Evidence("x", {"type": "MD", "value": 0.37}, "PMID:3",
                      "(MD -0.37, 95% CI -0.8 to 0.06)"), "effect size 0.37"),
            (Evidence("x", {"type": "MD", "value": -0.37}, "PMID:3",
                      "(MD 0.37, 95% CI 0.1 to 0.6)"), "effect size -0.37"),
        ]
        for record, reason in cases:
            e = self.verdicts(record)
            self.assertFalse(e.verified, record.effect)
            self.assertIn(reason, e.reason)

    def test_non_numeric_llm_output_is_rejected_not_a_crash(self):
        quote = "risk of exacerbation (OR 2.4, 95% CI 1.8-3.1)"
        records = [
            {"predictor": "string number", "quote": quote,
             "effect": {"type": "OR", "value": "2.4", "ci": ["1.8", "3.1"]}},
            {"predictor": "null value", "quote": quote,
             "effect": {"type": "OR", "value": None}},
            {"predictor": "bad ci", "quote": quote,
             "effect": {"type": "OR", "value": 2.4, "ci": [1.8]}},
            {"predictor": "words", "quote": quote,
             "effect": {"type": "OR", "value": "about two"}},
            {"predictor": "effect as text", "quote": quote, "effect": "OR 2.4"},
        ]
        kb = KnowledgeBase(llm=ScriptedLLM([_reply({"records": records})]))
        kb.add_documents([Document("PMID:1", ABSTRACT)])
        kb.extract(CARD)
        reasons = {e.predictor: (e.verified, e.reason) for e in kb.verify()}
        self.assertEqual(reasons["string number"], (True, ""))
        self.assertEqual(reasons["null value"],
                         (False, "the record has no effect size"))
        for name in ("bad ci", "words"):
            self.assertEqual(reasons[name],
                             (False, "the effect size is not a number"))
        self.assertEqual(reasons["effect as text"],
                         (False, "the effect is not a record"))

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
            self.assertEqual(records[0].reference, "[3]")
            self.assertEqual(records[0].population, ">=12 yrs")
            self.assertIsNone(records[2].effect["value"])  # a range: no estimate

    def test_table_ids_from_spreadsheet_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "refs.csv")
            with open(path, "w") as f:
                f.write('PMID,Abstract\n'
                        '"=HYPERLINK(""https://pubmed.ncbi.nlm.nih.gov/27825189/"",'
                        '""27825189"")",Text one.\n'
                        '31014587.0,Text two.\n')
            kb = KnowledgeBase()
            kb.add_table_documents(path, ["Abstract"], "PMID")
            self.assertEqual(sorted(kb.documents), ["PMID:27825189", "PMID:31014587"])

    def test_parse_effect(self):
        self.assertEqual(parse_effect("aHR 1.5 (1.1-2.0)")["type"], "aHR")
        self.assertEqual(parse_effect("AUROC 0.84")["type"], "AUC")

    def test_parse_effect_keeps_signs_and_skips_the_confidence_level(self):
        self.assertEqual(parse_effect("RR 1.9 (95% CI 1.6-2.3)")["ci"], [1.6, 2.3])
        md = parse_effect("MD -0.37 (95% CI -0.8 to 0.06)")
        self.assertEqual((md["value"], md["ci"]), (-0.37, [-0.8, 0.06]))
        self.assertEqual(parse_effect("AUC 84%")["value"], 0.84)
        crossing = parse_effect("OR 0.5–1.5")  # no point estimate, no direction
        self.assertEqual((crossing["value"], crossing["direction"], crossing["range"]),
                         (None, "", [0.5, 1.5]))
        self.assertEqual([n for n, _, _ in _numbers("covid-19, 1.8 - 3.1, -2")],
                         [19.0, 1.8, 3.1, -2.0])

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

    def test_fetch_by_id_sends_only_the_ids(self):
        urls = []

        def fetch(url):
            urls.append(url)
            return "Title. Abstract text."

        kb = KnowledgeBase(fetch=fetch)
        docs = kb.fetch_pubmed([33831622.0, "PMID: 31014587", None])
        self.assertEqual([d.id for d in docs], ["PMID:33831622", "PMID:31014587"])
        self.assertTrue(all(re.search(r"id=(\d+)&", u) for u in urls))


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
            # Before any trial, the evidence prompt lists the untried
            # predictor with its citation.
            prompt = build_feature_prompt(
                improver._context("a0", "evidence", None, kind="features"))
            self.assertIn("count_90d__visits", prompt)
            self.assertIn("PMID:1", prompt)
            result = improver.run()
            self.assertEqual([s["name"] for s in result.best.features],
                             ["count_90d__visits"])
            # Once tried, it is no longer offered as evidence.
            self.assertNotIn("count_90d__visits",
                             "\n".join(improver._evidence_lines()))
            text = improver.report().to_markdown()
            self.assertIn("## Literature", text)
            self.assertIn("Gap report", text)
            self.assertIn("PMID:1", text)


if __name__ == "__main__":
    unittest.main()
