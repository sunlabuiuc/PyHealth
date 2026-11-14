"""pyhealth.autoimprove.evidence: evidence in feature prompts, literature in reports."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from test_autoimprove_search import BASELINE, SPACE, World  # noqa: E402

from pyhealth.autoimprove import (  # noqa: E402
    AutoImprover,
    Document,
    Evidence,
    EvidenceProposer,
    FeatureSpace,
    KnowledgeBase,
    add_literature,
    evidence_lines,
    literature_section,
)
from pyhealth.autoimprove.proposers import (  # noqa: E402
    FEATURE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ProposalContext,
)
from pyhealth.tasks import TaskCard  # noqa: E402

CARD = TaskCard(population="test patients", outcome="the label")
VISITS = {"op": "count", "window": 90, "select": {"event_type": "visits"}}
LABS = {"op": "count", "window": 90, "select": {"event_type": "labs"}}


def _reply(data):
    return f"```json\n{json.dumps(data)}\n```"


def _kb(world, mappings):
    """Two verified predictors; the scripted LLM maps them with ``mappings``."""
    kb = KnowledgeBase(llm=lambda system, prompt: _reply({"mappings": mappings}))
    kb.add_documents([
        Document("PMID:1", "Recent visits raised the risk (OR 3.0)."),
        Document("PMID:2", "Lab draws raised the risk (OR 1.5). Wheeze intensity "
                           "discriminated exacerbations (AUC 0.84)."),
    ])
    kb.add_evidence([
        Evidence("recent visits", {"type": "OR", "value": 3.0}, "PMID:1",
                 "Recent visits raised the risk (OR 3.0)."),
        Evidence("lab draws", {"type": "OR", "value": 1.5}, "PMID:2",
                 "Lab draws raised the risk (OR 1.5)."),
        Evidence("wheeze intensity", {"type": "AUC", "value": 0.84}, "PMID:2",
                 "Wheeze intensity discriminated exacerbations (AUC 0.84)",
                 population="children with asthma"),
        Evidence("made up", {"type": "OR", "value": 9.0}, "PMID:9", "invented"),
    ])
    kb.verify()
    kb.map(FeatureSpace(world, time_key="t", min_patients=5), CARD)
    return kb


class ScriptedLLM:
    """Proposes the given feature groups in order; records every prompt."""

    def __init__(self, groups):
        self.groups = list(groups)
        self.calls = []

    def __call__(self, system, prompt):
        self.calls.append((system, prompt))
        if system == FEATURE_SYSTEM_PROMPT:
            spec = self.groups.pop(0) if self.groups else LABS
            return _reply({"features": [{**spec, "rationale": "r"}]})
        if "Rewrite the lessons" in prompt:
            return _reply({"lessons": []})
        return _reply({"model": "LogisticRegression", "model_params": {},
                       "train_params": {}, "rationale": "keep"})


class TestEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()
        cls.kb = _kb(cls.world, [
            {"predictor": "recent visits", "status": "build", "spec": VISITS,
             "population_match": 1.0, "data_support": 1.0},
            {"predictor": "lab draws", "status": "build", "spec": LABS,
             "population_match": 0.5, "data_support": 0.5},
            {"predictor": "wheeze intensity", "status": "not in data",
             "note": "no wheeze scores"},
        ])

    def test_evidence_lines_rank_and_skip_tried(self):
        lines = evidence_lines(self.kb)
        self.assertEqual([line.split(":")[0] for line in lines],
                         ["count_90d__visits", "count_90d__labs"])
        self.assertIn("PMID:1", lines[0])
        self.assertIn("direction +", lines[0])
        self.assertEqual([line.split(":")[0] for line in
                          evidence_lines(self.kb, tried={"count_90d__visits"})],
                         ["count_90d__labs"])
        self.assertEqual(len(evidence_lines(self.kb, k=1)), 1)
        self.assertEqual(evidence_lines(self.kb, k=0), [])

    def test_feature_prompts_list_untried_evidence_only(self):
        llm = ScriptedLLM([VISITS, LABS])
        improver = AutoImprover(
            self.world.train, self.world.val, EvidenceProposer(llm, self.kb),
            baseline=BASELINE, space=SPACE, cv_folds=5,
            features=FeatureSpace(self.world, time_key="t", min_patients=5),
            check=False, n_agents=1, n_cycles=3, feature_cycles=(1, 0))
        improver.run()
        feature_prompts = [p for s, p in llm.calls if s == FEATURE_SYSTEM_PROMPT]
        self.assertGreaterEqual(len(feature_prompts), 3)
        first, second, third = feature_prompts[:3]
        self.assertIn("Published predictors not tried yet", first)
        self.assertIn("count_90d__visits", first.split("not tried yet")[1])
        self.assertIn("PMID:1", first)
        # once proposed, a predictor leaves the evidence list
        self.assertNotIn("count_90d__visits", second.split("not tried yet")[1])
        self.assertIn("count_90d__labs", second.split("not tried yet")[1])
        self.assertNotIn("Published predictors not tried yet", third)  # all tried
        # model prompts are unchanged
        for system, prompt in llm.calls:
            if system == SYSTEM_PROMPT:
                self.assertNotIn("Published predictors", prompt)
        added = [t.added for t in improver.trials if t.added]
        self.assertIn(["count_90d__visits"], added)

    def test_an_adapted_predictor_leaves_the_list(self):
        # The LLM proposes the visits predictor with another window, keeping its
        # citation; the original no longer appears as untried evidence.
        adapted = {**VISITS, "window": 30, "citations": ["PMID:1"]}
        llm = ScriptedLLM([adapted, LABS])
        fs = FeatureSpace(self.world, time_key="t", min_patients=5)
        proposer = EvidenceProposer(llm, self.kb)
        context = ProposalContext("a0", "explore", SPACE, "task", "roc_auc", True,
                                  kind="features", features=fs)
        proposer.propose_features(context)
        proposer.propose_features(context)
        first, second = [p for s, p in llm.calls if s == FEATURE_SYSTEM_PROMPT]
        self.assertIn("count_90d__visits", first.split("not tried yet")[1])
        self.assertNotIn("count_90d__visits", second.split("not tried yet")[1])
        self.assertIn("count_90d__labs", second.split("not tried yet")[1])

    def test_variants_are_matched_on_the_whole_selection(self):
        from pyhealth.autoimprove.evidence import _adapts

        listed = {"name": "count_90d__labs", "op": "count", "window": 90,
                  "select": {"event_type": "labs", "where": {"code": ["eos"]}},
                  "citations": ["PMID:2"]}
        window = {**listed, "name": "count_30d__labs", "window": 30}
        other_lab = {**listed, "name": "count_90d__labs_ige",
                     "select": {"event_type": "labs", "where": {"code": ["ige"]}}}
        self.assertTrue(_adapts(window, listed))  # another window, same lab
        self.assertFalse(_adapts(other_lab, listed))  # same paper, another lab
        self.assertFalse(_adapts({**window, "citations": ["PMID:9"]}, listed))
        ratio = {"name": "ratio_labs", "op": "ratio", "a": listed,
                 "b": other_lab, "citations": ["PMID:2"]}
        self.assertFalse(_adapts(ratio, listed))  # a composite is not a variant

    def test_accepted_features_are_not_listed(self):
        llm = ScriptedLLM([LABS])
        fs = FeatureSpace(self.world, time_key="t", min_patients=5)
        accepted = [{"name": "count_90d__visits", "definition": "visits"}]
        context = ProposalContext("a0", "explore", SPACE, "task", "roc_auc", True,
                                  kind="features", features=fs, accepted=accepted)
        EvidenceProposer(llm, self.kb).propose_features(context)
        prompt = llm.calls[0][1]
        self.assertNotIn("count_90d__visits", prompt.split("not tried yet")[1])
        self.assertIn("count_90d__labs", prompt.split("not tried yet")[1])

    def test_literature_section_and_report(self):
        section = literature_section(self.kb)
        text = "\n".join(str(b) for b in section.blocks)
        self.assertIn("3 published effects passed the citation check (1 rejected", text)
        self.assertIn("Gap report", text)
        self.assertIn("no wheeze scores", text)
        self.assertIn("AUC 0.84 (PMID:2; children with asthma)", text)
        self.assertNotIn("This model on test", text)
        test = {"metric": "roc_auc", "best": {"estimate": 0.81}}
        with_test = "\n".join(str(b) for b in literature_section(self.kb, test).blocks)
        self.assertIn("This model on test: 0.810 (roc_auc).", with_test)

        improver = AutoImprover(
            self.world.train, self.world.val, EvidenceProposer(ScriptedLLM([]), self.kb),
            baseline=BASELINE, space=SPACE, cv_folds=5, check=False, n_agents=1,
            n_cycles=1)
        improver.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report = add_literature(improver.report(), self.kb, path=path)
            for suffix in (".html", ".md"):
                with open(path.removesuffix(".html") + suffix) as f:
                    self.assertIn("Literature", f.read())
        markdown = report.to_markdown()
        self.assertIn("## Literature", markdown)
        self.assertIn("recent visits", markdown)
        self.assertEqual(report.data["literature"]["verified"], 3)
        self.assertEqual(report.data["literature"]["rejected"], 1)
        selected = improver.report().data["selected"]
        matching = {"metric": "roc_auc", "best": {"trial_id": selected,
                                                  "estimate": 0.8}}
        stale = {"metric": "roc_auc", "best": {"trial_id": selected + 99,
                                               "estimate": 0.9}}
        self.assertIn("This model on test: 0.800",
                      add_literature(improver.report(), self.kb,
                                     matching).to_markdown())
        self.assertNotIn("This model on test",  # another trial's result
                         add_literature(improver.report(), self.kb,
                                        stale).to_markdown())
        pr = {"metric": "pr_auc", "best": {"trial_id": selected, "estimate": 0.4}}
        self.assertIn("Published AUC values are not on the scale of pr_auc",
                      add_literature(improver.report(), self.kb, pr).to_markdown())
        self.assertNotIn("not on the scale",  # AUC next to roc_auc: same scale
                         add_literature(improver.report(), self.kb,
                                        matching).to_markdown())


if __name__ == "__main__":
    unittest.main()
