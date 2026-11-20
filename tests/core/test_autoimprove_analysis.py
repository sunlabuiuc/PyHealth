"""pyhealth.autoimprove analysis turns: columns, small-cell suppression, findings."""

import json
import os
import random
import tempfile
import unittest

import numpy as np

from pyhealth.autoimprove import AnalysisSpec, AutoImprover, Param, PipelineConfig
from pyhealth.autoimprove import SearchSpace, sample_columns
from pyhealth.autoimprove.search.analysis import Out, format_results, run_analysis_job
from pyhealth.autoimprove.search.proposers import Proposer
from pyhealth.autoimprove.search.prompts import build_analysis_prompt
from pyhealth.autoimprove.utils.sandbox import ContractError
from pyhealth.datasets import create_sample_dataset

SPACE = SearchSpace(
    train_params={
        "epochs": Param("int", 1, 2, default=1),
        "lr": Param("float", 1e-4, 1e-1, log=True, default=1e-2),
        "batch_size": Param("choice", choices=(16, 32), default=16),
    }
)
SPACE.models = {k: SPACE.models[k] for k in ("LogisticRegression",)}

BY_LENGTH = """\
import numpy as np


def analyze(data, out):
    band = np.where(data.n_conditions > 2, "long", "short")
    out.table("label by record length", by={"band": band},
              values={"label": data.label}, stats=("rate",))
    out.value("training positives", data.label.sum(), n_patients=len(data.label))
    out.note("raw rows stay here: " + str(len(data.patient)))
    return data.patient  # ignored: only `out` leaves the process
"""


def _samples(n, seed, offset=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        codes = rng.sample([f"c{j}" for j in range(8)], rng.choice([2, 3, 3, 4]))
        out.append({"patient_id": f"p{offset + i}", "visit_id": f"v{offset + i}",
                    "conditions": codes, "age": [float(30 + i % 50)],
                    "vitals": [float(i % 7), float(i % 3)],
                    "label": int("c1" in codes)})
    return out


def _datasets():
    schema = ({"conditions": "sequence", "age": "tensor", "vitals": "tensor"},
              {"label": "binary"})
    train = create_sample_dataset(_samples(80, 0), *schema)
    kwargs = {"input_processors": train.input_processors,
              "output_processors": train.output_processors}
    val = create_sample_dataset(_samples(40, 1, 1000), *schema, **kwargs)
    return train, val


class TestOut(unittest.TestCase):
    def test_small_groups_are_withheld_with_their_complement(self):
        patients = np.array([f"a{i}" for i in range(20)] + [f"b{i}" for i in range(15)]
                            + [f"c{i}" for i in range(5)])
        group = np.array(["a"] * 20 + ["b"] * 15 + ["c"] * 5)
        out = Out(patients)
        out.table("t", by={"g": group}, values={"x": np.arange(40.0)}, stats=("mean",))
        rows = {r["g"]: r for r in out.results[0]["rows"]}
        self.assertEqual(rows["a"]["patients"], 20)
        self.assertAlmostEqual(rows["a"]["x_mean"], 9.5)
        # c has 5 patients; b is withheld too, so c can't be had from the total
        for g in ("b", "c"):
            self.assertTrue(rows[g]["withheld"])
            self.assertIsNone(rows[g]["patients"])
            self.assertIsNone(rows[g]["x_mean"])

    def test_groups_count_distinct_patients(self):
        patients = np.array(["p1", "p2", "p3"] * 10 + [f"q{i}" for i in range(12)])
        group = np.array(["few"] * 30 + ["many"] * 12)
        out = Out(patients)
        out.table("t", by={"g": group})
        rows = {r["g"]: r for r in out.results[0]["rows"]}
        self.assertTrue(rows["few"]["withheld"])   # 30 samples, 3 patients
        self.assertTrue(rows["many"]["withheld"])  # its complement

    def test_limits(self):
        out = Out(np.array([f"p{i}" for i in range(60)]))
        with self.assertRaisesRegex(ContractError, "not allowed"):
            out.table("t", by={"g": np.zeros(60)}, values={"x": np.ones(60)},
                      stats=("max",))
        with self.assertRaisesRegex(ContractError, "at most 40"):
            out.table("t", by={"g": np.arange(60)})
        with self.assertRaisesRegex(ContractError, "one per sample"):
            out.table("t", by={"g": np.zeros(5)})
        out.value("v", 3.0, n_patients=5)
        self.assertTrue(out.results[-1]["withheld"])
        out.note("x" * 1000)
        self.assertEqual(len(out.results[-1]["text"]), 300)
        for _ in range(100):
            out.note("y" * 200)
        self.assertTrue(out.serialized()["truncated"])


class TestColumnsAndJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val = _datasets()

    def test_sample_columns(self):
        cols = sample_columns(self.train)
        self.assertEqual(sorted(cols), ["age", "label", "n_conditions", "patient",
                                        "vitals_0", "vitals_1"])
        self.assertEqual(cols["age"][:2].tolist(), [30.0, 31.0])
        self.assertTrue(set(cols["n_conditions"].tolist()) <= {2.0, 3.0, 4.0})
        self.assertNotIn("p0", cols["patient"].tolist())  # hashed

    def test_analysis_runs_in_its_own_process(self):
        spec = AnalysisSpec.from_dict({"name": "length", "question": "q"}, BY_LENGTH)
        results = run_analysis_job(spec, sample_columns(self.train), timeout=120)
        kinds = [r["type"] for r in results["results"]]
        self.assertEqual(kinds, ["table", "value", "note"])
        text = format_results(results)
        self.assertIn("table label by record length by band", text)
        self.assertNotIn("p0", json.dumps(results))
        with self.assertRaisesRegex(ValueError, "import of os"):
            AnalysisSpec.from_dict({"name": "x"},
                                   "import os\ndef analyze(data, out):\n    pass\n")


class AnalysisProposer(Proposer):
    def __init__(self, analyses):
        self.analyses = list(analyses)
        self.contexts = []

    def propose(self, context):
        raise ValueError("no model turns here")

    def propose_analysis(self, context):
        self.contexts.append(context)
        return self.analyses.pop(0)


class TestAnalysisTurns(unittest.TestCase):
    def test_findings_reach_later_prompts(self):
        train, val = _datasets()
        failing = AnalysisSpec("broken", "def analyze(data, out):\n"
                                         "    out.table('t', by={'g': data.nope})\n")
        spec = AnalysisSpec.from_dict(
            {"name": "length", "question": "Is the label rarer in long records?"},
            BY_LENGTH)
        proposer = AnalysisProposer([failing, spec])
        with tempfile.TemporaryDirectory() as tmp:
            improver = AutoImprover(
                train, val, proposer,
                baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
                space=SPACE, n_agents=1, n_cycles=1, cv_folds=2,
                cycle_plan=("analysis",), run_dir=tmp, check=False,
            )
            improver.run()
            # the failure went back to the proposer, then the retry ran
            self.assertIn("your last analysis failed: ContractError: no column",
                          proposer.contexts[1].feedback)
            self.assertEqual(len(improver.findings), 1)
            self.assertIn("Is the label rarer in long records?", improver.findings[0])
            prompt = build_analysis_prompt(proposer.contexts[0])
            self.assertIn("oof_prob", prompt)  # the baseline's predictions
            self.assertIn("fold", prompt)
            context = improver._context("agent0", "explore", "")
            self.assertEqual(list(context.findings), improver.findings)
            with open(os.path.join(tmp, "findings.json")) as f:
                self.assertEqual(json.load(f), improver.findings)
            # numbered by finished analyses; the failed attempt keeps its code
            self.assertTrue(os.path.exists(os.path.join(tmp, "analysis",
                                                        "0_length.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "analysis",
                                                        "0_broken.py")))
            report = improver.report()
            self.assertIn("Findings from data analysis",
                          [s.title for s in report.sections])


if __name__ == "__main__":
    unittest.main()
