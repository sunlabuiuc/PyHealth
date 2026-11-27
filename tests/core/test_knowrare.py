import copy
import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import KnowRare
from pyhealth.models.knowrare import (
    ConditionKnowledgeGraph,
    TuckER,
    select_source_conditions,
)
from pyhealth.tasks import KnowRareICUMortalityMIMIC4
from pyhealth.tasks.knowrare_mimic4 import (
    _fill_both_ways,
    icd_category,
    select_rare_conditions,
)
from pyhealth.trainer import Trainer

T0 = datetime(2150, 1, 1, 8, 0, 0)  # noqa: DTZ001 - MIMIC times are naive
MICU = "Medical Intensive Care Unit (MICU)"
HR = KnowRareICUMortalityMIMIC4.DYNAMIC_FEATURES.index("hr")


# ------------------------------------------------------------------ helpers
def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _fmt(time):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _patient(hr_hours, diagnoses=(), drugs=(), hours=48):
    """One adult, one admission (hadm 1) and one ICU stay (stay 10)."""
    admit, disch = T0 - timedelta(hours=10), T0 + timedelta(hours=60)
    rows = [
        _event("patients", None, anchor_age=60, gender="F"),
        _event("admissions", admit, hadm_id=1, dischtime=_fmt(disch),
               deathtime=None, hospital_expire_flag=0),
        _event("transfers", admit, hadm_id=1, eventtype="admit", careunit=MICU),
        _event("transfers", disch, hadm_id=1, eventtype="discharge", careunit=None),
        _event("icustays", T0, stay_id=10, hadm_id=1, first_careunit=MICU,
               outtime=_fmt(T0 + timedelta(hours=hours))),
    ]
    for hour in hr_hours:
        rows.append(
            _event("chartevents", T0 + timedelta(hours=hour, minutes=5), stay_id=10,
                   hadm_id=1, itemid=220045, valuenum=80 + hour, valueuom="bpm")
        )
    for code, version, seq in diagnoses:
        rows.append(_event("diagnoses_icd", disch, hadm_id=1, icd_code=code,
                           icd_version=version, seq_num=seq))
    for drug in drugs:
        rows.append(_event("prescriptions", T0, hadm_id=1, drug=drug))
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


def _samples(n=24, steps=6, features=3, conditions="abc", seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        condition = conditions[i % len(conditions)]
        label = int(rng.random() < 0.4) if i > 1 else i  # both classes present
        series = rng.normal(size=(steps, features)) + label
        out.append({
            "patient_id": f"p{i}",
            "series": series.astype(np.float32),
            "static": [float(rng.normal()), float(i % 2)],
            "condition": condition,
            "diagnoses": [condition, conditions[(i + 1) % len(conditions)]],
            "drugs": [f"drug{i % 4}", "common"],
            "y": label,
        })
    return out


def _dataset(samples, output="binary"):
    return create_sample_dataset(
        samples, {"series": "tensor", "static": "tensor"}, {"y": output}
    )


def _batch(dataset, size=None):
    return next(iter(get_dataloader(dataset, batch_size=size or len(dataset))))


# --------------------------------------------------------------------- task
class TestTaskHelpers(unittest.TestCase):
    def test_icd_category(self):
        self.assertEqual(icd_category("0389", 9), "ICD9:038")
        self.assertEqual(icd_category("V4581", "9"), "ICD9:V45")
        self.assertEqual(icd_category("E8790", 9), "ICD9:E879")
        self.assertEqual(icd_category("a41.9", 10), "ICD10:A41")

    def test_fill_both_ways(self):
        x = np.array([[np.nan, np.nan], [1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]])
        out = _fill_both_ways(x)
        np.testing.assert_array_equal(out[:, 0], [1.0, 1.0, 1.0, 3.0])
        self.assertTrue(np.isnan(out[:, 1]).all())  # left to the processor
        self.assertTrue(np.isnan(x[0, 0]))  # input untouched

    def test_select_rare_conditions(self):
        conditions = ["a"] * 3 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2 + ["e"] * 9
        labels = [1, 0, 0] + [0, 1] + [0, 0] + [1, 0] + [1] * 9
        common = {"n": 10, "min_patients": 2, "max_prevalence": 0.2}
        # 18 patients: "e" is too common, "c" has no positive label.
        self.assertEqual(select_rare_conditions(conditions, labels, **common),
                         ["b", "d", "a"])
        self.assertEqual(
            select_rare_conditions(conditions, labels, required=[["a", "d"]],
                                   **common),
            ["d", "a"],
        )
        # Prevalence and the minimum count are per patient, not per sample.
        pids = ["x", "x", "y"] + [f"q{i}" for i in range(15)]
        self.assertEqual(
            select_rare_conditions(conditions, labels, patient_ids=pids, **common),
            ["a", "b", "d"],  # "a" now has 2 patients; ties break by name
        )
        self.assertEqual(
            select_rare_conditions(conditions, labels, patient_ids=pids, n=10,
                                   min_patients=3, max_prevalence=None),
            ["e"],
        )


class TestKnowRareICUMortalityMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = KnowRareICUMortalityMIMIC4()

    def test_sample_fields(self):
        patient = _patient(
            hr_hours=[2, *range(5, 48)],
            diagnoses=[("4280", 9, 2), ("0389", 9, 1), ("42731", 9, 3),
                       ("I214", 10, 4)],
            drugs=["Heparin ", "heparin", "Furosemide"],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["condition"], "ICD9:038")  # seq_num 1
        self.assertEqual(sample["diagnoses"],
                         ["ICD10:I21", "ICD9:038", "ICD9:427", "ICD9:428"])
        self.assertEqual(sample["drugs"], ["furosemide", "heparin"])
        self.assertEqual(sample["hadm_id"], 1)
        self.assertEqual(sample["mortality"], 0)

        dynamic = sample["dynamic"]
        self.assertEqual(dynamic.shape, (24, len(self.task.DYNAMIC_FEATURES)))
        # Backward fill before the first measurement, forward fill in gaps.
        np.testing.assert_allclose(dynamic[:5, HR], [82, 82, 82, 82, 82])
        np.testing.assert_allclose(dynamic[5:, HR], 80 + np.arange(5, 24))
        # Never measured: left missing for the processor (training mean).
        self.assertTrue(np.isnan(dynamic[:, HR + 1]).all())

    def test_observation_hours(self):
        task = KnowRareICUMortalityMIMIC4(observation_hours=12)
        patient = _patient(range(48), diagnoses=[("0389", 9, 1)])
        self.assertEqual(task(patient)[0]["dynamic"].shape[0], 12)
        with self.assertRaises(ValueError):
            KnowRareICUMortalityMIMIC4(observation_hours=0)

    def test_stay_without_diagnosis_is_excluded(self):
        self.assertEqual(self.task(_patient(range(48))), [])

    def test_short_stay_is_excluded(self):
        patient = _patient(range(29), diagnoses=[("0389", 9, 1)], hours=29)
        self.assertEqual(self.task(patient), [])

    def test_pre_filter_requires_graph_tables(self):
        columns = {
            "patient_id": ["p1"], "event_type": ["icustays"], "timestamp": [T0],
            "labevents/itemid": ["50912"], "chartevents/itemid": ["220045"],
            "outputevents/itemid": ["226559"], "transfers/careunit": [None],
        }
        with self.assertRaisesRegex(ValueError, "diagnoses_icd"):
            self.task.pre_filter(pl.LazyFrame(columns))
        columns["diagnoses_icd/icd_code"] = [None]
        columns["prescriptions/drug"] = [None]
        self.task.pre_filter(pl.LazyFrame(columns)).collect()

    def test_samples_fit_the_schema(self):
        patient = _patient(range(48), diagnoses=[("0389", 9, 1)], drugs=["x"])
        samples = self.task(patient)
        samples.append({**samples[0], "mortality": 1})
        dataset = create_sample_dataset(
            samples, self.task.input_schema, self.task.output_schema
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["dynamic"].shape), (24, 48))
        self.assertFalse(torch.isnan(sample["dynamic"]).any())
        self.assertEqual(sample["condition"], "ICD9:038")


# -------------------------------------------------------------------- graph
class TestConditionKnowledgeGraph(unittest.TestCase):
    def test_relations_follow_the_paper(self):
        samples = [
            {"condition": "a", "diagnoses": ["a", "b", "z"], "drugs": ["x", "y"],
             "record": [[0.0], [2.0]], "hadm_id": 1},
            # Same admission: its diagnoses are counted once.
            {"condition": "a", "diagnoses": ["a", "b", "z"], "drugs": [],
             "record": [[0.0], [2.0]], "hadm_id": 1},
            {"condition": "b", "diagnoses": ["b", "c"], "drugs": ["y"],
             "record": [[4.0], [4.0]], "hadm_id": 2},
            {"condition": "c", "diagnoses": ["c", "a"], "drugs": ["q"],
             "record": [[10.0], [10.0]], "hadm_id": 3},
        ]
        graph = ConditionKnowledgeGraph.from_samples(
            samples, record_key="record", record_top_ratio=1.0, drug_top_ratio=1.0
        )
        self.assertEqual(graph.conditions, ["a", "b", "c"])
        # Eq. 1: co-occurrence normalized by the row sum ("z" is not a node).
        np.testing.assert_allclose(
            graph.weights["diagnosis"],
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )
        # Eqs. 2-3: s_v = [mean, std]; a = [1, 1], b = [4, 0], c = [10, 0].
        expected_ab = 1.0 / (1.0 + np.hypot(3.0, 1.0))
        self.assertAlmostEqual(graph.weights["record"][0, 1], expected_ab)
        self.assertAlmostEqual(graph.weights["record"][1, 2], 1.0 / 7.0)
        self.assertEqual(graph.weights["record"][1, 1], 0.0)
        # Eq. 4: Jaccard of drug sets.
        self.assertAlmostEqual(graph.weights["drug"][0, 1], 0.5)
        self.assertEqual(graph.weights["drug"][0, 2], 0.0)

    def test_top_ratio_and_triples(self):
        samples = [
            {"condition": c, "record": [[float(v)]]}
            for c, v in [("a", 0), ("b", 1), ("c", 5)]
        ]
        graph = ConditionKnowledgeGraph.from_samples(
            samples, diagnoses_key=None, drugs_key=None, record_key="record",
            record_top_ratio=1 / 3,
        )
        self.assertEqual(graph.relations, ["record"])
        # Only the closest pair (a, b) survives, in both directions.
        self.assertEqual(graph.triples().tolist(), [[0, 0, 1], [1, 0, 0]])
        self.assertEqual(graph.triples(top_edge_ratio=0.5).shape, (1, 3))

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            ConditionKnowledgeGraph(["a", "b"], {"drug": np.zeros((3, 3))})


class TestTuckERAndSelection(unittest.TestCase):
    def test_fit_learns_the_graph(self):
        torch.manual_seed(0)
        triples = np.array([[h, 0, t] for h in range(4) for t in range(4)
                            if h != t and (h < 2) == (t < 2)])
        model = TuckER(4, 1, entity_dim=8, relation_dim=4, input_dropout=0.0,
                       hidden_dropout1=0.0, hidden_dropout2=0.0)
        losses = model.fit(triples, epochs=150, lr=0.01, batch_size=4, seed=0)
        self.assertLess(losses[-1], losses[0])
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.tensor([0]), torch.tensor([0])))[0]
        self.assertGreater(scores[1].item(), scores[2].item())
        self.assertEqual(model.entity_embeddings.shape, (4, 8))

    def test_fit_needs_triples(self):
        with self.assertRaises(ValueError):
            TuckER(2, 1).fit(np.zeros((0, 3), dtype=np.int64))

    def test_select_source_conditions(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [-1.0, 0.0]])
        names = ["t", "far", "near", "opposite"]
        self.assertEqual(select_source_conditions(emb, names, "t", 2),
                         ["near", "far"])
        self.assertEqual(len(select_source_conditions(emb, names, "t", 10)), 3)


# -------------------------------------------------------------------- model
class TestKnowRare(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset(_samples())
        self.model = KnowRare(self.dataset, temporal_key="series", hidden_dim=16)
        self.batch = _batch(self.dataset)

    def test_forward(self):
        out = self.model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (24, 1))
        self.assertEqual(out["embed"].shape, (24, 32))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertNotIn("loss_adv", out)

    def test_forward_without_labels(self):
        batch = {k: v for k, v in self.batch.items() if k != "y"}
        out = self.model(**batch)
        self.assertEqual(set(out), {"logit", "y_prob", "embed"})

    def test_without_static(self):
        model = KnowRare(self.dataset, temporal_key="series", static_key=None,
                         hidden_dim=8)
        self.assertEqual(model(**self.batch)["embed"].shape, (24, 8))

    def test_multiclass(self):
        samples = _samples()
        for i, s in enumerate(samples):
            s["y"] = i % 3
        dataset = _dataset(samples, "multiclass")
        model = KnowRare(dataset, temporal_key="series", hidden_dim=8)
        model.set_domains("a", ["b"], conditions=[s["condition"] for s in samples])
        model.train()
        out = model(**_batch(dataset))
        self.assertEqual(out["y_prob"].shape, (24, 3))
        self.assertIn("loss_adv", out)

    def test_rejects_regression_and_flat_series(self):
        samples = _samples()
        with self.assertRaises(ValueError):
            KnowRare(_dataset(samples, "regression"), temporal_key="series")
        with self.assertRaises(ValueError):
            KnowRare(self.dataset, temporal_key="static")

    def test_inverse_prevalence_weights(self):
        conditions = ["a"] * 1 + ["b"] * 3
        self.model.set_domains("a", ["b", "a"], conditions=conditions)
        self.assertEqual(self.model.domains, ["a", "b"])
        weights = self.model.condition_weights
        self.assertAlmostEqual(weights["a"], 2.0)
        self.assertAlmostEqual(weights["b"], 2.0 / 3.0)
        mean = np.mean([weights[c] for c in conditions])
        self.assertAlmostEqual(mean, 1.0)

    def test_weighted_loss(self):
        self.model.eval()
        plain = self.model(**self.batch)
        conditions = self.batch["condition"]
        self.model.set_domains("a", ["b", "c"], conditions=conditions[:3])
        weighted = self.model(**self.batch)  # eval: no adversarial term
        per_sample = F.binary_cross_entropy_with_logits(
            plain["logit"], self.batch["y"].float(), reduction="none"
        )[:, 0]
        w = torch.tensor([self.model.condition_weights[c] for c in conditions])
        self.assertTrue(torch.allclose(weighted["loss"], (per_sample * w).mean()))

    def test_gradient_reversal(self):
        conditions = self.batch["condition"]
        grads = {}
        for weight in (0.0, 0.5):
            model = copy.deepcopy(self.model)
            model.adversarial_weight = weight
            torch.manual_seed(1)
            model.set_domains("a", ["b"], conditions=conditions)
            model.train()
            out = model(**self.batch)
            out["loss"].backward()
            grads[weight] = (
                model.temporal_encoder.weight_ih_l0.grad.clone(),
                model.discriminator[0].weight.grad,
            )
            if weight > 0:
                # Only conditions "a" and "b" are domains; "c" is ignored.
                known = [c in ("a", "b") for c in conditions]
                self.assertIn("loss_adv", out)
                model.zero_grad()
                embed = model.encode(**self.batch)
                joint = torch.cat([embed, model.label_processor.activation(model.classifier(embed))],
                                  -1)[known]
                domain = torch.tensor([0 if c == "a" else 1
                                       for c, k in zip(conditions, known) if k])
                F.cross_entropy(model.discriminator(joint), domain).backward()
                adv_grad = model.temporal_encoder.weight_ih_l0.grad
                # Encoder gradient = prediction gradient - weight * adversarial.
                self.assertTrue(torch.allclose(
                    grads[weight][0], grads[0.0][0] - weight * adv_grad, atol=1e-6))
        self.assertIsNone(grads[0.0][1])  # weight 0: no adversarial term
        self.assertIsNotNone(grads[0.5][1])

    def test_pretrain_loss_matches_per_prefix_decoding(self):
        series = self.batch["series"]
        expected = []
        for t in range(1, series.shape[1]):
            prefix = {**self.batch, "series": series[:, :t]}
            expected.append(self.model.decoder(self.model.encode(**prefix)))
        expected = F.mse_loss(torch.stack(expected, 1), series[:, 1:])
        self.assertTrue(torch.allclose(self.model.pretrain_loss(**self.batch),
                                       expected, atol=1e-6))

    def test_pretrain(self):
        loader = get_dataloader(self.dataset, batch_size=8, shuffle=True, seed=0)
        classifier = copy.deepcopy(self.model.classifier.state_dict())
        history = self.model.pretrain(loader, loader, epochs=15, lr=1e-2, patience=None)
        self.assertEqual(len(history["train"]), 15)
        self.assertLess(min(history["val"]), history["val"][0])
        for key, value in self.model.classifier.state_dict().items():
            self.assertTrue(torch.equal(value, classifier[key]))  # untouched

    def test_trains_with_trainer(self):
        conditions = [s["condition"] for s in self.dataset]
        self.model.set_domains("a", ["b", "c"], conditions=conditions)
        trainer = Trainer(self.model, metrics=["roc_auc"], enable_logging=False)
        loader = get_dataloader(self.dataset, batch_size=8, shuffle=True, seed=0)
        trainer.train(loader, loader, epochs=2, monitor="roc_auc")
        self.assertIn("roc_auc", trainer.evaluate(loader))


if __name__ == "__main__":
    unittest.main()
