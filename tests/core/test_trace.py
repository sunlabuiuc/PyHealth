"""Tests for TRACE, the lab nowcasting tasks and the top-k multilabel metrics."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import multilabel_metrics_fn, multilabel_ranking_at_k
from pyhealth.models import TRACE, TRACEAttention, TRACETimeEmbedding
from pyhealth.tasks import LabNowcastingMIMIC3, LabNowcastingMIMIC4, lab_flag
from pyhealth.trainer import Trainer

T0 = datetime(2150, 1, 1, 8, 0)
DEMO_ROOT = Path(__file__).resolve().parents[2] / "test-resources/core/mimic4demo"


# ------------------------------------------------------------------ metrics
class TestRankingMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        self.y_prob = np.array(
            [[0.9, 0.8, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]]
        )

    def test_values_by_hand(self):
        out = multilabel_ranking_at_k(self.y_true, self.y_prob, 2)
        # precision over all 3 samples: hits 1, 0, 0 out of 2 each
        self.assertAlmostEqual(out["precision"], 1 / 6)
        # recall and NDCG over the 2 samples with a true label
        self.assertAlmostEqual(out["recall"], (1 / 2 + 0) / 2)
        ideal = 1 + 1 / np.log2(3)
        self.assertAlmostEqual(out["ndcg"], (1 / ideal + 0) / 2)

    def test_perfect_ranking_and_cutoff_clipping(self):
        y_true = np.array([[1, 1, 0]])
        out = multilabel_ranking_at_k(y_true, np.array([[0.9, 0.8, 0.1]]), 10)
        self.assertAlmostEqual(out["ndcg"], 1.0)
        self.assertAlmostEqual(out["recall"], 1.0)
        self.assertAlmostEqual(out["precision"], 2 / 3)  # k clipped to 3

    def test_through_multilabel_metrics_fn(self):
        out = multilabel_metrics_fn(
            self.y_true[:2], self.y_prob[:2], metrics=["precision@1", "ndcg@3"]
        )
        self.assertEqual(set(out), {"precision@1", "ndcg@3"})
        self.assertAlmostEqual(out["precision@1"], 0.5)
        for bad in ("ndcg@0", "precision@x"):
            with self.assertRaises(ValueError):
                multilabel_metrics_fn(self.y_true, self.y_prob, metrics=[bad])


# ------------------------------------------------------------------ the task
def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _lab(hours, itemid, value, lower=None, upper=None, flag=None):
    return _event(
        "labevents", T0 + timedelta(hours=hours), itemid=itemid, valuenum=value,
        ref_range_lower=lower, ref_range_upper=upper, flag=flag,
    )


def _patient(rows, discharge_hours=100):
    rows = [
        _event("patients", None, anchor_age=60, gender="F"),
        _event("admissions", T0, hadm_id=7,
               dischtime=(T0 + timedelta(hours=discharge_hours)).strftime(
                   "%Y-%m-%d %H:%M:%S")),
    ] + rows
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


def _trajectory():
    return [
        _lab(-5, 1, 9.0, 3.5, 5.1),           # before admission: excluded
        _lab(1, 1, 6.0, 3.5, 5.1),            # high
        _lab(1, 2, 100.0, 70, 100),           # normal (bound inclusive)
        _event("prescriptions", T0 + timedelta(hours=2), drug="Heparin"),
        _lab(10, 1, 3.0, 3.5, 5.1),           # low
        _lab(10, 3, None, None, None, "abnormal"),  # flag column only
        _lab(30, 2, 120.0, 70, 100),          # last draw: high
        _lab(30, 1, 4.0, 3.5, 5.1, "abnormal"),  # range wins over flag
        _lab(200, 1, 4.0, 3.5, 5.1),          # after discharge: excluded
    ]


class TestLabNowcastingTask(unittest.TestCase):
    def test_lab_flag(self):
        self.assertEqual(lab_flag("6.2", "3.5", "5.1"), "H")
        self.assertEqual(lab_flag(3.0, 3.5, None), "L")
        self.assertEqual(lab_flag(5.1, 3.5, 5.1), "N")
        self.assertEqual(lab_flag(None, 3.5, 5.1, "abnormal"), "A")
        self.assertEqual(lab_flag("pending", None, None, None), "N")

    def test_last_draw_sample(self):
        (sample,) = LabNowcastingMIMIC4()(_patient(_trajectory()))
        self.assertEqual(sample["visit_id"], "7")
        self.assertEqual(sample["labs"], ["LAB:1:N", "LAB:2:H"])
        self.assertEqual(
            sample["events"],
            ["LAB:1:H", "LAB:2:N", "RX:Heparin", "LAB:1:L", "LAB:3:A"],
        )
        self.assertEqual(sample["event_times"], [29.0, 29.0, 28.0, 20.0, 20.0])

    def test_all_draws_min_history_and_truncation(self):
        patient = _patient(_trajectory())
        samples = LabNowcastingMIMIC4(target="all")(patient)
        # the first draw (hour 1) has no earlier event in the admission
        self.assertEqual([s["labs"] for s in samples],
                         [["LAB:1:L", "LAB:3:A"], ["LAB:1:N", "LAB:2:H"]])
        self.assertEqual(LabNowcastingMIMIC4(target="all", min_history=4)(patient),
                         samples[1:])
        (short,) = LabNowcastingMIMIC4(max_events=2)(patient)
        self.assertEqual(short["events"], ["LAB:1:L", "LAB:3:A"])
        self.assertEqual(short["event_times"], [20.0, 20.0])

    def test_mimic3_flags_and_tables(self):
        rows = [
            _event("labevents", T0 + timedelta(hours=1), itemid=1, valuenum=9,
                   flag="abnormal"),
            _event("labevents", T0 + timedelta(hours=1), itemid=2, valuenum=1,
                   flag=None),
            _event("prescriptions", T0 + timedelta(hours=2), drug="Heparin"),
            _event("labevents", T0 + timedelta(hours=5), itemid=1, valuenum=4,
                   flag=None),
        ]
        (sample,) = LabNowcastingMIMIC3()(_patient(rows))
        self.assertEqual(sample["events"], ["LAB:1:A", "LAB:2:N"])  # labs only
        self.assertEqual(sample["labs"], ["LAB:1:N"])
        (with_rx,) = LabNowcastingMIMIC3(
            event_tables=["labevents", "prescriptions"]
        )(_patient(rows))
        self.assertIn("RX:Heparin", with_rx["events"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            LabNowcastingMIMIC4(event_tables=["prescriptions"])
        with self.assertRaises(ValueError):
            LabNowcastingMIMIC4(event_tables=["labevents", "unknown"])
        with self.assertRaises(ValueError):
            LabNowcastingMIMIC4(target="first")

    def test_mimic4_demo_fixture(self):
        from pyhealth.datasets import MIMIC4EHRDataset

        with tempfile.TemporaryDirectory() as tmp:
            dataset = MIMIC4EHRDataset(
                root=str(DEMO_ROOT), tables=["labevents", "prescriptions"],
                cache_dir=tmp,
            )
            samples = dataset.set_task(LabNowcastingMIMIC4(), num_workers=1)
            self.assertGreater(len(samples), 0)
            labels = samples.output_processors["labs"].label_vocab
            # reference ranges reach the task: some labs are flagged H or L
            self.assertTrue(any(c.endswith((":H", ":L")) for c in labels))
            samples.close()


# ------------------------------------------------------------------ the model
def _dataset(n=12, seed=0, with_labels=True):
    rng = np.random.default_rng(seed)
    codes = [f"LAB:{i}:{f}" for i in range(4) for f in "HLN"] + ["RX:a", "RX:b"]
    samples = []
    for i in range(n):
        length = int(rng.integers(1, 9))
        sample = {
            "patient_id": f"p{i}",
            "events": list(rng.choice(codes, size=length)),
            "event_times": sorted(rng.uniform(0.5, 72, size=length).tolist(),
                                  reverse=True),
        }
        if with_labels:
            sample["labs"] = sorted(set(rng.choice(codes[:12], size=3)))
        samples.append(sample)
    output = {"labs": "multilabel"} if with_labels else {}
    return create_sample_dataset(
        samples, {"events": "sequence", "event_times": "tensor"}, output
    )


def _small(dataset, **kwargs):
    defaults = {"embedding_dim": 16, "num_layers": 2, "num_heads": 2, "max_len": 12}
    defaults.update(kwargs)
    return TRACE(dataset, **defaults)


class TestTRACEPieces(unittest.TestCase):
    def test_time_embedding_is_periodic_in_its_periodic_part(self):
        emb = TRACETimeEmbedding(8, period=24.0, decay_scale=1e9)
        t = torch.tensor([[1.0, 25.0]])
        out = emb(t)
        self.assertEqual(out.shape, (1, 2, 8))
        # with a huge decay scale the decay part is constant, so t and t + 24
        # differ only through float error
        self.assertTrue(torch.allclose(out[0, 0], out[0, 1], atol=1e-5))

    def test_attention_masks_padding_and_uses_mask_block(self):
        torch.manual_seed(0)
        attention = TRACEAttention(8, 2, max_len=6, dropout=0.0)
        x = torch.randn(1, 4, 8)
        keep = torch.tensor([[False, True, True, True]])
        changed = x.clone()
        changed[0, 0] = 100.0  # the padded position must not matter
        out = attention(x, keep)
        self.assertTrue(torch.allclose(out[0, 1:], attention(changed, keep)[0, 1:],
                                       atol=1e-5))
        self.assertEqual(attention.smoothed_mask(4).shape, (4, 4))
        self.assertTrue(torch.equal(attention.smoothed_mask(4),
                                    attention.smoothed_mask()[-4:, -4:]))


class TestTRACEModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()

    def test_forward_backward(self):
        torch.manual_seed(0)
        model = _small(self.dataset)
        self.assertEqual(model.mode, "multilabel")
        self.assertEqual((model.feature_key, model.time_key),
                         ("events", "event_times"))
        batch = next(iter(get_dataloader(self.dataset, batch_size=5)))
        out = model(**batch, embed=True)
        n_labels = self.dataset.output_processors["labs"].size()
        self.assertEqual(out["y_prob"].shape, (5, n_labels))
        self.assertEqual(out["embed"].shape, (5, 16))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
        out["loss"].backward()
        mask = model.layers[0].attention.mask_logits
        self.assertIsNotNone(mask.grad)

    def test_denoise_term(self):
        model = _small(self.dataset).eval()
        batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        with torch.no_grad():
            with_penalty = model(**batch)["loss"]
            model.denoise_weight = 0.0
            without = model(**batch)["loss"]
        self.assertAlmostEqual(
            (with_penalty - without).item(), 0.01 * model.denoise_loss().item(),
            places=4,
        )

    def test_padding_and_batch_invariance(self):
        torch.manual_seed(0)
        model = _small(self.dataset).eval()
        batch = next(iter(get_dataloader(self.dataset, batch_size=12)))
        with torch.no_grad():
            together = model(**batch)["logit"]
            for i in (0, 5):
                n = int((batch["events"][i] != 0).sum())
                alone = model(events=batch["events"][i:i + 1, :n],
                              event_times=batch["event_times"][i:i + 1, :n])["logit"]
                self.assertTrue(torch.allclose(together[i], alone[0], atol=1e-5))

    def test_long_sequences_keep_the_last_events(self):
        torch.manual_seed(0)
        model = _small(self.dataset, max_len=4).eval()
        codes = torch.tensor([[5, 6, 7, 8, 9, 3]])
        times = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
        aligned, aligned_times = model._left_align(codes, times)
        self.assertEqual(aligned.tolist(), [[8, 9, 3, model.mask_index]])
        self.assertEqual(aligned_times.tolist(), [[6.0, 5.0, 4.0, 0.0]])
        with torch.no_grad():
            full = model(events=codes, event_times=times)["logit"]
            last = model(events=codes[:, -3:], event_times=times[:, -3:])["logit"]
        self.assertTrue(torch.allclose(full, last, atol=1e-5))

    def test_inference_without_labels(self):
        model = _small(self.dataset).eval()
        unlabeled = _dataset(n=3, with_labels=False)
        batch = next(iter(get_dataloader(unlabeled, batch_size=3)))
        out = model(**batch)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape[0], 3)

    def test_time_embedding_options(self):
        batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        for option in ("input", "none"):
            out = _small(self.dataset, time_embedding=option)(**batch)
            self.assertTrue(torch.isfinite(out["loss"]))
        with self.assertRaises(ValueError):
            _small(self.dataset, time_embedding="bad")

    def test_trainer_runs(self):
        torch.manual_seed(0)
        model = _small(self.dataset)
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=True)
        trainer = Trainer(model, metrics=["pr_auc_micro", "precision@5", "ndcg@5"],
                          enable_logging=False)
        trainer.train(loader, epochs=1)
        scores = trainer.evaluate(get_dataloader(self.dataset, batch_size=4))
        self.assertIn("ndcg@5", scores)
        self.assertTrue(np.isfinite(scores["loss"]))


if __name__ == "__main__":
    unittest.main()
