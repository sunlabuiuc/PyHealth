"""Tests for HeartLang and its PTB-XL task, with tiny synthetic ECGs."""

import argparse
import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import (
    PTBXLDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.models import HeartLang, HeartLangBackbone, ecg_to_heartbeat_sentence
from pyhealth.models.heartlang import (
    HF_REVISION,
    heartlang_state_dict,
    load_heartlang_checkpoint,
)
from pyhealth.tasks import PTBXLHeartLangClassification

TINY = {"seq_len": 8, "token_len": 16, "embed_dim": 32, "depth": 1, "heads": 2,
        "dim_head": 8, "mlp_dim": 64}
HAS_WFDB = importlib.util.find_spec("wfdb") is not None
PTBXL_FIXTURE = (
    Path(__file__).resolve().parents[2] / "test-resources" / "core" / "ptbxl"
)


def make_dataset(n: int = 6):
    rng = np.random.default_rng(0)
    samples = [
        {
            "patient_id": f"p{i}",
            "ecg_sentence": rng.standard_normal((8, 16)).astype(np.float32),
            "lead_ids": np.arange(8) % 12 + 1,
            "time_ids": np.arange(8) % 10 + 1,
            "labels": ["NORM"] if i % 2 else ["MI", "STTC"],
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples,
        input_schema={"ecg_sentence": "tensor", "lead_ids": "tensor",
                      "time_ids": "tensor"},
        output_schema={"labels": "multilabel"},
    )


def synthetic_ecg(rate: int = 100, seconds: int = 10, leads: int = 12):
    """Sharp R peaks every 0.8 s on a flat line, the same on every lead."""
    signal = np.zeros((leads, rate * seconds), dtype=np.float32)
    peaks = np.arange(rate // 2, rate * seconds - rate // 4, int(0.8 * rate))
    for p in peaks:
        signal[:, p - 2 : p + 3] = [0.2, 0.6, 1.5, 0.6, 0.2]
    return signal, peaks


class TestHeartbeatSentence(unittest.TestCase):
    def test_words_are_centred_on_the_qrs(self):
        signal = np.zeros((2, 1000), dtype=np.float32)
        signal[0, 300], signal[1, 300] = 1.0, 2.0
        words, lead_ids, time_ids = ecg_to_heartbeat_sentence(
            signal, qrs_inds=[100, 300, 500], max_len=10, token_len=96)
        self.assertEqual(words.shape, (10, 96))
        self.assertEqual(lead_ids.tolist(), [1, 1, 1, 2, 2, 2, 0, 0, 0, 0])
        self.assertEqual(time_ids.tolist(), [2, 4, 6, 2, 4, 6, 0, 0, 0, 0])
        self.assertEqual(words[1, 48], 1.0)  # QRS at the centre of its word
        self.assertEqual(words[4, 48], 2.0)
        self.assertFalse(words[6:].any())  # padding

    def test_truncation_and_lead_subset(self):
        signal = np.random.default_rng(0).standard_normal((12, 1000))
        qrs = list(range(50, 1000, 80))
        words, lead_ids, _ = ecg_to_heartbeat_sentence(
            signal, qrs_inds=qrs, leads=[1, 7], max_len=16)
        self.assertEqual(words.shape, (16, 96))
        # lead ids keep the original lead index (II = 2, V2 = 8)
        self.assertEqual(set(lead_ids.tolist()), {2, 8})
        self.assertEqual(lead_ids.tolist().count(2), len(qrs))

    def test_long_recordings_and_bad_leads_are_refused(self):
        signal = np.zeros((12, 2000), dtype=np.float32)
        # the last QRS before 15 s still fits the 16-row time embedding
        _, _, time_ids = ecg_to_heartbeat_sentence(signal, qrs_inds=[100, 1499])
        self.assertEqual(time_ids.max(), 15)
        with self.assertRaisesRegex(ValueError, "crop the recording"):
            ecg_to_heartbeat_sentence(signal, qrs_inds=[100, 1500])
        with self.assertRaises(ValueError):
            ecg_to_heartbeat_sentence(
                np.zeros((16, 1000)), qrs_inds=[100], leads=[15])
        # every index the tokenizer emits is accepted by the backbone
        words, lead_ids, time_ids = ecg_to_heartbeat_sentence(
            signal, qrs_inds=[100, 1499], max_len=8, token_len=16)
        out = HeartLangBackbone(**TINY)(
            torch.from_numpy(words)[None], torch.from_numpy(lead_ids)[None],
            torch.from_numpy(time_ids)[None])
        self.assertEqual(out.shape, (1, 9, 32))

    def test_no_qrs_warns_and_gives_padding(self):
        with self.assertWarnsRegex(UserWarning, "no QRS complex"):
            words, lead_ids, time_ids = ecg_to_heartbeat_sentence(
                np.zeros((12, 1000)), qrs_inds=[])
        self.assertFalse(words.any() or lead_ids.any() or time_ids.any())

    @unittest.skipUnless(HAS_WFDB, "wfdb not installed")
    def test_qrs_detection(self):
        signal, peaks = synthetic_ecg()
        words, lead_ids, _ = ecg_to_heartbeat_sentence(signal)
        detected = int((lead_ids == 1).sum())
        self.assertGreaterEqual(detected, len(peaks) - 2)
        self.assertLessEqual(detected, len(peaks) + 1)
        self.assertEqual(words.shape, (256, 96))


class TestHeartLangModel(unittest.TestCase):
    def test_backbone_shapes(self):
        backbone = HeartLangBackbone(**TINY)
        words = torch.randn(3, 8, 16)
        ids = torch.ones(3, 8, dtype=torch.long)
        self.assertEqual(backbone(words, ids, ids).shape, (3, 9, 32))
        self.assertEqual(backbone(words).shape, (3, 9, 32))

    def test_forward_and_backward(self):
        dataset = make_dataset()
        model = HeartLang(dataset, pretrained=False, **TINY)
        batch = next(iter(get_dataloader(dataset, batch_size=6)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (6, 3))
        self.assertEqual(out["embed"].shape, (6, 32))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
        out["loss"].backward()
        self.assertIsNotNone(model.backbone.cls_token.grad)
        # without labels (inference) there is no loss
        unlabeled = {k: v for k, v in batch.items() if k != "labels"}
        self.assertNotIn("loss", model(**unlabeled))

    def test_freeze_backbone_trains_only_the_head(self):
        dataset = make_dataset()
        model = HeartLang(dataset, pretrained=False, freeze_backbone=True, **TINY)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        batch = next(iter(get_dataloader(dataset, batch_size=6)))
        model(**batch)["loss"].backward()
        self.assertIsNone(model.backbone.cls_token.grad)
        self.assertIsNotNone(model.mlp_head.weight.grad)
        # Trainer calls train(); the frozen backbone keeps dropout off
        model.train()
        self.assertTrue(model.training and model.mlp_head.training)
        self.assertFalse(any(m.training for m in model.backbone.modules()))
        model.eval()
        self.assertFalse(model.mlp_head.training)
        # a trainable backbone follows train() as usual
        tuned = HeartLang(dataset, pretrained=False, **TINY).train()
        self.assertTrue(tuned.backbone.training)

    def test_missing_input_is_an_error(self):
        dataset = make_dataset()
        with self.assertRaises(ValueError):
            HeartLang(dataset, pretrained=False, words_key="signal", **TINY)


class TestCheckpoint(unittest.TestCase):
    def test_download_is_pinned_to_a_revision(self):
        with mock.patch(
            "huggingface_hub.hf_hub_download",
            return_value="/nonexistent/checkpoint.pth",
        ) as download, self.assertRaises(FileNotFoundError):
            load_heartlang_checkpoint()
        self.assertEqual(download.call_args.kwargs["revision"], HF_REVISION)
        self.assertRegex(HF_REVISION, r"^[0-9a-f]{40}$")

    def test_state_dict_mapping(self):
        state = {"module.backbone.cls_token": torch.zeros(4),
                 "lm_head.weight": torch.zeros(2, 4), "pos_embed": torch.ones(1)}
        self.assertEqual(sorted(heartlang_state_dict({"model": state})),
                         ["cls_token", "pos_embed"])
        self.assertEqual(sorted(heartlang_state_dict(state)),
                         ["cls_token", "pos_embed"])
        # the original feed-forward names map onto PyHealth's components
        original = {
            f"transformer.layers.{i}.1.net.{j}.{p}": torch.zeros(1)
            for i in (0, 11) for j in (0, 1, 4) for p in ("weight", "bias")
        }
        original["transformer.layers.0.0.norm.weight"] = torch.zeros(1)
        self.assertEqual(sorted(heartlang_state_dict(original)), sorted(
            [f"transformer.layers.{i}.1.{m}.{p}" for i in (0, 11)
             for m in ("sublayer.norm", "ff.w_1", "ff.w_2")
             for p in ("weight", "bias")]
            + ["transformer.layers.0.0.norm.weight"]))
        # a model trained with the original layout loads into ours
        backbone = HeartLangBackbone(**TINY)
        renamed = {
            k.replace("1.sublayer.norm.", "1.net.0.")
            .replace("1.ff.w_1.", "1.net.1.").replace("1.ff.w_2.", "1.net.4."): v
            for k, v in backbone.state_dict().items()
        }
        self.assertTrue(any(".net.4." in k for k in renamed))
        HeartLangBackbone(**TINY).load_state_dict(heartlang_state_dict(renamed))

    def test_loads_a_training_checkpoint(self):
        # a checkpoint shaped like the published one: weights under "model",
        # the pre-training head, and argparse arguments with numpy scalars
        source = HeartLangBackbone(**TINY)
        weights = dict(source.state_dict())
        weights["lm_head.weight"] = torch.zeros(10, 32)
        args = argparse.Namespace(lr=np.float64(5e-4), seed=np.int64(0))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.pth")
            torch.save({"model": weights, "epoch": 200, "args": args}, path)
            self.assertEqual(len(load_heartlang_checkpoint(path)), len(weights) - 1)
            model = HeartLang(make_dataset(), pretrained=path, **TINY)
            self.assertTrue(torch.equal(model.backbone.pos_embed, source.pos_embed))
            # a checkpoint for another architecture is refused
            with self.assertRaises(ValueError):
                HeartLang(make_dataset(), pretrained=path,
                          **{**TINY, "depth": 2})
            with self.assertRaises(FileNotFoundError):
                load_heartlang_checkpoint(os.path.join(tmp, "missing.pth"))


class TestPTBXLHeartLangTask(unittest.TestCase):
    def setUp(self):
        self.statements = os.path.join(
            os.path.dirname(__file__), "..", "..", "test-resources", "core",
            "ptbxl", "scp_statements.csv")

    def test_schema_and_options(self):
        task = PTBXLHeartLangClassification(self.statements)
        self.assertEqual(task.label_set, "superdiagnostic")
        self.assertEqual(set(task.input_schema),
                         {"ecg_sentence", "lead_ids", "time_ids"})
        with self.assertRaises(ValueError):
            PTBXLHeartLangClassification(self.statements, signal_mean=0.0,
                                         signal_std=None)
        with self.assertRaises(ValueError):
            PTBXLHeartLangClassification(self.statements, signal_std=0.0)

    def test_global_standardization(self):
        task = PTBXLHeartLangClassification(
            self.statements, signal_mean=1.0, signal_std=2.0)
        raw = np.full((12, 1000), 3.0, dtype=np.float32)
        with mock.patch("pyhealth.datasets.ptbxl.load_ptbxl_record",
                        return_value=raw):
            out = task._prepare_signal("record", 100)
        np.testing.assert_allclose(out, 1.0)

    def test_only_100_hz_by_default(self):
        task = PTBXLHeartLangClassification(self.statements)
        raw = np.zeros((12, 5000), dtype=np.float32)
        with mock.patch("pyhealth.datasets.ptbxl.load_ptbxl_record",
                        return_value=raw):
            with self.assertRaisesRegex(ValueError, "500 Hz"):
                task._prepare_signal("record", 500)
            # sampling_rate=None is the documented opt-out
            free = PTBXLHeartLangClassification(
                self.statements, sampling_rate=None)
            self.assertEqual(free._prepare_signal("record", 500).shape,
                             (12, 5000))

    def test_samples_are_sentences(self):
        task = PTBXLHeartLangClassification(self.statements, leads=[0, 1])
        signal, peaks = synthetic_ecg()
        parent = [{"patient_id": "1", "strat_fold": 3, "signal": signal,
                   "labels": ["NORM"]}]
        real = ecg_to_heartbeat_sentence
        with mock.patch(
            "pyhealth.tasks.ptbxl_multilabel_classification."
            "PTBXLMultilabelClassification.__call__", return_value=parent
        ), mock.patch(
            "pyhealth.models.heartlang.ecg_to_heartbeat_sentence",
            side_effect=lambda s, **kw: real(s, qrs_inds=peaks, **kw),
        ):
            (sample,) = task(patient=None)
        self.assertNotIn("signal", sample)
        self.assertEqual(sample["ecg_sentence"].shape, (256, 96))
        self.assertEqual(int((sample["lead_ids"] == 2).sum()), len(peaks))
        self.assertEqual(sample["strat_fold"], 3)


def write_ptbxl_100hz(root: Path) -> None:
    """The committed PTB-XL fixture CSVs plus 10 s, 100 Hz WFDB records with
    R peaks (the 500 Hz records are not needed)."""
    root.mkdir(parents=True)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        shutil.copy(PTBXL_FIXTURE / name, root / name)
    signal, _ = synthetic_ecg()
    data = np.round(signal.T * 1000).astype(np.int16)  # mV -> uV
    for rel in pd.read_csv(root / "ptbxl_database.csv")["filename_lr"]:
        base = root / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{base}.dat").write_bytes(data.tobytes())
        lines = [f"{base.name} 12 100 {len(data)}"] + [
            f"{base.name}.dat 16 1000.0(0)/uV 16 0 0 0 0 {i}" for i in range(12)
        ]
        Path(f"{base}.hea").write_text("\n".join(lines) + "\n")


@unittest.skipUnless(HAS_WFDB, "wfdb not installed")
class TestPTBXLHeartLangSetTask(unittest.TestCase):
    """The real set_task and split path on the PTB-XL fixture (no mocks)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_ptbxl_100hz(self.tmp / "data")
        self.dataset = PTBXLDataset(
            root=str(self.tmp / "data"),
            metadata_cache_dir=self.tmp / "meta_cache",
            sampling_rate=100,
            cache_dir=self.tmp / "pyhealth_cache",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_set_task_split_and_model(self):
        task = PTBXLHeartLangClassification(self.dataset.scp_statements_path)
        samples = self.dataset.set_task(task, num_workers=1)
        # ecg 1 NORM (fold 1), 2 MI (fold 9), 4 HYP+MI (fold 2); ecg 3 is
        # PACE only and ecg 5 has no label, so both are dropped
        self.assertEqual(len(samples), 3)
        by_fold = {int(s["strat_fold"]): s for s in samples}
        self.assertEqual(set(by_fold), {1, 2, 9})
        sample = by_fold[1]
        self.assertEqual(tuple(sample["ecg_sentence"].shape), (256, 96))
        # 12 peaks per lead (synthetic_ecg), every word on a detected beat
        lead_ids = sample["lead_ids"].long()
        self.assertGreaterEqual(int((lead_ids == 1).sum()), 10)
        self.assertEqual(set(lead_ids.tolist()) - {0}, set(range(1, 13)))
        self.assertLessEqual(int(sample["time_ids"].max()), 10)
        self.assertEqual(by_fold[2]["labels"].tolist(), [1.0, 1.0, 0.0])

        train, val, test = split_by_strat_fold(
            samples, train_folds=(1,), val_folds=(2,), test_folds=(9,))
        self.assertEqual([len(train), len(val), len(test)], [1, 1, 1])
        self.assertEqual(int(test[0]["strat_fold"]), 9)

        model = HeartLang(samples, pretrained=False, freeze_backbone=True,
                          **{**TINY, "seq_len": 256, "token_len": 96})
        batch = next(iter(get_dataloader(train, batch_size=1)))
        out = model(**batch)
        self.assertEqual(tuple(out["y_prob"].shape), (1, 3))
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
