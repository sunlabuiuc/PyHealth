"""Tests for the PhysioNet 2012 dataset, the forecasting task and APN."""

import io
import math
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pyhealth.datasets import (
    PhysioNet2012Dataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.datasets.physionet2012 import (
    EVENTS_CSV,
    PATIENTS_CSV,
    PHYSIONET2012_URL,
    download_physionet2012,
    prepare_physionet2012,
    read_record,
)
from pyhealth.models import APN, TimeAwarePatchAggregation, masked_mse
from pyhealth.tasks import (
    PhysioNet2012Forecasting,
    physionet2012_benchmark_split,
    physionet2012_statistics,
)
from pyhealth.trainer import Trainer

HEADER = "Time,Parameter,Value\n"
DESCRIPTORS = "00:00,Age,54\n00:00,Gender,0\n00:00,Height,-1\n00:00,ICUType,4\n00:00,Weight,80\n"


def _record(record_id, lines):
    return HEADER + f"00:00,RecordID,{record_id}\n" + DESCRIPTORS + "".join(lines)


def _write_fake_root(root: Path) -> None:
    """Three records in set-a with outcomes; one has no data after hour 36."""
    folder = root / "set-a"
    folder.mkdir(parents=True)
    # 132539: HR at hours 0 (two values, averaged), 1 and 35; targets at 36, 37, 40, 41.
    (folder / "132539.txt").write_text(_record(132539, [
        "00:10,HR,70\n", "00:40,HR,80\n", "01:05,HR,90\n", "01:05,Temp,37\n",
        "35:00,HR,100\n", "36:30,HR,110\n", "37:00,Temp,38\n", "40:00,HR,90\n",
        "41:00,HR,95\n", "41:00,Weight,81\n",
    ]))
    # 132540: a target hour with only Weight (not a forecast variable).
    (folder / "132540.txt").write_text(_record(132540, [
        "02:00,HR,60\n", "36:00,Weight,70\n", "38:00,HR,65\n",
    ]))
    # 132541: nothing after hour 36, so no target and no sample.
    (folder / "132541.txt").write_text(_record(132541, ["03:00,HR,75\n"]))
    (root / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,6,1,5,-1,0\n132540,16,8,8,-1,0\n132541,10,3,4,3,1\n"
    )


class TestPhysioNet2012Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "p12"
        _write_fake_root(cls.root)
        cls.dataset = PhysioNet2012Dataset(
            root=str(cls.root), cache_dir=Path(cls.tmp.name) / "cache"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_read_record(self):
        record_id, descriptors, events = read_record(self.root / "set-a/132539.txt")
        self.assertEqual(record_id, "132539")
        self.assertTrue(math.isnan(descriptors["Height"]))  # -1 is missing
        self.assertEqual(descriptors["Weight"], 80)
        self.assertEqual(events[0], (10, "HR", 70.0))
        self.assertIn((41 * 60, "Weight", 81.0), events)  # later weights are series

    def test_tables(self):
        self.assertTrue((self.root / EVENTS_CSV).exists())
        self.assertTrue((self.root / PATIENTS_CSV).exists())
        self.assertEqual(sorted(self.dataset.unique_patient_ids),
                         ["132539", "132540", "132541"])
        patient = self.dataset.get_patient("132541")
        info = patient.get_events(event_type="patients")[0]
        self.assertEqual(float(info.in_hospital_death), 1.0)
        self.assertEqual(info.set, "a")

    def test_prepare_requires_records(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(
            FileNotFoundError
        ):
            prepare_physionet2012(empty)

    def test_statistics(self):
        mean, std = physionet2012_statistics(self.dataset, variables=["HR"])
        # hourly means of HR: 75, 90, 100, 110, 90, 95 | 60, 65 | 75
        hourly = [75, 90, 100, 110, 90, 95, 60, 65, 75]
        self.assertAlmostEqual(mean["HR"], np.mean(hourly), places=5)
        self.assertAlmostEqual(std["HR"], np.std(hourly), places=5)
        only = physionet2012_statistics(self.dataset, ["132540"], variables=["HR"])[0]
        self.assertAlmostEqual(only["HR"], 62.5)

    def test_task_windows_and_targets(self):
        task = PhysioNet2012Forecasting(variables=["HR", "Temp"])
        samples = {s["patient_id"]: s for p in self.dataset.iter_patients()
                   for s in task(p)}
        self.assertEqual(sorted(samples), ["132539", "132540"])  # 132541: no target
        s = samples["132539"]
        self.assertEqual(s["x"].shape, (36, 2))
        self.assertEqual(s["x"][0, 0], 75.0)  # 70 and 80 averaged in hour 0
        self.assertEqual(s["x"][1, 1], 37.0)
        self.assertEqual(s["x"][35, 0], 100.0)
        self.assertEqual(np.isfinite(s["x"]).sum(), 4)
        np.testing.assert_allclose(s["y_time"], np.array([36, 37, 40]) / 48)
        np.testing.assert_array_equal(
            s["y"], np.array([[110, np.nan], [np.nan, 38], [90, np.nan]], dtype="f4")
        )
        np.testing.assert_allclose(s["x_time"], np.arange(36) / 48)
        # 132540: the Weight-only hour is no forecast step; one target, padded.
        s = samples["132540"]
        np.testing.assert_allclose(s["y_time"], [38 / 48, 0, 0])
        self.assertEqual(np.isfinite(s["y"]).sum(), 1)

    def test_task_normalization_and_outliers(self):
        mean, std = {"HR": 90.0, "Temp": 37.0}, {"HR": 5.0, "Temp": 1.0}
        task = PhysioNet2012Forecasting(mean, std, variables=["HR", "Temp"])
        (s,) = task(self.dataset.get_patient("132539"))
        self.assertAlmostEqual(s["x"][0, 0], (75 - 90) / 5)  # -3 SD: kept
        self.assertAlmostEqual(s["y"][0, 0], 4.0)  # 110 is +4 SD: kept
        strict = PhysioNet2012Forecasting(mean, std, variables=["HR", "Temp"],
                                          outlier_sd=3.5)
        (s,) = strict(self.dataset.get_patient("132539"))
        self.assertTrue(np.isnan(s["y"][0, 0]))  # +4 SD dropped
        with self.assertRaises(ValueError):
            PhysioNet2012Forecasting(mean=mean)

    def test_set_task_and_split(self):
        samples = self.dataset.set_task(PhysioNet2012Forecasting(variables=["HR", "Temp"]))
        self.assertEqual(len(samples), 2)
        train, val, test = physionet2012_benchmark_split(samples, 0.5, 0.0)
        self.assertEqual((len(train), len(val), len(test)), (1, 0, 1))
        self.assertEqual(test[0]["patient_id"], "132540")  # the last RecordID
        self.assertTrue(torch.isnan(samples[0]["y"]).any())  # NaN kept for the mask


def _fake_tarball(records: dict[str, str]) -> bytes:
    """A gzipped tar holding ``set-a/<id>.txt`` for each record."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for record_id, text in records.items():
            data = text.encode()
            info = tarfile.TarInfo(f"set-a/{record_id}.txt")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class TestPhysioNet2012Download(unittest.TestCase):
    """The download and extraction path, with the network patched out."""

    OUTCOMES = (
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,6,1,5,-1,0\n"
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "p12"
        self.files = {
            "set-a.tar.gz": _fake_tarball(
                {"132539": _record(132539, ["00:10,HR,70\n", "37:00,HR,80\n"])}
            ),
            "Outcomes-a.txt": self.OUTCOMES.encode(),
        }
        self.calls = []
        self.failures = {}  # filename -> attempts that fail before success
        sleep = mock.patch("pyhealth.datasets.physionet2012.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def _urlretrieve(self, url, filename):
        name = url.removeprefix(PHYSIONET2012_URL)
        self.calls.append(name)
        if self.failures.get(name, 0) > 0:
            self.failures[name] -= 1
            Path(filename).write_bytes(b"partial")  # a half-written file
            raise OSError("connection reset")
        Path(filename).write_bytes(self.files[name])
        return filename, None

    def _download(self, **kwargs):
        with mock.patch(
            "urllib.request.urlretrieve", side_effect=self._urlretrieve
        ):
            download_physionet2012(self.root, sets=["a"], **kwargs)

    def test_download_and_extract(self):
        self._download()
        self.assertEqual(self.calls, ["set-a.tar.gz", "Outcomes-a.txt"])
        self.assertEqual((self.root / "Outcomes-a.txt").read_text(), self.OUTCOMES)
        record_id, _, events = read_record(self.root / "set-a/132539.txt")
        self.assertEqual((record_id, events), ("132539", [(10, "HR", 70.0),
                                                          (37 * 60, "HR", 80.0)]))
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            ["Outcomes-a.txt", "set-a", "set-a.tar.gz"],
        )  # no .part files or temporary folders left

        # Files already present are not downloaded again.
        self.calls.clear()
        self._download()
        self.assertEqual(self.calls, [])

    def test_dataset_download(self):
        with mock.patch(
            "urllib.request.urlretrieve", side_effect=self._urlretrieve
        ):
            dataset = PhysioNet2012Dataset(
                root=str(self.root),
                download=True,
                sets=["a"],
                cache_dir=Path(self.tmp.name) / "cache",
            )
        self.assertEqual(dataset.unique_patient_ids, ["132539"])

    def test_retry_discards_partial_file(self):
        self.failures["set-a.tar.gz"] = 2
        self._download(retries=3)
        self.assertEqual(self.calls.count("set-a.tar.gz"), 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.assertTrue((self.root / "set-a/132539.txt").exists())
        self.assertFalse((self.root / "set-a.tar.gz.part").exists())

    def test_gives_up_after_retries(self):
        self.failures["Outcomes-a.txt"] = 5
        with self.assertRaises(OSError):
            self._download(retries=2)
        self.assertEqual(self.calls.count("Outcomes-a.txt"), 2)
        self.assertFalse((self.root / "Outcomes-a.txt").exists())
        self.assertFalse((self.root / "Outcomes-a.txt.part").exists())
        self.assertFalse((self.root / "set-a").exists())

    def test_interrupted_extraction_is_redone(self):
        self.root.mkdir(parents=True)
        partial = self.root / ".set-a.partial" / "set-a"
        partial.mkdir(parents=True)
        (partial / "999999.txt").write_text("truncated")
        with (
            mock.patch("tarfile.TarFile.extractall", side_effect=OSError("full")),
            self.assertRaises(OSError),
        ):
            self._download()
        self.assertFalse((self.root / "set-a").exists())  # nothing half-made
        self.assertFalse((self.root / ".set-a.partial").exists())

        self._download()
        self.assertEqual(
            [p.name for p in (self.root / "set-a").iterdir()], ["132539.txt"]
        )


def _synthetic_samples(n=64, length=12, steps=3, n_vars=3, seed=0):
    """Sine waves with per-sample phase, observed at random points."""
    rng = np.random.default_rng(seed)
    samples = []
    total = length + steps
    for i in range(n):
        phase = rng.uniform(0, 2 * np.pi, size=n_vars)
        t = np.arange(total) / total
        values = np.sin(2 * np.pi * t[:, None] + phase).astype("float32")
        x = values[:length].copy()
        x[rng.random(x.shape) < 0.3] = np.nan
        samples.append({
            "patient_id": f"p{i}",
            "x": x,
            "x_time": t[:length].astype("float32"),
            "y_time": t[length:].astype("float32"),
            "y": values[length:],
        })
    return samples


def _dataset(samples):
    return create_sample_dataset(
        samples,
        {"x": "tensor", "x_time": "tensor", "y_time": "tensor"},
        {"y": "tensor"},
    )


class TestAPN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_patch_weights_follow_boundaries(self):
        tapa = TimeAwarePatchAggregation(1, 2, feature_dim=1, hidden_dim=4)
        with torch.no_grad():
            tapa.raw_tau.fill_(-6.0)  # sharp edges
        left, right = tapa.boundaries()
        np.testing.assert_allclose(left[0].tolist(), [0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(right[0].tolist(), [0.5, 1.0], atol=1e-5)
        t = torch.tensor([[[0.1, 0.4, 0.7, 0.9]]])
        w = tapa.weights(t, torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]))
        self.assertEqual(w.shape, (1, 1, 2, 4))
        np.testing.assert_allclose(w[0, 0].round().tolist(),
                                   [[1, 1, 0, 0], [0, 0, 1, 0]])  # masked point: 0

    def test_patch_is_weighted_average(self):
        tapa = TimeAwarePatchAggregation(1, 1, feature_dim=1, hidden_dim=1)
        captured = {}
        tapa.project.register_forward_hook(
            lambda module, args, out: captured.setdefault("pooled", args[0])
        )
        t = torch.tensor([[[0.2, 0.5, 0.8]]])
        feats = torch.tensor([[[[1.0], [2.0], [9.0]]]])
        tapa(t, feats, torch.tensor([[[1.0, 1.0, 0.0]]]))
        w = tapa.weights(t, torch.tensor([[[1.0, 1.0, 0.0]]]))[0, 0, 0]
        expected = (w[0] * 1 + w[1] * 2) / (w.sum() + 1e-9)
        self.assertAlmostEqual(captured["pooled"].item(), expected.item(), places=5)

    def test_masked_mse(self):
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[0.0, float("nan")], [3.0, float("nan")]])
        loss = masked_mse(pred, target)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertEqual(pred.grad[0, 1].item(), 0.0)
        empty = masked_mse(pred, torch.full((2, 2), float("nan")))
        self.assertEqual(empty.item(), 0.0)
        self.assertTrue(empty.requires_grad)

    def test_loss_and_probability_helpers(self):
        dataset = _dataset(_synthetic_samples(n=8))
        model = APN(dataset, hidden_dim=8, num_patches=4, time_dim=4)
        self.assertEqual(model.mode, "regression")
        logits = torch.randn(2, 3, 3)
        target = torch.randn(2, 3, 3)
        target[0, 0, 0] = float("nan")
        torch.testing.assert_close(
            model.label_processor.loss(logits, target), masked_mse(logits, target)
        )
        self.assertIs(model.label_processor.activation(logits), logits)

    def test_forward_backward_and_inference(self):
        dataset = _dataset(_synthetic_samples(n=8))
        model = APN(dataset, hidden_dim=8, num_patches=4, time_dim=4)
        self.assertEqual(model.mode, "regression")
        self.assertEqual(model.num_variables, 3)
        batch = next(iter(get_dataloader(dataset, batch_size=8)))
        out = model(**batch)
        self.assertEqual(out["forecast"].shape, (8, 3, 3))
        n_obs = int(torch.isfinite(batch["y"]).sum())
        self.assertEqual(out["y_true"].shape, (n_obs,))
        self.assertEqual(out["y_prob"].shape, (n_obs,))
        out["loss"].backward()
        self.assertIsNotNone(model.patching.left_offset.grad)
        self.assertIsNotNone(model.queries.grad)
        unlabeled = {k: v for k, v in batch.items() if k != "y"}
        out = model(**unlabeled)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (8, 3, 3))

    def test_training_lowers_loss_and_trainer_scores(self):
        dataset = _dataset(_synthetic_samples(n=64))
        loader = get_dataloader(dataset, batch_size=16, shuffle=True)
        model = APN(dataset, hidden_dim=16, num_patches=4, time_dim=4, dropout=0.0)
        trainer = Trainer(model=model, metrics=["mse", "mae"], enable_logging=False)
        before = trainer.evaluate(get_dataloader(dataset, batch_size=64))["mse"]
        trainer.train(loader, epochs=30, optimizer_params={"lr": 1e-2})
        after = trainer.evaluate(get_dataloader(dataset, batch_size=64))
        self.assertLess(after["mse"], 0.5 * before)
        self.assertIn("mae", after)


if __name__ == "__main__":
    unittest.main()
