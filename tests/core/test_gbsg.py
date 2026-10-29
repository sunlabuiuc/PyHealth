# Authors: Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html
"""GBSG source parsing, actual PyHealth tasks/batches and model integration."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import (
    GBSGDataset,
    gbsg_paper_time_transform,
    gbsg_released_split_ids,
    get_dataloader,
    prepare_gbsg_csv,
)
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventGBSG

try:
    import h5py
except ImportError:
    h5py = None


def write_csv(root, rows):
    fields = [
        "source_id",
        "source_group",
        "source_row",
        *[f"x{i}" for i in range(7)],
        "time",
        "event",
    ]
    with (root / "gbsg.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rows_fixture():
    # Distinct outcomes, legitimately equal outcomes, and a full record match
    # under another source ID. None should be silently dropped or copied.
    return [
        dict(
            source_id=f"train_{i}",
            source_group="train",
            source_row=i,
            **{f"x{j}": value + j for j in range(7)},
            time=time,
            event=event,
        )
        for i, (value, time, event) in enumerate(
            [(1, 12, 1), (10, 30, 0), (3, 12, 1), (1, 12, 1)]
        )
    ]


def patient_from_rows(rows):
    events = [
        {
            "patient_id": "train_0",
            "event_type": "gbsg",
            "timestamp": None,
            **{f"gbsg/{k}": v for k, v in row.items() if k != "source_id"},
        }
        for row in rows
    ]
    df = (
        pl.DataFrame(events)
        if events
        else pl.DataFrame(
            schema={
                "patient_id": pl.String,
                "event_type": pl.String,
                "timestamp": pl.Datetime,
            }
        )
    )
    return Patient("train_0", df)


class TestGBSGTask(unittest.TestCase):
    def test_duration_event_contract_and_optional_standardization(self):
        task = SurvivalTimeToEventGBSG([1] * 7, [2] * 7)
        sample = task(patient_from_rows(rows_fixture()[:1]))[0]
        self.assertEqual(sample["features"], [0, 0.5, 1, 1.5, 2, 2.5, 3])
        self.assertEqual(sample["duration"] + sample["event"], [12, 1])
        self.assertEqual(task.input_schema, {"features": "tensor"})
        self.assertEqual(task.output_schema, {"duration": "tensor", "event": "tensor"})

    def test_invalid_preprocessing_state(self):
        cases = [
            ([0] * 7, None),
            ([0] * 6, [1] * 7),
            ([0] * 7, [0] * 7),
            ([float("nan")] * 7, [1] * 7),
            ([0] * 7, [float("inf")] * 7),
        ]
        for mean, scale in cases:
            with self.subTest(mean=mean, scale=scale), self.assertRaises(ValueError):
                SurvivalTimeToEventGBSG(mean, scale)

    def test_missing_or_multiple_records_are_rejected(self):
        for rows in ([], rows_fixture()[:2]):
            with (
                self.subTest(n=len(rows)),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                SurvivalTimeToEventGBSG()(patient_from_rows(rows))

    def test_invalid_raw_values_are_rejected(self):
        for field, value in [
            ("time", 0),
            ("time", -1),
            ("time", 1e-100),
            ("time", float("nan")),
            ("event", 2),
            ("event", 0.5),
            ("x3", None),
            ("x4", float("inf")),
            ("x2", 1e100),
        ]:
            row = rows_fixture()[0]
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                SurvivalTimeToEventGBSG()(patient_from_rows([row]))


class TestGBSGDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = rows_fixture()
        write_csv(self.root, self.rows)

    def dataset(self):
        return GBSGDataset(str(self.root), cache_dir=self.root / "cache", num_workers=1)

    def test_actual_loading_shuffling_and_model(self):
        dataset = self.dataset()
        self.assertEqual(
            set(dataset.unique_patient_ids), {r["source_id"] for r in self.rows}
        )
        samples = dataset.set_task()
        self.assertEqual(len(samples), 4)
        expected = {r["source_id"]: r for r in self.rows}
        torch.manual_seed(3)
        batch = next(iter(get_dataloader(samples, batch_size=4, shuffle=True)))
        self.assertEqual(batch["features"].shape, (4, 7))
        self.assertEqual(batch["duration"].shape, (4, 1))
        self.assertEqual(batch["event"].shape, (4, 1))
        for i, source_id in enumerate(batch["source_id"]):
            row = expected[source_id]
            self.assertEqual(batch["patient_id"][i], source_id)
            self.assertEqual(
                batch["features"][i].tolist(), [row[f"x{j}"] for j in range(7)]
            )
            self.assertEqual(
                batch["duration"][i].tolist() + batch["event"][i].tolist(),
                [row["time"], row["event"]],
            )
        model = SurvivalMDN(samples, num_components=3)
        self.assertIsNone(model.mode)
        loss = model(**batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        curves = model.predict_survival(
            batch["features"], torch.tensor([0.001, 10, 20, 40])
        )
        self.assertTrue(((curves >= 0) & (curves <= 1)).all())
        self.assertTrue((curves[:, 1:] <= curves[:, :-1]).all())

    def test_reordering_source_preserves_ids_and_feature_order(self):
        # CSV column order and physical row order are irrelevant to IDs.
        path = self.root / "gbsg.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reversed(self.rows[0])))
            writer.writeheader()
            writer.writerows(reversed(self.rows))
        dataset = self.dataset()
        sample = SurvivalTimeToEventGBSG()(dataset.get_patient("train_1"))[0]
        self.assertEqual(sample["features"], list(range(10, 17)))
        self.assertEqual(sample["duration"] + sample["event"], [30, 0])

    def test_invalid_csv_fails_before_caching(self):
        for field, value in [
            ("source_id", ""),
            ("source_row", -1),
            ("source_group", "other"),
            ("time", 0),
            ("event", 3),
            ("x0", "NaN"),
        ]:
            rows = rows_fixture()
            rows[0][field] = value
            write_csv(self.root, rows)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.dataset()

    def test_duplicate_ids_rejected_but_equal_values_allowed(self):
        self.dataset()  # Equal outcomes and full records under distinct IDs are valid.
        self.rows[1] = dict(self.rows[0])
        write_csv(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.dataset()

    def test_wrong_table_or_identifier_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supports only"):
            GBSGDataset(str(self.root), tables=["other"], cache_dir=self.root / "cache")
        import pyhealth.datasets.gbsg as module

        text = (Path(module.__file__).parent / "configs/gbsg.yaml").read_text()
        config = self.root / "bad.yaml"
        config.write_text(
            text.replace('patient_id: "source_id"', 'patient_id: "source_row"')
        )
        with self.assertRaisesRegex(ValueError, "config must use source_id"):
            GBSGDataset(
                str(self.root), config_path=str(config), cache_dir=self.root / "cache"
            )

    def test_empty_missing_column_and_missing_file(self):
        write_csv(self.root, [])
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.dataset()
        (self.root / "gbsg.csv").write_text("source_id,time,event\ntrain_0,1,1\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.dataset()
        (self.root / "gbsg.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()

    def test_training_fit_isolation_and_saved_configuration(self):
        from examples.gbsg_survival_loading import load_gbsg_splits, write_synthetic_csv

        original = self.root / "original"
        write_synthetic_csv(original)
        parts, first = load_gbsg_splits(original, self.root / "first")
        membership = first["split_source_ids"]
        self.assertEqual(sum(map(len, parts)), 40)
        sets = [set(ids) for ids in membership.values()]
        self.assertTrue(
            all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])
        )
        with (original / "gbsg.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        heldout = set(membership["valid"] + membership["test"])
        for row in rows:
            if row["source_id"] in heldout:
                for j in range(7):
                    row[f"x{j}"] = float(row[f"x{j}"]) * 17 + 123
                row["time"] = float(row["time"]) * 100
                row["event"] = 1 - int(row["event"])
        altered = self.root / "altered"
        altered.mkdir()
        write_csv(altered, rows)
        changed, second = load_gbsg_splits(
            altered, self.root / "second", split_source_ids=membership
        )
        self.assertEqual(first["preprocessing"], second["preprocessing"])
        self.assertEqual(first["model"], second["model"])
        self.assertEqual(first["split_source_ids"], second["split_source_ids"])
        for a, b in zip(parts[0], changed[0]):
            self.assertEqual(a["source_id"], b["source_id"])
            torch.testing.assert_close(a["features"], b["features"], rtol=0, atol=0)
            torch.testing.assert_close(a["duration"], b["duration"], rtol=0, atol=0)
            torch.testing.assert_close(a["event"], b["event"], rtol=0, atol=0)
        saved = json.loads((self.root / "first/run.json").read_text())
        restored_task = SurvivalTimeToEventGBSG(**saved["preprocessing"])
        restored = GBSGDataset(str(original), cache_dir=self.root / "third").set_task(
            restored_task
        )
        expected = {s["source_id"]: s for part in parts for s in part}
        for sample in restored:
            torch.testing.assert_close(
                sample["features"],
                expected[sample["source_id"]]["features"],
                rtol=0,
                atol=0,
            )


@unittest.skipUnless(h5py, "optional h5py is required for HDF5 preparation tests")
class TestGBSGPreparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.h5"
        with h5py.File(self.source, "w") as handle:
            for group, start in [("train", 1), ("test", 20)]:
                g = handle.create_group(group)
                g["x"] = np.arange(start, start + 14).reshape(2, 7)
                g["t"] = [12.5, 30]
                g["e"] = [1, 0]

    def test_conversion_values_provenance_and_actual_loader(self):
        out = self.root / "prepared"
        metadata = prepare_gbsg_csv(self.source, out)
        self.assertEqual(metadata["rows"], 4)
        self.assertEqual(metadata["source_groups"], {"train": 2, "test": 2})
        self.assertEqual(
            metadata["source_sha256"],
            hashlib.sha256(self.source.read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata, json.loads((out / "gbsg-source.json").read_text()))
        dataset = GBSGDataset(str(out), cache_dir=self.root / "cache")
        sample = dataset.default_task(dataset.get_patient("test_0"))[0]
        self.assertEqual(sample["features"], list(range(20, 27)))
        self.assertEqual(sample["duration"] + sample["event"], [12.5, 1])
        with self.assertRaises(FileExistsError):
            prepare_gbsg_csv(self.source, out)

    def test_bad_source_shapes_and_events(self):
        for field, value in [
            ("x", np.zeros((2, 6))),
            ("x", np.array(1)),
            ("t", [1]),
            ("e", [1, 2]),
        ]:
            with self.subTest(field=field):
                source = self.root / f"bad-{field}.h5"
                with h5py.File(self.source) as src, h5py.File(source, "w") as dst:
                    for group in ("train", "test"):
                        src.copy(group, dst)
                    del dst[f"train/{field}"]
                    dst[f"train/{field}"] = value
                out = self.root / f"out-{field}"
                with self.assertRaises(ValueError):
                    prepare_gbsg_csv(source, out)
                self.assertFalse((out / "gbsg.csv").exists())


def write_released(directory, rows, parts, repeat=1, max_time=None):
    """Write synthetic files in the authors' released layout."""
    max_time = max(float(r["time"]) for r in rows) if max_time is None else max_time
    for split, members in zip(("train", "valid", "test"), parts):
        x = np.asarray([[rows[i][f"x{j}"] for j in range(7)] for i in members], float)
        y = np.asarray(
            [[rows[i]["time"] / max_time + 0.001, rows[i]["event"]] for i in members],
            float,
        ).reshape(-1, 2)
        np.savez(directory / f"{split}_{repeat}.npz", x.reshape(-1, 7), y)


class TestGBSGReleasedSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = rows_fixture()
        write_csv(self.root, self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_paper_time_transform(self):
        self.assertEqual(
            gbsg_paper_time_transform(self.root),
            {"time_divisor": 30.0, "time_offset": 0.002},
        )

    def test_released_mapping_rejects_invalid_matching_controls(self):
        for tolerance in (float("nan"), float("inf"), -1):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                gbsg_released_split_ids(self.root, self.root, 1, tolerance=tolerance)
        for repeat in (True, 0, 11, 1.5):
            with self.subTest(repeat=repeat), self.assertRaises(ValueError):
                gbsg_released_split_ids(self.root, self.root, repeat=repeat)

    def test_released_mapping_rejects_duplicate_source_ids(self):
        self.rows[1]["source_id"] = self.rows[0]["source_id"]
        write_csv(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "source IDs"):
            gbsg_released_split_ids(self.root, self.root, 1)

    def test_mapping_keeps_released_order_and_assigns_twins_once(self):
        # train_0 and train_3 are identical records; train_2 differs only in x.
        write_released(self.root, self.rows, [[3, 1], [0], [2]])
        ids = gbsg_released_split_ids(self.root, self.root, repeat=1)
        self.assertEqual(ids["train"][1], "train_1")
        self.assertEqual(ids["test"], ["train_2"])
        self.assertEqual(
            sorted(ids["train"][:1] + ids["valid"]), ["train_0", "train_3"]
        )

    def test_unmatched_or_incomplete_released_files_fail(self):
        write_released(self.root, self.rows, [[0, 1], [2], [2]])
        with self.assertRaisesRegex(ValueError, "no matching source record"):
            gbsg_released_split_ids(self.root, self.root, repeat=1)
        write_released(self.root, self.rows, [[0, 1], [2], []])
        with self.assertRaisesRegex(ValueError, "1 source rows unused"):
            gbsg_released_split_ids(self.root, self.root, repeat=1)
        # Times converted with a different divisor do not match.
        write_released(self.root, self.rows, [[0, 1], [2], [3]], max_time=31.0)
        with self.assertRaisesRegex(ValueError, "no matching source record"):
            gbsg_released_split_ids(self.root, self.root, repeat=1)
        np.savez(self.root / "train_2.npz", np.zeros((2, 6)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, r"expected arr_0 \[N, 7\]"):
            gbsg_released_split_ids(self.root, self.root, repeat=2)

    def test_task_applies_paper_time_conversion(self):
        task = SurvivalTimeToEventGBSG(**gbsg_paper_time_transform(self.root))
        sample = task(patient_from_rows(self.rows[1:2]))[0]
        self.assertAlmostEqual(sample["duration"][0], 30 / 30 + 0.002)
        self.assertEqual(sample["features"], [10.0 + j for j in range(7)])
        for bad in (
            {"time_divisor": 0},
            {"time_divisor": float("inf")},
            {"time_offset": -0.1},
            {"time_offset": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SurvivalTimeToEventGBSG(**bad)

        for transform in ({"time_divisor": 1e-300}, {"time_offset": 1e300}):
            with (
                self.subTest(transform=transform),
                self.assertRaisesRegex(ValueError, "finite positive float32"),
            ):
                SurvivalTimeToEventGBSG(**transform)(patient_from_rows(self.rows[:1]))

    def test_example_runs_released_protocol(self):
        from examples.gbsg_survival_loading import load_gbsg_splits

        write_released(self.root, self.rows, [[0, 1], [2], [3]])
        parts, metadata = load_gbsg_splits(
            self.root, self.root / "out", released_dir=self.root, repeat=1
        )
        self.assertEqual(metadata["protocol"], "authors-released-repeat-1")
        self.assertEqual([len(p) for p in parts], [2, 1, 1])
        self.assertNotIn("time_scale", metadata["model"])
        released = np.load(self.root / "train_1.npz")["arr_1"][:, 0] + 0.001
        loaded = [s["duration"][0].item() for s in parts[0]]
        np.testing.assert_allclose(sorted(loaded), sorted(released), rtol=1e-6)
        with self.assertRaises(ValueError):
            load_gbsg_splits(
                self.root,
                self.root / "out2",
                split_source_ids={"train": [], "valid": [], "test": []},
                released_dir=self.root,
            )


if __name__ == "__main__":
    unittest.main()
