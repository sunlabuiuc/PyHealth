# Authors: Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html
"""SUPPORT2 time-to-event task and the authors' released-split mapping."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from pyhealth.data import Patient
from pyhealth.datasets import (
    Support2Dataset,
    get_dataloader,
    support2_paper_time_transform,
    support2_released_split_ids,
)
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventSupport2
from pyhealth.tasks.survival_time_to_event import SUPPORT2_FEATURES

NUMERIC = ["age", "meanbp", "hrt", "resp", "temp", "sod", "wblc", "crea"]
COLUMNS = [
    "sno",
    *NUMERIC,
    "sex",
    "num.co",
    "race",
    "diabetes",
    "dementia",
    "ca",
    "d.time",
    "death",
]
# Support2Dataset reads every column in its config; the rest stay blank.
CONFIG = Path(__file__).parents[2] / "pyhealth/datasets/configs/support2.yaml"
ALL_COLUMNS = ["sno"] + yaml.safe_load(CONFIG.read_text())["tables"]["support2"][
    "attributes"
]


def row(sno, **overrides):
    values = dict(
        sno=str(sno),
        age=f" {50 + sno}.5",
        meanbp="97.0",
        hrt=" 80",
        resp="20",
        temp="37.2",
        sod="140",
        wblc=f"  {5 + sno}.25",
        crea="1.1",
        sex="male",
        **{"num.co": str(sno % 3)},
        race="white",
        diabetes="0",
        dementia="0",
        ca="no",
        **{"d.time": str(30 * sno)},
        death=str(sno % 2),
    )
    values.update(overrides)
    return values


def rows_fixture():
    return [
        row(1),
        row(2, sex="female", race="", ca="metastatic"),  # missing race kept
        row(3, wblc=""),  # missing numeric: outside the cohort
        row(4, race="asian", diabetes="1", **{"num.co": "9"}),
        row(5, ca="yes", dementia="1"),
        row(6),
    ]


def write_csv(root, rows):
    with (root / "support2.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ALL_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def patient(rows):
    events = [
        {
            "patient_id": r["sno"],
            "event_type": "support2",
            "timestamp": None,
            **{
                f"support2/{k}": (v if v != "" else None)
                for k, v in r.items()
                if k != "sno"
            },
        }
        for r in rows
    ]
    return Patient(rows[0]["sno"] if rows else "0", pl.DataFrame(events))


def encode(r):
    """Independent re-encoding of a fixture row, for comparison."""
    out = [float(r[c]) for c in NUMERIC]
    out += [r["sex"] == "female"]
    out += [r["num.co"] == str(i) for i in range(1, 10)]
    race = r["race"] or "missing"
    out += [race == lv for lv in ("white", "black", "hispanic", "asian", "missing")]
    out += [
        r["diabetes"] == "1",
        r["dementia"] == "1",
        r["ca"] == "no",
        r["ca"] == "yes",
    ]
    return [float(v) for v in out]


def write_released(directory, rows, parts, repeat=1, offset_days=0.0):
    for split, members in zip(("train", "valid", "test"), parts):
        x = np.asarray([encode(rows[i]) for i in members], float).reshape(-1, 27)
        y = np.asarray(
            [
                [
                    (float(rows[i]["d.time"]) + offset_days) / 365.25,
                    float(rows[i]["death"]),
                ]
                for i in members
            ],
            float,
        ).reshape(-1, 2)
        np.savez(
            directory / f"{split}_{repeat}.npz",
            (x - x.mean(0)) if len(x) else x,
            y,
            x.astype(np.float32),
        )


class TestSupport2Task(unittest.TestCase):
    def test_encoding_and_cohort(self):
        task = SurvivalTimeToEventSupport2()
        self.assertEqual(task.output_schema, {"duration": "tensor", "event": "tensor"})
        self.assertEqual(len(SUPPORT2_FEATURES), 27)
        for r in rows_fixture():
            samples = task(patient([r]))
            if r["wblc"] == "":
                self.assertEqual(samples, [])
                continue
            (sample,) = samples
            self.assertEqual(sample["patient_id"], r["sno"])
            self.assertEqual(sample["features"], encode(r))
            self.assertEqual(
                sample["duration"] + sample["event"],
                [float(r["d.time"]), float(r["death"])],
            )
        missing_race = task(patient([rows_fixture()[1]]))[0]["features"]
        self.assertEqual(missing_race[SUPPORT2_FEATURES.index("race=missing")], 1.0)

    def test_paper_time_and_scaling(self):
        transform = support2_paper_time_transform()
        self.assertEqual(transform, {"time_divisor": 365.25, "time_offset": 0.001})
        mean, scale = list(range(27)), [2.0] * 27
        task = SurvivalTimeToEventSupport2(mean, scale, **transform)
        r = rows_fixture()[0]
        (sample,) = task(patient([r]))
        np.testing.assert_allclose(
            sample["features"], (np.asarray(encode(r)) - mean) / 2.0
        )
        self.assertAlmostEqual(sample["duration"][0], 30 / 365.25 + 0.001)

    def test_invalid_inputs_are_rejected(self):
        for transform in ({"time_divisor": 1e-300}, {"time_offset": 1e300}):
            with (
                self.subTest(transform=transform),
                self.assertRaisesRegex(ValueError, "finite positive float32"),
            ):
                SurvivalTimeToEventSupport2(**transform)(patient([row(1)]))
        for kwargs in (
            {"time_divisor": 0},
            {"time_offset": -1.0},
            {"feature_mean": [0.0] * 27},
            {"feature_mean": [0.0] * 7, "feature_scale": [1.0] * 7},
            {"feature_mean": [0.0] * 27, "feature_scale": [0.0] * 27},
        ):
            with self.assertRaises(ValueError):
                SurvivalTimeToEventSupport2(**kwargs)
        task = SurvivalTimeToEventSupport2()
        for bad in (
            {"sex": ""},
            {"ca": "maybe"},
            {"num.co": "12"},
            {"age": "old"},
            {"d.time": "0"},
            {"death": "2"},
        ):
            with self.assertRaises(ValueError):
                task(patient([row(1, **bad)]))
        with self.assertRaises(ValueError):
            task(patient([row(1), row(1)]))


class TestSupport2ReleasedSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = rows_fixture()
        write_csv(self.root, self.rows)
        self.cohort = [r for r in self.rows if r["wblc"] != ""]

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapping_keeps_released_order(self):
        write_released(self.root, self.cohort, [[3, 0], [4, 1], [2]])
        ids = support2_released_split_ids(self.root, self.root, 1)
        self.assertEqual(ids, {"train": ["5", "1"], "valid": ["6", "2"], "test": ["4"]})

    def test_mapping_rejects_invalid_controls_and_duplicate_ids(self):
        for tolerance in (float("nan"), float("inf"), -1):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                support2_released_split_ids(self.root, self.root, 1, tolerance=tolerance)
        for repeat in (True, 0, 11, 1.5):
            with self.subTest(repeat=repeat), self.assertRaises(ValueError):
                support2_released_split_ids(self.root, self.root, repeat=repeat)
        self.rows[1]["sno"] = self.rows[0]["sno"]
        write_csv(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "sno values"):
            support2_released_split_ids(self.root, self.root, 1)

    def test_mismatch_and_leftover_fail(self):
        write_released(self.root, self.cohort, [[0, 1], [2], [3]])
        with self.assertRaisesRegex(ValueError, "1 cohort patients unused"):
            support2_released_split_ids(self.root, self.root, 1)
        write_released(self.root, self.cohort, [[0, 1], [2, 4], [3]], offset_days=1)
        with self.assertRaisesRegex(ValueError, "no matching SUPPORT2 patient"):
            support2_released_split_ids(self.root, self.root, 1)

    def test_dataset_task_and_model(self):
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"],
            cache_dir=self.root / "cache",
            num_workers=1,
        )
        samples = dataset.set_task(
            SurvivalTimeToEventSupport2(**support2_paper_time_transform())
        )
        self.assertEqual(len(samples), 5)  # sno 3 is outside the cohort
        batch = next(iter(get_dataloader(samples, batch_size=5)))
        self.assertEqual(tuple(batch["features"].shape), (5, 27))
        model = SurvivalMDN(
            samples, num_components=2, hidden_dim=8, num_hidden_layers=1
        )
        out = model(**batch)
        self.assertTrue(out["loss"].isfinite())


if __name__ == "__main__":
    unittest.main()
