"""Tests for PTBXLLabelGroupClassification and the label-grouping helpers.

Author: Hanho Ryu (NetID: hanhor2)

Uses three synthetic patients whose WFDB records are written to a temporary
directory at test time (no real PTB-XL data). Patients are built directly as
:class:`~pyhealth.data.Patient` objects so that every test runs in
milliseconds.
"""

import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.models import PTBXL_BRANCH_LABELS
from pyhealth.tasks import PTBXLLabelGroupClassification
from pyhealth.tasks.ptbxl_label_group_classification import (
    PHYSICIAN_LABEL_GROUPS,
    label_cooccurrence,
    match_label_groups,
    random_label_groups,
    spectral_label_groups,
    validate_label_groups,
)

try:
    import wfdb  # noqa: F401

    _WFDB = True
except ImportError:  # pragma: no cover - optional extra
    _WFDB = False

STATEMENTS = ["NORM", "IMI", "LVH", "SR", "AFIB", "PVC", "EL"]
GROUPS = {"rhythm": ["SR", "AFIB"], "morphology": ["IMI", "LVH", "PVC"]}


def _write_wfdb(record_base: Path, n_samples: int = 1000, seed: int = 0) -> None:
    """Writes a tiny 12-lead int16 WFDB record (header + .dat)."""
    record_base.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(-500, 500, size=(n_samples, 12), dtype=np.int16)
    Path(f"{record_base}.dat").write_bytes(data.tobytes())
    name = record_base.name
    lines = [f"{name} 12 100 {n_samples}"]
    lines += [f"{name}.dat 16 1000.0(0)/uV 16 0 0 0 0 {i}" for i in range(12)]
    Path(f"{record_base}.hea").write_text("\n".join(lines) + "\n")


def _write_statements(path: Path) -> None:
    """Writes a minimal ``scp_statements.csv`` covering :data:`STATEMENTS`."""
    rows = ["scp,diagnostic,form,rhythm,diagnostic_class,diagnostic_subclass"]
    kinds = {"SR": "rhythm", "AFIB": "rhythm", "PVC": "form", "EL": "form"}
    for code in STATEMENTS:
        kind = kinds.get(code, "diagnostic")
        flags = ["1.0" if kind == k else "" for k in ("diagnostic", "form", "rhythm")]
        rows.append(",".join([code, *flags, "", ""]))
    path.write_text("\n".join(rows) + "\n")


def _patient(patient_id: str, records: list[tuple[str, str, int]]) -> Patient:
    """Builds a Patient with one ``records`` event per (scp_codes, file, fold)."""
    count = len(records)
    frame = pl.DataFrame(
        {
            "patient_id": [patient_id] * count,
            "event_type": ["records"] * count,
            # PyHealth event timestamps are naive datetimes.
            "timestamp": [
                datetime(2000, 1, 1 + i)  # noqa: DTZ001
                for i in range(count)
            ],
            "records/scp_codes": [r[0] for r in records],
            "records/signal_file": [r[1] for r in records],
            "records/sampling_rate": ["100"] * count,
            "records/strat_fold": [str(r[2]) for r in records],
        }
    )
    return Patient(patient_id=patient_id, data_source=frame)


class TestValidateLabelGroups(unittest.TestCase):
    """Validation and normalization of label groupings."""

    def test_sorts_members_and_keeps_group_order(self):
        out = validate_label_groups({"b": ["Z", "Y"], "a": ["X"]})
        self.assertEqual(list(out), ["b", "a"])
        self.assertEqual(out["b"], ["Y", "Z"])

    def test_rejects_overlap(self):
        with self.assertRaisesRegex(ValueError, "both group"):
            validate_label_groups({"a": ["X"], "b": ["X", "Y"]})

    def test_rejects_empty_and_bad_names(self):
        for bad in ({}, {"a": []}, {"": ["X"]}, {"a.b": ["X"]}):
            with self.assertRaises(ValueError):
                validate_label_groups(bad)

    def test_rejects_string_group(self):
        with self.assertRaises(TypeError):
            validate_label_groups({"a": "XY"})

    def test_physician_groups_match_protoecgnet(self):
        self.assertEqual(
            validate_label_groups(PHYSICIAN_LABEL_GROUPS),
            validate_label_groups(PTBXL_BRANCH_LABELS),
        )
        sizes = [len(v) for v in PHYSICIAN_LABEL_GROUPS.values()]
        self.assertEqual(sizes, [16, 52, 3])


class TestLabelCooccurrence(unittest.TestCase):
    """Jaccard co-occurrence used by the contrastive loss and data-driven grouping."""

    def test_values(self):
        c = label_cooccurrence([["A", "B"], ["A"], ["B", "C"]], ["A", "B", "C"])
        # A: {0,1}, B: {0,2}, C: {2}
        self.assertAlmostEqual(c[0, 1], 1 / 3)
        self.assertAlmostEqual(c[1, 2], 1 / 2)
        self.assertEqual(c[0, 2], 0.0)
        np.testing.assert_allclose(np.diag(c), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(c, c.T)

    def test_unseen_label_and_unknown_codes(self):
        c = label_cooccurrence([["A", "ZZZ"]], ["A", "B"])
        np.testing.assert_allclose(c, [[1.0, 0.0], [0.0, 0.0]])

    def test_empty_samples(self):
        self.assertEqual(label_cooccurrence([], ["A", "B"]).sum(), 0.0)


class TestGroupingStrategies(unittest.TestCase):
    """Spectral (data-driven) and random (control) groupings."""

    def setUp(self):
        # Two blocks of labels that co-occur only within the block.
        self.labels = ["A", "B", "C", "D", "E"]
        self.c = np.array(
            [
                [1.0, 0.8, 0.7, 0.0, 0.0],
                [0.8, 1.0, 0.9, 0.0, 0.0],
                [0.7, 0.9, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.6],
                [0.0, 0.0, 0.0, 0.6, 1.0],
            ]
        )

    def test_spectral_recovers_blocks(self):
        groups = spectral_label_groups(self.c, self.labels, num_groups=2)
        self.assertEqual(groups, {"cluster0": ["A", "B", "C"], "cluster1": ["D", "E"]})

    def test_spectral_is_partition_and_deterministic(self):
        first = spectral_label_groups(self.c, self.labels, num_groups=3, seed=1)
        again = spectral_label_groups(self.c, self.labels, num_groups=3, seed=1)
        self.assertEqual(first, again)
        flat = sorted(x for members in first.values() for x in members)
        self.assertEqual(flat, self.labels)

    def test_spectral_single_group_and_errors(self):
        one = spectral_label_groups(self.c, self.labels, num_groups=1)
        self.assertEqual(one, {"cluster0": self.labels})
        with self.assertRaises(ValueError):
            spectral_label_groups(self.c, self.labels, num_groups=6)
        with self.assertRaises(ValueError):
            spectral_label_groups(self.c[:2, :2], self.labels)

    def test_random_groups_sizes_and_seed(self):
        groups = random_label_groups(self.labels, sizes=[3, 2], seed=4)
        self.assertEqual([len(g) for g in groups.values()], [3, 2])
        self.assertEqual(groups, random_label_groups(self.labels, [3, 2], seed=4))
        flat = sorted(x for members in groups.values() for x in members)
        self.assertEqual(flat, self.labels)

    def test_random_groups_errors(self):
        with self.assertRaises(ValueError):
            random_label_groups(self.labels, sizes=[3, 3])
        with self.assertRaises(ValueError):
            random_label_groups(self.labels, sizes=[5, 0])

    def test_match_is_one_to_one_when_possible(self):
        match = match_label_groups(
            {"x": ["D", "E"], "y": ["A", "B", "C"]},
            {"big": ["A", "B", "C"], "small": ["D", "E"], "unused": ["F"]},
        )
        self.assertEqual(match, {"x": "small", "y": "big"})

    def test_match_falls_back_for_extra_groups(self):
        match = match_label_groups(
            {"x": ["A"], "y": ["B"], "z": ["D"]},
            {"left": ["A", "B"], "right": ["D"]},
        )
        self.assertEqual(set(match), {"x", "y", "z"})
        self.assertEqual(match["z"], "right")
        self.assertEqual(sorted(match.values()).count("left"), 2)


@unittest.skipUnless(_WFDB, "wfdb not installed; install pyhealth[ptbxl]")
class TestPTBXLLabelGroupClassification(unittest.TestCase):
    """Sample processing of the task on three synthetic patients."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.scp_path = cls.tmp / "scp_statements.csv"
        _write_statements(cls.scp_path)
        for i in range(4):
            _write_wfdb(cls.tmp / "records100" / f"{i:05d}_lr", seed=i)
        rec = [str(cls.tmp / "records100" / f"{i:05d}_lr") for i in range(4)]
        cls.patients = [
            # Rhythm + morphology labels, plus a label outside every group.
            _patient("p0", [("{'SR': 0.0, 'IMI': 80.0, 'NORM': 100.0}", rec[0], 1)]),
            # Two ECGs: rhythm only, and labels outside every group only.
            _patient(
                "p1",
                [("{'AFIB': 100.0}", rec[1], 9), ("{'EL': 100.0}", rec[2], 9)],
            ),
            # Morphology only, one low-likelihood statement.
            _patient("p2", [("{'LVH': 50.0, 'PVC': 100.0}", rec[3], 10)]),
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, **kwargs) -> list[dict]:
        task = PTBXLLabelGroupClassification(str(self.scp_path), **kwargs)
        return [s for patient in self.patients for s in task(patient)]

    def test_schema(self):
        self.assertEqual(
            PTBXLLabelGroupClassification.input_schema, {"signal": "tensor"}
        )
        self.assertEqual(
            PTBXLLabelGroupClassification.output_schema, {"labels": "multilabel"}
        )

    def test_all_groups_restrict_labels_and_skip_unlabelled(self):
        samples = self._run(label_groups=GROUPS)
        self.assertEqual(len(samples), 3)  # the EL-only ECG is skipped
        by_patient = {s["patient_id"]: s for s in samples}
        self.assertEqual(by_patient["p0"]["labels"], ["IMI", "SR"])  # NORM dropped
        self.assertEqual(by_patient["p0"]["groups"], ["morphology", "rhythm"])
        self.assertEqual(by_patient["p1"]["groups"], ["rhythm"])
        self.assertEqual(by_patient["p2"]["labels"], ["LVH", "PVC"])

    def test_single_group_reproduces_branch_rows(self):
        samples = self._run(label_groups=GROUPS, group="rhythm")
        self.assertEqual([s["patient_id"] for s in samples], ["p0", "p1"])
        for sample in samples:
            self.assertTrue(set(sample["labels"]) <= {"SR", "AFIB"})
            self.assertEqual(sample["groups"], ["rhythm"])

    def test_skipped_ecgs_load_no_signal(self):
        task = PTBXLLabelGroupClassification(
            str(self.scp_path), label_groups=GROUPS, group="rhythm"
        )
        loaded = []
        original = task._prepare_signal

        def spy(signal_file, sampling_rate):
            loaded.append(Path(signal_file).name)
            return original(signal_file, sampling_rate)

        task._prepare_signal = spy
        samples = [s for patient in self.patients for s in task(patient)]
        self.assertEqual(len(samples), 2)
        # Only the two rhythm ECGs are read; EL-only and p2 are skipped first.
        self.assertEqual(loaded, ["00000_lr", "00001_lr"])

    def test_signal_fold_and_dtype(self):
        sample = self._run(label_groups=GROUPS)[0]
        self.assertEqual(sample["signal"].shape, (12, 1000))
        self.assertEqual(sample["signal"].dtype, np.float32)
        self.assertTrue(np.isfinite(sample["signal"]).all())
        # Per-lead z-scoring (default) gives zero mean, unit variance.
        np.testing.assert_allclose(sample["signal"].mean(axis=1), 0.0, atol=1e-4)
        self.assertEqual(sample["strat_fold"], 1)

    def test_min_likelihood_filters_statements(self):
        samples = self._run(label_groups=GROUPS, min_likelihood=60.0)
        by_patient = {s["patient_id"]: s for s in samples}
        self.assertEqual(by_patient["p0"]["labels"], ["IMI"])  # SR has 0.0
        self.assertEqual(by_patient["p2"]["labels"], ["PVC"])  # LVH has 50.0

    def test_default_grouping_is_physician(self):
        task = PTBXLLabelGroupClassification(str(self.scp_path))
        self.assertEqual(task.label_groups, validate_label_groups(PTBXL_BRANCH_LABELS))
        self.assertEqual(len(task.labels), 71)
        self.assertTrue(task.remove_baseline)
        labels = {tuple(s["labels"]) for s in self._run()}
        self.assertIn(("EL",), labels)  # EL is a physician "global" label

    def test_group_changes_cache_identity(self):
        a = PTBXLLabelGroupClassification(str(self.scp_path), group="rhythm")
        b = PTBXLLabelGroupClassification(str(self.scp_path), group="global")
        self.assertNotEqual(vars(a), vars(b))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            PTBXLLabelGroupClassification(str(self.scp_path), group="nope")
        with self.assertRaises(ValueError):
            PTBXLLabelGroupClassification(
                str(self.scp_path), label_groups={"a": ["SR"], "b": ["SR"]}
            )
        with self.assertRaises(ValueError):
            PTBXLLabelGroupClassification(str(self.scp_path), standardize="global")


if __name__ == "__main__":
    unittest.main()
