"""Tests for RESPectCCEPDataset and SeizureOnsetZoneLocalisation.

Run with:
    python -m pytest tests/core/test_respect_ccep.py -v
or:
    python -m unittest tests.core.test_respect_ccep -v

All tests use synthetic in-memory data only — no real ds004080 files are
required, and no network access is performed.  MNE and BrainVision I/O are
bypassed by patching ``RESPectCCEPDataset._process_run`` where needed.

Test organisation
-----------------
TestRESPectCCEPDatasetHelpers
    Unit tests for the static helper methods that do not require a real BIDS
    directory (entity extraction, sidecar discovery, coordinate parsing, …).

TestRESPectCCEPDatasetPrepareData
    Integration-style tests for ``prepare_data`` and the CSV index builder,
    using a minimal synthetic BIDS tree written to a temporary directory.

TestSeizureOnsetZoneLocalisationHelpers
    Unit tests for every private helper in soz_localisation.py
    (_canonical_stim_key, _extract_coords, _parse_response_ts, etc.).

TestSeizureOnsetZoneLocalisationTask
    Tests for SeizureOnsetZoneLocalisation.__call__ against synthetic Patient
    objects built from real Polars DataFrames (matching the actual PyHealth API).
"""

import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Imports under test — adjust paths if your project layout differs.
# ---------------------------------------------------------------------------
# Dataset helpers are accessed via the class; we import the class directly.
from pyhealth.datasets.respectccep import RESPectCCEPDataset  # noqa: E402

# Task + private helpers
from pyhealth.tasks.ccep_detect_soz import (  # noqa: E402
    SeizureOnsetZoneLocalisation,
    _DIST_UNKNOWN,
    _DEFAULT_MIN_DISTANCE_MM,
    _canonical_stim_key,
    _euclidean_mm,
    _extract_coords,
    _mean_responses,
    _parse_response_ts,
    _safe_str,
    _stim_distance,
)
from pyhealth.data import Patient  # noqa: E402

# ---------------------------------------------------------------------------
# Shared synthetic constants
# ---------------------------------------------------------------------------

_T = 256  # timeseries length used in all synthetic responses
_PATIENT_ID = "sub-01"
_SESSION_ID = "ses-1"
_EVENT_TYPE = "respectccep"


# ---------------------------------------------------------------------------
# Helpers for building synthetic Polars DataFrames
# ---------------------------------------------------------------------------

def _make_response_ts(seed: int = 0, length: int = _T) -> str:
    """Return a JSON-encoded synthetic 1-D float32 timeseries."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(length).astype(np.float32)
    return json.dumps(arr.tolist(), separators=(",", ":"))


def _make_event_rows(
    rows: List[Dict[str, Any]],
    event_type: str = _EVENT_TYPE,
) -> pl.DataFrame:
    """Build a Polars DataFrame in the format PyHealth's Patient expects.

    Column names follow the ``"{event_type}/{attr}"`` convention used by
    ``Event.from_dict``.  A synthetic ``timestamp`` column is added so that
    ``Patient.__init__`` can sort and partition the data.
    """
    if not rows:
        return pl.DataFrame(
            schema={
                "event_type": pl.Utf8,
                "timestamp": pl.Datetime,
            }
        )

    # Prefix every key that is not "timestamp" or "event_type".
    prefixed: List[Dict[str, Any]] = []
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    for i, row in enumerate(rows):
        d: Dict[str, Any] = {
            "event_type": event_type,
            "timestamp": base_time,  # all share the same timestamp for simplicity
        }
        for k, v in row.items():
            d[f"{event_type}/{k}"] = v
        prefixed.append(d)

    return pl.DataFrame(prefixed)


def _make_patient(
    rows: List[Dict[str, Any]],
    patient_id: str = _PATIENT_ID,
    event_type: str = _EVENT_TYPE,
) -> Patient:
    """Construct a real ``pyhealth.data.Patient`` from a list of attribute dicts."""
    df = _make_event_rows(rows, event_type=event_type)
    return Patient(patient_id=patient_id, data_source=df)


# ---------------------------------------------------------------------------
# Canonical synthetic event row factory
# ---------------------------------------------------------------------------

def _row(
    recording_electrode: str,
    stim_1: str,
    stim_2: str,
    soz_label: int = 0,
    session_id: str = _SESSION_ID,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    response_seed: int = 0,
) -> Dict[str, Any]:
    """Return a single event attribute dict matching the dataset's CSV schema."""
    return {
        "recording_electrode": recording_electrode,
        "stim_1": stim_1,
        "stim_2": stim_2,
        "soz_label": soz_label,
        "session_id": session_id,
        "recording_x": x,
        "recording_y": y,
        "recording_z": z,
        "response_ts": _make_response_ts(seed=response_seed),
        "response_ts_std": _make_response_ts(seed=response_seed+100),
    }


# ===========================================================================
# 1. Dataset helper tests (no filesystem, no MNE)
# ===========================================================================

class TestRESPectCCEPDatasetHelpers(unittest.TestCase):
    """Unit tests for RESPectCCEPDataset static helper methods."""

    # --- _extract_bids_entities -------------------------------------------

    def test_extract_bids_entities_full(self):
        path = Path("sub-02_ses-1_task-SPES_run-01_events.tsv")
        entities = RESPectCCEPDataset._extract_bids_entities(path)
        self.assertEqual(entities["participant_id"], "sub-02")
        self.assertEqual(entities["session_id"], "ses-1")
        self.assertEqual(entities["run_id"], "run-01")

    def test_extract_bids_entities_missing_session(self):
        path = Path("sub-05_task-SPES_events.tsv")
        entities = RESPectCCEPDataset._extract_bids_entities(path)
        self.assertEqual(entities["participant_id"], "sub-05")
        self.assertIsNone(entities["session_id"])
        self.assertIsNone(entities["run_id"])

    def test_extract_bids_entities_missing_all(self):
        path = Path("events.tsv")
        entities = RESPectCCEPDataset._extract_bids_entities(path)
        self.assertIsNone(entities["participant_id"])
        self.assertIsNone(entities["session_id"])
        self.assertIsNone(entities["run_id"])

    def test_extract_bids_entities_run_without_session(self):
        path = Path("sub-03_run-02_events.tsv")
        entities = RESPectCCEPDataset._extract_bids_entities(path)
        self.assertEqual(entities["participant_id"], "sub-03")
        self.assertIsNone(entities["session_id"])
        self.assertEqual(entities["run_id"], "run-02")

    # --- _find_run_sidecar ------------------------------------------------

    def test_find_run_sidecar_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "sub-01_ses-1_task-SPES_events.tsv"
            channels = Path(tmp) / "sub-01_ses-1_task-SPES_channels.tsv"
            events.touch()
            channels.touch()
            result = RESPectCCEPDataset._find_run_sidecar(events, "_channels.tsv")
            self.assertEqual(result, channels)

    def test_find_run_sidecar_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "sub-01_ses-1_task-SPES_events.tsv"
            events.touch()
            result = RESPectCCEPDataset._find_run_sidecar(events, "_channels.tsv")
            self.assertIsNone(result)

    # --- _find_session_sidecar --------------------------------------------

    def test_find_session_sidecar_sub_ses_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            electrodes = Path(tmp) / "sub-01_ses-1_electrodes.tsv"
            electrodes.touch()
            result = RESPectCCEPDataset._find_session_sidecar(
                Path(tmp), "sub-01", "ses-1", "electrodes.tsv"
            )
            self.assertEqual(result, electrodes)

    def test_find_session_sidecar_fallback_to_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Only the generic glob form exists
            electrodes = Path(tmp) / "sub-01_electrodes.tsv"
            electrodes.touch()
            result = RESPectCCEPDataset._find_session_sidecar(
                Path(tmp), "sub-01", "ses-1", "electrodes.tsv"
            )
            self.assertEqual(result, electrodes)

    def test_find_session_sidecar_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = RESPectCCEPDataset._find_session_sidecar(
                Path(tmp), "sub-01", "ses-1", "electrodes.tsv"
            )
            self.assertIsNone(result)

    # --- _safe_float ------------------------------------------------------

    def test_safe_float_valid(self):
        self.assertEqual(RESPectCCEPDataset._safe_float("3.14"), 3.14)
        self.assertEqual(RESPectCCEPDataset._safe_float(42), 42.0)

    def test_safe_float_invalid_returns_nan(self):
        result = RESPectCCEPDataset._safe_float("not_a_number")
        self.assertTrue(math.isnan(result))

    def test_safe_float_none_returns_nan(self):
        result = RESPectCCEPDataset._safe_float(None)
        self.assertTrue(math.isnan(result))

    # --- _canonical_stim_site --------------------------------------------

    def test_canonical_stim_site_sorted(self):
        result = RESPectCCEPDataset._canonical_stim_site("P30-P29")
        self.assertEqual(result, "P29-P30")

    def test_canonical_stim_site_already_sorted(self):
        result = RESPectCCEPDataset._canonical_stim_site("P29-P30")
        self.assertEqual(result, "P29-P30")

    def test_canonical_stim_site_with_spaces(self):
        result = RESPectCCEPDataset._canonical_stim_site(" P30 - P29 ")
        self.assertEqual(result, "P29-P30")

    def test_canonical_stim_site_single_electrode_returns_none(self):
        result = RESPectCCEPDataset._canonical_stim_site("P29")
        self.assertIsNone(result)

    def test_canonical_stim_site_non_string_returns_none(self):
        result = RESPectCCEPDataset._canonical_stim_site(None)
        self.assertIsNone(result)

    # --- _to_json_1d / round-trip -----------------------------------------

    def test_to_json_1d_roundtrip(self):
        arr = np.array([1.0, 2.5, -3.0], dtype=np.float32)
        encoded = RESPectCCEPDataset._to_json_1d(arr)
        decoded = np.array(json.loads(encoded), dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, decoded, decimal=5)

    def test_to_json_1d_is_string(self):
        arr = np.zeros(10, dtype=np.float32)
        self.assertIsInstance(RESPectCCEPDataset._to_json_1d(arr), str)

    # --- _coord_tuple -----------------------------------------------------

    def test_coord_tuple_found(self):
        import pandas as pd
        df = pd.DataFrame({"name": ["P22", "P23"], "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]})
        result = RESPectCCEPDataset._coord_tuple(df, "P22")
        self.assertEqual(result, (1.0, 3.0, 5.0))

    def test_coord_tuple_not_found_returns_nans(self):
        import pandas as pd
        df = pd.DataFrame({"name": ["P22"], "x": [1.0], "y": [2.0], "z": [3.0]})
        x, y, z = RESPectCCEPDataset._coord_tuple(df, "P99")
        self.assertTrue(math.isnan(x))
        self.assertTrue(math.isnan(y))
        self.assertTrue(math.isnan(z))

    # --- _is_overlap ------------------------------------------------------

    def test_is_overlap_true(self):
        import pandas as pd
        a = pd.Series({"sample_start": 100, "sample_end": 200})
        b = pd.Series({"sample_start": 150, "sample_end": 250})
        self.assertTrue(RESPectCCEPDataset._is_overlap(a, b))

    def test_is_overlap_false_disjoint(self):
        import pandas as pd
        a = pd.Series({"sample_start": 100, "sample_end": 200})
        b = pd.Series({"sample_start": 300, "sample_end": 400})
        self.assertFalse(RESPectCCEPDataset._is_overlap(a, b))

    def test_is_overlap_touching_not_overlapping(self):
        """Intervals that touch at a single endpoint should not count as overlapping."""
        import pandas as pd
        a = pd.Series({"sample_start": 100, "sample_end": 200})
        b = pd.Series({"sample_start": 200, "sample_end": 300})
        # 100 < 300 AND 200 < 200 is False, so no overlap
        self.assertFalse(RESPectCCEPDataset._is_overlap(a, b))


# ===========================================================================
# 2. Dataset prepare_data tests (minimal synthetic BIDS tree, no MNE)
# ===========================================================================

class TestRESPectCCEPDatasetPrepareData(unittest.TestCase):
    """Tests for prepare_data using a synthetic BIDS directory.

    MNE/BrainVision I/O is avoided by patching _process_run to return
    synthetic pre-built rows, matching the behaviour of the real method.
    """

    # ---- helpers ---------------------------------------------------------

    def _make_bids_root(self, tmp: str) -> Path:
        """Write a minimal BIDS tree with participants.tsv and one run."""
        root = Path(tmp)
        import pandas as pd

        pd.DataFrame({
            "participant_id": ["sub-01", "sub-02"],
            "age": [25, 30],
            "sex": ["M", "F"],
        }).to_csv(root / "participants.tsv", sep="\t", index=False)

        ieeg_dir = root / "sub-01" / "ses-1" / "ieeg"
        ieeg_dir.mkdir(parents=True)
        # Minimal events.tsv — content does not matter because _process_run
        # is patched.
        (ieeg_dir / "sub-01_ses-1_task-SPES_run-01_events.tsv").write_text(
            "trial_type\tsample_start\tsample_end\telectrical_stimulation_site\n"
            "electrical_stimulation\t100\t200\tP29-P30\n"
        )
        return root

    def _synthetic_run_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "participant_id": "sub-01",
                "session_id": "ses-1",
                "run_id": "run-01",
                "age": 25,
                "sex": "M",
                "recording_electrode": "P22",
                "stim_1": "P29",
                "stim_2": "P30",
                "response_ts": _make_response_ts(seed=1),
                "response_ts_std": _make_response_ts(seed=101),
                "soz_label": 1,
                "recording_x": 10.0,
                "recording_y": 20.0,
                "recording_z": 30.0,
            },
            {
                "participant_id": "sub-01",
                "session_id": "ses-1",
                "run_id": "run-01",
                "age": 25,
                "sex": "M",
                "recording_electrode": "P23",
                "stim_1": "P29",
                "stim_2": "P30",
                "response_ts": _make_response_ts(seed=2),
                "response_ts_std": _make_response_ts(seed=102),
                "soz_label": 0,
                "recording_x": 11.0,
                "recording_y": 21.0,
                "recording_z": 31.0,
            },
        ]

    # ---- tests -----------------------------------------------------------

    def test_prepare_data_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_bids_root(tmp)
            with patch.object(
                RESPectCCEPDataset, "_process_run",
                return_value=self._synthetic_run_rows()
            ):
                inst = object.__new__(RESPectCCEPDataset)
                inst._pyhealth_csv = str(root / "respect_ccep_data-pyhealth.csv")
                inst.tmin_s = 0.0
                inst.tmax_s = 1.0
                inst.filter_low_hz = 1.0
                inst.filter_high_hz = 150.0
                inst.resample_hz = 512.0
                inst.min_trials = 5
                inst.prepare_data(str(root))

            import pandas as pd
            df = pd.read_csv(inst._pyhealth_csv)
            self.assertFalse(df.empty, "CSV should contain at least one row")

    def test_prepare_data_csv_has_required_columns(self):
        required_cols = {
            "participant_id", "session_id", "run_id",
            "recording_electrode", "stim_1", "stim_2",
            "response_ts", "soz_label",
            "recording_x", "recording_y", "recording_z",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_bids_root(tmp)
            with patch.object(
                RESPectCCEPDataset, "_process_run",
                return_value=self._synthetic_run_rows()
            ):
                inst = object.__new__(RESPectCCEPDataset)
                inst._pyhealth_csv = str(root / "respect_ccep_data-pyhealth.csv")
                inst.tmin_s = inst.tmax_s = inst.filter_low_hz = 0.0
                inst.filter_high_hz = inst.resample_hz = inst.min_trials = 1
                inst.prepare_data(str(root))

            import pandas as pd
            df = pd.read_csv(inst._pyhealth_csv)
            self.assertTrue(
                required_cols.issubset(set(df.columns)),
                f"Missing columns: {required_cols - set(df.columns)}",
            )

    def test_prepare_data_no_participants_tsv_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            inst = object.__new__(RESPectCCEPDataset)
            inst._pyhealth_csv = str(Path(tmp) / "out.csv")
            inst.tmin_s = inst.tmax_s = inst.filter_low_hz = 0.0
            inst.filter_high_hz = inst.resample_hz = inst.min_trials = 1
            with self.assertRaises(FileNotFoundError):
                inst.prepare_data(tmp)

    def test_prepare_data_skips_run_with_no_participant_id(self):
        """_process_run should not be called for events files lacking sub- entity."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import pandas as pd

            pd.DataFrame({"participant_id": ["sub-01"], "age": [25], "sex": ["M"]}).to_csv(
                root / "participants.tsv", sep="\t", index=False
            )
            # Create an events file with no sub- entity in the name.
            ieeg = root / "sub-01" / "ses-1" / "ieeg"
            ieeg.mkdir(parents=True)
            (ieeg / "events.tsv").write_text("trial_type\n")

            inst = object.__new__(RESPectCCEPDataset)
            inst._pyhealth_csv = str(root / "out.csv")
            inst.tmin_s = inst.tmax_s = inst.filter_low_hz = 0.0
            inst.filter_high_hz = inst.resample_hz = 1.0
            inst.min_trials = 1

            with patch.object(
                RESPectCCEPDataset, "_process_run", return_value=[]
            ) as mock_run:
                inst.prepare_data(str(root))
                mock_run.assert_not_called()

    def test_prepare_data_empty_tree_writes_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import pandas as pd
            pd.DataFrame({"participant_id": ["sub-01"]}).to_csv(
                root / "participants.tsv", sep="\t", index=False
            )
            inst = object.__new__(RESPectCCEPDataset)
            inst._pyhealth_csv = str(root / "out.csv")
            inst.tmin_s = inst.tmax_s = inst.filter_low_hz = 0.0
            inst.filter_high_hz = inst.resample_hz = 1.0
            inst.min_trials = 1
            inst.prepare_data(str(root))

            df = pd.read_csv(inst._pyhealth_csv)
            self.assertTrue(df.empty)


# ===========================================================================
# 3. Task private-helper unit tests
# ===========================================================================

class TestSeizureOnsetZoneLocalisationHelpers(unittest.TestCase):
    """Unit tests for the private helper functions in soz_localisation.py."""

    # --- _canonical_stim_key -------------------------------------------

    def test_canonical_stim_key_sorted(self):
        self.assertEqual(_canonical_stim_key("P30", "P29"), "P29-P30")

    def test_canonical_stim_key_already_sorted(self):
        self.assertEqual(_canonical_stim_key("P29", "P30"), "P29-P30")

    def test_canonical_stim_key_strips_whitespace(self):
        self.assertEqual(_canonical_stim_key(" P30 ", " P29 "), "P29-P30")

    def test_canonical_stim_key_alphabetic_sort(self):
        self.assertEqual(_canonical_stim_key("Da5", "Da1"), "Da1-Da5")

    # --- _safe_str --------------------------------------------------------

    def test_safe_str_normal(self):
        self.assertEqual(_safe_str("ses-1"), "ses-1")

    def test_safe_str_none_returns_none(self):
        self.assertIsNone(_safe_str(None))

    def test_safe_str_nan_returns_none(self):
        self.assertIsNone(_safe_str(float("nan")))

    def test_safe_str_int(self):
        self.assertEqual(_safe_str(42), "42")

    # --- _euclidean_mm ----------------------------------------------------

    def test_euclidean_mm_zero(self):
        self.assertAlmostEqual(_euclidean_mm((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), 0.0)

    def test_euclidean_mm_known_value(self):
        # 3-4-5 right triangle
        self.assertAlmostEqual(_euclidean_mm((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), 5.0)

    def test_euclidean_mm_symmetric(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        self.assertAlmostEqual(_euclidean_mm(a, b), _euclidean_mm(b, a))

    # --- _extract_coords --------------------------------------------------

    def test_extract_coords_valid(self):
        class FakeEvent:
            recording_x = 1.0
            recording_y = 2.0
            recording_z = 3.0

        result = _extract_coords(FakeEvent())
        self.assertEqual(result, (1.0, 2.0, 3.0))

    def test_extract_coords_nan_returns_none(self):
        class FakeEvent:
            recording_x = float("nan")
            recording_y = 2.0
            recording_z = 3.0

        self.assertIsNone(_extract_coords(FakeEvent()))

    def test_extract_coords_missing_attr_returns_none(self):
        class FakeEvent:
            pass  # no coordinate attributes at all

        self.assertIsNone(_extract_coords(FakeEvent()))

    def test_extract_coords_string_values_parsed(self):
        class FakeEvent:
            recording_x = "10.5"
            recording_y = "20.0"
            recording_z = "30.0"

        result = _extract_coords(FakeEvent())
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        self.assertAlmostEqual(result[0], 10.5)

    # --- _parse_response_ts -----------------------------------------------

    def test_parse_response_ts_json_string(self):
        data = [0.1, 0.2, 0.3]
        encoded = json.dumps(data)
        result = _parse_response_ts(encoded)
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, np.array(data, dtype=np.float32))

    def test_parse_response_ts_list_passthrough(self):
        result = _parse_response_ts([1.0, 2.0, 3.0])
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        self.assertEqual(result.shape, (3,))

    def test_parse_response_ts_none_returns_none(self):
        self.assertIsNone(_parse_response_ts(None))

    def test_parse_response_ts_invalid_json_returns_none(self):
        self.assertIsNone(_parse_response_ts("{not valid json"))

    def test_parse_response_ts_empty_list_returns_none(self):
        self.assertIsNone(_parse_response_ts("[]"))

    def test_parse_response_ts_2d_array_returns_none(self):
        encoded = json.dumps([[1.0, 2.0], [3.0, 4.0]])
        self.assertIsNone(_parse_response_ts(encoded))

    def test_parse_response_ts_numpy_array_passthrough(self):
        arr = np.array([0.5, 1.5], dtype=np.float64)
        result = _parse_response_ts(arr)
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        self.assertEqual(result.dtype, np.float32)

    # --- _stim_distance ---------------------------------------------------

    def test_stim_distance_known(self):
        rec_coords = (0.0, 0.0, 0.0)
        coord_lookup = {"P29": (3.0, 4.0, 0.0), "P30": (100.0, 0.0, 0.0)}
        # min distance is to P29 = 5.0
        dist = _stim_distance("P29-P30", rec_coords, coord_lookup)
        self.assertAlmostEqual(dist, 5.0)

    def test_stim_distance_rec_coords_none_returns_unknown(self):
        coord_lookup = {"P29": (3.0, 4.0, 0.0), "P30": (1.0, 0.0, 0.0)}
        dist = _stim_distance("P29-P30", None, coord_lookup)
        self.assertEqual(dist, _DIST_UNKNOWN)

    def test_stim_distance_missing_pole_returns_unknown(self):
        rec_coords = (0.0, 0.0, 0.0)
        coord_lookup = {"P29": (3.0, 4.0, 0.0)}  # P30 missing
        dist = _stim_distance("P29-P30", rec_coords, coord_lookup)
        self.assertEqual(dist, _DIST_UNKNOWN)

    def test_stim_distance_same_location_is_zero(self):
        rec_coords = (5.0, 5.0, 5.0)
        coord_lookup = {"P29": (5.0, 5.0, 5.0), "P30": (100.0, 100.0, 100.0)}
        dist = _stim_distance("P29-P30", rec_coords, coord_lookup)
        self.assertAlmostEqual(dist, 0.0)

    # --- _mean_responses --------------------------------------------------

    def test_mean_responses_single(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _mean_responses([arr])
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        np.testing.assert_array_almost_equal(result, arr)

    def test_mean_responses_multiple(self):
        a = np.array([1.0, 3.0], dtype=np.float32)
        b = np.array([3.0, 1.0], dtype=np.float32)
        result = _mean_responses([a, b])
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        np.testing.assert_array_almost_equal(result, np.array([2.0, 2.0]))

    def test_mean_responses_empty_returns_none(self):
        self.assertIsNone(_mean_responses([]))

    def test_mean_responses_length_mismatch_skips_bad(self):
        good = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        bad = np.array([1.0, 2.0], dtype=np.float32)  # wrong length
        result = _mean_responses([good, bad])
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        # Only the good array should survive
        np.testing.assert_array_almost_equal(result, good)

    def test_mean_responses_all_mismatched_returns_none(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _mean_responses([a, b])
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        # First array is the reference; second is mismatched and dropped.
        np.testing.assert_array_almost_equal(result, a)

    def test_mean_responses_output_dtype_is_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = _mean_responses([arr])
        self.assertIsNotNone(result)
        assert result is not None  # narrows type for Pylance
        self.assertEqual(result.dtype, np.float32)


# ===========================================================================
# 4. SeizureOnsetZoneLocalisation.__call__ integration tests
# ===========================================================================

class TestSeizureOnsetZoneLocalisationTask(unittest.TestCase):
    """Tests for SeizureOnsetZoneLocalisation.__call__ using real Patient objects."""

    def setUp(self):
        self.task = SeizureOnsetZoneLocalisation()

    # ---- schema / instantiation ------------------------------------------

    def test_task_name(self):
        self.assertEqual(self.task.task_name, "SeizureOnsetZoneLocalisation")

    def test_input_schema_keys(self):
        self.assertIn("spes_responses", SeizureOnsetZoneLocalisation.input_schema)
        self.assertNotIn("stim_distances", SeizureOnsetZoneLocalisation.input_schema)
        self.assertEqual(SeizureOnsetZoneLocalisation.input_schema["spes_responses"], "tensor")

    def test_output_schema(self):
        self.assertIn("soz_label", SeizureOnsetZoneLocalisation.output_schema)
        self.assertEqual(SeizureOnsetZoneLocalisation.output_schema["soz_label"], "binary")

    def test_default_min_distance(self):
        self.assertEqual(self.task.min_distance_mm, _DEFAULT_MIN_DISTANCE_MM)

    def test_custom_min_distance(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=5.0)
        self.assertEqual(task.min_distance_mm, 5.0)

    def test_negative_min_distance_raises(self):
        with self.assertRaises(ValueError):
            SeizureOnsetZoneLocalisation(min_distance_mm=-1.0)

    # ---- empty / missing event cases ------------------------------------

    def test_empty_patient_returns_empty_list(self):
        patient = _make_patient([], patient_id="sub-empty")
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_wrong_event_type_returns_empty_list(self):
        """Events stored under a different type should be invisible to the task."""
        rows = [_row("P22", "P29", "P30", soz_label=1, x=0.0, y=0.0, z=0.0)]
        df = _make_event_rows(rows, event_type="wrong_type")
        patient = Patient(patient_id="sub-01", data_source=df)
        result = self.task(patient)
        self.assertEqual(result, [])

    # ---- basic sample generation ----------------------------------------

    def test_single_electrode_one_stim_site_no_filtering(self):
        """One electrode, one stim site, distance filtering disabled."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30", soz_label=0, x=0.0, y=0.0, z=0.0)]
        patient = _make_patient(rows)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s["patient_id"], _PATIENT_ID)
        self.assertEqual(s["electrode_id"], "P22")
        self.assertEqual(s["soz_label"], 0)

    def test_soz_electrode_label_is_one(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30", soz_label=1, x=0.0, y=0.0, z=0.0)]
        patient = _make_patient(rows)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["soz_label"], 1)

    def test_multiple_electrodes_produce_one_sample_each(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=1, x=0.0, y=0.0, z=0.0),
            _row("P23", "P29", "P30", soz_label=0, x=5.0, y=0.0, z=0.0),
            _row("P24", "P29", "P30", soz_label=0, x=10.0, y=0.0, z=0.0),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        self.assertEqual(len(samples), 3)
        electrode_ids = {s["electrode_id"] for s in samples}
        self.assertEqual(electrode_ids, {"P22", "P23", "P24"})

    def test_multiple_stim_sites_stacked_into_channels(self):
        """Multiple stim sites for the same recording electrode → C > 1."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=0, x=0.0, y=0.0, z=0.0, response_seed=1),
            _row("P22", "P31", "P32", soz_label=0, x=0.0, y=0.0, z=0.0, response_seed=2),
            _row("P22", "Da1", "Da2", soz_label=0, x=0.0, y=0.0, z=0.0, response_seed=3),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        values = samples[0]["spes_responses"]
        shape = values.shape
        self.assertEqual(shape[0], 3, "Expected 3 input channels (one per stim site)")
        self.assertEqual(shape[2], _T + 1)

    # ---- output schema compliance ----------------------------------------

    def test_sample_keys_complete(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30", soz_label=0)]
        patient = _make_patient(rows)
        sample = task(patient)[0]
        required = {"patient_id", "visit_id", "electrode_id",
                    "spes_responses", "soz_label"}
        self.assertTrue(required.issubset(set(sample.keys())))

    def test_spes_responses_dtype_float32(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30")]
        patient = _make_patient(rows)
        sample = task(patient)[0]
        values = sample["spes_responses"]
        self.assertEqual(values.dtype, np.float32)

    def test_spes_responses_is_3d(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30")]
        patient = _make_patient(rows)
        sample = task(patient)[0]
        values = sample["spes_responses"]
        self.assertEqual(values.ndim, 3)

    def test_channel_shape_dimensions(self):
        """C dim of spes_responses is correctly formatted with 2 modes (mean, std)."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", response_seed=1),
            _row("P22", "P31", "P32", response_seed=2),
        ]
        patient = _make_patient(rows)
        sample = task(patient)[0]
        response_values = sample["spes_responses"]
        self.assertEqual(
            response_values.shape[0],
            2, # C = 2 specific rows were added (stim sites)
        )
        self.assertEqual(
            response_values.shape[1],
            2, # 2 means modes (mean, std)
        )
        self.assertEqual(
            response_values.shape[2],
            _T + 1, # T + 1 for distance prepended
        )



    def test_visit_id_populated_from_session(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30", session_id="ses-2")]
        patient = _make_patient(rows)
        sample = task(patient)[0]
        self.assertEqual(sample["visit_id"], "ses-2")

    def test_visit_id_unknown_when_none(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30", session_id="")]
        # Override session_id to None via a modified row
        row = _row("P22", "P29", "P30")
        row["session_id"] = None
        patient = _make_patient([row])
        sample = task(patient)[0]
        self.assertEqual(sample["visit_id"], "unknown")

    # ---- distance filtering ---------------------------------------------

    def test_distance_filter_excludes_close_stim_site(self):
        """Stim site at distance 5 mm should be excluded when threshold is 13 mm."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=13.0)
        rows = [
            # Recording electrode at origin; P29 is 5 mm away — too close.
            # P29 coords will be in coord_lookup because P29 appears as a recording electrode below.
            _row("P29", "Da1", "Da2", soz_label=0, x=3.0, y=4.0, z=0.0, response_seed=10),  # dist=5
            # This is the electrode under test.
            _row("P22", "P29", "P30", soz_label=0, x=0.0, y=0.0, z=0.0, response_seed=1),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        # P22's only stim site is P29-P30; P29 is at distance 5 < 13 mm → excluded.
        # P30 has no coords → unknown → included. Min distance to pair is min(5, unknown)
        # → unknown sentinel triggers inclusion (distance unknown → include).
        # But P29 IS in coord_lookup, P30 is NOT → _DIST_UNKNOWN → include.
        p22_samples = [s for s in samples if s["electrode_id"] == "P22"]
        self.assertEqual(len(p22_samples), 1, "P22 sample should exist (P30 missing coords → unknown → include)")

    def test_distance_filter_excludes_when_both_poles_known_and_close(self):
        """When both poles of a stim pair are known and both are < threshold, exclude."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=13.0)
        rows = [
            # P29 at (3,4,0) → 5 mm from origin
            _row("P29", "Da1", "Da2", soz_label=0, x=3.0, y=4.0, z=0.0, response_seed=10),
            # P30 at (0,5,0) → 5 mm from origin
            _row("P30", "Da1", "Da2", soz_label=0, x=0.0, y=5.0, z=0.0, response_seed=11),
            # Target electrode P22 at origin; stim site P29-P30 → both poles < 13 mm → exclude
            _row("P22", "P29", "P30", soz_label=0, x=0.0, y=0.0, z=0.0, response_seed=1),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        # P22 has no valid stim sites after filtering → no sample
        p22_samples = [s for s in samples if s["electrode_id"] == "P22"]
        self.assertEqual(len(p22_samples), 0,
                         "P22 should produce no sample: its only stim pair is too close")

    def test_distance_filter_zero_includes_all(self):
        """min_distance_mm=0 should include every stim site."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=0, x=0.0, y=0.0, z=0.0),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        values = samples[0]["spes_responses"]
        self.assertEqual(values.shape[0], 1)

    def test_distance_filter_unknown_coords_includes_site(self):
        """NaN coordinates → distance unknown → site is always included."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=50.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=0,
                 x=float("nan"), y=float("nan"), z=float("nan")),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        # No coords → _DIST_UNKNOWN → included despite large threshold
        self.assertEqual(len(samples), 1)

    def test_distances_clamped_to_zero_when_unknown(self):
        """Entries in spes_responses[c,0,0] should be 0.0 when coordinates were unavailable."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=0,
                 x=float("nan"), y=float("nan"), z=float("nan")),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        values = samples[0]["spes_responses"]
        self.assertEqual(values[0, 0, 0], 0.0)

    # ---- stim key canonicalisation --------------------------------------

    def test_reversed_stim_pair_merged_with_forward(self):
        """P30-P29 and P29-P30 are the same site and should map to one channel."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=0, response_seed=1),
            _row("P22", "P30", "P29", soz_label=0, response_seed=2),  # same site, reversed
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        # Two responses for the same canonical key → averaged → still 1 channel
        values = samples[0]["spes_responses"]
        self.assertEqual(values.shape[0], 1)

    # ---- response timeseries handling -----------------------------------

    def test_null_response_ts_row_skipped(self):
        """A row with response_ts=None should not contribute a channel."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        row1 = _row("P22", "P29", "P30", soz_label=0, response_seed=1)
        row2 = _row("P22", "Da1", "Da2", soz_label=0)
        row2["response_ts"] = None  # explicitly null

        df = _make_event_rows([row1, row2])
        patient = Patient(patient_id=_PATIENT_ID, data_source=df)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        # Only row1's stim site contributes → C=1
        values = samples[0]["spes_responses"]
        self.assertEqual(values.shape[0], 1)

    def test_timeseries_length_preserved(self):
        """T dimension of spes_responses must equal the serialised timeseries length."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30", response_seed=7)]
        patient = _make_patient(rows)
        sample = task(patient)[0]
        values = sample["spes_responses"]
        self.assertEqual(values.shape[2], _T + 1)

    # ---- class balance tracking -----------------------------------------

    def test_mixed_soz_labels(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [
            _row("P22", "P29", "P30", soz_label=1),
            _row("P23", "P29", "P30", soz_label=0),
            _row("P24", "P29", "P30", soz_label=0),
        ]
        patient = _make_patient(rows)
        samples = task(patient)
        labels = [s["soz_label"] for s in samples]
        self.assertEqual(labels.count(1), 1)
        self.assertEqual(labels.count(0), 2)

    # ---- patient_id propagation ----------------------------------------

    def test_patient_id_in_samples(self):
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)
        rows = [_row("P22", "P29", "P30")]
        patient = _make_patient(rows, patient_id="sub-99")
        sample = task(patient)[0]
        self.assertEqual(sample["patient_id"], "sub-99")

    # ---- variable channel count across patients -------------------------

    def test_variable_channel_count_between_patients(self):
        """Two patients with different electrode counts produce different C values."""
        task = SeizureOnsetZoneLocalisation(min_distance_mm=0.0)

        # Patient A: 1 stim site → C=1
        rows_a = [_row("P22", "P29", "P30", response_seed=1)]
        patient_a = _make_patient(rows_a, patient_id="sub-01")

        # Patient B: 3 stim sites → C=3
        rows_b = [
            _row("P22", "P29", "P30", response_seed=2),
            _row("P22", "P31", "P32", response_seed=3),
            _row("P22", "Da1", "Da2", response_seed=4),
        ]
        patient_b = _make_patient(rows_b, patient_id="sub-02")

        values_a = task(patient_a)[0]["spes_responses"]
        values_b = task(patient_b)[0]["spes_responses"]
        c_a = values_a.shape[0]
        c_b = values_b.shape[0]
        self.assertNotEqual(c_a, c_b,
                            "C should differ between patients with different channel counts")
        self.assertEqual(c_a, 1)
        self.assertEqual(c_b, 3)


if __name__ == "__main__":
    unittest.main()
