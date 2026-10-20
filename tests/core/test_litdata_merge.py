"""_litdata_merge must keep samples in worker order for any worker count.

Workers write contiguous sample ranges, and patient_to_index (built before
processing) assumes merged sample i is input sample i. Sorting the per-worker
index files as strings puts rank 10 before rank 2, so this uses 12 ranks.
"""

import tempfile
import unittest
from pathlib import Path

import litdata
from litdata.streaming.writer import BinaryWriter

from pyhealth.datasets.base_dataset import _litdata_merge
from pyhealth.utils import set_env

N_WORKERS = 12
PER_WORKER = 3


def _write_worker_outputs(cache_dir: Path) -> None:
    """Writes N_WORKERS contiguous ranges, as _proc_transform_fn does."""
    with set_env(DATA_OPTIMIZER_NUM_WORKERS=str(N_WORKERS)):
        for rank in range(N_WORKERS):
            with set_env(DATA_OPTIMIZER_GLOBAL_RANK=str(rank)):
                writer = BinaryWriter(cache_dir=str(cache_dir), chunk_bytes="64MB")
                for k in range(PER_WORKER):
                    i = rank * PER_WORKER + k
                    writer.add_item(k, {"position": i, "patient_id": f"p{i:03d}"})
                writer.done()


class TestLitdataMerge(unittest.TestCase):
    def test_merge_keeps_worker_order_beyond_ten_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _write_worker_outputs(cache)
            _litdata_merge(cache)
            positions = [s["position"] for s in litdata.StreamingDataset(str(cache))]
            self.assertEqual(positions, list(range(N_WORKERS * PER_WORKER)))
            self.assertEqual(
                sorted(p.name for p in cache.glob("*index.json")), ["index.json"]
            )

    def test_merge_is_noop_when_index_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _write_worker_outputs(cache)
            _litdata_merge(cache)
            before = (cache / "index.json").read_text()
            _litdata_merge(cache)
            self.assertEqual((cache / "index.json").read_text(), before)

    def test_merge_without_worker_indexes_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _litdata_merge(Path(tmp))


if __name__ == "__main__":
    unittest.main()
