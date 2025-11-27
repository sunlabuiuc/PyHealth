import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pyhealth.datasets.base_dataset import StreamingParquetWriter
from tests.base import BaseTestCase


class TestStreamingParquetWriter(BaseTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.schema = pa.schema(
            [
                ("id", pa.int64()),
                ("value", pa.string()),
            ]
        )
        self.output_path = Path(self.tmpdir.name) / "stream.parquet"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_append_flush_close_and_context_manager(self):
        rows = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
            {"id": 4, "value": "d"},
        ]

        with StreamingParquetWriter(
            self.output_path, self.schema, chunk_size=2
        ) as writer:
            # First two appends trigger an automatic flush due to chunk_size=2.
            writer.append(rows[0])
            writer.append(rows[1])

            # Flush again after adding a third row to ensure flushing appends
            # rather than overwriting previous row groups.
            writer.append(rows[2])
            writer.flush()

            # Leave data in the buffer to verify close() flushes it.
            writer.append(rows[3])

        # Context manager should have closed and flushed remaining buffered rows.
        self.assertTrue(self.output_path.exists())

        written_rows = pq.read_table(self.output_path).to_pylist()

        # Every append should be present as a distinct row in order.
        self.assertEqual(written_rows, rows)
