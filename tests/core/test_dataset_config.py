import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from pyhealth.datasets.base_dataset import BaseDataset


class DemoDataset(BaseDataset):
    def __init__(self, root: str, config_path: str):
        super().__init__(root=root, tables=["demo"], config_path=config_path)


class DatasetConfigTest(unittest.TestCase):
    def test_wildcard_attributes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "demo.csv"
            csv_path.write_text(
                "subject_id,value,charttime\n1,foo,2020-01-01 00:00:00\n",
                encoding="utf-8",
            )

            config_path = root / "demo.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    version: "1.0"
                    tables:
                      demo:
                        file_path: "demo.csv"
                        patient_id: "subject_id"
                        timestamp: "charttime"
                        attributes:
                          - "*"
                    """
                ),
                encoding="utf-8",
            )

            dataset = DemoDataset(str(root), str(config_path))
            df = dataset.collected_global_event_df

            self.assertIn("demo/value", df.columns)
            self.assertNotIn("demo/subject_id", df.columns)

    def test_json_loader_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "demo.json"
            records = [
                {"text": "ctx", "summary": "sum"},
                {"text": "ctx2", "summary": "sum2", "labels": [{"label": "foo"}]},
            ]
            json_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            config_path = root / "demo_json.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    version: "1.0"
                    tables:
                      demo:
                        file_path: "demo.json"
                        patient_id: null
                        timestamp: null
                        attributes:
                          - "*"
                    """
                ),
                encoding="utf-8",
            )

            dataset = DemoDataset(str(root), str(config_path))
            df = dataset.collected_global_event_df

            self.assertIn("demo/text", df.columns)
            self.assertIn("demo/summary", df.columns)
            self.assertIn("demo/labels", df.columns)


if __name__ == "__main__":
    unittest.main()
