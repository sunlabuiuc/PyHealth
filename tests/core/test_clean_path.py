import ntpath
import unittest
from unittest.mock import patch

from pyhealth.datasets import base_dataset
from pyhealth.datasets.base_dataset import clean_path

# (input URL, expected output). A URL path is POSIX whatever the host OS.
URL_CASES = [
    (
        "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/PATIENTS.csv.gz",
        "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/PATIENTS.csv.gz",
    ),
    (
        "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III//PATIENTS.csv",
        "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/PATIENTS.csv",
    ),
    (
        "https://example.org/a/b/../c.csv",
        "https://example.org/a/c.csv",
    ),
    (
        "s3://bucket/key/file.csv.gz",
        "s3://bucket/key/file.csv.gz",
    ),
]


class TestCleanPathURL(unittest.TestCase):
    def test_url_path_is_normalised_as_posix(self):
        for url, expected in URL_CASES:
            with self.subTest(url=url):
                self.assertEqual(clean_path(url), expected)

    def test_url_path_is_unaffected_by_windows_os_path(self):
        # Reproduce the Windows behaviour on any OS: ntpath.normpath rewrites
        # "/" as "\\", which must not leak into a URL.
        with patch.object(base_dataset.os, "path", ntpath):
            for url, expected in URL_CASES:
                with self.subTest(url=url):
                    self.assertEqual(clean_path(url), expected)


if __name__ == "__main__":
    unittest.main()
