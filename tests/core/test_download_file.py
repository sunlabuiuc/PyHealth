"""Tests for pyhealth.utils.download_file (atomic downloads with a timeout)."""

import inspect
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from pyhealth.medcode import utils as medcode_utils
from pyhealth.utils import download_file

PAYLOAD = b"code,name\n001,Cholera\n" * 100


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/ok.csv":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        elif self.path == "/truncated.csv":
            # Promise more bytes than we send, then drop the connection.
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD) * 2))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            self.wfile.flush()
            self.connection.close()
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


class TestDownloadFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp.name, "data.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_downloads_complete_file(self):
        self.assertEqual(download_file(f"{self.base}/ok.csv", self.dest), self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_http_error_leaves_no_file(self):
        with self.assertRaises(Exception):
            download_file(f"{self.base}/missing.csv", self.dest)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_interrupted_download_leaves_no_file(self):
        with self.assertRaises(Exception):
            download_file(f"{self.base}/truncated.csv", self.dest)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_existing_file_survives_failed_refresh(self):
        with open(self.dest, "wb") as f:
            f.write(b"old")
        with self.assertRaises(Exception):
            download_file(f"{self.base}/truncated.csv", self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), b"old")

    def test_medcode_json_uses_cache_by_default(self):
        default = inspect.signature(
            medcode_utils.download_and_read_json
        ).parameters["refresh_cache"].default
        self.assertFalse(default)


if __name__ == "__main__":
    unittest.main()
