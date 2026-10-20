"""Tests for the package-level ``pyhealth`` logger setup."""

import io
import logging
import unittest

import pyhealth


class TestPyHealthLogging(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("pyhealth")
        self.root = logging.getLogger()
        self.root_stream = io.StringIO()
        self.root_handler = logging.StreamHandler(self.root_stream)
        self.root.addHandler(self.root_handler)
        self.pyhealth_stream = io.StringIO()
        self.old_stream = pyhealth.handler.setStream(self.pyhealth_stream)

    def tearDown(self):
        pyhealth.handler.setStream(self.old_stream)
        self.root.removeHandler(self.root_handler)

    def test_line_printed_once_when_root_has_handler(self):
        # Jupyter and Colab configure a root handler; the line must not
        # appear there as well as on PyHealth's own handler.
        logging.getLogger("pyhealth.datasets").info("hello once")
        self.assertEqual(self.pyhealth_stream.getvalue(), "hello once\n")
        self.assertEqual(self.root_stream.getvalue(), "")

    def test_info_level_is_still_printed(self):
        logging.getLogger("pyhealth.trainer").info("epoch 1")
        self.assertIn("epoch 1", self.pyhealth_stream.getvalue())
        logging.getLogger("pyhealth.trainer").debug("hidden")
        self.assertNotIn("hidden", self.pyhealth_stream.getvalue())

    def test_opt_back_in_to_root_logging(self):
        self.logger.removeHandler(pyhealth.handler)
        self.logger.propagate = True
        try:
            logging.getLogger("pyhealth.models").info("via root")
            self.assertEqual(self.root_stream.getvalue(), "via root\n")
            self.assertEqual(self.pyhealth_stream.getvalue(), "")
        finally:
            self.logger.propagate = False
            self.logger.addHandler(pyhealth.handler)

    def test_trainer_file_log_still_written(self):
        import tempfile
        from pathlib import Path

        from pyhealth.trainer import set_logger

        with tempfile.TemporaryDirectory() as tmp:
            trainer_logger = logging.getLogger("pyhealth.trainer")
            before = list(trainer_logger.handlers)
            set_logger(tmp)
            try:
                trainer_logger.info("to file")
            finally:
                for h in trainer_logger.handlers:
                    if h not in before:
                        h.close()
                        trainer_logger.removeHandler(h)
            self.assertIn("to file", (Path(tmp) / "log.txt").read_text())
        self.assertEqual(self.pyhealth_stream.getvalue().count("to file"), 1)


if __name__ == "__main__":
    unittest.main()
