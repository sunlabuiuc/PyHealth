from typing import Type
import unittest
import logging


class BaseTestCase(unittest.TestCase):
    @staticmethod
    def _setup_logging(level: int = logging.INFO):
        logging.basicConfig(level=level)

    @classmethod
    def _set_debug(cls: Type):
        cls._setup_logging(logging.DEBUG)
        print()
        print('_' * 80)
