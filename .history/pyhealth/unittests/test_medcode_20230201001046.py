import unittest
from pyhealth.medcode import InnerMap


class TestInnerMap(unittest.TestCase):
    def setUp(self) -> None:
        self.inner_map = InnerMap.load("ICD9CM")
    
    