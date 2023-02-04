import unittest
from pyhealth.medcode import InnerMap


class TestInnerMap(unittest.TestCase):
    def setUp(self):
        self.inner_map = InnerMap.load("ICD9CM")
    
    def test_lookup(self):
        return

    def test_get_ancestors(self):
        return

    def test_get_descendants(self):
        return 
        