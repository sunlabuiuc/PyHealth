import unittest
from pyhealth.medcode import InnerMap


class TestInnerMap(unittest.TestCase):
    def setUp(self):
        self.inner_map = InnerMap.load("ICD9CM")

    def test_contain(self):
        self.assertTrue("428.0" in self.inner_map, )
    
    def test_lookup(self):
        self.assertEqual(
            self.inner_map.lookup("428.0"),
            'Congestive heart failure, unspecified'
            )
        
        return

    def test_get_ancestors(self):
        return

    def test_get_descendants(self):
        return 


class TestInnerMapATC(unittest.TestCase):
    def setUp(self):
        self.inner_map = InnerMap.load("ATC")
    
    def test_lookup(self):
        return

    def test_get_ancestors(self):
        return

    def test_get_descendants(self):
        return 
