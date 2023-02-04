import unittest
from pyhealth.medcode import InnerMap


class TestInnerMap(unittest.TestCase):
    def setUp(self):
        self.inner_map = InnerMap.load("ICD9CM")

    def test_contain(self):
        self.assertTrue(
            "428.0" in self.inner_map, 
            msg="contain function of InnerMap failed"
            )
    
    def test_lookup(self):
        self.assertEqual(
            self.inner_map.lookup("428.0"),
            'Congestive heart failure, unspecified',
            msg="lookup function of InnerMap failed"
            )

    def test_get_ancestors(self):
        self.assertEqual(
            self.inner_map.get_ancestors("428.0"),
            ['428', '420-429.99', '390-459.99', '001-999.99'],
            msg="get_ancestors function of InnerMap failed"
            )

    def test_get_descendants(self):
        self.assert
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
