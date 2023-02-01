import unittest
import sys 
import os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current)))

from pyhealth.medcode import InnerMap, CrossMap

class TestInnerMap(unittest.TestCase):

    def setUp(self):
        map_name = "ICD9CM"
        self.inner_map = InnerMap.load(map_name)

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
        self.assertEqual(
            self.inner_map.get_descendants("428"),
            ['428.0', '428.1', '428.2', '428.3', '428.4', '428.9', '428.20', '428.21', '428.22', '428.23', '428.30', '428.31', '428.32', '428.33', '428.40', '428.41', '428.42', '428.43'],            
            msg="get_descendants function of InnerMap failed"        
            )


class TestInnerMapATC(unittest.TestCase):
    def setUp(self):
        self.inner_map = InnerMap.load("ATC")
    
    def test_lookup(self):
        self.assertEqual(
            self.inner_map.lookup("M01AE51"),
            'ibuprofen, combinations',
            msg="lookup function of InnerMap (ATC) failed"
            )
        self.assertEqual(
            self.inner_map.lookup("M01AE51", "drugbank_id"),
            'DB01050',
            msg="lookup function of InnerMap (ATC) failed"
            )
        self.assertEqual(
            self.inner_map.lookup("M01AE51", "smiles"),
            'CC(C)CC1=CC=C(C=C1)C(C)C(O)=O',
            msg="lookup function of InnerMap (ATC) failed"
            )
        
            
    def test_convert(self):
        self.assertEqual(
            self.inner_map.convert("A12CE02", level=3),
            "A12C",
            msg="convert function of InnerMap (ATC) failed"
        )


class TestCrossMap(unittest.TestCase):
    def setUp(self):
        self.cross_map = CrossMap.load(source_vocabulary="ICD9CM", target_vocabulary="CCSCM")

    def test_map(self):
        self.assertEqual(
            self.cross_map.map("428.0"),
            ["108"],
            msg="map function of CrossMap failed"
        )


if __name__ == "__main__":
    unittest.main()
