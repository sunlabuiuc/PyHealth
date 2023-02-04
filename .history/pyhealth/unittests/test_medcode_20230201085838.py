import unittest
import sys 
import os
import logging

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current)))

from pyhealth.medcode import InnerMap

class TestInnerMap(unittest.TestCase):
    logging.info("Starting testing InnerMap")

    def setUp(self):
        map_name = "ICD9CM"
        logging.info(f"loading {map_name}")
        self.inner_map = InnerMap.load(map_name)

    def test_contain(self):
        self.assertTrue(
            "428.0" in self.inner_map, 
            msg="contain function of InnerMap failed"
            )
        logging.info("Done testing contain function ...")
    
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

# class TestInnerMapATC(unittest.TestCase):
#     def setUp(self):
#         self.inner_map = InnerMap.load("ATC")
    
#     def test_lookup(self):
#         return

#     def test_get_ancestors(self):
#         return

#     def test_get_descendants(self):
#         return 


if __name__ == "__main__":
    unittest.main()
