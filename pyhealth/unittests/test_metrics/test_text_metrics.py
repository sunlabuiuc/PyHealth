import unittest
import numpy as np

from pyhealth.metrics import calculate_program_inconsistency_score

class TestTextMetrics(unittest.TestCase):

    def test_calculate_program_inconsistency_score(self):
        # Test case 1: Identical programs
        programs_identical = [
            "max_litset(gen_litset(table_name='labevents', literal_column='valuenum'))",
            "max_litset(gen_litset(table_name='labevents', literal_column='valuenum'))"
        ]
        pis_identical = calculate_program_inconsistency_score(programs_identical)
        self.assertEqual(pis_identical, 0.0, "PIS should be 0 for identical programs")

        # Test case 2: Varied programs
        programs_varied = [
            "count_entset(gen_entset_equal(table_name='patients', column_name='gender', value='F'))",
            "count_entset(gen_entset_equal(table_name='patients', column_name='gender', value='M'))",
            "count_entset(gen_entset_all(table_name='admissions'))"
        ]
        pis_varied = calculate_program_inconsistency_score(programs_varied)
        self.assertGreater(pis_varied, 0.0, "PIS should be > 0 for varied programs")

        # Test case 3: Very different programs
        programs_different = [
            "filter_entset_comparison(source_entset_df=A, attribute_col='age', comparison_operator='>', comparison_value=65)",
            "gen_entset_down(source_entset_df=B, target_table='diagnoses', desired_target_entity_col='icd_code')",
            "min_litset(C)"
        ]
        pis_different = calculate_program_inconsistency_score(programs_different)
        self.assertGreater(pis_different, pis_varied, "PIS should be higher for more different programs")

        # Test case 4: Single program
        program_single = [
            "count_entset(gen_entset_equal(table_name='patients', column_name='gender', value='F'))"
        ]
        pis_single = calculate_program_inconsistency_score(program_single)
        self.assertEqual(pis_single, 0.0, "PIS should be 0 for a single program")

        # Test case 5: Empty list
        programs_empty = []
        pis_empty = calculate_program_inconsistency_score(programs_empty)
        self.assertEqual(pis_empty, 0.0, "PIS should be 0 for an empty list")

        # Test case 6: Programs including an empty string
        programs_with_empty_str = [
            "count_entset(gen_entset_equal(table_name='patients', column_name='gender', value='F'))",
            "",
            "count_entset(gen_entset_all(table_name='admissions'))"
        ]
        pis_with_empty_str = calculate_program_inconsistency_score(programs_with_empty_str)
        self.assertGreater(pis_with_empty_str, 0.0, "PIS should be > 0 even with empty strings")

        # Test case 7: Two empty strings
        programs_two_empty = ["", ""]
        pis_two_empty = calculate_program_inconsistency_score(programs_two_empty)
        self.assertEqual(pis_two_empty, 0.0, "PIS should be 0 for two empty strings")

if __name__ == '__main__':
    unittest.main() 