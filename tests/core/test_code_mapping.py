import unittest
from unittest.mock import patch, MagicMock

from pyhealth.processors import SequenceProcessor


class TestCodeMappingSequenceProcessor(unittest.TestCase):
    """Tests for the code_mapping feature in SequenceProcessor.

    Verifies backward compatibility when code_mapping=None (default)
    and correct vocabulary collapsing when a mapping is provided.
    """

    # -- Backward compatibility (code_mapping=None) --

    def test_default_no_mapping_fit(self):
        """Without code_mapping, fit builds vocabulary from raw codes."""
        proc = SequenceProcessor()
        samples = [
            {"codes": ["A", "B", "C"]},
            {"codes": ["D", "E"]},
        ]
        proc.fit(samples, "codes")
        vocab_keys = set(proc.code_vocab.keys())
        self.assertTrue({"A", "B", "C", "D", "E"}.issubset(vocab_keys))
        # 5 codes + <pad> + <unk>
        self.assertEqual(proc.size(), 7)

    def test_default_no_mapping_process(self):
        """Without code_mapping, process returns correct indices."""
        proc = SequenceProcessor()
        samples = [{"codes": ["A", "B", "C"]}]
        proc.fit(samples, "codes")

        result = proc.process(["A", "B", "C"])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].item(), proc.code_vocab["A"])
        self.assertEqual(result[1].item(), proc.code_vocab["B"])
        self.assertEqual(result[2].item(), proc.code_vocab["C"])

    def test_default_no_mapping_unknown_token(self):
        """Without code_mapping, unknown tokens map to <unk>."""
        proc = SequenceProcessor()
        proc.fit([{"codes": ["A"]}], "codes")

        result = proc.process(["A", "UNKNOWN"])
        self.assertEqual(result[1].item(), proc.code_vocab["<unk>"])

    # -- With code_mapping --

    def _make_mock_crossmap(self, mapping_dict):
        """Helper: create a mock CrossMap that maps according to a dict."""
        mock_cm = MagicMock()
        mock_cm.map.side_effect = lambda code: mapping_dict.get(code, [])
        return mock_cm

    def test_mapping_collapses_vocabulary(self):
        """Multiple raw codes mapping to the same target produce one vocab entry."""
        mapping = {
            "428.0": ["108"],
            "428.1": ["108"],
            "428.20": ["108"],
            "250.00": ["49"],
            "250.01": ["49"],
        }
        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("ICD9CM", "CCSCM"))

        samples = [
            {"codes": ["428.0", "428.1", "428.20"]},
            {"codes": ["250.00", "250.01"]},
        ]
        proc.fit(samples, "codes")
        vocab_keys = set(proc.code_vocab.keys())
        # Should contain mapped codes, not raw codes
        self.assertIn("108", vocab_keys)
        self.assertIn("49", vocab_keys)
        self.assertNotIn("428.0", vocab_keys)
        self.assertNotIn("250.00", vocab_keys)
        # 2 mapped codes + <pad> + <unk>
        self.assertEqual(proc.size(), 4)

    def test_mapping_process_uses_mapped_codes(self):
        """process() maps codes before looking up indices."""
        mapping = {
            "428.0": ["108"],
            "250.00": ["49"],
        }
        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("ICD9CM", "CCSCM"))

        proc.fit([{"codes": ["428.0", "250.00"]}], "codes")
        result = proc.process(["428.0", "250.00"])

        self.assertEqual(result[0].item(), proc.code_vocab["108"])
        self.assertEqual(result[1].item(), proc.code_vocab["49"])

    def test_unmapped_codes_fall_through(self):
        """Codes without a mapping are kept as-is (fallback to raw code)."""
        mapping = {
            "428.0": ["108"],
            # "UNKNOWN_CODE" has no mapping
        }
        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("ICD9CM", "CCSCM"))

        samples = [{"codes": ["428.0", "UNKNOWN_CODE"]}]
        proc.fit(samples, "codes")
        vocab_keys = set(proc.code_vocab.keys())
        # Mapped code
        self.assertIn("108", vocab_keys)
        # Unmapped code kept as-is
        self.assertIn("UNKNOWN_CODE", vocab_keys)

        result = proc.process(["428.0", "UNKNOWN_CODE"])
        self.assertEqual(result[0].item(), proc.code_vocab["108"])
        self.assertEqual(result[1].item(), proc.code_vocab["UNKNOWN_CODE"])

    def test_one_to_many_mapping(self):
        """A single code mapping to multiple targets expands correctly."""
        mapping = {
            "COMBO_CODE": ["TARGET_A", "TARGET_B"],
        }
        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("SRC", "TGT"))

        proc.fit([{"codes": ["COMBO_CODE"]}], "codes")
        vocab_keys = set(proc.code_vocab.keys())
        self.assertIn("TARGET_A", vocab_keys)
        self.assertIn("TARGET_B", vocab_keys)
        self.assertNotIn("COMBO_CODE", vocab_keys)

        result = proc.process(["COMBO_CODE"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].item(), proc.code_vocab["TARGET_A"])
        self.assertEqual(result[1].item(), proc.code_vocab["TARGET_B"])

    def test_vocab_size_reduction(self):
        """Demonstrates the real-world impact: many raw codes → few mapped codes."""
        # Simulate 100 raw ICD9 codes all mapping to 5 CCS categories
        mapping = {}
        for i in range(100):
            category = str(i % 5)
            mapping[f"ICD9_{i}"] = [category]

        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("ICD9CM", "CCSCM"))

        samples = [{"codes": [f"ICD9_{i}" for i in range(100)]}]
        proc.fit(samples, "codes")
        # 5 CCS categories + <pad> + <unk> = 7
        self.assertEqual(proc.size(), 7)

    def test_none_tokens_skipped_with_mapping(self):
        """None tokens are still skipped when mapping is active."""
        mapping = {"A": ["X"]}
        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("SRC", "TGT"))

        proc.fit([{"codes": ["A", None, "A"]}], "codes")
        # Only "X" + <pad> + <unk>
        self.assertEqual(proc.size(), 3)

    def test_repr_unchanged(self):
        """__repr__ still works with or without mapping."""
        proc = SequenceProcessor()
        self.assertIn("code_vocab_size=2", repr(proc))

    def test_remove_retain_add_work_with_mapping(self):
        """Existing vocab methods (remove, retain, add) work on mapped codes."""
        mapping = {
            "428.0": ["108"],
            "250.00": ["49"],
            "401.9": ["99"],
        }
        with patch(
            "pyhealth.medcode.CrossMap"
        ) as MockCrossMap:
            MockCrossMap.load.return_value = self._make_mock_crossmap(mapping)
            proc = SequenceProcessor(code_mapping=("ICD9CM", "CCSCM"))

        proc.fit([{"codes": ["428.0", "250.00", "401.9"]}], "codes")
        self.assertEqual(proc.size(), 5)  # 3 codes + pad + unk

        # Remove mapped code "108"
        proc.remove({"108"})
        self.assertNotIn("108", proc.code_vocab)
        self.assertIn("49", proc.code_vocab)

        # Retain only "49"
        proc.retain({"49"})
        self.assertIn("49", proc.code_vocab)
        self.assertNotIn("99", proc.code_vocab)

        # Add new code
        proc.add({"NEW"})
        self.assertIn("NEW", proc.code_vocab)


class TestTaskCodeMappingInit(unittest.TestCase):
    """Tests for passing code_mapping as a task __init__ argument."""

    def test_no_code_mapping_leaves_schema_unchanged(self):
        """Tasks without code_mapping keep simple string schema."""
        from pyhealth.tasks import MortalityPredictionMIMIC3

        task = MortalityPredictionMIMIC3()
        self.assertEqual(task.input_schema["conditions"], "sequence")
        self.assertEqual(task.input_schema["procedures"], "sequence")
        self.assertEqual(task.input_schema["drugs"], "sequence")

    def test_code_mapping_upgrades_schema_to_tuples(self):
        """code_mapping converts string schema entries to (type, kwargs) tuples."""
        from pyhealth.tasks import MortalityPredictionMIMIC3

        task = MortalityPredictionMIMIC3(
            code_mapping={
                "conditions": ("ICD9CM", "CCSCM"),
                "procedures": ("ICD9PROC", "CCSPROC"),
            }
        )
        # Mapped fields become tuples
        self.assertEqual(
            task.input_schema["conditions"],
            ("sequence", {"code_mapping": ("ICD9CM", "CCSCM")}),
        )
        self.assertEqual(
            task.input_schema["procedures"],
            ("sequence", {"code_mapping": ("ICD9PROC", "CCSPROC")}),
        )
        # Unmapped fields stay as strings
        self.assertEqual(task.input_schema["drugs"], "sequence")

    def test_code_mapping_ignores_unknown_fields(self):
        """code_mapping silently ignores fields not in input_schema."""
        from pyhealth.tasks import MortalityPredictionMIMIC3

        task = MortalityPredictionMIMIC3(
            code_mapping={"nonexistent_field": ("SRC", "TGT")}
        )
        # Schema unchanged
        self.assertEqual(task.input_schema["conditions"], "sequence")

    def test_code_mapping_does_not_mutate_class_attribute(self):
        """code_mapping creates instance schema, doesn't modify class attribute."""
        from pyhealth.tasks import MortalityPredictionMIMIC3

        task_with = MortalityPredictionMIMIC3(
            code_mapping={"conditions": ("ICD9CM", "CCSCM")}
        )
        task_without = MortalityPredictionMIMIC3()

        # Class attribute unchanged
        self.assertEqual(task_without.input_schema["conditions"], "sequence")
        # Instance attribute changed
        self.assertIsInstance(task_with.input_schema["conditions"], tuple)

    def test_code_mapping_with_readmission_task(self):
        """Tasks with existing __init__ params also accept code_mapping."""
        from pyhealth.tasks import ReadmissionPredictionMIMIC3

        task = ReadmissionPredictionMIMIC3(
            code_mapping={"conditions": ("ICD9CM", "CCSCM")}
        )
        self.assertEqual(
            task.input_schema["conditions"],
            ("sequence", {"code_mapping": ("ICD9CM", "CCSCM")}),
        )
        # Original params still work
        from datetime import timedelta
        self.assertEqual(task.window, timedelta(days=15))

    def test_code_mapping_mimic4(self):
        """code_mapping works with MIMIC4 mortality task too."""
        from pyhealth.tasks import MortalityPredictionMIMIC4

        task = MortalityPredictionMIMIC4(
            code_mapping={
                "conditions": ("ICD9CM", "CCSCM"),
                "drugs": ("NDC", "ATC"),
            }
        )
        self.assertEqual(
            task.input_schema["conditions"],
            ("sequence", {"code_mapping": ("ICD9CM", "CCSCM")}),
        )
        self.assertEqual(
            task.input_schema["drugs"],
            ("sequence", {"code_mapping": ("NDC", "ATC")}),
        )
        self.assertEqual(task.input_schema["procedures"], "sequence")


if __name__ == "__main__":
    unittest.main()
