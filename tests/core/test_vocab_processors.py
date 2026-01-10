import unittest
from typing import List, Dict, Any
from pyhealth.processors import (
    SequenceProcessor,
    StageNetProcessor,
    NestedSequenceProcessor,
    DeepNestedSequenceProcessor,
)

class TestVocabProcessors(unittest.TestCase):
    """
    Test remove and retain methods for processors with vocabulary support.
    covers: SequenceProcessor, StageNetProcessor, NestedSequenceProcessor, DeepNestedSequenceProcessor
    """

    def test_sequence_processor_remove(self):
        processor = SequenceProcessor()
        samples = [
            {"codes": ["A", "B", "C"]},
            {"codes": ["D", "E"]},
        ]
        processor.fit(samples, "codes")
        original_vocab = set(processor.code_vocab.keys())
        self.assertTrue({"A", "B", "C", "D", "E"}.issubset(original_vocab))

        # Remove "A" and "B"
        processor.remove({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        self.assertIn("<unk>", new_vocab)
        self.assertIn("<pad>", new_vocab)
        
        # Verify processing still works (A and B become <unk>)
        res = processor.process(["A", "C"])
        unk_idx = processor.code_vocab["<unk>"]
        c_idx = processor.code_vocab["C"]
        self.assertEqual(res[0].item(), unk_idx)
        self.assertEqual(res[1].item(), c_idx)

    def test_sequence_processor_retain(self):
        processor = SequenceProcessor()
        samples = [
            {"codes": ["A", "B", "C"]},
            {"codes": ["D", "E"]},
        ]
        processor.fit(samples, "codes")
        
        # Retain "A" and "B"
        processor.retain({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)
        self.assertNotIn("E", new_vocab)
        self.assertIn("<unk>", new_vocab)
        self.assertIn("<pad>", new_vocab)

    def test_stagenet_processor_remove(self):
        processor = StageNetProcessor()
        # Flat codes
        samples = [
            {"data": ([0.0, 1.0, 2.0], ["A", "B", "C"])},
            {"data": ([0.0, 1.0], ["D", "E"])},
        ]
        processor.fit(samples, "data")
        
        processor.remove({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        # Test processing
        time, res = processor.process(([0.0, 1.0], ["A", "C"]))
        unk_idx = processor.code_vocab["<unk>"]
        c_idx = processor.code_vocab["C"]
        self.assertEqual(res[0].item(), unk_idx)
        self.assertEqual(res[1].item(), c_idx)

    def test_stagenet_processor_retain(self):
        processor = StageNetProcessor()
        # Nested codes
        samples = [
            {"data": ([0.0, 1.0], [["A", "B"], ["C"]])},
            {"data": ([0.0], [["D", "E"]])},
        ]
        processor.fit(samples, "data")
        
        processor.retain({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)

    def test_nested_sequence_processor_remove(self):
        processor = NestedSequenceProcessor()
        samples = [
            {"codes": [["A", "B"], ["C", "D"]]},
            {"codes": [["E"]]},
        ]
        processor.fit(samples, "codes")
        
        processor.remove({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        res = processor.process([["A", "C"]])
        unk_idx = processor.code_vocab["<unk>"]
        c_idx = processor.code_vocab["C"]
        # res shape (1, max_inner_len)
        # First code in first visit should be unk, second C
        # Note: processor padds to max_inner_len
        visit = res[0]
        self.assertEqual(visit[0].item(), unk_idx)
        self.assertEqual(visit[1].item(), c_idx)

    def test_nested_sequence_processor_retain(self):
        processor = NestedSequenceProcessor()
        samples = [
            {"codes": [["A", "B"], ["C", "D"]]},
            {"codes": [["E"]]},
        ]
        processor.fit(samples, "codes")
        
        processor.retain({"E"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("E", new_vocab)
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)

    def test_deep_nested_sequence_processor_remove(self):
        processor = DeepNestedSequenceProcessor()
        samples = [
            {"codes": [[["A", "B"], ["C"]], [["D"]]]},
        ]
        processor.fit(samples, "codes")
        
        processor.remove({"A"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        
        # Test process
        # Input [[[A]]] -> [[[<unk>]]] (padded)
        res = processor.process([[["A"]]])
        unk_idx = processor.code_vocab["<unk>"]
        # res shape (1, max_visits, max_codes)
        # first group, first visit, first code
        self.assertEqual(res[0, 0, 0].item(), unk_idx)

    def test_deep_nested_sequence_processor_retain(self):
        processor = DeepNestedSequenceProcessor()
        samples = [
            {"codes": [[["A", "B"], ["C"]], [["D"]]]},
        ]
        processor.fit(samples, "codes")
        
        processor.retain({"A"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)

if __name__ == "__main__":
    unittest.main()
