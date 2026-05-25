import unittest


class TestStageNetTimeTensor(unittest.TestCase):
    def test_stagenet_processor_time_never_none(self):
        from pyhealth.processors.stagenet_processor import StageNetProcessor

        proc = StageNetProcessor()
        proc.fit([{"codes": (None, ["A", "B", "C"])}], "codes")
        time_tensor, value_tensor = proc.process((None, ["A", "B", "C"]))

        self.assertIsNotNone(time_tensor)
        self.assertEqual(time_tensor.shape[0], value_tensor.shape[0])

    def test_stagenet_tensor_processor_time_never_none(self):
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor()
        proc.fit([{"labs": (None, [[1.0, 2.0], [3.0, 4.0]])}], "labs")
        time_tensor, value_tensor = proc.process((None, [[1.0, 2.0], [3.0, 4.0]]))

        self.assertIsNotNone(time_tensor)
        self.assertEqual(time_tensor.shape[0], value_tensor.shape[0])


if __name__ == "__main__":
    unittest.main()
