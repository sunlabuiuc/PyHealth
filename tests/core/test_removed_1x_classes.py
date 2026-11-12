"""PyHealth 1.x classes fail at once with a pointer to the 2.x API.

They used to be stubs whose ``__init__`` only warned, so a subclass built
an object with none of its attributes set and failed much later (#1201).
"""

import unittest

from pyhealth.data import Visit
from pyhealth.datasets import (
    BaseEHRDataset,
    BaseSignalDataset,
    SampleEHRDataset,
    SampleSignalDataset,
)

REMOVED = {
    BaseEHRDataset: "BaseDataset",
    BaseSignalDataset: "BaseDataset",
    SampleEHRDataset: "SampleDataset",
    SampleSignalDataset: "SampleDataset",
    Visit: "Patient.get_events",
}


class TestRemoved1xClasses(unittest.TestCase):
    def test_instantiating_raises_with_replacement(self):
        for cls, replacement in REMOVED.items():
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(NotImplementedError) as ctx:
                    cls(root="data", tables=["DIAGNOSES_ICD"])
                message = str(ctx.exception)
                self.assertIn(f"{cls.__name__} was removed in PyHealth 2.0", message)
                self.assertIn(replacement, message)

    def test_subclass_raises_too(self):
        for cls in REMOVED:
            with self.subTest(cls=cls.__name__):
                subclass = type("Legacy" + cls.__name__, (cls,), {})
                with self.assertRaises(NotImplementedError):
                    subclass()


if __name__ == "__main__":
    unittest.main()
