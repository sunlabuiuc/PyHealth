"""Every series task schema spec must instantiate like SampleDataset does.

Regression tests for the W5 cache-A/B blocker: the multimodal MIMIC-IV
tasks declared ``"labs_mask": ("stagenet_tensor", {"forward_fill": False})``
while ``StageNetTensorProcessor`` took no kwargs, so ``set_task`` raised
``TypeError`` on every real cohort. The existing series tests only called
the task ``__call__`` (which never touches processor specs), so nothing
caught it. These tests instantiate each ``input_schema``/``output_schema``
spec through the exact resolution helper ``SampleDataset``/``SampleBuilder``
use, for every task class in the series.
"""

import unittest


def _all_specs(task):
    return list(task.input_schema.items()) + list(task.output_schema.items())


def _instantiate(spec):
    from pyhealth.datasets.sample_dataset import SampleBuilder

    # Same resolution logic SampleDataset.fit / SampleBuilder.fit use to
    # build processors (string alias, (alias-or-class, kwargs), class,
    # or instance). `self` is unused by the helper.
    return SampleBuilder._get_processor_instance(None, spec)


class TestMultimodalTaskSchemaSpecs(unittest.TestCase):
    def _check_task(self, task):
        specs = _all_specs(task)
        self.assertGreater(len(specs), 0)
        for field, spec in specs:
            with self.subTest(task=type(task).__name__, field=field):
                processor = _instantiate(spec)
                self.assertIsNotNone(processor)

    def test_notes_labs_specs_instantiate(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        self._check_task(NotesLabsMIMIC4())

    def test_notes_labs_icd_specs_instantiate(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        # The leaky-ICD ablation variant adds the ("stagenet", {...}) spec.
        self._check_task(NotesLabsMIMIC4(include_icd=True))

    def test_labs_specs_instantiate(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        self._check_task(LabsMIMIC4())


if __name__ == "__main__":
    unittest.main()
