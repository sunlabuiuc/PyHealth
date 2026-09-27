"""Tests that ``model.mode`` is always a plain mode string.

An ``output_schema`` entry may be a string (``"multiclass"``), a processor class
(``MultiClassLabelProcessor``), a processor instance, or a ``(name, kwargs)``
tuple. ``Trainer.evaluate`` and the calibration methods compare ``model.mode``
against strings, so every form must resolve to the same string on every model.
See https://github.com/sunlabuiuc/PyHealth/issues/914.
"""

import unittest

import torch

from pyhealth.calib.calibration import TemperatureScaling
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    GAT,
    GCN,
    GRASP,
    MICRON,
    MLP,
    RETAIN,
    RNN,
    TCN,
    AdaCare,
    Agent,
    BaseModel,
    CNN,
    ConCare,
    Deepr,
    EHRMamba,
    JambaEHR,
    LogisticRegression,
    MultimodalAdaCare,
    MultimodalRETAIN,
    MultimodalRNN,
    StageAttentionNet,
    StageNet,
    Transformer,
)
from pyhealth.processors import MultiClassLabelProcessor, MultiLabelProcessor
from pyhealth.trainer import Trainer

# Every model that runs on a plain {"conditions": "sequence"} input with a
# multiclass label. MICRON only supports multilabel and is tested separately.
MODELS = [
    AdaCare,
    Agent,
    CNN,
    ConCare,
    Deepr,
    EHRMamba,
    GAT,
    GCN,
    GRASP,
    JambaEHR,
    LogisticRegression,
    MLP,
    MultimodalAdaCare,
    MultimodalRETAIN,
    MultimodalRNN,
    RETAIN,
    RNN,
    StageAttentionNet,
    StageNet,
    TCN,
    Transformer,
]

SCHEMA_FORMS = {
    "string": "multiclass",
    "class": MultiClassLabelProcessor,
    "tuple": ("multiclass", {}),
}


def _dataset(label_spec, multilabel=False):
    samples = [
        {
            "patient_id": f"patient-{i}",
            "visit_id": f"visit-{i}",
            "conditions": ["cond-1", "cond-2", "cond-3", "cond-4"][: 1 + i % 4],
            "label": [["x"], ["y"], ["x", "y"]][i % 3] if multilabel else i % 3,
        }
        for i in range(9)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "sequence"},
        output_schema={"label": label_spec},
        dataset_name="test_model_mode",
    )


class _TinyModel(BaseModel):
    def forward(self, **kwargs):
        raise NotImplementedError


class TestModelMode(unittest.TestCase):
    def test_mode_setter_resolves_every_schema_form(self):
        model = _TinyModel(dataset=None)
        forms = [
            ("MultiClass", "multiclass"),
            (MultiClassLabelProcessor, "multiclass"),
            (MultiClassLabelProcessor(), "multiclass"),
            (("multiclass", {}), "multiclass"),
            ((MultiClassLabelProcessor, {}), "multiclass"),
            (None, None),
        ]
        for value, expected in forms:
            with self.subTest(value=value):
                model.mode = value
                self.assertEqual(model.mode, expected)

    def test_mode_setter_warns_and_clears_on_non_label_entries(self):
        model = _TinyModel(dataset=None)
        for value in ("sequence", "not-a-processor"):
            with self.subTest(value=value):
                with self.assertLogs("pyhealth.models.base_model", "WARNING"):
                    model.mode = value
                self.assertIsNone(model.mode)

    def test_models_evaluate_with_every_schema_form(self):
        for form, spec in SCHEMA_FORMS.items():
            dataset = _dataset(spec)
            loader = get_dataloader(dataset, batch_size=9, shuffle=False)
            for model_cls in MODELS:
                with self.subTest(model=model_cls.__name__, schema=form):
                    torch.manual_seed(0)
                    model = model_cls(dataset=dataset)
                    self.assertEqual(model.mode, "multiclass")
                    trainer = Trainer(
                        model=model, metrics=["accuracy"], enable_logging=False
                    )
                    scores = trainer.evaluate(loader)
                    self.assertIn("accuracy", scores)

    def test_micron_evaluates_with_every_multilabel_schema_form(self):
        forms = {
            "string": "multilabel",
            "class": MultiLabelProcessor,
            "tuple": ("multilabel", {}),
        }
        for form, spec in forms.items():
            with self.subTest(schema=form):
                dataset = _dataset(spec, multilabel=True)
                model = MICRON(dataset=dataset)
                self.assertEqual(model.mode, "multilabel")
                trainer = Trainer(
                    model=model, metrics=["jaccard_samples"], enable_logging=False
                )
                loader = get_dataloader(dataset, batch_size=9, shuffle=False)
                self.assertIn("jaccard_samples", trainer.evaluate(loader))

    def test_calibration_with_processor_class_schema(self):
        dataset = _dataset(MultiClassLabelProcessor)
        cal_model = TemperatureScaling(RNN(dataset=dataset))
        cal_model.calibrate(cal_dataset=dataset)


if __name__ == "__main__":
    unittest.main()
