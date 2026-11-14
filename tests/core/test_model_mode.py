"""Tests that ``model.mode`` is the label processor's kind on every model.

An ``output_schema`` entry may be a string (``"multiclass"``), a processor class
(``MultiClassLabelProcessor``), a processor instance, or a ``(name, kwargs)``
tuple. Each gives the same fitted label processor, so ``model.mode``,
``Trainer.evaluate`` and the calibration methods work the same with every form.
See https://github.com/sunlabuiuc/PyHealth/issues/914.
"""

import unittest

import torch

from pyhealth.calib.calibration import DirichletCalibration, TemperatureScaling
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
    def test_mode_is_the_label_processors_for_every_schema_form(self):
        forms = [
            "multiclass",
            MultiClassLabelProcessor,
            MultiClassLabelProcessor(),
            ("multiclass", {}),
            (MultiClassLabelProcessor, {}),
        ]
        for spec in forms:
            with self.subTest(spec=spec):
                model = _TinyModel(dataset=_dataset(spec))
                self.assertIsInstance(model.label_processor, MultiClassLabelProcessor)
                self.assertEqual(model.mode, "multiclass")

    def test_mode_is_read_only(self):
        model = _TinyModel(dataset=_dataset("multiclass"))
        with self.assertRaises(AttributeError):
            model.mode = "binary"

    def test_no_label_processor_has_no_mode(self):
        model = _TinyModel(dataset=None)
        self.assertEqual(model.label_processors, {})
        self.assertIsNone(model.mode)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.label_processor

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

    def test_calibration_targets_keep_the_label_dtype(self):
        # The calibration losses get y_true as the label processor emits it:
        # class indices (long) for multiclass, floats for binary.
        loader_batch = lambda ds: next(iter(get_dataloader(ds, batch_size=9)))
        multiclass = _dataset("multiclass")
        for calibrator in (TemperatureScaling, DirichletCalibration):
            with self.subTest(calibrator=calibrator.__name__):
                torch.manual_seed(0)
                cal_model = calibrator(RNN(dataset=multiclass))
                cal_model.calibrate(cal_dataset=multiclass)
                out = cal_model(**loader_batch(multiclass))
                self.assertTrue(torch.isfinite(out["loss"]))
        samples = [
            {"patient_id": f"p{i}", "conditions": ["cond-1", "cond-2"][: 1 + i % 2],
             "label": i % 2}
            for i in range(8)
        ]
        binary = create_sample_dataset(
            samples, {"conditions": "sequence"}, {"label": "binary"}
        )
        cal_model = TemperatureScaling(RNN(dataset=binary))
        cal_model.calibrate(cal_dataset=binary)
        self.assertTrue(torch.isfinite(cal_model(**loader_batch(binary))["loss"]))


if __name__ == "__main__":
    unittest.main()
